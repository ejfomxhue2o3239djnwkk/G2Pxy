import argparse
import os
import os.path as osp
from copy import deepcopy
from re import A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv

from torch.optim.lr_scheduler import MultiStepLR

from utils import ensure_path, progress_bar
from models.utils import pprint, ensure_path
from torch.distributions import Categorical 
import copy
import random

import logging

from sklearn.model_selection import train_test_split 
from models import GCN_model2

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def cal_center_loss(matrix, indices):
    center_all = torch.tensor([])
    center_all = center_all.to(device)
    center_loss = torch.tensor([0.])
    center_loss = center_loss.to(device)
    for i in range(args.known_class):
        new_indices = torch.where(g.y[indices] == i)[0]
        new_matrix = matrix[new_indices]
        center = torch.mean(new_matrix, dim = 0)
        center = torch.unsqueeze(center, dim = 0)
        center_distance = new_matrix - center
        center_loss += torch.norm(center_distance)
        center_all = torch.cat([center_all, center], dim = 0)
    return center_all, center_loss / len(indices)
    
def find_margin_nodes_1(adj_matrix, indices):
    degree = torch.sum(adj_matrix, dim = 0)
    indices1 = torch.where(degree == 1)[0]
    new_indices = indices[indices1]
    return new_indices

def find_margin_nodes_2(net, seq, edge_index, adj_matrix, indices):
    res = net(seq, edge_index)[indices]
    values, indexs = torch.max(res, dim = 1) 
    indexs1, indexs2 = torch.sort(values)
    return indices[indexs2.cpu()[:100]]

def show_distribution(indices):
    labels = g.y[indices]
    label = torch.unique(labels)
    lens = []
    for item in label:
        length = len(torch.where(labels == item)[0])
        lens.append(length)
    return label, lens, len(labels)

def cal_cosine_distance(vector1, vector2):
    a = torch.sum(vector1 * vector2)
    vector1 = vector1 * vector1
    vector2 = vector2 * vector2
    b = torch.sqrt(torch.sum(vector1) * torch.sum(vector2))
    return 1. * a / b

def cal_E_distance(vector1, vector2):
    vector3 = vector1 - vector2
    vector4 = vector3 * vector3
    return torch.sqrt(torch.sum(vector4))

def cal_loss_5(logits, indices, margin = 0):
    res = torch.softmax(logits, dim = 1)
    loss = 0.
    for i in range(len(indices)):
        loss += res[i, new_labels[indices[i]]] - res[i, args.known_class]
    loss /= len(indices)
    if loss < margin:
        loss = 0
    return loss

def traindummy(epoch, net, g, train_indices):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = args.lr , momentum = 0.9, weight_decay = 5e-4)

    print('\n Epoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = args.alpha

    optimizer.zero_grad()
    beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
    
    pre_indices, later_indices = train_test_split(train_indices, test_size = 0.5)
    seq = g.x
    edge_index = g.edge_index
    pre2embeddings = pre2block(net, seq, edge_index)

    train_adj = adj[pre_indices]
    train_adj = train_adj[:,pre_indices]
    sum = 0
    sum_extra = 0 

    new_edge_index = deepcopy(edge_index)
    new_mixed_embeddings = torch.tensor([]).to(device)
    new_parents = torch.tensor([]).to(device)
    node_num = g.x.shape[0]
    centers_all, center_loss = cal_center_loss(pre2embeddings, pre_indices)
    
    margin_nodes_indices1 = find_margin_nodes_1(train_adj, pre_indices)
    margin_nodes_indices2 = find_margin_nodes_2(net, seq, edge_index, train_adj, pre_indices)    
    print(margin_nodes_indices1.shape)
    print(margin_nodes_indices2.shape)

    margin_nodes_indices = np.concatenate([margin_nodes_indices1, margin_nodes_indices2])
    print(margin_nodes_indices.shape)

    cos_distance_p_max = [] # 统计较大的正余弦距离
    cos_distance_p_min = [] # 统计较小的正余弦距离
    cos_distance_n = [] # 统计负余弦距离
    for i in range(len(pre_indices)): # 该两层循环针对生成的类间点
        for j in range(len(pre_indices)): # 针对cora以及citeseer数据集
            if train_adj[i][j] == 1  and g.y[pre_indices[i]] != g.y[pre_indices[j]]:
                p1 = pre_indices[i]
                p2 = pre_indices[j]
                mixed_embeddings = beta * pre2embeddings[p1] + (1 - beta) * pre2embeddings[p2]
                mixed_embedding = mixed_embeddings.unsqueeze(0)
                sum += 1 
                new_mixed_embeddings = torch.cat([new_mixed_embeddings, mixed_embedding], dim = 0) # 为了之后挑选一定比例的mix_embedding (new code)
                distance1 = cal_cosine_distance(pre2embeddings[p1], mixed_embedding)
                distance2 = cal_cosine_distance(pre2embeddings[p2], mixed_embedding)
                if distance1 > distance2:
                    cos_distance_p_max.append(distance1)
                    cos_distance_p_min.append(distance2)
                else:
                    cos_distance_p_max.append(distance2)
                    cos_distance_p_min.append(distance1)
                p3 = torch.randint(0, len(later_indices), (1,1))[0,0]
                distance3 = cal_cosine_distance(pre2embeddings[later_indices[p3]], mixed_embedding)
                cos_distance_n.append(distance3)

                new_parents = torch.cat([new_parents, torch.tensor([[p1], [p2]]).to(device)], dim = 1)
    
    for i in range(len(margin_nodes_indices)):
        index = (margin_nodes_indices[i])
        label = g.y[index]
        coef1 = torch.rand(1) + 1
        coef1 = coef1.to(device)
        coef2 = torch.rand(1) + 1
        coef2 = coef2.to(device)
        
        mix_embeddings = pre2embeddings[index] - centers_all[label]
        mix_embedding = mix_embeddings.unsqueeze(0)
        new_mixed_embeddings = torch.cat([new_mixed_embeddings, mixed_embedding], dim = 0)
        sum_extra += 1
        new_parents = torch.cat([new_parents, torch.tensor([[-1], [index]]).to(device)], dim = 1)
        distance1 = cal_cosine_distance(mix_embeddings, pre2embeddings[index])
        distance2 = cal_cosine_distance(mix_embeddings, centers_all[label])
        distance3 = cal_cosine_distance(pre2embeddings[index], centers_all[label])
        cos_distance_p_max.append(distance1)
        cos_distance_p_min.append(distance2)

    print(sum)
    print(sum_extra)
    
    print(new_mixed_embeddings.shape)
    
    num_new_point1.append(sum_extra)

    new_sum = 0
    cos_distance_p_max = torch.tensor(cos_distance_p_max)
    length = int((sum + sum_extra)* 1.0)
    new_indices = torch.sort(cos_distance_p_max)[1][:length]
    average_distance = torch.mean(cos_distance_p_max[new_indices])
    print(f'average_distance = {average_distance}')
    average_distance_list.append(average_distance)
    for i in range(length):
        new_sum += 1
        new_id = int(node_num + new_sum - 1)
        pre2embeddings = torch.cat([pre2embeddings, new_mixed_embeddings[new_indices[i]].unsqueeze(0)], dim = 0)
        p1 = int(new_parents[0][new_indices[i]])
        p2 = int(new_parents[1][new_indices[i]])
        if p1!=-1:
            a = torch.tensor([[new_id], [p1]]).to(device)
            b = torch.tensor([[new_id], [p2]]).to(device)
            c = torch.tensor([[p1], [new_id]]).to(device)
            d = torch.tensor([[p2], [new_id]]).to(device)
            new_edge_index = torch.cat([new_edge_index, a, b, c, d], dim = 1)
        else:
            b = torch.tensor([[new_id], [p2]]).to(device)
            d = torch.tensor([[p2], [new_id]]).to(device)
            new_edge_index = torch.cat([new_edge_index, b, d], dim = 1)
    
        
    sum = new_sum

    print(f'sum of new point :{sum}')


    cos_distance_p_min = torch.tensor(cos_distance_p_min)
    # cos_distance_p_max = torch.tensor(cos_distance_p_max)
    cos_distance_n = torch.tensor(cos_distance_n)
    max_p_distance_max = torch.max(cos_distance_p_max)
    max_p_distance_min = torch.min(cos_distance_p_max)
    max_p_distance_mean = torch.mean(cos_distance_p_max)
    min_p_distance_max = torch.max(cos_distance_p_min)
    min_p_distance_min = torch.min(cos_distance_p_min)
    min_p_distance_mean = torch.mean(cos_distance_p_min)
    max_n_distance = torch.max(cos_distance_n)
    min_n_distance = torch.min(cos_distance_n)
    mean_n_distance = torch.mean(cos_distance_n)
    print(max_p_distance_max, max_p_distance_min, max_p_distance_mean)
    print(min_p_distance_max, min_p_distance_min, min_p_distance_mean)
    print(max_n_distance, min_n_distance, mean_n_distance)
    num_new_point.append(sum)
    max_p_distance_max_list.append(max_p_distance_max)
    max_p_distance_min_list.append(max_p_distance_min)
    max_p_distance_mean_list.append(max_p_distance_mean)
    min_p_distance_max_list.append(min_p_distance_max)
    min_p_distance_min_list.append(min_p_distance_min)
    min_p_distance_mean_list.append(min_p_distance_mean)
    max_n_distance_list.append(max_n_distance)
    min_n_distance_list.append(min_n_distance)
    mean_n_distance_list.append(mean_n_distance)


    cos_distance_max_extra_mean = torch.mean(cos_distance_p_max[-sum_extra:])
    cos_distance_max_extra_median = torch.median(cos_distance_p_max[-sum_extra:])


    cos_distance_min_extra_mean = torch.mean(cos_distance_p_min[-sum_extra:])
    cos_distance_min_extra_median = torch.median(cos_distance_p_min[-sum_extra:])


    cos_distance_max_extra_mean_list.append(cos_distance_max_extra_mean)
    cos_distance_max_extra_median_list.append(cos_distance_max_extra_median)
    cos_distance_min_extra_mean_list.append(cos_distance_min_extra_mean)
    cos_distance_min_extra_median_list.append(cos_distance_min_extra_median)

    dummylogit = dummypredict(net, seq, edge_index)
    later_dummylogit = dummylogit[later_indices]
    pre_dummylogit = dummylogit[pre_indices]

    outputs = net(seq, edge_index)
    later_outputs = outputs[later_indices]
    pre_outputs = outputs[pre_indices]
    

    laterhalfoutput = torch.cat([later_outputs, later_dummylogit], dim = 1)
    prehalfoutput = torch.cat([pre_outputs, pre_dummylogit], dim = 1)
    
    new_output = torch.cat((latter2blockclf1(net, pre2embeddings, new_edge_index),
                               latter2blockclf2(net, pre2embeddings, new_edge_index)), 
                               dim = 1)

    f1 = torch.cat([new_output[pre_indices], new_output[-sum:]], dim = 0)
    print(f1.shape)
    f2 = torch.argmax(f1, dim = 1)
    print(f2.shape)

    dummpyoutputs = laterhalfoutput.clone()

    for i in range(len(dummpyoutputs)):
        nowlabel = new_labels[later_indices[i]]
        dummpyoutputs[i][nowlabel] = -1e9
    dummytargets = torch.ones_like(new_labels[later_indices]) * args.known_class

    pre_later_indices = np.concatenate([pre_indices, later_indices], axis=0)

    outputs = torch.cat((prehalfoutput, laterhalfoutput), dim = 0)
    loss1 = criterion(new_output[-sum : -sum_extra], (torch.ones(sum - sum_extra) * args.known_class).long().cuda())
    loss2 = criterion(laterhalfoutput, new_labels[later_indices].long().cuda())
    loss3 = criterion(dummpyoutputs, dummytargets.long().cuda())
    loss7 = criterion(new_output[-sum_extra:], (torch.ones(sum_extra) * args.known_class).long().cuda()) 
    loss =  0.01 * loss1 + args.lamda1 * loss2 + args.lamda2 * loss3 + 0.01 * loss7
    loss_list.append(loss.item())
    loss1_list.append(loss1.item())
    loss2_list.append(loss2.item())
    loss3_list.append(loss3.item())
    loss7_list.append(loss7.item())

    loss.backward()
    optimizer.step()
    _, predicted = outputs.max(1)
    total = len(train_indices)
    new_labels_gpu = new_labels.to(device)
    print(type(new_labels_gpu))
    correct = predicted.eq(new_labels_gpu[pre_later_indices]).sum().item()

    if args.shmode == False:
        print(f'epoch = {epoch}, correct = {correct}, total = {total}')
        print(f'Acc : {correct*1. / total}')
        print(f'L1 : {loss1.item()}')
        print(f'L2 : {loss2.item()}')
        print(f'L3 : {loss3.item()}')
        print(f'L7 : {loss7.item()}')
        
def cal_max(array):
    data = torch.tensor(array)
    data = torch.unsqueeze(data, dim = 0)
    max, epoch = torch.max(data, dim = 1)
    return max, epoch

def valdummy_extra(epoch, net, closeset_indices, openset_indices):
    net.eval()
    seq = g.x
    edge_index = g.edge_index
    outputs = net(seq, edge_index)

    close_outputs = outputs[closeset_indices]
    open_outputs = outputs[openset_indices]

    dummylogit = dummypredict(net, seq, edge_index)
    close_logit = dummylogit[closeset_indices]
    open_logit = dummylogit[openset_indices]
    close_logit = torch.max(close_logit, dim = 1)[0]
    open_logit = torch.max(open_logit, dim = 1)[0]
    close_logit = torch.unsqueeze(close_logit, dim = 1)
    open_logit = torch.unsqueeze(open_logit, dim = 1)
    closelogits = torch.cat([close_outputs, close_logit], dim = 1)
    openlogits = torch.cat([open_outputs, open_logit], dim = 1)
    
    close_correct = torch.max(closelogits, dim = 1)[1]
    res1 = close_correct.eq(new_labels[closeset_indices].to('cuda')).sum()
    close_ratio = 1. * res1 / len(closeset_indices)

    openlogits = torch.softmax(openlogits, dim = 1)
    open_correct = torch.max(openlogits, dim = 1)[1]
    open_true = torch.ones(len(openset_indices)) * args.known_class
    res2 = open_correct.eq(open_true.to('cuda')).sum()
    open_ratio = 1. * res2 / len(openset_indices)
    overall_ratio = 1. * (res1 + res2) / (len(openset_indices) + len(closeset_indices))
    acc_seen_valid.append(close_ratio.cpu().item())
    acc_unseen_valid.append(open_ratio.cpu().item())
    acc_overall_valid.append(overall_ratio.cpu().item())
    valid_f1_seen_score = f1_score(new_labels[closeset_indices].to('cpu').numpy(), close_correct.to('cpu').numpy(), average = 'macro')
    valid_f1_unseen_score = f1_score(open_true.to('cpu').numpy(), open_correct.to('cpu').numpy(), average = 'macro')
    valid_seen_f1.append(valid_f1_seen_score)
    valid_unseen_f1.append(valid_f1_unseen_score)
    targets = torch.cat([new_labels[closeset_indices], open_true])
    correct = torch.cat([close_correct, open_correct])
    valid_acc = accuracy_score(targets.to('cpu').numpy(), correct.to('cpu').numpy())
    valid_f1 = f1_score(targets.to('cpu').numpy(), correct.to('cpu').numpy(), average = 'macro')
    valid_f1_score.append(valid_f1)
    valid_f1_none = f1_score(targets.to('cpu').numpy(), correct.to('cpu').numpy(), average = None)
    open_true2 = torch.ones(len(openset_indices))
    abc = torch.where(open_correct != args.known_class)[0]
    abc1 = torch.where(open_correct == args.known_class)[0]
    open_correct[abc] = 0
    open_correct[abc1] = 1
    valid_f1_unseen_score2 = f1_score(open_true2.to('cpu').numpy(), open_correct.to('cpu').numpy(), average = 'binary')
    print(f'valid : accuracy_score = {valid_acc}, f1_score = {valid_f1}')

def test_extra(epoch, net, closeset_indices, openset_indices):
    net.eval()
    seq = g.x
    edge_index = g.edge_index
    outputs = net(seq, edge_index)

    close_outputs = outputs[closeset_indices]
    open_outputs = outputs[openset_indices]

    dummylogit = dummypredict(net, seq, edge_index)
    close_logit = dummylogit[closeset_indices]
    open_logit = dummylogit[openset_indices]
    close_logit = torch.max(close_logit, dim = 1)[0]
    open_logit = torch.max(open_logit, dim = 1)[0]
    close_logit = torch.unsqueeze(close_logit, dim = 1)
    open_logit = torch.unsqueeze(open_logit, dim = 1)
    closelogits = torch.cat([close_outputs, close_logit], dim = 1)
    openlogits = torch.cat([open_outputs, open_logit], dim = 1)
    
    close_correct = torch.max(closelogits, dim = 1)[1]
    res1 = close_correct.eq(new_labels[closeset_indices].to('cuda')).sum()
    close_ratio = 1. * res1 / len(closeset_indices)

    openlogits = torch.softmax(openlogits, dim = 1)
    open_correct = torch.max(openlogits, dim = 1)[1]
    open_true = torch.ones(len(openset_indices)) * args.known_class
    res2 = open_correct.eq(open_true.to('cuda')).sum()
    open_ratio = 1. * res2 / len(openset_indices)
    overall_ratio = 1. * (res1 + res2) / (len(openset_indices) + len(closeset_indices))
    acc_seen_test.append(close_ratio.cpu().item())
    acc_unseen_test.append(open_ratio.cpu().item())
    acc_overall_test.append(overall_ratio.cpu().item())
    test_f1_seen_score = f1_score(new_labels[closeset_indices].to('cpu').numpy(), close_correct.to('cpu').numpy(), average = 'macro')
    test_f1_unseen_score = f1_score(open_true.to('cpu').numpy(), open_correct.to('cpu').numpy(), average = 'macro')
    test_seen_f1.append(test_f1_seen_score)
    test_unseen_f1.append(test_f1_unseen_score)
    targets = torch.cat([new_labels[closeset_indices], open_true]) # 真实标签
    correct = torch.cat([close_correct, open_correct]) # 预测值
    test_acc = accuracy_score(targets.to('cpu').numpy(), correct.to('cpu').numpy())
    test_f1 = f1_score(targets.to('cpu').numpy(), correct.to('cpu').numpy(), average = 'macro')
    test_f1_score.append(test_f1)
    test_f1_none = f1_score(targets.to('cpu').numpy(), correct.to('cpu').numpy(), average = None)
    open_true2 = torch.ones(len(openset_indices))
    abc = torch.where(open_correct != args.known_class)[0]
    abc1 = torch.where(open_correct == args.known_class)[0]
    open_correct[abc] = 0
    open_correct[abc1] = 1
    test_f1_unseen_score2 = f1_score(open_true2.to('cpu').numpy(), open_correct.to('cpu').numpy(), average = 'binary')
    print(f'test: accuracy_score = {test_acc}, f1_score = {test_f1}')
    
def getmodel(args):
    print('==> Building model..')
    if args.backbone == "GCN_model2":
        net = GCN_model2(args.dim_feat, 512, 128, args.known_class)
    net = net.cuda()
    return net

def finetune_proser(epoch = 59):
    print('Now processing epoch', epoch)

    net = getmodel(args)
    print('==> Resuming from checkpoints..')
    print(model_path)
    assert os.path.isdir(model_path)

    modelname = 'Modelof_Epoch' + str(epoch) + '.pth'
    print(osp.join(model_path, save_path2, modelname))
    checkpoint = torch.load(osp.join(model_path, save_path2, modelname))

    net.load_state_dict(checkpoint['net'])
    if args.backbone == 'GCN_model2':
        net.clf2 = nn.Linear(128, args.dummynumber)
    net = net.cuda()

    Finetune_MAX_EPOCH = args.epoch
    wholebestacc = 0
    for finetune_epoch in range(Finetune_MAX_EPOCH):
        traindummy(finetune_epoch, net, g, train_indices)
        valdummy_extra(finetune_epoch, net, valid_closeset_indices, valid_openset_indices)
        test_extra(finetune_epoch, net, test_closeset_indices, test_openset_indices)

def dummypredict(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv1(seq, edge_index)
        out = net.Conv2(out, edge_index)
        out = net.clf2(out) 
    return out

def adjacency_matrix(edge_index, node_num):
    adjacency_matrix = torch.zeros(node_num, node_num)
    length = edge_index.shape[1]
    for i in range(length):
        x = edge_index[0][i]
        y = edge_index[1][i]
        adjacency_matrix[x,y] = 1
    return adjacency_matrix

def relabel_new(args, old_labels):
    old_label = torch.unique(old_labels)
    new_labels = deepcopy(old_labels)
    new_label = deepcopy(old_label)
    seen_label = new_label[:args.known_class]
    unseen_label = new_label[args.known_class:]
    seen_indices = []
    unseen_indices = []
    for label in unseen_label:
        indices = np.where(old_labels == label)[0]
        new_labels[indices] = args.known_class
    seen_indices = np.where(new_labels!=args.known_class)[0]
    unseen_indices = np.where(new_labels == args.known_class)[0]

    seen_train_indices, seen_test_indices = train_test_split(seen_indices, test_size = 1 - args.train_rate, random_state = args.seed)
    
    train_indices = seen_train_indices
    test_valid_indices = np.concatenate([seen_test_indices, unseen_indices], axis=0)
    valid_indices, test_indices = train_test_split(test_valid_indices, test_size = args.valid_rate / (1 - args.train_rate), random_state = args.seed)
    return g.y, seen_label, unseen_label, train_indices, test_indices, valid_indices

def pre2block(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv1(seq, edge_index)
    return out

def latter2blockclf1(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
        out = net.clf1(out)
    return out

def latter2blockclf2(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
        out = net.clf2(out)
    return out

def Conv1toConv2(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
    return out

def pretoConv2(net, seq, edge_index):
    out = net.Conv1(seq, edge_index)
    out = net.Conv2(out, edge_index)
    return out

def decompose(args, indices):
    indexs = []

    indices = torch.tensor(indices)

    for label in unseen_label:
        index = np.where(new_labels[indices] == label)[0]
        index = list(index)
        indexs = indexs + index

    old_indexs = list(np.arange(len(indices)))
    new_indexs = list(set(old_indexs) - set(indexs))

    new_indexs = torch.tensor(new_indexs, dtype = torch.long)
    indexs = torch.tensor(indexs, dtype = torch.long)

    closeset_indices = indices[new_indexs]
    openset_indices = indices[indexs]

    return closeset_indices, openset_indices

if __name__=="__main__":
    # 构建参数列表
    parser = argparse.ArgumentParser(description = 'PyTorch Cora Training')
    parser.add_argument('--dataset', default = 'cora')
    parser.add_argument('--lr', default = 0.1, type = float, help = 'learning rate')
    parser.add_argument('--model_type', default = 'Proser', type = str, help = 'Recognition Method')
    parser.add_argument('--known_class', default = 6, type = int, help = 'number of known class')
    parser.add_argument('--backbone', default = 'GCN_model2', type = str)
    parser.add_argument('--seed', default = '9', type = int, help = 'random seed for dataset generation')
    parser.add_argument('--lamda1', default = '1', type = float, help = 'trade-off between loss')
    parser.add_argument('--lamda2', default = '1', type = float, help = 'trade-off between loss')
    parser.add_argument('--alpha', default = '1', type = float)
    parser.add_argument('--dummynumber', default = 1, type = int, help = 'number of dummy label.')
    parser.add_argument('--shmode', action = 'store_true')
    parser.add_argument('--ratio1', default = 0.5, type = float)
    parser.add_argument('--train_rate', default = 0.7)
    parser.add_argument('--valid_rate', default = 0.1)
    parser.add_argument('--epoch', default = 50)
    args = parser.parse_args()

    args.seed = 100 

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pprint(vars(args)) # 将args的全部argument输出
    if args.dataset == 'cora':
        graph = Planetoid(root = '/vs code/vs code for python 3.9/GNN/simplt-PYG/datasets', name = 'Cora')
    g = graph[0]

    print("adj begin`")
    adj = adjacency_matrix(g.edge_index, g.x.shape[0])
    print("adj end")
    node_num =g.y.shape[0]
    edge_index = g.edge_index
    adjacent_matrix = adjacency_matrix(edge_index, node_num)
    edge_num = torch.sum(adjacent_matrix)
    args.dim_feat = g.x.shape[1]
    new_labels, seen_label, unseen_label, train_indices, valid_indices, test_indices = relabel_new(args, g.y)

    valid_closeset_indices, valid_openset_indices = decompose(args, valid_indices) #
    test_closeset_indices, test_openset_indices = decompose(args, test_indices) #

    g = g.to(device)
    save_path1 = osp.join('results', 'D{}-M{}-B{}-C'.format(args.dataset, args.model_type, args.backbone,)) 
    model_path = osp.join('results', 'D{}-M{}-B{}-C'.format(args.dataset, 'softmax', args.backbone,))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), seen_label, unseen_label, str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    print(model_path)
    ensure_path(save_path1, remove = False)
    ensure_path(args.save_path, remove = False)

    num_new_point = []
    max_p_distance_max_list = [] 
    max_p_distance_min_list = []
    max_p_distance_mean_list = []
    min_p_distance_max_list = []
    min_p_distance_min_list = []
    min_p_distance_mean_list = []
    max_n_distance_list = []
    min_n_distance_list = []
    mean_n_distance_list = []
    average_distance_list = []

    num_new_point1 = []
    cos_distance_max_extra_mean_list = []
    cos_distance_max_extra_median_list = []
    cos_distance_min_extra_mean_list = []
    cos_distance_min_extra_median_list = []

    acc_seen_valid = []
    acc_unseen_valid = []
    acc_overall_valid = []
    acc_seen_test = []
    acc_unseen_test = []
    acc_overall_test = []
    valid_f1_score = []
    test_f1_score = []
    valid_seen_f1 = []
    valid_unseen_f1 = []
    test_seen_f1 = []
    test_unseen_f1 = []
    valid_auc1_list = []
    test_auc1_list = []
    valid_auc2_list = []
    test_auc2_list = []
    valid_auc3_list = []
    test_auc3_list = []

    loss_list = []
    loss1_list = []
    loss2_list = []
    loss3_list = []
    loss4_list = []
    loss5_list = []
    loss6_list = []
    loss7_list = []

    x = np.arange(args.epoch) + 1

    globalacc = 0
    finetune_proser(59) 
    average_distance_list = torch.tensor(average_distance_list)

    max_acc_seen_valid, epoch1 = cal_max(acc_seen_valid)
    max_acc_unseen_valid, epoch2 = cal_max(acc_unseen_valid)
    max_acc_overall_valid, epoch3 = cal_max(acc_overall_valid)
    max_acc_seen_test, epoch4 = cal_max(acc_seen_test)
    max_acc_unseen_test, epoch5 = cal_max(acc_unseen_test)
    max_acc_overall_test, epoch6 = cal_max(acc_overall_test)
    max_f1_score_valid, epoch7 = cal_max(valid_f1_score)
    max_f1_score_test, epoch8 = cal_max(test_f1_score)

    print(max_acc_seen_valid, epoch1 + 1)
    print(max_acc_unseen_valid, epoch2 + 1)
    print(max_acc_overall_valid, epoch3 + 1)
    print(max_acc_seen_test, epoch4 + 1)
    print(max_acc_unseen_test, epoch5 + 1)
    print(max_acc_overall_test, epoch6 + 1)
    print(max_f1_score_valid, epoch7 + 1)    
    print(max_f1_score_test, epoch8 + 1)
    print(acc_overall_test[epoch3])
    print(f'valid_epoch : {epoch3}')
    print(f'test epoch : {epoch6}')
    print(f'max_seen_acc_by_valid = {acc_seen_test[epoch3]}')
    print(f'max_unseen_acc_by_valid = {acc_unseen_test[epoch3]}')
    print(f'max_overall_acc_by_valid = {acc_overall_test[epoch3]}')
    print(f'max_seen_acc_by_test = {acc_seen_test[epoch6]}')
    print(f'max_unseen_acc_by_test = {acc_unseen_test[epoch6]}')
    print(f'max_overall_acc_by_test = {acc_overall_test[epoch6]}')
    print(f'max_seen_f1_by_valid = {test_seen_f1[epoch3]}')
    print(f'max_unseen_f1_by_valid = {test_unseen_f1[epoch3]}')
    print(f'max_overall_f1_by_valid = {test_f1_score[epoch3]}')
    print(f'max_seen_f1_by_test = {test_seen_f1[epoch6]}')
    print(f'max_unseen_f1_by_test = {test_unseen_f1[epoch6]}')
    print(f'max_overall_f1_by_test = {test_f1_score[epoch6]}')



