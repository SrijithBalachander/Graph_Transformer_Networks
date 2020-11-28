import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score, plot_lines
import random
import time
import os
import networkx as nx

folder = "saved_models/"
plt_folder = "saved_plots/"

if not os.path.exists(plt_folder):
    os.mkdir(plt_folder)

def test(args):
    node_dim = args.node_dim
    num_channels = args.num_channels
    # lr = args.lr
    # weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    # adaptive_lr = args.adaptive_lr

    if args.ogb_arxiv:
        print("Using OGB arxiv")

    else:
        with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
            node_features = pickle.load(f)
        with open('data/'+args.dataset+'/edges.pkl','rb') as f:
            edges = pickle.load(f)
        with open('data/'+args.dataset+'/labels.pkl','rb') as f:
            labels = pickle.load(f)

    num_nodes = edges[0].shape[0]
    # print("Current Dataset : ",args.dataset)
    # print("Number of nodes : ",num_nodes)
    # print("Node Feature shape , sample: ", node_features.shape, node_features[0])
    # print("Edges shape , sample : ", edges[0].shape, edges[0])
    # print("labels shape , sample : ", labels[0].shape, labels[0])
    # print("Node Feature shape : ", node_features.shape)

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

    model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)

    print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model))

    loss = nn.CrossEntropyLoss()

    dur_test = time.time()
    best_model.eval()
    with torch.no_grad():
        test_loss, y_test,W = best_model.forward(A, node_features, test_node, test_target)
        # # print("W matrix shape", W.shape)
        # print("W type and length",type(W), len(W))
        # print("W[0] type and length", type(W[0]), len(W[0]))
        # print("W[0][0] type and shape", type(W[0][0]), W[0][0].shape)
        # # print("W[0][0][0] type and len", type(W[0][0][0]),len(W[0][0][0]))
        # print(W[0][0])
        test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
    # best_test_loss = min(best_test_loss,test_loss.detach().cpu().numpy())
    # best_test_f1 = max(best_test_f1,test_f1)
    dur_test = (time.time() - dur_test)/60.0

    print('Test - Loss: {}, Macro_F1: {}, Duration: {}'.format(test_loss.detach().cpu().numpy(), test_f1, dur_test))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    # parser.add_argument('--epoch', type=int, default=40,
    #                     help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    # parser.add_argument('--lr', type=float, default=0.005,
    #                     help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0.001,
    #                     help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    # parser.add_argument('--adaptive_lr', type=str, default='false',
    #                     help='adaptive learning rate')

    parser.add_argument('--experiment_name', 
                        help='Where to store logs and models')
    # parser.add_argument('--log_csv', 
    #                     help='csv file storing logs')

    parser.add_argument('--ogb_arxiv', 
                        help='Using OGB arxiv dataset')

    # parser.add_argument

    parser.add_argument('--total_runs', type=int, default=1,
                        help='number of runs')

    parser.add_argument('--saved_model', required=True, 
                        help="path to saved_model to evaluation")

    args = parser.parse_args()

    if not args.experiment_name:
        # exp = random.randint()
        args.experiment_name = f'GTN-Experiment'

    # os.makedirs(f'./saved_models/{args.experiment_name}', exist_ok=True)

    print(args)

    test(args)