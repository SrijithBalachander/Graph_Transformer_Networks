import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score, plot_lines, plot_testf1
import random
import time
import os
import networkx as nx
import datetime

folder = "saved_models/"
plt_folder = "saved_plots/"

if not os.path.exists(plt_folder):
    os.mkdir(plt_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')

    parser.add_argument('--experiment_name', 
                        help='Where to store logs and models')
    # parser.add_argument('--log_csv', 
    #                     help='csv file storing logs')

    parser.add_argument('--ogb_arxiv', 
                        help='Using OGB arxiv dataset')

    # parser.add_argument

    parser.add_argument('--total_runs', type=int, default=1,
                        help='number of runs')

    args = parser.parse_args()

    if not args.experiment_name:
        # exp = random.randint()
        args.experiment_name = f'GTN-Experiment'

    os.makedirs(f'./saved_models/{args.experiment_name}', exist_ok=True)
    os.makedirs(f'./results/', exist_ok=True)

    print(args)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

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
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)
    
    num_classes = torch.max(train_target).item()+1
    final_f1 = 0

    total_f1_list = []
    total_loss_list = []
    best_f1_list = []

    # log_file = open(f'log_{args.experiment_name}_run{l}.csv',"w")

    total_runs = args.total_runs

    for l in range(total_runs):
        log_csv = open(f'./results/csv_{args.experiment_name}_run{str(l)}.csv',"w")
        log_file = open(f'./results/log_{args.experiment_name}_run{str(l)}.txt',"a")
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params':model.weight},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        loss = nn.CrossEntropyLoss()

        print(model)

        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        best_ep = 0
        dur_train = 0
        dur_valid = 0
        dur_test = 0
        dur_total = 0
        val_f1_list = []
        val_loss_list = []
        train_f1_list = []
        train_loss_list = []
        
        for i in range(epochs):
            ep_time = time.time()
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1)
            # print("Duration for param setting : ", time.time() - ep_time)
            dur_train = time.time()
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(A, node_features, train_node, train_target)
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            dur_train = (time.time() - dur_train)/60.0
            train_f1_list.append(train_f1*100)
            train_loss_list.append(loss)
            print('Train - Loss: {}, Macro_F1: {}, Duration: {}'.format(loss.detach().cpu().numpy(), train_f1, dur_train))
            loss.backward()
            optimizer.step()
            # if i == 0:
            #     print("loss : ",loss)
            #     print("Target : ", train_target)
            #     print("Pred : ", y_train)
            dur_valid = time.time()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                dur_valid = (time.time() - dur_valid)/60.0
                val_loss_list.append(val_loss)
                val_f1_list.append(val_f1*100)
                print('Valid - Loss: {}, Macro_F1: {}, Duration: {}'.format(val_loss.detach().cpu().numpy(), val_f1, dur_valid))
                # dur_test = time.time()
                # test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                # test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                # dur_test = (time.time() - dur_test)/60.0
                # print('Test - Loss: {}, Macro_F1: {}, Duration: {}'.format(test_loss.detach().cpu().numpy(), test_f1, dur_test))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                torch.save(model.state_dict(), f'./saved_models/{args.experiment_name}/best_val_f1_run_{l}.pth')
                best_model = model
                # best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_ep = i+1
                # best_test_f1 = test_f1

            # print('Test - Loss: {}, Macro_F1: {}, Duration: {}'.format(test_loss.detach().cpu().numpy(), test_f1, dur_test))
            ep_time = (time.time() - ep_time)/60.0
            print("Epoch Duration : {}\n".format(ep_time))
            dur_total += ep_time

        total_f1_list.append(val_f1_list)
        total_loss_list.append(val_loss_list)
        dur_test = time.time()
        best_model.eval()
        with torch.no_grad():
            test_loss, y_test,W = best_model.forward(A, node_features, test_node, test_target)
            # print("W matrix shape", W.shape)
            # print("W type and length",type(W), len(W))
            # print("W[0] type and length", type(W[0]), len(W[0]))
            # print("W[0][0] type and shape", type(W[0][0]), W[0][0].shape)
            # # print("W[0][0][0] type and len", type(W[0][0][0]),len(W[0][0][0]))
            # print(W[0][0])
            test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
        best_test_loss = min(best_test_loss,test_loss.detach().cpu().numpy())
        best_test_f1 = max(best_test_f1,test_f1)
        dur_test = (time.time() - dur_test)/60.0

        print('---------------Best Results--------------------')
        print("Epoch: ", best_ep)
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        print('Total Duration : ', dur_total+dur_test)
        final_f1 += best_test_f1
        best_f1_list.append(best_test_f1)
        log_file.write(f'\n\nDate/Time : {datetime.datetime.now()}\n')
        log_file.write(f'Test run: {l+1}, Best Val F1: {best_val_f1*100}, Best Val loss: {best_val_loss}, Best Test F1: {best_test_f1*100}, Best Test loss: {best_test_loss}')
        log_csv.write(f'{l+1},{best_val_f1*100},{best_val_loss},{best_test_f1*100},{best_test_loss}')
        log_file.close()
        log_csv.close()

        plot_lines(train_f1_list,val_f1_list,train_loss_list,val_loss_list,epochs,args.experiment_name,l,plt_folder)

    print("Macro F1 scores: ", best_f1_list)
    print(f"Average Macro_F1 score over {total_runs} iterations : {final_f1*100/total_runs}")
    plot_testf1(best_f1_list,total_runs,args.experiment_name,plt_folder)
    