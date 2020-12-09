import torch
import numpy as np
# import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score#, plot_lines, plot_testf1
import random
import time
# import os
import networkx as nx
import datetime
import copy

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
from cora_utils import load_data

from ogb.nodeproppred import NodePropPredDataset

# folder = "saved_models/"
# plt_folder = "saved_plots/"

# if not os.path.exists(plt_folder):
#     os.mkdir(plt_folder)

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

    parser.add_argument('--ogb_mag', action='store_true',
                        help='Using OGB-MAG dataset')

    parser.add_argument('--cora', action='store_true',
                        help='Using CORA dataset')

    # parser.add_argument

    parser.add_argument('--total_runs', type=int, default=1,
                        help='number of runs')

    args = parser.parse_args()

    if not args.experiment_name:
        # exp = random.randint()
        args.experiment_name = f'GTN-Experiment'

    # os.makedirs(f'./saved_models/{args.experiment_name}', exist_ok=True)
    # os.makedirs(f'./results/', exist_ok=True)

    log_out = open(f"log_output_{args.experiment_name}.txt","w")

    print("******* starting **********",file=log_out,flush=True)

    # log_out.write(args)
    print(args,file=log_out,flush=True)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    if args.ogb_mag:
        print("Using OGB MAG",flush=True)
        dataset = NodePropPredDataset(name = "ogbn-mag")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0] # graph: library-agnostic graph object

        AvsI = graph['edge_index_dict'][('author', 'affiliated_with', 'institution')]
        AvsP = graph['edge_index_dict'][('author', 'writes', 'paper')]
        PvsP = graph['edge_index_dict'][('paper', 'cites', 'paper')]
        PvsS = graph['edge_index_dict'][('paper', 'has_topic', 'field_of_study')]

        # empty_lists = [ [] for _ in range(len(AvsI[0])) ]
        # AvsIdict = dict(zip(AvsI[0],empty_lists))
        empty_lists = [ [] for _ in range(len(AvsI[1])) ]
        IvsAdict = dict(zip(AvsI[1],empty_lists))
        empty_lists = [ [] for _ in range(len(AvsP[0])) ]
        AvsPdict = dict(zip(AvsP[0],empty_lists))
        empty_lists = [ [] for _ in range(len(PvsS[0])) ]
        PvsSdict = dict(zip(PvsS[0],empty_lists))
        empty_lists = [ [] for _ in range(len(PvsS[1])) ]
        SvsPdict = dict(zip(PvsS[1],empty_lists))

        for i in range(len(AvsP[0])):
            AvsPdict[AvsP[0][i]].append(AvsP[1][i])

        for i in range(len(PvsS[1])):
            SvsPdict[PvsS[1][i]].append(PvsS[0][i])

        for i in range(len(AvsI[1])):
            IvsAdict[AvsI[1][i]].append(AvsI[0][i])

        # AvsP = np.column_stack(graph['edge_index_dict'][('author', 'writes', 'paper')])
        # PvsP = np.column_stack(graph['edge_index_dict'][('paper', 'cites', 'paper')])
        # PvsS = np.column_stack(graph['edge_index_dict'][('paper', 'has_topic', 'field_of_study')])

        num_paper = graph['num_nodes_dict']['paper']
        num_auth = graph['num_nodes_dict']['author']
        num_inst = graph['num_nodes_dict']['institution']
        num_field = graph['num_nodes_dict']['field_of_study']
        
        paper_feat = graph['node_feat_dict']['paper']
        auth_feat = np.zeros(shape=(num_auth,128))
        inst_feat = np.zeros(shape=(num_inst,128))
        field_feat = np.zeros(shape=(num_field,128))

        for i in AvsPdict:
            auth_feat[i] = np.mean(paper_feat[AvsPdict[i]],axis=0)

        for i in SvsPdict:
            field_feat[i] = np.mean(paper_feat[SvsPdict[i]],axis=0)

        for i in IvsAdict:
            inst_feat[i] = np.mean(auth_feat[IvsAdict[i]],axis=0)

        split_auth = num_paper
        split_inst = split_auth + num_auth
        split_field = split_inst + num_inst

        G = nx.Graph()
        DiG = nx.DiGraph()

        new_AvsI = np.column_stack((AvsI[0] + split_auth, AvsI[1] + split_inst))
        new_AvsP = np.column_stack((AvsP[0] + split_auth, AvsP[1]))
        new_PvsP = np.column_stack((PvsP[0], PvsP[1]))
        new_PvsS = np.column_stack((PvsS[0], PvsS[1] + split_field))

        G.add_edges_from(new_AvsI)
        G.add_edges_from(new_PvsS)
        G.add_edges_from(new_PvsP)
        G.add_edges_from(new_AvsP)

        DiG.add_edges_from(new_AvsI)
        DiG.add_edges_from(new_PvsS)
        DiG.add_edges_from(new_PvsP)
        DiG.add_edges_from(new_AvsP)

        bfs_dict = dict(nx.bfs_successors(G, 736389, depth_limit=3))

        total_nodes = list()
        total_nodes += list(bfs_dict.keys())

        for i in bfs_dict:
            total_nodes += bfs_dict[i][:100]

        diG2 = nx.subgraph(DiG,set(total_nodes))

        print(len(diG2.nodes)) # 3633
        print(len(diG2.edges())) # 18610

        set_papers = set()
        set_authors = set()
        set_inst = set()
        set_fields = set()
        for i in list(diG2.nodes()):
            if i < split_auth:
                set_papers.add(i)
            elif i < split_inst:
                set_authors.add(i)
            elif i < split_field:
                set_inst.add(i)
            else:
                set_fields.add(i)

        label_papers = label['paper'][list(set_papers)]
        label_papers = label_papers.squeeze()       ## no. 205

        # labels_dict = dict(zip(list(set_papers),label_papers))

        norm_auth_feat = auth_feat[sorted(np.array(list(set_authors))-split_auth)]
        norm_paper_feat = paper_feat[sorted(list(set_papers))]
        norm_inst_feat = inst_feat[sorted(np.array(list(set_inst))-split_inst)]
        norm_field_feat = field_feat[sorted(np.array(list(set_fields))-split_field)]

        norm_paper_feat = (norm_paper_feat-np.min(norm_paper_feat))/np.ptp(norm_paper_feat)
        norm_auth_feat = ((norm_auth_feat-np.min(norm_auth_feat))/np.ptp(norm_auth_feat))
        norm_inst_feat = ((norm_inst_feat-np.min(norm_inst_feat))/np.ptp(norm_inst_feat))
        norm_field_feat = ((norm_field_feat-np.min(norm_field_feat))/np.ptp(norm_field_feat))

        matAvsI = np.zeros(shape=(3633,3633))
        matAvsP = np.zeros(shape=(3633,3633))
        matPvsP = np.zeros(shape=(3633,3633))
        matPvsS = np.zeros(shape=(3633,3633))

        norm_nodes = dict(zip(np.arange(3633),np.zeros(shape=(3633,))))

        total_nodes = sorted(np.unique(total_nodes))

        for i in range(len(total_nodes)):
            norm_nodes[total_nodes[i]] = i

        norm_split_auth = 2849
        norm_split_inst = norm_split_auth + 418
        norm_split_fields = norm_split_inst + 192

        # Al = nx.adjlist.generate_adjlist(diG2)

        all_edges = list(diG2.edges())

        for i,j in all_edges:
            if norm_nodes[i] < norm_split_auth:
                if norm_nodes[j] < norm_split_auth:
                    matPvsP[norm_nodes[i]][norm_nodes[i]] = 1
                # elif j < norm_split_inst:
                # elif j < norm_split_field:
                elif norm_nodes[j] > norm_split_fields:
                    matPvsS[norm_nodes[i]][norm_nodes[j]] = 1
            elif norm_nodes[i] < norm_split_inst:
                if norm_nodes[j] < norm_split_auth:
                    matAvsP[norm_nodes[i]][norm_nodes[j]] = 1
                # elif j < norm_split_inst:
                elif norm_nodes[j] < norm_split_fields:
                    matAvsI[norm_nodes[i]][norm_nodes[j]] = 1
                # elif j > norm_split_field:
                #     matPvsS[i][j] = 1
            # elif i < norm_split_field:
            #     if j < norm_split_auth:
            #         matAvsP[i][j] = 1
            #     # elif j < norm_split_inst:
            #     elif j < norm_split_field:
            #         matAvsI[i][j] = 1
            #     # elif j > norm_split_field:
            #     #     matPvsS[i][j] = 1
            # else:
            #     set_fields.add(i)

        # for i,edge in enumerate(edges):
        #     if i ==0:
        #         A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        #     else:
        #         A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        A = torch.from_numpy(matAvsP).type(torch.FloatTensor).unsqueeze(-1)
        A = torch.cat([A,torch.from_numpy(matAvsI).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.from_numpy(matPvsP).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.from_numpy(matPvsS).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        num_nodes = matAvsI.shape[0]

        A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        print("A shape and sample: ", A.shape,flush=True)
        print(A,flush=True)

        label_map = dict(zip(sorted(np.unique(label_papers)),np.arange(205)))

        for i in range(len(label_papers)):
            label_papers[i] = label_map[label_papers[i]]

        train_node, test_node = model_selection.train_test_split(total_nodes[:norm_split_auth], train_size=140, test_size=None)#, stratify=node_nodes)
        valid_node, test_node = model_selection.train_test_split(test_node, train_size=500, test_size=1000)#, stratify=test_nodes)

        for i in range(len(train_node)):
            train_node[i] = norm_nodes[train_node[i]]

        for i in range(len(valid_node)):
            valid_node[i] = norm_nodes[valid_node[i]]

        for i in range(len(test_node)):
            test_node[i] = norm_nodes[test_node[i]]

        train_target = torch.from_numpy(np.array(label_papers[train_node])).type(torch.LongTensor)

        valid_target = torch.from_numpy(np.array(label_papers[valid_node])).type(torch.LongTensor)

        test_target = torch.from_numpy(np.array(label_papers[test_node])).type(torch.LongTensor)

        node_features = torch.from_numpy(np.vstack((norm_paper_feat,norm_auth_feat,norm_inst_feat,norm_field_feat))).type(torch.FloatTensor)

        num_classes = torch.max(torch.cat((train_target,valid_target,test_target))).item()+1

        print("Num classes: ", num_classes)


    elif args.cora:
        print("Using Cora",file=log_out,flush=True)
        # edgelist = pd.read_csv(os.path.join("cora", "cora.cites"), sep='\t', header=None, names=["target", "source"])
        # edgelist["label"] = "cites"
        # Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
        # nx.set_node_attributes(Gnx, "paper", "label")
        # feature_names = ["w_{}".format(ii) for ii in range(1433)]
        # column_names =  feature_names + ["subject"]
        # node_data = pd.read_csv(os.path.join("cora", "cora.content"), sep='\t', header=None, names=column_names)

        # num_nodes = node_data.shape[0]
        
        # A = nx.adjacency_matrix(Gnx)
        # A = torch.from_numpy(A.todense()).type(torch.FloatTensor).unsqueeze(-1)
        # A = torch.cat([matA,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        # node_list = node_data.index

        # node_features = node_data.loc[:,node_data.columns != 'subject']
        # node_labels = node_data.loc[:,node_data.columns == 'subject']

        # # train_subjects, test_subjects = model_selection.train_test_split(node_subjects, train_size=140, test_size=None, stratify=node_subjects)
        # # val_subjects, test_subjects = model_selection.train_test_split(test_subjects, train_size=500, test_size=None, stratify=test_subjects)

        # onehot_labels = LabelBinarizer().fit_transform(node_data.loc[:,node_data.columns == 'subject'])

        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

        num_nodes = adj.shape[0]
        A = torch.from_numpy(adj.todense()).type(torch.FloatTensor).unsqueeze(-1)
        A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        node_features = torch.from_numpy(features.todense()).type(torch.FloatTensor)

        train_node = [] #torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
        train_target = []
        valid_node = []
        valid_target = []
        test_node = []
        test_target = []

        # for i in range(len(train_mask)):
        #     if train_mask[i]:
        #         train_node.append(i)
        #     if valid_mask[i]:
        #         valid_node.append(i)
        #     if test_mask[i]:
        #         test_node.append(i)

        # for i in range(len(y_train)):

        node_list = np.arange(2708)
        label_list = np.arange(7)
        train_node = torch.from_numpy(node_list[train_mask]).type(torch.LongTensor)
        valid_node = torch.from_numpy(node_list[val_mask]).type(torch.LongTensor)
        test_node = torch.from_numpy(node_list[test_mask]).type(torch.LongTensor)

        train_target = torch.from_numpy(np.array([label_list[i!=0].item() for i in y_train[:140]])).type(torch.LongTensor)        
        valid_target = torch.from_numpy(np.array([label_list[i!=0].item() for i in y_val[140:valid_node[-1]+1]])).type(torch.LongTensor)
        
        test_target = torch.from_numpy(np.array([label_list[i!=0].item() for i in y_test[test_node[0]:test_node[-1]+1]])).type(torch.LongTensor)

        num_classes = torch.max(train_target).item()+1


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
        # log_csv = open(f'./results/csv_{args.experiment_name}_run{str(l)}.csv',"w")
        # log_file = open(f'./results/log_{args.experiment_name}_run{str(l)}.txt',"a")
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

        print(model,file=log_out,flush=True)

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
        best_Ws = []
        
        for i in range(epochs):
            ep_time = time.time()
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1,file=log_out,flush=True)
            # log_out.write(f'Epoch: {i+1}\n')
            # print("Duration for param setting : ", time.time() - ep_time)
            dur_train = time.time()
            model.zero_grad()
            model.train()
            loss,y_train,Ws = model(A, node_features, train_node, train_target)
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            # print(Ws,flush=True)
            # dur_train = (time.time() - dur_train)/60.0
            train_f1_list.append(train_f1*100)
            train_loss_list.append(loss)
            print('Train - Loss: {}, Macro_F1: {}, Duration: {}'.format(loss.detach().cpu().numpy(), train_f1, (time.time() - dur_train)/60.0),file=log_out,flush=True)
            # log_out.write('Train - Loss: {}, Macro_F1: {}, Duration: {}'.format(loss.detach().cpu().numpy(), train_f1, dur_train)+'\n')
            loss.backward()
            optimizer.step()
            dur_train = (time.time() - dur_train)/60.0
            print("Total train epoch duration : {}".format(dur_train),file=log_out,flush=True)
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
                print('Valid - Loss: {}, Macro_F1: {}, Duration: {}'.format(val_loss.detach().cpu().numpy(), val_f1, dur_valid),file=log_out,flush=True)
                # log_out.write('Valid - Loss: {}, Macro_F1: {}, Duration: {}'.format(val_loss.detach().cpu().numpy(), val_f1, dur_valid)+'\n')
                # dur_test = time.time()
                # test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                # test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                # dur_test = (time.time() - dur_test)/60.0
                # print('Test - Loss: {}, Macro_F1: {}, Duration: {}'.format(test_loss.detach().cpu().numpy(), test_f1, dur_test))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                torch.save(model.state_dict(), f'{args.experiment_name}_best_val_f1_run_{l}.pth')
                # best_model = type(model)(args)
                # best_model.load_state_dict(best_model.state_dict())
                best_model = copy.deepcopy(model)
                # best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_ep = i+1
                best_trainW = Ws.copy()#.detach().clone()
                # best_test_f1 = test_f1

            # print('Test - Loss: {}, Macro_F1: {}, Duration: {}'.format(test_loss.detach().cpu().numpy(), test_f1, dur_test))
            ep_time = (time.time() - ep_time)/60.0
            print("Epoch Duration : {}\n".format(ep_time),file=log_out,flush=True)
            # log_out.write("Epoch Duration : {}\n".format(ep_time))
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
            # _, y_valid,_ = best_model.forward(A, node_features, valid_node, valid_target)
            # val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
            # torch.save(best_model.state_dict(), f'{args.experiment_name}_best_val_f1_run_{l}.pth')
        best_test_loss = min(best_test_loss,test_loss.detach().cpu().numpy())
        best_test_f1 = max(best_test_f1,test_f1)
        dur_test = (time.time() - dur_test)/60.0

        torch.save(best_trainW,f'attn_wgt_{args.experiment_name}_run_{l}.pt')

        print('---------------Best Results--------------------',file=log_out,flush=True)
        print("Epoch: ", best_ep,file=log_out,flush=True)
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1),file=log_out,flush=True)
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1),file=log_out,flush=True)
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1),file=log_out,flush=True)
        # print("Validating best model: {}".format(val_f1),file=log_out,flush=True)
        print('Total Duration : ', dur_total+dur_test,file=log_out,flush=True)
        final_f1 += best_test_f1
        best_f1_list.append(best_test_f1)
        # log_file.write(f'\n\nDate/Time : {datetime.datetime.now()}\n')
        # log_file.write(f'Test run: {l+1}, Best Val F1: {best_val_f1*100}, Best Val loss: {best_val_loss}, Best Test F1: {best_test_f1*100}, Best Test loss: {best_test_loss}')
        # log_csv.write(f'{l+1},{best_val_f1*100},{best_val_loss},{best_test_f1*100},{best_test_loss}')
        # log_file.close()
        # log_csv.close()

        # plot_lines(train_f1_list,val_f1_list,train_loss_list,val_loss_list,epochs,args.experiment_name,l,plt_folder)

    print("Macro F1 scores: ", best_f1_list,file=log_out,flush=True)
    print(f"Average Macro_F1 score over {total_runs} iterations : {final_f1*100/total_runs}",file=log_out,flush=True)
    log_out.close()
    # plot_testf1(best_f1_list,total_runs,args.experiment_name,plt_folder)
    