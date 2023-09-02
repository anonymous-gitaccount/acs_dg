#feature extraction for graph input
#in the original deep game, we don't need to specify the name of the dataset
#but here we need.
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
import numpy as np

class FeatureExtraction_lb:
    def __init__(self, dataset='KarateClub'):
        #self.PATTERN = pattern
        self.NUM_PARTITION = 10
        self.ENTRY_BOUNDS = (0, 1)
        self.NUM_OF_ENTRY_MANIPULATION = 2
        self.dataset=dataset
        dataset = KarateClub()
        self.data=dataset[0]
    def get_partitions(self, node_index,hop, num_partition=10):
        #feature partition in 2 hop neighbors according to 1's appearance.
        #need to be updated by feature influence score.
        data=self.data
        partitions = {}
        '''
        neighbor_1hop, _, _, _=k_hop_subgraph(node_idx=i, num_hops=1, edge_index=data.edge_index)
        print("1-hop neighbor", len(neighbor_1hop))
        partitions[0]=neighbor_1hop
        print(partitions[0])
        neighbor_2hop, _, _, _=k_hop_subgraph(node_idx=i, num_hops=2, edge_index=data.edge_index)
        partitions[1]=list(set(neighbor_2hop.tolist()) - set(neighbor_1hop.tolist()))
        print("2-hop neighbor",len(partitions[1]))
        print(partitions[1])
        '''
        neighbors, _, _, _=k_hop_subgraph(node_idx=node_index, num_hops=hop, edge_index=data.edge_index)
        neighbors=neighbors.clone().detach().numpy()
        print("neighbors are",neighbors)
        #print(len(neighbors))
        #partitions[2]=list(set(neighbor_3hop.tolist())-set(neighbor_2hop.tolist())-set(neighbor_1hop.tolist()))
        #print("3-hop neighbor",len(partitions[2]))
        X=self.data.x.clone().detach().numpy()
        #no shared memory and in numpy
        selected_rows = X[neighbors]
        #counts = selected_rows.sum(axis=0)
        columns = list(range(X.shape[1]))
        #ranked_attributes = np.argsort(counts)[::-1]
        #print("FeaturExtraction",ranked_attributes[0])
        #print(counts[ranked_attributes[0]],counts[ranked_attributes[1]])
        column_partition = np.array_split(columns, num_partition)
        for k in range(num_partition):
            cols=column_partition[k]
            partition_k=[]
            for row in range(len(neighbors)):
                for col in cols:
                    partition_k.append([row,col])
            partitions[k]=partition_k
        return partitions, neighbors
    def get_key_points(self,node_index,num_partition=10):
        self.NUM_PARTITION=num_partition
        key_points = [key for key in range(self.NUM_PARTITION)]
        return key_points