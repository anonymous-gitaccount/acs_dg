#feature extraction for graph input
#in the original deep game, we don't need to specify the name of the dataset
#but here we need.
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph

import numpy as np

class FeatureExtraction:
    def __init__(self, dataset='Cora'):
        #self.PATTERN = pattern
        self.NUM_PARTITION = 10
        self.ENTRY_BOUNDS = (0, 1)
        self.NUM_OF_ENTRY_MANIPULATION = 2
        self.dataset=dataset
        dataset = Planetoid(root='.', name=dataset)
        self.data=dataset[0]
    def get_partitions(self, node_index, num_partition=10):
        data=self.data
        partitions = {}
        neighbors, _, _, _=k_hop_subgraph(node_idx=node_index, num_hops=0, edge_index=data.edge_index)
        neighbors=neighbors.clone().detach().numpy()
        print(neighbors)
        #print(len(neighbors))
        #partitions[2]=list(set(neighbor_3hop.tolist())-set(neighbor_2hop.tolist())-set(neighbor_1hop.tolist()))
        #print("3-hop neighbor",len(partitions[2]))
        #no shared memory and in numpy
        selected_rows = X[neighbors]
        counts = selected_rows.sum(axis=0)
        ranked_attributes = np.argsort(counts)[::-1]
        #print("FeaturExtraction",ranked_attributes[0])
        #print(counts[ranked_attributes[0]],counts[ranked_attributes[1]])
        column_partition = np.array_split(ranked_attributes, num_partition)
        for k in range(num_partition):
            cols=column_partition[k]
            partition_k=[]
            for row in neighbors:
                for col in cols:
                    partition_k.append([row,col])
            partitions[k]=partition_k
            print("partition length", len(partitions[k]))
        return partitions
    def get_key_points(self,node_index,num_partition=10):
        self.NUM_PARTITION=num_partition
        key_points = [key for key in range(self.NUM_PARTITION)]
        return key_points