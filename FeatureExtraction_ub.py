#feature extraction for graph input
#in the original deep game, we don't need to specify the name of the dataset
#but here we need.
import argparse
import os.path as osp
import random
import torch
import torch.nn.functional as F
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from FeatureScore import *
import numpy as np

class FeatureExtraction_ub:
    def __init__(self, dataset='Cora'):
        #self.PATTERN = pattern
        self.NUM_PARTITION = 10
        self.ENTRY_BOUNDS = (0, 1)
        self.NUM_OF_ENTRY_MANIPULATION = 2
        self.dataset=dataset
        dataset = Planetoid(root='.', name=dataset)
        self.data=dataset[0]
        self.neighbors=[]
        self.top_partition=0
    #flip a whole node
    def get_partitions_featurewise(self, node_index, hop_neighbors,num_partition=10):        
        data=self.data
        partitions = {}
        neighbors, _, _, _=k_hop_subgraph(node_idx=node_index, num_hops=hop_neighbors, edge_index=data.edge_index)
        #in the case of indirect attack, remove self.
        if hop_neighbors!=0:
            neighbors=neighbors[neighbors!=node_index]
        neighbors=neighbors.clone().detach().numpy()
        print("neighbors are",neighbors)
        #check for reduced
        X=self.data.x.clone().detach().numpy()
        #the partition now stores: partition[k]={list of feature matrix column index}
        print(X.shape)
        self.neighbors=neighbors
        columns = list(range(X.shape[1]))
        random.shuffle(columns)
        partition_size = X.shape[1] // num_partition
        remainder = X.shape[1] % num_partition
        start = 0
        for i in range(num_partition - 1):
            partitions[i] = columns[start: start + partition_size]
            start += partition_size
        partitions[num_partition - 1] = columns[start:]
    
        return partitions,neighbors
    def get_partitions(self, node_index, hop_neighbors,num_partition=10):        
        if hop_neighbors == 0:      
            data=self.data
            partitions = {}
            score_partitions={}
            neighbors, _, _, _=k_hop_subgraph(node_idx=node_index, num_hops=0, edge_index=data.edge_index)
            #in the case of indirect attack, remove self.
            neighbors=neighbors.clone().detach().numpy()
            X=self.data.x.clone().detach().numpy()
            selected_rows = X[neighbors]
            columns = list(range(X.shape[1]))
            random.shuffle(columns)
            column_partition = np.array_split(columns, num_partition)
            for k in range(num_partition):
                cols=column_partition[k]
                partition_k=[]
                for row in range(len(neighbors)):
                    for col in cols:
                        partition_k.append([row,col])
                partitions[k]=partition_k
            return partitions, neighbors,score_partitions
        elif hop_neighbors != 0:   
            scoreComputation = FeatureScore()
            #use the overloaded score function
            sorted_ixs,scores =scoreComputation.computeScore(target_node=node_index)
            top_list=sorted_ixs.tolist()
            scores=scores.tolist()
            data=self.data
            partitions = {}
            score_partitions={}
            #obtain the list of m-hop neighbours
            neighbors, _, _, _=k_hop_subgraph(node_idx=node_index, num_hops=hop_neighbors, edge_index=data.edge_index)
            #in the case of indirect attack, remove self node.
            neighbors=neighbors[neighbors!=node_index]
            neighbors=neighbors.clone().detach().numpy()
            index_mapping={value: i for i, value in enumerate(neighbors)}
            # Update the entries using the index mapping
            update_top_list=[(index_mapping[entry[0]], entry[1]) for entry in top_list]
            X=self.data.x.clone().detach().numpy()
            #the partition now stores: partition[k]={list of feature matrix column index}
            self.neighbors=neighbors
            new_list=update_top_list 
            partition_size=len(new_list)//num_partition
            remainder=len(new_list)%num_partition
            start=0
            #generate partitions of entries and partitions of scores
            for i in range(num_partition-1):
                partitions[i] = new_list[start: start+partition_size]
                start += partition_size
            partitions[num_partition - 1] = new_list[start:]
            return partitions,neighbors,score_partitions
    def get_key_points(self,node_index,num_partition=10):
        self.NUM_PARTITION=num_partition
        key_points = [key for key in range(self.NUM_PARTITION)]
        return key_points