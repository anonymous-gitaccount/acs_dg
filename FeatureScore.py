from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
import torch.nn.functional as F
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from scipy import sparse
import scipy.sparse as sp
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
#Overloaded model from DeepRobust
class ModifiedNettack(Nettack):
    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super().__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def feature_scores(self):
        """Compute feature scores for all possible feature changes.
        """
        ###from DeepRobust
        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]

        gradient = self.gradient_wrt_x(self.label_u) - self.gradient_wrt_x(best_wrong_class)
        # gradients_flipped = (gradient * -1).tolil()
        gradients_flipped = sp.lil_matrix(gradient * -1)
        gradients_flipped[self.modified_features.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.modified_features.shape)
        X_influencers[self.influencer_nodes] = self.modified_features[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply((self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

class FeatureScore:
    def __init__(self, dataset='Cora'):
        self.dataset_name=dataset
    def computeSelfScore(self,target_node):
        data = Planetoid(root='.', name='Cora')
        edge_index=data[0].edge_index
        data = Pyg2Dpr(data)
        #data = Dataset(root='/tmp/', name='cora')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        #print(features.shape)
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, with_relu=False, with_bias=False, device='cpu').to('cpu')
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        print("Feature",features[0])
        # Setup Attack Model
        model = ModifiedNettack(surrogate, nnodes=adj.shape[0], attack_structure=False,attack_features=True, device='cpu').to('cpu')
        neighbors,_, _, _=k_hop_subgraph(node_idx=target_node, num_hops=0, edge_index=edge_index)
        
        model.attack(sparse.csr_matrix(features), adj, labels, target_node, direct=True, n_influencers= 1,n_perturbations=20)
        model.influencer_nodes=neighbors
        #print(model.influencer_nodes)
        sorted_ixs, scores=model.feature_scores()
        #print(sorted_ixs)
        return sorted_ixs
    def computeScore(self,target_node):
        data = Planetoid(root='.', name='Cora')
        edge_index=data[0].edge_index
        #convert torch geometric dataset to deeprobust
        data = Pyg2Dpr(data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        #create attack method
        model = ModifiedNettack(surrogate, nnodes=adj.shape[0], attack_structure=False,attack_features=True, device='cpu').to('cpu')
        #restrcting attacking node to hop-neighbours
        neighbors,_, _, _=k_hop_subgraph(node_idx=target_node, num_hops=2, edge_index=edge_index) 
        model.attack(sparse.csr_matrix(features), adj, labels, target_node, direct=False, n_influencers= 1,n_perturbations=1)
        model.influencer_nodes=neighbors[neighbors!= target_node]
        #obtain entries with high influence score
        sorted_ixs, scores=model.feature_scores()
        return sorted_ixs,scores