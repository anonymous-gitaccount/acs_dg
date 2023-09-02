'''
extended from a pytorch implementation of CLEVER for graph structured data
original program is under MIT License
'''

import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min
import argparse
import os.path as osp
import math


import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
#load data
def save_grad(save):
    def hook(grad):
        # note: clear gradient after saving it each time
        save['grads'] = grad.detach().clone()
        grad.data.zero_()
    return hook
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = KarateClub()
data=dataset[0]
data = data.to(device)
model = torch.load("models/KarateClub")
model.eval()
y=model(data.x, data.edge_index, data.edge_attr)
pred_class = y.argmax(dim=-1)
x=data.x
labels=data.y
skip_x_idx = set([idx for idx in range(x.shape[0]) if not (pred_class[idx].eq(labels[idx]))])
print(len(skip_x_idx))
input_grads = {'grads': None}
(values, indices)=torch.topk(y, k=y.shape[1], dim=1)
targeted_classes=indices[:, 1:].cpu().numpy()
print(targeted_classes.shape)
device = x.device
num_node,num_feature=x.shape
dims=x.shape[0]*x.shape[1]
single_dim=1*x.shape[1]
iterations=100
sample_size=100
radius=5
count=0
for idx in [10]:
    if idx in skip_x_idx:
        continue
    #print(idx)
    node_pred_class=pred_class[idx]
    targeted_classes_node=targeted_classes[idx]
    max_grad_norms = np.zeros((y.shape[1], iterations))
    for itr in range(iterations):
        norm_list=np.zeros((y.shape[1],sample_size))
        for sample in range(sample_size):
        #randomly select a point
            neighbors, _, _, _=k_hop_subgraph(node_idx=idx, num_hops=2, edge_index=data.edge_index)
            random_pert_node=x.clone().detach()
            #for n_idx in neighbors:
            #    rand_pert = torch.randn(1, (single_dim), device=device)
            #    r_pert = torch.norm(rand_pert, 2, dim=1, keepdim=True)
            #    random_pert_node[n_idx]= torch.clamp(((rand_pert / r_pert)[:, :single_dim] * radius/math.sqrt(len(neighbors))).view(1,num_feature) + x[n_idx], 0.0, 1.0)
            rand_pert = torch.randn(len(neighbors), (single_dim+1), device=device)
            r_pert = torch.norm(rand_pert, 2, dim=1, keepdim=True)
            random_pert_node[neighbors]= torch.clamp(((rand_pert / r_pert)[:, :single_dim] * radius).view(len(neighbors),num_feature) + x[neighbors], 0.0, 1.0)
            #print(random_pert_node[0])
            #print(torch.norm(random_pert_node)- torch.norm(x))
            random_pert_node.requires_grad = True
            random_pert_node=random_pert_node.to(device)
            preds = model(random_pert_node,data.edge_index, data.edge_attr)
            model.zero_grad()
            hook = random_pert_node.register_hook(save_grad(input_grads))
            node_flatten=torch.reshape(random_pert_node, (-1,))
            #print(torch.tensor([1.0] * len(preds[:, 0])).shape)
            for target_class in targeted_classes_node:
                #print(preds[:, node_pred_class])
                #print(preds[:, target_class])

                torch.autograd.backward(preds[idx:idx+1, node_pred_class] - preds[idx:idx+1, target_class],
                                        torch.tensor([1.0] * len(preds[idx:idx+1, 0])).to(next(model.parameters()).device),
                                        retain_graph=True)
                norm_of_sample=torch.norm(
                        (input_grads['grads']).view(1, -1), p=2, dim=1).item()
                #print(norm_of_sample)
                norm_list[target_class][sample]=norm_of_sample
        max_grad_norms[target_class][itr] = np.max(norm_list[target_class])
        #print(max_grad_norms[target_class][itr])
        count+=1
        #print("iteration", itr)
    #loc_classes=[]
    loc_x_0=[]
    for target_class in targeted_classes_node:
        #print("size",max_grad_norms[target_class].shape)
        [_, loc, _] = weibull_min.fit(-max_grad_norms[target_class])
        loc_x_0.append(-loc)
    print(idx)
    print("lipschitz estimate for a node: ",max(loc_x_0))

    
