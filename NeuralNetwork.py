import os.path as osp
import copy
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
#from FeatureExtraction import *
seed=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataset='KarateClub'
hidden_channels=16
lr=0.01
epochs=200
#parser.add_argument('--wandb', action='store_true', help='Track experiment')
dataset = KarateClub()
data=dataset[0]

print(data.x.shape)
print(data.x[0])

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.gcn = GCNConv(in_channels, 3)
        #self.out = Linear(3, dataset.num_classes)

    #def forward(self, x, edge_index):
    #    h = self.gcn(x, edge_index).relu()
    #    z = self.out(h)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.softmax(x, dim=1)
        return x
    def softmax_logits(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    def embedding(self,x,edge_index,edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        #x=self.gcn(x,edge_index).relu()
        return x
# Define a Neural Network class.
class NeuralNetwork:
    # Specify which dataset at initialisation.
    def __init__(self, dataset_name,hidden_channels=16):
        self.dataset=dataset_name
        dataset = KarateClub()
        data=dataset[0]
        model = GCN(dataset.num_features, hidden_channels, dataset.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("cuda")
        self.model, self.data = model.to(device), data.to(device)
        self.device=device
        self.X=data.x

    def predict(self, X):
        X=X.to(self.device)
        model=self.model
        model.eval()
        predict_value = model(X, self.data.edge_index, self.data.edge_attr).detach().cpu().numpy()
        new_classes = np.argmax(predict_value, axis=1)
        confident = np.amax(predict_value, axis=1)
        return new_classes, confident
    #overload predict
    def predict_perturb(self, X_perturb, neighbors):
        X=self.data.x.clone()
        X[neighbors,:]=X_perturb.to(self.device)
        model=self.model
        model.eval()
        predict_value = model(X, self.data.edge_index, self.data.edge_attr).detach().cpu().numpy()
        new_classes = np.argmax(predict_value, axis=1)
        confident = np.amax(predict_value, axis=1)
        return new_classes, confident

        #equivalent function for calling model.predict, using perturbation rows only
    def predict_prob(self, X_perturb, neighbors):
        X=self.data.x.clone()
        X[neighbors,:]=X_perturb.to(self.device)
        model=self.model
        model.eval()
        predict_value = model(X, self.data.edge_index, self.data.edge_attr)#.detach().cpu().numpy()
        return predict_value
    def predict_with_margin(self, X):
        X=X.to(self.device)
        model=self.model
        model.eval()
        predict_value = model(X, self.data.edge_index, self.data.edge_attr).detach().cpu().numpy()
        new_classes = np.argmax(predict_value, axis=1)
        confident = np.amax(predict_value, axis=1)
        sorted_values = np.sort(predict_value, axis=1)
        margin = confident - sorted_values[:, -2]
        return new_classes, confident, margin
    def train_network(self, epochs):
        optimizer = torch.optim.Adam([
        dict(params=self.model.conv1.parameters(), weight_decay=5e-4),
        dict(params=self.model.conv2.parameters(), weight_decay=0)
        ], lr=lr)
        for epoch in range(1, epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x,self.data.edge_index, self.data.edge_attr)
            loss = F.cross_entropy(out[self.data.train_mask],self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
        train_acc, val_acc, test_acc = NN.test_network()
        print("train accuracy",train_acc)
        print("val accuracy",val_acc)
        print("test accuracy",test_acc)
    

    def train_network(self, epochs):
        optimizer = torch.optim.Adam([
        dict(params=self.model.conv1.parameters(), weight_decay=5e-4),
        dict(params=self.model.conv2.parameters(), weight_decay=0)
        ], lr=lr)
        for epoch in range(1, epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x,self.data.edge_index, self.data.edge_attr)
            loss = F.cross_entropy(out[self.data.train_mask],self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
        train_acc = NN.test_network()
        print("train accuracy",train_acc)
        #print("val accuracy",val_acc)
        self.model.eval()
        pred = self.model(self.data.x, self.data.edge_index, self.data.edge_attr).argmax(dim=-1)
        accs = []
        new_mask = ~self.data.train_mask
        accs=int((pred[new_mask] == self.data.y[new_mask]).sum()) / 30
        print(accs)
    def test_network(self):
        #first switch to evaluation mode
        self.model.eval()
        pred = self.model(self.data.x, self.data.edge_index, self.data.edge_attr).argmax(dim=-1)
        accs = []
        accs=int((pred[self.data.train_mask] == self.data.y[self.data.train_mask]).sum()) / int(self.data.train_mask.sum())
        return accs
    def save_network(self):
        if self.dataset == 'KarateClub':
            torch.save(self.model,'models/KarateClub')
            print("Neural network {} saved to disk.".format(self.dataset))
        else:
            print("save_network: Unsupported dataset.")

    def load_network(self):
        if self.dataset == 'KarateClub':
            self.model=torch.load('models/KarateClub')
            print("Neural network {} loaded from disk.".format(self.dataset))
        else:
            print("load_network: Unsupported dataset.") 
    def get_label(self, index):
        if self.dataset == 'KarateClub':
            labels = ['0','1','2','3']
            return labels[index]
    def softmax_logits(self,X):
        self.model.eval() 
        return self.model.softmax_logits(X, self.data.edge_index, self.data.edge_attr)

NN=NeuralNetwork('KarateClub',16)
NN.train_network(epochs=200)

clas,confident,margin= NN.predict_with_margin(data.x)
print(clas[17],confident[17],margin[17])
'''
data=data.to(device)
out1=NN.model(data.x,data.edge_index)
data.x[0,0]=1-data.x[0,0]
out=NN.model(data.x,data.edge_index)
print(out1[0],out[0])


dataset = Planetoid(root='.', name='KarateClub')
data=dataset[0]
new_classes, confident=NN.predict(data.x)
print(new_classes[0])
X=torch.clone(data.x)
print("new_classes.shape: {}, new_classes[1]: {}, confident.shape: {},confident[1]: {}".format(new_classes.shape, new_classes[1], confident.shape, confident[1]))
for i in range(1000,1400):
    X[1,i]=1-X[1,i]
new_classes, confident=NN.predict(X)
print("new_classes.shape: {}, new_classes[1]: {}, confident.shape: {},confident[1]: {}".format(new_classes.shape, new_classes[1], confident.shape, confident[1]))
'''
#new_classes, confident, margin=NN.predict_with_margin(data.x)
#print(margin[0])
#print("new_classes.shape: {}, new_classes[0]: {}, confident.shape: {}, confident[0]: {}, margin.shape: {}, margin[0]: {}".format(new_classes.shape, new_classes[0], confident.shape, confident[0],margin.shape,margin[0]))

#NN.load_network()
#logits=NN.softmax_logits(data.x)
#print(logits[0])
#numpyX=data.x.clone().detach().numpy()
#new_classes, confident=NN.predict(data.x)
'''
print(margin)

FE=FeatureExtraction()
FE.get_partitions(0)
for i in range(len(data.test_mask)):
    if data.test_mask[i]== False:
        continue
    neighbors, _, _, _=k_hop_subgraph(node_idx=i, num_hops=2, edge_index=data.edge_index)
    print(len(neighbors))
'''
