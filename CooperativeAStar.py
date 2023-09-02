#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a CooperativeAStar class to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative.

"""

import heapq

from FeatureExtraction_lb import *
from basics import *
import time
import pickle
import collections
import matplotlib.pyplot as plt
class CooperativeAStar:
    def __init__(self, dataset,X, node_index, model, eta, tau,hop_neighbor,bounds=(0, 1)):
        #within the function, X is no longer a tensor. feel free to use it as a numpy array
        self.DATASET = dataset
        self.node_index=node_index
        self.IMAGE_BOUNDS = bounds
        self.MODEL = model
        #currently pass in this metric but no use
        self.DIST_METRIC = eta[0]
        self.DIST_VAL = eta[1]
        self.TAU = tau
        self.X_whole=X.clone().detach().numpy()
        LABEL, _ = self.MODEL.predict(torch.from_numpy(self.X_whole))
        self.LABEL=LABEL[node_index]
        print("CooperativeAStar, current node_label is:", self.LABEL)
        feature_extraction = FeatureExtraction_lb(dataset='Cora')
        self.PARTITIONS,self.NEIGHBORS = feature_extraction.get_partitions(self.node_index, hop_neighbor,num_partition=10)
        self.X=self.X_whole.take(self.NEIGHBORS,axis=0)
        self.DIST_EVALUATION = {}
        self.ADV_MANIPULATION = ()
        self.ADVERSARY_FOUND = None
        self.ADVERSARY = None
        self.hop=hop_neighbor
        self.explored={}
        self.CURRENT_SAFE = [0]

        print("Distance metric %s, with bound value %s." % (self.DIST_METRIC, self.DIST_VAL))

    # from previous deepgame implementations. Uncomment to remove permutations
    '''
    def add_to_exp(self,advatomic):
        atomic_list=[advatomic[i:i+3] for i in range(0,len(advatomic),3)]
        size=len(atomic_list)
        if size >0:
            if size in self.explored:
                self.explored[size].append(atomic_list)
            else:
                self.explored[size]=[atomic_list]
    '''

    def target_pixels(self, X, hop_neighbor_list):
        #print("target_pixels")
        # tau = self.TAU
        # model = self.MODEL
        start=time.time()
        (node,attr) = X.shape
        atomic_manipulations = []
        manipulated_matrices = []
        for (x,y) in hop_neighbor_list:   
            atomic = (x,y, 1 * self.TAU)
            valid, atomic_mat = self.apply_atomic_manipulation(X, atomic)
            if valid is True:
                manipulated_matrices.append(atomic_mat)
                atomic_manipulations.append(atomic)
            
            atomic = (x, y,  -1 * self.TAU)
            valid, atomic_mat = self.apply_atomic_manipulation(X, atomic)
            if valid is True:
                manipulated_matrices.append(atomic_mat)
                atomic_manipulations.append(atomic)
        manipulated_matrices = np.asarray(manipulated_matrices)
        #print(manipulated_matrices.shape)
        probabilities={}
        
        for i in range(len(manipulated_matrices)):       
            probabilities[i] = self.MODEL.predict_prob(torch.tensor(manipulated_matrices[i]).to(self.MODEL.device), self.NEIGHBORS)[self.node_index].detach().cpu().numpy()
            
        # softmax_logits = self.MODEL.softmax_logits(manipulated_images)
        #print("COOPERATIVEASTAR",probabilities)
        if self.ADV_MANIPULATION:
            atomic_list = [self.ADV_MANIPULATION[i:i + 3] for i in range(0, len(self.ADV_MANIPULATION), 3)]
        
        #print(len(manipulated_matrices))
        for idx in range(len(manipulated_matrices)):
            start=time.time()
            #if not diffMatrix(manipulated_matrices[idx], self.X) or not diffMatrix(manipulated_matrices[idx], X):
            if np.array_equal(manipulated_matrices[idx], self.X) or np.array_equal(manipulated_matrices[idx], X):
                continue
            end=time.time()
            #print("prediction_time ", end-start)
            cost = self.cal_distance(manipulated_matrices[idx], self.X)
            #print("probability",probabilities[idx].shape)
            [p_max, p_2dn_max] = heapq.nlargest(2, probabilities[idx])
            heuristic = (p_max - p_2dn_max) * 2 * self.TAU  # heuristic value determines Admissible (lb) or not (ub)
            #heuristic = (p_max - p_2dn_max) * 2 * 50  # heuristic value determines Admissible (lb) or not (ub)
            estimation = cost + heuristic
            #print(estimation)
            end=time.time()
            
            valid = True
            if self.ADV_MANIPULATION:
                for atomic in atomic_list:  # atomic: [x, y,  +/-tau]
                    if atomic_manipulations[idx][0:2] == atomic[0:2] and atomic_manipulations[idx][2] == -atomic[2]:
                        valid = False

            if valid is True:
                
                self.DIST_EVALUATION.update({self.ADV_MANIPULATION + atomic_manipulations[idx]: estimation})
        
            # self.DIST_EVALUATION.update({self.ADV_MANIPULATION + atomic_manipulations[idx]: estimation})
        # print("Atomic manipulations of target pixels done.")

    def apply_atomic_manipulation(self, X, atomic):
        atomic_mat = X.copy()
        idx = atomic[0:2]
        #print("appy_atomic_manipulation",idx)
        manipulate = atomic[2]
        atomic_list = [self.ADV_MANIPULATION[i:i + 3] for i in range(0, len(self.ADV_MANIPULATION), 3)]
        length=len(atomic_list)
        atomic_list.append(atomic)
        # from previous deepgame implementations. Uncomment to remove permutations
        '''
        current_count=collections.Counter(atomic_list)
        if length in self.explored:
            for ato in self.explored[length]:
                dif=current_count-collections.Counter(ato)
                if sum(dif.values())==1:
                    return False, atomic_mat
        '''
        if (atomic_mat[idx] >= max(self.IMAGE_BOUNDS) and manipulate >= 0) or (
                atomic_mat[idx] <= min(self.IMAGE_BOUNDS) and manipulate <= 0):
            valid = False
            return valid, atomic_mat
        else:
            if atomic_mat[idx] + manipulate > max(self.IMAGE_BOUNDS):
                atomic_mat[idx] = max(self.IMAGE_BOUNDS)
            elif atomic_mat[idx] + manipulate < min(self.IMAGE_BOUNDS):
                atomic_mat[idx] = min(self.IMAGE_BOUNDS)
            else:
                atomic_mat[idx] += manipulate
            valid = True
            return valid, atomic_mat

    def cal_distance(self, X1, X2):
        return l2Distance(X1, X2)
        

    def play_game(self, node_index):
        #print("play_game")
        #new_X=copy.deepcopy(self.X_whole)
        cur_max=0
        new_label, new_confidence = self.MODEL.predict(torch.from_numpy(self.X_whole))
        new_label=new_label[node_index]
        new_confidence=new_confidence[node_index]
        new_X=copy.deepcopy(self.X)
        distance_list=[]
        itr=0
        while self.cal_distance(self.X, new_X) <= self.DIST_VAL and new_label == self.LABEL:
            itr=itr+1
            print(itr)
            if itr > 2000:
                break
            start=time.time()
            #print("in loop")
            # for partitionID in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            for partitionID in self.PARTITIONS.keys():
                #print(len(self.PARTITIONS[partitionID]))
                #print(partitionID)
                hop_entry_list = self.PARTITIONS[partitionID]
                #hop_entry_list contains a list of 2 tuples (x,y)-(node, feature)
                self.target_pixels(new_X, hop_entry_list)
            #print(self.explored)
            #uncomment: eliminate permutation
            #self.add_to_exp(self.ADV_MANIPULATION)
            self.ADV_MANIPULATION = min(self.DIST_EVALUATION, key=self.DIST_EVALUATION.get)
            print("Current best manipulations:", self.ADV_MANIPULATION)
            # print("%s distance (estimated): %s" % (self.DIST_METRIC, self.DIST_EVALUATION[self.ADV_MANIPULATION]))
            self.DIST_EVALUATION.pop(self.ADV_MANIPULATION)
            new_X = copy.deepcopy(self.X)
            atomic_list = [self.ADV_MANIPULATION[i:i + 3] for i in range(0, len(self.ADV_MANIPULATION), 3)]

            #start=time.time()
            for atomic in atomic_list:
                #print("line134",atomic)
                valid, new_X = self.apply_atomic_manipulation(new_X, atomic)
            dist = self.cal_distance(self.X, new_X)
            cur_max=max(dist,cur_max)
            distance_list.append(cur_max)
            print("%s distance (actual): %s" % (self.DIST_METRIC, dist))
            #end=time.time()
            #print("atomic_loop", end-start)
            new_label, new_confidence = self.MODEL.predict_perturb(torch.from_numpy(new_X).to(self.MODEL.device),self.NEIGHBORS)
            new_label=new_label[node_index]
            new_confidence=new_confidence[node_index]
            if self.cal_distance(self.X, new_X) > self.DIST_VAL:
                # print("Adversarial distance exceeds distance budget.")
                self.ADVERSARY_FOUND = False
                break
            elif new_label != self.LABEL:
                # print("Adversarial image is found.")
                self.ADVERSARY_FOUND = True
                self.ADVERSARY = new_X
                break

            if self.CURRENT_SAFE[-1] != dist:
                self.CURRENT_SAFE.append(dist)
                #path = "%s_pic/idx_%s_Safe_currentBest_%s.png" % (self.DATASET, self.IDX, len(self.CURRENT_SAFE) - 1)
                #self.MODEL.save_input(new_image, path)
            end=time.time()
            print(end-start)
        if self.hop==0:
            directory_path = 'distance_list/lowerbound/direct'
        else:
            directory_path = 'distance_list/lowerbound/indirect'
        os.makedirs(directory_path, exist_ok=True)
        file_name = os.path.join(directory_path, f'{node_index}.pkl')
        #save distance list
        with open(file_name, 'wb') as file:
            pickle.dump(distance_list, file)
        #save plot
        plt.plot(distance_list)
        plt.xlabel('iteration')
        plt.ylabel('L2 distance')
        plt.title('Lowerbound')
        if self.hop == 0:
            os.makedirs('plots/lowerbound/direct', exist_ok=True)
            file_name = 'plots/lowerbound/direct/' + str(node_index) + '.png'
        else:
            os.makedirs('plots/lowerbound/indirect', exist_ok=True)
            file_name = 'plots/lowerbound/indirect/' + str(node_index) + '.png'
        plt.savefig(file_name)  



