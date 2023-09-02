import sys
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import numpy as np

from scipy.stats import truncnorm, norm

from basics import *
from FeatureExtraction_ub import *
import collections

#UNTESTED

############################################################
#
#  initialise possible moves for a two-player game
#
################################################################


class GameMoves:

    def __init__(self, data_set, model, node_index, tau, hop_neighbor):
        #dataset = Planetoid(root='.', name=data_set)
        print("GameMoves ub used")
        self.data_set = data_set
        self.model = model
        self.node_index = node_index
        self.tau = tau
        self.maxVal = 1
        self.minVal = 0
        self.hop_neighbor=hop_neighbor
        #remember to chang
        feature_extraction = FeatureExtraction_ub(dataset='Cora')
        kps = feature_extraction.get_key_points(self.node_index, num_partition=10)
        partitions,neighbors,scores = feature_extraction.get_partitions(self.node_index,self.hop_neighbor, num_partition=10)
        actions = dict()
        actions[0] = kps
        s = 1
        kp2 = []

        # construct moves according to the obtained the partitions 
        num_of_manipulations = 0
        #all_atomic_manipulations[k] contains all the manipulations for that block of pixels.
        for k, blocks in partitions.items():
            all_atomic_manipulations = []
            #all_atomic_maniulations now contains i-hop neighbor manipulations
            #each node has 2 manilpulations
            for i in range(len(blocks)):
                #entry [node_idx, feature_idx]
                x,y = blocks[i]
                atomic_manipulation = dict()
                atomic_manipulation[(x,y)] = self.tau
                all_atomic_manipulations.append(atomic_manipulation)
                #can comment out the following when tau=1
                atomic_manipulation = dict()
                atomic_manipulation[(x,y)] = - self.tau
                all_atomic_manipulations.append(atomic_manipulation)

            actions[s] = all_atomic_manipulations
            kp2.append(kps[s - 1])

            s += 1
            # print("%s manipulations have been initialised for keypoint (%s,%s), whose response is %s."
            #       % (len(all_atomic_manipulations), int(kps[k - 1].pt[0] / img_enlarge_ratio),
            #          int(kps[k - 1].pt[1] / img_enlarge_ratio), kps[k - 1].response))
            num_of_manipulations += len(all_atomic_manipulations)

        # index-0 keeps the keypoints, actual actions start from 1
        actions[0] = kp2
        print("the number of all manipulations initialised: %s\n" % num_of_manipulations)
        self.moves = actions

    def applyManipulation(self, X_in, manipulation):
        X = copy.deepcopy(X_in)
        if len(manipulation.keys())>0:        
            values = np.array(list(manipulation.values()))
            if values[0]==1:
                a = np.array(list(manipulation.keys()))
                X[a[:,0],a[:,1]]=1-X[a[:,0],a[:,1]]
                return X
            else:
                a = np.array(list(manipulation.keys()))
                X[a[:, 0], a[:, 1]] += values
                X = np.clip(X, 0, 1)
                return X
        else:
            #print("length<0")
            return X