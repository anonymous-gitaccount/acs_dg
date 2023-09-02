import sys
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import numpy as np

from scipy.stats import truncnorm, norm

from basics import *
from FeatureExtraction_lb import *
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
        self.data_set = data_set
        self.model = model
        self.node_index = node_index
        self.tau = tau
        self.maxVal = 1
        self.minVal = 0
        self.hop_neighbor=hop_neighbor
        #remember to chang
        feature_extraction = FeatureExtraction_lb(dataset='KarateClub')
        kps = feature_extraction.get_key_points(self.node_index, num_partition=10)
        partitions,neighbors = feature_extraction.get_partitions(self.node_index,self.hop_neighbor, num_partition=10)

        # path = "%s_pic/%s_Saliency_(%s).png" % (self.data_set, self.image_index, feature_extraction.PATTERN)
        # feature_extraction.plot_saliency_map(self.image, partitions=partitions, path=path)


        img_enlarge_ratio = 1
        #image1 = copy.deepcopy(self.image)


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
                atomic_manipulation = dict()
                atomic_manipulation[(x,y)] = -1 * self.tau
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
        # this is a python copy instruction... maybe update to numpy?
        X = copy.deepcopy(X_in)
       # print("Xshape ",X.shape)
      #  print("apply manipulation",len(manipulation.keys()))
        #currently add or deduct tau to all elements in node's attribute vector
        #if len(manipulation.keys())>0 and self.hop_neighbor==0:
        if len(manipulation.keys())>0:
            a = np.array(list(manipulation.keys()))
            values = np.array(list(manipulation.values()))
            X[a[:, 0], a[:, 1]] += values
            X = np.clip(X, 0, 1)
            return X
        else:
            #print("length<0")
            return X