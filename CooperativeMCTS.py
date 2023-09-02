import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math
import sys
#sys.setrecursionlimit(100000)
from basics import *
from GameMoves_ub import *

MCTS_multi_samples = 1
effectiveConfidenceWhenChanging = 0.0
explorationRate = math.sqrt(2)

class MCTSCooperative:

    def __init__(self, data_set, model, node_index, X, tau, eta,hop):
        self.data_set = data_set
        self.node_index = node_index
        self.X_whole=X.clone().detach().numpy()
        self.hop=hop
        self.model = model
        self.tau = tau
        self.eta = eta
        feature_extraction = FeatureExtraction_ub(dataset=data_set)
        self.PARTITIONS,self.NEIGHBORS,self.SCORES = feature_extraction.get_partitions(self.node_index, self.hop,num_partition=10)
        self.X=self.X_whole.take(self.NEIGHBORS,axis=0)

        originalClass, originalConfident = self.model.predict(torch.from_numpy(self.X_whole))
        self.originalClass, self.originalConfident = originalClass[self.node_index], originalConfident[self.node_index]
        
        self.moves = GameMoves(self.data_set, self.model, self.node_index, self.tau,self.hop)

        self.cost = {}
        self.numberOfVisited = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}

        self.indexToNow = 0
        # current root node
        self.rootIndex = 0

        self.manipulation = {}
        # initialise root node
        self.manipulation[-1] = {}
        self.initialiseLeafNode(0, -1, {})

        # record all the keypoints: index -> kp
        self.keypoints = {}
        # mapping nodes to keypoints
        self.keypoint = {}
        self.keypoint[0] = 0

        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (sys.maxsize, {})
        print("best case init",self.bestCase[0])
        self.numConverge = 0

        # number of adversarial examples
        self.numAdv = 0

        # how many sampling is conducted
        self.numOfSampling = 0

        # temporary variables for sampling 
        self.atomicManipulationPath = []
        self.depth = 0
        self.availableActionIDs = []
        self.usedActionIDs = []
        self.count=0
        if hop==0:
            directory_path = 'distance_list/upperbound/direct'
        else:
            directory_path = 'distance_list/upperbound/indirect'
        os.makedirs(directory_path, exist_ok=True)
        self.file_name = os.path.join(directory_path, f'{node_index}.txt')
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

        
        

    def initialiseMoves(self):
        # initialise actions according to the type of manipulations
        actions = self.moves.moves
        self.keypoints[0] = 0
        i = 1
        for k in actions[0]:
            self.keypoints[i] = k
            i += 1

        for i in range(len(actions)):
            ast = {}
            for j in range(len(actions[i])):
                ast[j] = actions[i][j]
            self.actions[i] = ast
        nprint("%s actions have been initialised." % (len(self.actions)))

    def initialiseLeafNode(self, index, parentIndex, newAtomicManipulation):
        nprint("initialising a leaf node %s from the node %s" % (index, parentIndex))
        self.manipulation[index] = mergeTwoDicts(self.manipulation[parentIndex], newAtomicManipulation)
        self.cost[index] = 0
        self.parent[index] = parentIndex
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0

    def destructor(self):
        self.X = 0
        self.X = 0
        self.model = 0
        self.model = 0
        self.manipulatedDimensions = {}
        self.manipulation = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}

        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self, newRootIndex):
        if self.keypoint[newRootIndex] != 0:
            player = "the first player"
        else:
            player = "the second player"
        print("%s making a move into the new root %s, whose value is %s and visited number is %s" % (
            player, newRootIndex, self.cost[newRootIndex], self.numberOfVisited[newRootIndex]))
        self.removeChildren(self.rootIndex, [newRootIndex])
        self.rootIndex = newRootIndex

    def removeChildren(self, index, indicesToAvoid):
        if self.fullyExpanded[index] is True:
            for childIndex in self.children[index]:
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex, [])
        self.manipulation.pop(index, None)
        self.cost.pop(index, None)
        self.parent.pop(index, None)
        self.keypoint.pop(index, None)
        self.children.pop(index, None)
        self.fullyExpanded.pop(index, None)
        self.numberOfVisited.pop(index, None)

    def bestChild(self, index):
        allValues = {}
        for childIndex in self.children[index]:
            allValues[childIndex] = float(self.numberOfVisited[childIndex]) / self.cost[childIndex]
        nprint("finding best children from %s" % allValues)
        # for cooperative
        return max(allValues.items(), key=operator.itemgetter(1))[0]
    '''
    def treeTraversal(self, index):
        print("Tree Traversal", self.fullyExpanded[index])
        if self.fullyExpanded[index] is True:
            nprint("tree traversal on node %s with children %s" % (index, self.children[index]))
            allValues = {}
            if self.first_step==0:
                for childIndex in self.children[index]:
                    # UCB values
                    print("fully expanded",len(self.children[index]))
                    allValues[childIndex] = ((float(self.numberOfVisited[childIndex]) / self.cost[childIndex]) * self.eta[1]
                                                + explorationRate * math.sqrt(
                                    math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex])))
                #print(allValues[childIndex])
            # for cooperative
            elif self.first_step==1:

                for childIndex in self.children[index]:
                    # UCB values
                    print(self.children[index])
                    print("first step, fully expanded",len(self.children[index]))
                    print(len(self.init_policy))
                    allValues[childIndex] = self.init_policy[childIndex-1]
                self.first_step==0
            print("length of all values", len(allValues))
            nextIndex = np.random.choice(list(allValues.keys()), 1,
                                         p=[x / sum(allValues.values()) for x in allValues.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]

            return self.treeTraversal(nextIndex)

        else:
            print("not fully expanded",len(self.children[index]))
            nprint("tree traversal terminated on node %s" % index)
            availableActions = copy.deepcopy(self.actions)
            # for k in self.usedActionsID.keys():
            #    for i in self.usedActionsID[k]: 
            #        availableActions[k].pop(i, None)
            return index, availableActions
        '''
    def treeTraversal(self, index):
        #print("Tree Traversal", self.fullyExpanded[index])
        if self.fullyExpanded[index] is True:
            nprint("tree traversal on node %s with children %s" % (index, self.children[index]))
            allValues = {}
            for childIndex in self.children[index]:
                # UCB values
                #print("fully expanded",len(self.children[index]))
                allValues[childIndex] = ((float(self.numberOfVisited[childIndex]) / self.cost[childIndex]) * min(5,self.eta[1])
                                         + explorationRate * math.sqrt(
                            math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex])))
                #print(allValues[childIndex])
            # for cooperative
            #print("length of all values", len(allValues))
            nextIndex = np.random.choice(list(allValues.keys()), 1,
                                         p=[x / sum(allValues.values()) for x in allValues.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]

            return self.treeTraversal(nextIndex)

        else:
            #print("not fully expanded",len(self.children[index]))
            nprint("tree traversal terminated on node %s" % index)
            availableActions = copy.deepcopy(self.actions)
            # for k in self.usedActionsID.keys():
            #    for i in self.usedActionsID[k]: 
            #        availableActions[k].pop(i, None)
            return index, availableActions

    def usefulAction(self, ampath, am):
        newAtomicManipulation = mergeTwoDicts(ampath, am)
        activations0 = self.moves.applyManipulation(self.X, ampath)
        newClass0, newConfident0 = self.model.predict_perturb(torch.from_numpy(activations0),self.NEIGHBORS)
        newClass0, newConfident0 = newClass0[self.node_index], newConfident0[self.node_index]
        activations1 = self.moves.applyManipulation(self.X, newAtomicManipulation)
        newClass1, newConfident1 = self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
        newClass1, newConfident1 = newClass1[self.node_index], newConfident1[self.node_index]
        if abs(newConfident0 - newConfident1) < 10 ** -6:
            return False
        else:
            return True

    def usefulActionNew(self, ampath, am, oldConfident):
        newAtomicManipulation = mergeTwoDicts(ampath, am)
        #print("usefulActionNew{} len of manipulation".format(len(newAtomicManipulation)))
        activations1 = self.moves.applyManipulation(self.X, newAtomicManipulation)
        dist = self.computeDistance(activations1)
        #print("dist is ",dist)
        newClass1, newConfident1 = self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
        newClass1, newConfident1 = newClass1[self.node_index], newConfident1[self.node_index]
        #print("usefulActionNew {} {}".format(newClass1,newConfident1))
        if abs(oldConfident - newConfident1) < 10 ** -6:
            return (False, (newClass1, newConfident1), dist)
        else:
            return (True, (newClass1, newConfident1), dist)

    def initialiseExplorationNode(self, index, availableActions):
        nprint("expanding %s" % index)
        if self.keypoint[index] != 0:
            for (actionId, am) in availableActions[self.keypoint[index]].items():
                if self.usefulAction(self.manipulation[index], am) == True:
                    self.indexToNow += 1
                    self.keypoint[self.indexToNow] = 0
                    self.indexToActionID[self.indexToNow] = actionId
                    self.initialiseLeafNode(self.indexToNow, index, am)
                    self.children[index].append(self.indexToNow)
        else:
            for kp in list(set(self.keypoints.keys()) - set([0])):
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = kp
                self.indexToActionID[self.indexToNow] = 0
                self.initialiseLeafNode(self.indexToNow, index, {})
                self.children[index].append(self.indexToNow)

        self.fullyExpanded[index] = True
        self.usedActionsID = {}
        return self.children[index]

    def backPropagation(self, index, value):
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent:
            nprint("start backPropagating the value %s from node %s, whose parent node is %s" % (
                value, index, self.parent[index]))
            self.backPropagation(self.parent[index], value)
        else:
            nprint("backPropagating ends on node %s" % index)

    # start random sampling and return the Euclidean value as the value
    def sampling(self, index, availableActions):
        nprint("start sampling node %s" % index)
        availableActions2 = copy.deepcopy(availableActions)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples):
            self.atomicManipulationPath = self.manipulation[index]
            self.depth = 0
            self.availableActionIDs = {}
            for k in self.keypoints.keys():
                self.availableActionIDs[k] = list(availableActions2[k].keys())
            self.usedActionIDs = {}
            for k in self.keypoints.keys():
                self.usedActionIDs[k] = []
            activations1 = self.moves.applyManipulation(self.X, self.atomicManipulationPath)
            result1,result2 = self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
            result1=result1[self.node_index]
            result2=result2[self.node_index]
            result=(result1,result2)
            #print(result)
            dist = self.computeDistance(activations1)
            
            #print("samplenext {} {} {}".format(self.keypoint[index],result,dist))
            (childTerminated, val) = self.sampleNext(self.keypoint[index],result,dist)
            self.numOfSampling += 1
            sampleValues.append(val)
            i += 1
        return childTerminated, min(sampleValues)

    def computeDistance(self, newImage):
        (distMethod, _) = self.eta
        if distMethod == "L2":
            dist = l2Distance(newImage, self.X)
        elif distMethod == "L1":
            dist = l1Distance(newImage, self.X)
        elif distMethod == "Percentage":
            dist = diffPercent(newImage, self.X)
        elif distMethod == "NumDiffs":
            dist = diffPercent(newImage, self.X) * self.X.size
        return dist

    def sampleNext(self, k, newResult, dist):
        self.count=self.count+1
        #if self.count==100:
            #return (self.depth == 0, dist)
        (newClass, newConfident) = newResult

        (distMethod, distVal) = self.eta
        # need not only class change, but also high confidence adversary examples
        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            #print("class changed to {} from {}".format(newClass, self.originalClass))
            #print("with probability {}".format(newConfident))
            nprint("sampling a path ends in a terminal node with depth %s... " % self.depth)
            #print("before scrutinize", self.atomicManipulationPath)
            self.atomicManipulationPath = self.scrutinizePath(self.atomicManipulationPath)
            #print("afterscrutinize", self.atomicManipulationPath)
            self.numAdv += 1
            #print("PATH",self.atomicManipulationPath)
            nprint("current best %s, considered to be replaced by %s" % (self.bestCase[0], dist))

            if self.bestCase[0] > dist:
                print("update best case from %s to %s" % (self.bestCase[0], dist))
                self.numConverge += 1
                self.bestCase = (dist, self.atomicManipulationPath)
                #(len(self.atomicManipulationPath))
                #path0 = "%s_pic/%s_Unsafe_currentBest_%s.png" % (self.data_set, self.image_index, self.numConverge)
                #print("PATH",self.atomicManipulationPath)
                activations1 = self.moves.applyManipulation(self.X, self.atomicManipulationPath)
                temp_class,temp_confident=self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
                temp_class, temp_confident = temp_class[self.node_index], temp_confident[self.node_index]
                #print("temp class {} and confid {}".format(temp_class,temp_confident))
                #self.model.save_input(activations1, path0)
            if self.bestCase[0]<sys.maxsize:
                with open(self.file_name, 'a') as f:
                    f.write(str(self.bestCase[0])+ '\n')
            return (self.depth == 0, dist)

        elif dist > distVal:   ##########################
            #print("sampling a path ends by eta with depth %s ... " % self.depth)
            return (self.depth == 0, distVal)

        elif (not list(set(self.availableActionIDs[k]) - set(self.usedActionIDs[k]))) or len(self.availableActionIDs[k])==0: ####################
            print("sampling a path ends with depth %s because no more actions can be taken ... " % self.depth)
            return (self.depth == 0, distVal)

        else:
            #print("continue sampling node ... ")
            # randomActionIndex = random.choice(list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k])))

            i = 0
            while True:
                '''
                if k!=0 and self.hop!=0:
                    #print(len(self.availableActionIDs[k]))
                    #print(len(self.SCORES[k]))
                    randomActionIndex = np.random.choice(self.availableActionIDs[k],p=self.SCORES[k-1])
                else:
                    ''' 
                randomActionIndex = random.choice(self.availableActionIDs[k])
                #print("length of self available id", len(self.availableActionIDs[k]))
                if k == 0:
                    nextAtomicManipulation = {}
                else:
                    nextAtomicManipulation = self.actions[k][randomActionIndex]
                newResult = self.usefulActionNew(self.atomicManipulationPath,nextAtomicManipulation,newConfident)
                if nextAtomicManipulation == {} or i > 10 or newResult[0] or len(self.availableActionIDs[k])==0:
                    #if(k!=0):
                       #self.availableActionIDs[k].remove(randomActionIndex)
                       #self.usedActionIDs[k].append(randomActionIndex)
                    break

                i += 1

                
            newManipulationPath = mergeTwoDicts(self.atomicManipulationPath, nextAtomicManipulation)
            #activations2 = self.moves.applyManipulation(self.image, newManipulationPath)
            #(newClass2, newConfident2) = self.model.predict(activations2)

            self.atomicManipulationPath = newManipulationPath
            self.depth = self.depth + 1
            #print(self.depth)
            if k == 0:
                #print("recursive k==0, newResults[1]{},[2]{}".format(newResult[1],newResult[2]))
                return self.sampleNext(randomActionIndex,newResult[1],newResult[2])
            else:
                #print("recursive k!=0, newResults[1]{},[2]{}".format(newResult[1],newResult[2]))
                #print(newResult[2])
                return self.sampleNext(0,newResult[1],newResult[2])

    def scrutinizePath(self, manipulations):
        flag = False
        tempManipulations = copy.deepcopy(manipulations)
        #print("apply manipulation", manipulations.keys())
        for k, v in manipulations.items():
            tempManipulations.pop(k)
            activations1 = self.moves.applyManipulation(self.X, tempManipulations)
            newClass, newConfident = self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
            newClass=newClass[self.node_index]
            newConfident=newConfident[self.node_index]
            if newClass != self.originalClass:
                manipulations.pop(k)
                flag = True
                break

        if flag is True:
            return self.scrutinizePath(manipulations)
        else:
            return manipulations

    def terminalNode(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        (newClass, _) = self.model.predict_perturb(torch.from_numpy(activations1),self.NEIGHBORS)
        newClass=newClass[self.node_index]
        return newClass != self.originalClass

    def terminatedByEta(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        dist = self.computeDistance(activations1)
        nprint("terminated by controlled search: distance = %s" % dist)
        return dist > self.eta[1]

    def applyManipulation(self, manipulation):
        activations1 = self.moves.applyManipulation(self.X, manipulation)
        return activations1

    def l2Dist(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        return l2Distance(self.X, activations1)

    def l1Dist(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        return l1Distance(self.X, activations1)

    def l0Dist(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        return l0Distance(self.X, activations1)

    def diffImage(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        return diffImage(self.X, activations1)

    def diffPercent(self, index):
        activations1 = self.moves.applyManipulation(self.X, self.manipulation[index])
        return diffPercent(self.X, activations1)