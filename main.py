from __future__ import print_function

import sys
import os
sys.setrecursionlimit(10000)
from NeuralNetwork import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# the first way of defining parameters
if len(sys.argv) == 9:

    if sys.argv[1] == 'Cora' or sys.argv[1] == 'CiteSeer' or sys.argv[1]== 'KarateClub' :
        dataSetName = sys.argv[1]
    else:
        print("please specify as the 1st argument the dataset: Cora or CiteSeer")
        exit

    if sys.argv[2] == 'ub' or sys.argv[2] == 'lb':
        bound = sys.argv[2]
    else:
        print("please specify as the 2nd argument the bound: ub or lb")
        exit

    if sys.argv[3] == 'cooperative' or sys.argv[3] == 'competitive':
        gameType = sys.argv[3]
    else:
        print("please specify as the 3nd argument the game mode: cooperative or competitive")
        exit

    if isinstance(int(sys.argv[4]), int):
        image_index = int(sys.argv[4])
    else:
        print("please specify as the 4th argument the index of the image: [int]")
        exit

    if sys.argv[5] == 'L0' or sys.argv[5] == 'L1' or sys.argv[5] == 'L2':
        distanceMeasure = sys.argv[5]
    else:
        print("please specify as the 5th argument the distance measure: L0, L1, or L2")
        exit

    if isinstance(float(sys.argv[6]), float):
        distance = float(sys.argv[6])
    else:
        print("please specify as the 6th argument the distance: [int/float]")
        exit
    eta = (distanceMeasure, distance)

    if isinstance(float(sys.argv[7]), float):
        tau = float(sys.argv[7])
    else:
        print("please specify as the 7th argument the tau: [int/float]")
        exit
    
    if isinstance(int(sys.argv[8]), int):
        hop_neighbor=int(sys.argv[8])
    else:
        print("please specify as the 8th argument the number of hops to attacke: [int]")
    
#by default, direct attack
elif len(sys.argv) == 1:
    # the second way of defining parameters
    dataSetName = 'Cora'
    bound = 'lb'
    gameType = 'cooperative'
    image_index = 7
    eta = ('L2', 10)
    tau = 1
    hop_neighbor=0

# calling algorithms
# dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, tau, gameType, image_index, eta[0], eta[1]))
# dc.initialiseIndex(image_index)

if bound == 'ub':
    (elapsedTime, newConfident, percent, l2dist, maxFeatures) = upperbound(dataSetName,image_index,
                                                                                           gameType,eta,tau,hop_neighbor)

    # dc.addRunningTime(elapsedTime)dataset_name, node_index,gameType, eta, tau,bound
    # dc.addConfidence(newConfident)
    # dc.addManipulationPercentage(percent)
    # dc.addl2Distance(l2dist)
    # dc.addl1Distance(l1dist)
    # dc.addl0Distance(l0dist)
    # dc.addMaxFeatures(maxFeatures)

elif bound == 'lb':
    lowerbound(dataSetName, image_index, gameType, eta, tau,hop_neighbor)

else:
    print("Unrecognised bound setting.\n"
          "Try 'ub' for upper bound or 'lb' for lower bound.\n")
    exit

# dc.provideDetails()
# dc.summarise()
# dc.close()

