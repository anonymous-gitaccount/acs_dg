#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a 'lowerbound' function to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative, or Player I's maximum
adversary distance whilst Player II being competitive.
Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""
import pickle
from CooperativeAStar import *
from NeuralNetwork import *
from FeatureExtraction_lb import *
import random
def lowerbound(dataset_name, node_index, game_type, eta, tau,hop_neighbor):
    dataset = Planetoid(root='.', name=dataset_name)
    data=dataset[0]
    NN = NeuralNetwork(dataset_name)
    NN.load_network()
    print("Dataset is %s." % NN.dataset)
    #NN.model.summary()
    X=data.x
    print(X.shape)
    (label, confidence) = NN.predict(X)
    label=label[node_index]
    confidence=confidence[node_index]
    label_str = NN.get_label(int(label))
    print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (node_index, label_str, confidence))
    print("The second player is being %s." % game_type)

    #currently NN.save_input undefined
    #path = "%s_pic/idx_%s_label_[%s]_with_confidence_%s.png" % (
        #dataset_name, image_index, label_str, confidence)
    #NN.save_input(image, path)

    if game_type == 'cooperative':
        tic = time.time()
        cooperative = CooperativeAStar(dataset_name, X, node_index, NN, eta, tau,hop_neighbor)
        cooperative.play_game(node_index)
        if cooperative.ADVERSARY_FOUND is True:
            elapsed = time.time() - tic
            adversary = cooperative.ADVERSARY
            #print(adversary.shape)
            feature_extraction = FeatureExtraction_lb(dataset=dataset_name)
            _,neighbors = feature_extraction.get_partitions(node_index,hop_neighbor, num_partition=10)
        
            adv_label, adv_confidence = NN.predict_perturb(torch.from_numpy(adversary),neighbors)
            
            adv_label=adv_label[node_index]
            adv_confidence=adv_confidence[node_index]
            adv_label_str = NN.get_label(int(adv_label))

            print("\nFound an adversary within pre-specified bounded computational resource. "
                  "\nThe following is its information: ")
            print("difference between images: %s" % (diffMatrix(X[neighbors,:], adversary)))
            l2dist = l2Distance(X[neighbors,:], adversary)
            #l1dist = l1Distance(image, adversary)
            #l0dist = l0Distance(image, adversary)
            percent = diffPercent(X[neighbors,:], adversary)
            print("L2 distance %s" % l2dist)
            #print("L1 distance %s" % l1dist)
            #print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (adv_label_str, adv_confidence))

            #path = "%s_pic/idx_%s_modified_into_[%s]_with_confidence_%s.png" % (
            #    dataset_name, image_index, adv_label_str, adv_confidence)
            #NN.save_input(adversary, path)
            if eta[0] == 'L0':
                dist = l0dist
            elif eta[0] == 'L1':
                dist = l1dist
            elif eta[0] == 'L2':
                dist = l2dist
            else:
                print("Unrecognised distance metric.")
            #path = "%s_pic/idx_%s_modified_diff_%s=%s_time=%s.png" % (
            #    dataset_name, image_index, eta[0], dist, elapsed)
            #NN.save_input(np.absolute(image - adversary), path)
        else:
            print("Adversarial distance exceeds distance budget.")

    elif game_type == 'competitive':
        competitive = CompetitiveAlphaBeta(dataset_name, data.x, node_index, NN, eta, tau)
        competitive.play_game(node_index)

    else:
        print("Unrecognised game type. Try 'cooperative' or 'competitive'.")