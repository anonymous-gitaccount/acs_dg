from __future__ import print_function
from NeuralNetwork import *
from CooperativeMCTS import *
from FeatureExtraction_ub import *
def upperbound(dataset_name, node_index,gameType, eta, tau,hop_neighbor):
    start_time = time.time()
    MCTS_all_maximal_time = 300
    MCTS_level_maximal_time = 60
    dataset = Planetoid(root='.', name=dataset_name)
    data=dataset[0]
    NN = NeuralNetwork(dataset_name)
    NN.load_network()
    data=dataset[0]
    #dataset = DataSet(dataSetName, 'testing')
    X=data.x
    (label, confident) = NN.predict(X)
    #print()
    label = label[node_index]
    confident = confident[node_index]
    origClassStr = NN.get_label(int(label))
    feature_extraction = FeatureExtraction_ub(dataset=dataset_name)
    _,neighbors,_ = feature_extraction.get_partitions(node_index,hop_neighbor, num_partition=10)
        
    print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (node_index, origClassStr, confident))
    print("the second player is %s." % gameType)

    # tau = 1
    X1=0
    # choose between "cooperative" and "competitive"
    if gameType == 'cooperative':
        # data_set, model, node_index, X, tau, eta
        mctsInstance = MCTSCooperative(dataset_name, NN, node_index, X, tau, eta,hop_neighbor)
        mctsInstance.initialiseMoves()
        start_time_all = time.time()
        runningTime_all = 0
        start_time_level = time.time()
        runningTime_level = 0 
        currentBest = eta[1]
        while runningTime_all <= MCTS_all_maximal_time:
        
            '''
            if runningTime_level > MCTS_level_maximal_time: 
                bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
                # pick the current best move to take  
                mctsInstance.makeOneMove(bestChild)
                start_time_level = time.time()
            '''
             

            # Here are three steps for MCTS
            #print("rootIndex is", mctsInstance.rootIndex)
            (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
            #print("length of available actions",len(availableActions))
            newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
            #print(len(newNodes))
            for node in newNodes:
                #print("newNodes")
                (_, value) = mctsInstance.sampling(node, availableActions)
                mctsInstance.backPropagation(node, value)
            #print("one sampling finished")
            #print(currentBest)
            #print(mctsInstance.bestCase[0])
            if currentBest > mctsInstance.bestCase[0]:
                print("best distance up to now is %s" % (str(mctsInstance.bestCase[0])))
                currentBest = mctsInstance.bestCase[0]
            bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            X1 = mctsInstance.applyManipulation(bestManipulation)
            #path0 = "%s_pic/%s_Unsafe_currentBest.png" % (dataSetName, image_index)
            #NN.save_input(image1, path0)

            runningTime_all = time.time() - start_time_all
            runningTime_level = time.time() - start_time_level

        (_, bestManipulation) = mctsInstance.bestCase
        #print(len(bestManipulation))
        #print(bestManipulation)
        #print(bestManipulation)
        print("the number of sampling: %s" % mctsInstance.numOfSampling)
        print("the number of adversarial examples: %s\n" % mctsInstance.numAdv)
        
        X_final = mctsInstance.applyManipulation(bestManipulation)
        newClass, newConfident = NN.predict_perturb(torch.from_numpy(X_final),neighbors)
        newClass, newConfident = newClass[node_index], newConfident[node_index]
        newClassStr = NN.get_label(int(newClass))
        predict_prob=NN.predict_prob(torch.from_numpy(X_final),neighbors)[node_index]
        print(predict_prob)
        print(newClass)
        print(newConfident)
        print(confident)
        print(label)
        if newClass != label:
            #path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
               # dataSetName, NODE_index, origClassStr, newClassStr, newConfident)
            #NN.save_input(image1, path0)
            #path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
            #NN.save_input(np.absolute(image - image1), path0)
            print("\nfound an adversary example within pre-specified bounded computational resource. "
                  "The following is its information: ")
            print("difference between feature matrix: %s" % (len(diffMatrix(data.x[neighbors], X1))))

            print("number of adversarial examples found: %s" % mctsInstance.numAdv)

            l2dist = l2Distance(mctsInstance.X, X1)
            #l1dist = l1Distance(mctsInstance.X, X1)
            #l0dist = l0Distance(mctsInstance.X, X1)
            percent = diffPercent(mctsInstance.X, X1)
            print("L2 distance %s" % l2dist)
            #print("L1 distance %s" % l1dist)
            #print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))
            with open(mctsInstance.file_name, 'r') as f:
                lines = f.readlines()
            data_list = [float(line.strip()) for line in lines]

            plt.plot(data_list)
            plt.xlabel('iteration')
            plt.ylabel('L2 distance')
            plt.title('Upperbound')
            if hop_neighbor == 0:
                os.makedirs('plots/upperbound/direct/', exist_ok=True)
                file_name = 'plots/upperbound/direct/' + str(node_index) + '.png'
            else:
                os.makedirs('plots/upperbound/indirect/', exist_ok=True)
                file_name = 'plots/upperbound/indirect/' + str(node_index) + '.png'
            plt.savefig(file_name)  

            return time.time() - start_time_all, newConfident, percent, l2dist, 0

        else:
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
            return 0, 0, 0, 0, 0, 0

    else:
        print("Unrecognised game type. Try 'cooperative'.")

    runningTime = time.time() - start_time


'''

    if gameType == 'cooperative':
        mctsInstance = MCTSCooperative(dataSetName, NN, image_index, image, tau, eta)
        mctsInstance.initialiseMoves()

        start_time_all = time.time()
        runningTime_all = 0
        numberOfMoves = 0
        while (not mctsInstance.terminalNode(mctsInstance.rootIndex) and
               not mctsInstance.terminatedByEta(mctsInstance.rootIndex) and
               runningTime_all <= MCTS_all_maximal_time):
            print("the number of moves we have made up to now: %s" % numberOfMoves)
            l2dist = mctsInstance.l2Dist(mctsInstance.rootIndex)
            l1dist = mctsInstance.l1Dist(mctsInstance.rootIndex)
            l0dist = mctsInstance.l0Dist(mctsInstance.rootIndex)
            percent = mctsInstance.diffPercent(mctsInstance.rootIndex)
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("manipulated dimensions %s" % diffs)

            start_time_level = time.time()
            runningTime_level = 0
            childTerminated = False
            currentBest = eta[1]
            while runningTime_level <= MCTS_level_maximal_time:
                # Here are three steps for MCTS
                (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
                newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
                for node in newNodes:
                    (childTerminated, value) = mctsInstance.sampling(node, availableActions)
                    mctsInstance.backPropagation(node, value)
                runningTime_level = time.time() - start_time_level
                if currentBest > mctsInstance.bestCase[0]: 
                    print("best possible distance up to now is %s" % (str(mctsInstance.bestCase[0])))
                    currentBest = mctsInstance.bestCase[0]
            bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
            # pick the current best move to take  
            mctsInstance.makeOneMove(bestChild)

            image1 = mctsInstance.applyManipulation(mctsInstance.manipulation[mctsInstance.rootIndex])
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            path0 = "%s_pic/%s_temp_%s.png" % (dataSetName, image_index, len(diffs))
            NN.save_input(image1, path0)
            (newClass, newConfident) = NN.predict(image1)
            print("confidence: %s" % newConfident)

            # break if we found that one of the children is a misclassification
            if childTerminated is True:
                break

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            image1 = mctsInstance.applyManipulation(bestManipulation)
            path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
            NN.save_input(image1, path0)

            numberOfMoves += 1
            runningTime_all = time.time() - start_time_all

        (_, bestManipulation) = mctsInstance.bestCase

        image1 = mctsInstance.applyManipulation(bestManipulation)
        (newClass, newConfident) = NN.predict(image1)
        newClassStr = NN.get_label(int(newClass))

        if newClass != label:
            path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
                dataSetName, image_index, origClassStr, newClassStr, newConfident)
            NN.save_input(image1, path0)
            path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
            NN.save_input(np.subtract(image, image1), path0)
            print("\nfound an adversary image within pre-specified bounded computational resource. "
                  "The following is its information: ")
            print("difference between images: %s" % (diffImage(image, image1)))

            print("number of adversarial examples found: %s" % mctsInstance.numAdv)

            l2dist = l2Distance(mctsInstance.image, image1)
            l1dist = l1Distance(mctsInstance.image, image1)
            l0dist = l0Distance(mctsInstance.image, image1)
            percent = diffPercent(mctsInstance.image, image1)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))

            return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, 0

        else:
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
            return 0, 0, 0, 0, 0, 0, 0


'''
