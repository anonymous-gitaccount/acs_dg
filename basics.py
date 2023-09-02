#!/usr/bin/env python


import numpy as np
import math
import time
import os
import copy
from scipy.sparse.linalg import norm

## current implementation: assume all inputs are numpy already
def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def current_milli_time():
    return int(round(time.time() * 1000) % 4294967296)


def diffMatrix(matrix1, matrix2):
    #print(list(zip(*np.nonzero(np.subtract(matrix1, matrix2)))))
    return list(zip(*np.nonzero(np.subtract(matrix1, matrix2))))


def diffPercent(matrix1, matrix2):
    return len(diffMatrix(matrix1, matrix2)) / float(np.prod(matrix1.shape))


def numDiffs(matrix1, matrix2):
    return len(diffMatrix(matrix1, matrix2))


def l2Distance(matrix1, matrix2):
    #print(np.subtract(matrix1,matrix2).shape)
    #print(np.square(np.subtract(matrix1,matrix2)).shape)
    #start=time.time()
    x=np.linalg.norm(matrix1-matrix2) # 0.004
    #x=np.sqrt(np.sum(np.square(np.subtract(matrix1, matrix2))))
    #end=time.time()
    #print(end-start)
    return x
    #return np.sqrt(np.sum(np.square(np.subtract(matrix1, matrix2))))


def l1Distance(matrix1, matrix2):
    return np.sum(np.absolute(np.subtract(matrix1, matrix2)))

def l0Distance(matrix1, matrix2):
    return np.count_nonzero(np.absolute(np.subtract(matrix1, matrix2)))


def mergeTwoDicts(x, y):
    z = x.copy()
    for key, value in y.items():
        if key in z.keys():
            z[key] += y[key]
        else:
            z[key] = y[key]
    # z.update(y)
    return z


def nprint(str):
    return 0


def printDict(dictionary):
    for key, value in dictionary.items():
        print("%s : %s" % (key, value))
    print("\n")