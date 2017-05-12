#coding=utf-8
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import lmdb
import numpy as np
import os
import sys
from numpy import linalg as la
import matplotlib.pyplot as plt 

CAFFE_HOME = "/opt/caffe/"
RESULT_DIR = "./result/"

deploy = "./proto/cifar10_quick.prototxt"
caffe_model = CAFFE_HOME + "/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5" 
train_db = "./examples/DeepID2_train_lmdb" 
test_db = "./examples/DeepID2_test_lmdb" 
mean_proto = "./examples/DeepID2_mean.proto" 
#mean_npy = "./mean.npy"
#mean_pic = np.load(mean_npy)

def read_db(db_name):
    lmdb_env = lmdb.open(db_name)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    X = []
    y = []
    cnts = {}
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        data = data.swapaxes(0, 2).swapaxes(0, 1)
        X.append(data)
        y.append(label)
        if label not in cnts:
            cnts[label] = 0
        cnts[label] += 1
        data[:,:,[0,1,2]] = data[:,:,[2,1,0]]
        print (label)
        plt.imshow(data)
        plt.show()
    return X, np.array(y), cnts

testX, testy, cnts = read_db(test_db)
#testX, testy, cnts = read_db(train_db)
print ("#train set: ", len(testX))
print ("the size of sample:", testX[0].shape)
