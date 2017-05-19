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

deploy = "./DeepID2_train_test.prototxt" 
caffe_model = "./model/_iter_35273.caffemodel"
train_db = "./examples/DeepID2_train_lmdb" 
test_db = "./examples/DeepID2_test_lmdb" 
mean_proto = "./examples/DeepID2_mean.proto" 

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
        #data = data.swapaxes(0, 2).swapaxes(0, 1)
        X.append(data)
        y.append(label)
        if label not in cnts:
            cnts[label] = 0
        cnts[label] += 1
        #plt.imshow(data)
        #plt.show()
    return X, np.array(y), cnts

testX, testy, cnts = read_db(test_db)
#testX, testy, cnts = read_db(train_db)
print ("#train set: ", len(testX))
print ("the size of sample:", testX[0].shape)
print ("kinds: ", cnts)

# Load model and network
net = caffe.Net(deploy, caffe_model, caffe.TEST) 
for layer_name, param in net.params.items():
#    # 0 is weight, 1 is biases
    print (layer_name, param[0].data.shape,net.blobs[layer_name].data.shape)
    print param[0].data

#print (type(net.params))
#print (net.params.keys())
#print (net.params["ip2"][0].data.shape)
#print ("BIASES:")
#print (net.params["ip2"][1].data.shape)
