#coding=utf-8

# DeepID2 FaceVerification

import caffe
#import lmdb
import numpy as np
import random
import cv2
import json
import os

def crop_img(im, tsize = (128,128), crop_min_ratio = 0.5, gray_ratio = 0.05):
    #positions, scales, color channels, and horizontal flipping
    ratio = random.random() * (1 - crop_min_ratio) + crop_min_ratio 
    rows, cols, ts = im.shape
    nr = int(rows * ratio)
    nc = int(cols * ratio)
    sr = int(random.random() * (rows - nr))
    sc = int(random.random() * (cols - nc))
    nim = im[sr:sr+nr, sc:sc+nc, :]
    if random.random() > 0.5:
        # horizontal flipping
        nim = nim[:,::-1,:]
    if random.random() < gray_ratio:
        nim = cv2.cvtColor(cv2.cvtColor(nim, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    nim = cv2.resize(nim, tsize)
    return nim

def get_rand2(ma):
    # [0, ma)
    ma -= 1
    e1 = random.randint(0, ma)
    e2 = random.randint(0, ma)
    while e1 == e2:
        e2 = random.randint(0, ma)
    return (e1, e2)


class data_pair_layer(caffe.Layer):
    def setup(self, bottom, top):
        params = json.loads(self.param_str.replace("\'", "\""))
        self.dir = params["dir"]
        self.batch_size = params["batch_size"]
        self.ratio = params["ratio"]
        self.mean_file = params["mean_file"]
        self.source = params["source"]
        self.rows = params["rows"]
        self.cols = params["cols"]
        self.label_num = params["label_num"]
        top[0].reshape(self.batch_size, 3, self.rows, self.cols) 
        top[1].reshape(self.batch_size, 1)


        # npy_mean
        blob = caffe.proto.caffe_pb2.BlobProto()
        bin_mean = open(self.mean_file, 'rb').read()
        blob.ParseFromString(bin_mean)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        self.npy_mean = arr[0]

        # Buffer
        self.buffer = [[] for _ in range(self.label_num)]
        fin = open(self.source, "r")
        tsize = (self.rows, self.cols)
        for line in fin.readlines():
            filename, label = line.split(" ")
            im = cv2.imread(os.path.join(self.dir, filename))
            im = cv2.resize(im, tsize).swapaxes(0,2)
            me = im - self.npy_mean
            self.buffer[int(label)].append(me)
    def forward(self, bottom, top):
        self.ids = []
        for i in range(self.batch_size // 2):
            if random.random() < self.ratio:
                # + 
                t = random.randint(0, self.label_num - 1)
                e1, e2 = get_rand2(len(self.buffer[t]))
                self.ids.append((t,e2)) 
            else:
                t1, t2 = get_rand2(self.label_num)
                e1 = random.randint(0, len(self.buffer[t1]) - 1)
                e2 = random.randint(0, len(self.buffer[t2]) - 1)
                self.ids.append((t1,e1)) 
                self.ids.append((t2,e2)) 
        tsize = (self.rows, self.cols)
        top[0].data[...] = np.require(map(lambda t : crop_img(self.buffer[t[0]][e[1]], tsize), self.ids))
        top[1].data[...] = np.require(map(lambda t : t[0], self.ids))
    def backward(self, bottom, top):
        pass
    def reshape(self, bottom, top):
        pass

if __name__ == "__main__":
    '''
    im = cv2.imread("../lfw-deepfunneled/Abdullah/Abdullah_0001.jpg")
    nim = crop_img(im)
    cv2.imshow("nim", nim)
    cv2.waitKey(0)
    '''
    '''
    db_name = "../DeepID2_train_test.prototxt"
    lmdb_env = lmdb.open(db_name)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    print (lmdb_cursor)
    '''
    pass
