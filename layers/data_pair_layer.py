#coding=utf-8

# DeepID2 FaceVerification

import caffe
import lmdb
import numpy as np
import random
import cv2

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

class data_pair_layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

if __name__ == "__main__":
    im = cv2.imread("../lfw-deepfunneled/Abdullah/Abdullah_0001.jpg")
    nim = crop_img(im)
    cv2.imshow("nim", nim)
    cv2.waitKey(0)
