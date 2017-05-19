import os
import random
import math
import copy
DATASET = "./lfw-deepfunneled"
TRAIN_TXT = "train.txt" 
VAL_TXT = "val.txt" 
PIC_FORMATS = ["jpg", "png", "bmp", "jpeg"]
RATIO = 0.8
MIN_PIC_NUM = 5
PERSON_ID = 0

def get_data(person_name):
    global PERSON_ID
    pid = PERSON_ID
    res = []
    dir_name = "%s/" % person_name
    for name in os.listdir(DATASET + "/" + dir_name):
        if name.split(".")[-1] in PIC_FORMATS:
            res.append((pid, dir_name + name))
    if len(res) < MIN_PIC_NUM:
        pid -= 1
        return []
    PERSON_ID += 1
    random.shuffle(res)
    return res[:MIN_PIC_NUM]

def split_data(data):
    random.shuffle(data)
    i = int(math.ceil(len(data) * RATIO))
    return data[:i], data[i:]

def write_file(data, fn):
    fout = open(DATASET + "/" + fn, "w")
    for idx, name in data:
        fout.write("%s %d\n" % (name, idx))

train = []
val = []
people = 0
for i in os.listdir(DATASET):
    try:
        train_p,val_p = split_data(get_data(i))
        if (len(val_p)):
            people += 1
        train.extend(train_p)
        val.extend(val_p)
    except:
        print ("Ignore: %s" % i)

print ("People: %d" % people)
print (len(train), len(val))
'''
rtrain = copy.copy(train)
for i in range(10):
    random.shuffle(rtrain)
    train.extend(rtrain)
'''
write_file(train, TRAIN_TXT)
write_file(val, VAL_TXT)
