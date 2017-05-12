import os
import random
import math
DATASET = "./lfw-deepfunneled"
TRAIN_TXT = "train.txt" 
VAL_TXT = "val.txt" 
PIC_FORMATS = ["jpg", "png", "bmp", "jpeg"]
RATIO = 0.7
PERSONS = {}

def get_data(person_name):
    if person_name not in PERSONS:
        pid = len(PERSONS)
        PERSONS[person_name] = pid
    else:
        pid = PERSONS[person_name]
        print ("duplication!")
    res = []
    dir_name = "%s/" % person_name
    for name in os.listdir(DATASET + "/" + dir_name):
        if name.split(".")[-1] in PIC_FORMATS:
            res.append((pid, dir_name + name))
    return res

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
        people += 1
        train_p,val_p = split_data(get_data(i))
        train.extend(train_p)
        val.extend(val_p)
    except:
        print ("Ignore: %s" % i)

print ("People: %d" % people)
print (len(train), len(val))
write_file(train, TRAIN_TXT)
write_file(val, VAL_TXT)
