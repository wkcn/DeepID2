import os
import random
TRAIN_TXT = "train.txt" 
VAL_TXT = "val.txt" 
PIC_FORMATS = ["jpg", "png", "bmp", "jpeg"]
RATIO = 0.7

def get_data(person_id):
    res = []
    dir_name = "%d/" % person_id
    for name in os.listdir(dir_name):
        if name.split(".")[-1] in PIC_FORMATS:
            res.append((person_id, dir_name + name))
    return res

def split_data(data):
    random.shuffle(data)
    i = int(len(data) * RATIO)
    return data[:i], data[i:]

def write_file(data, fn):
    fout = open(fn, "w")
    for idx, name in data:
        fout.write("%s %d\n" % (name, idx))

train = []
val = []
for i in range(3):
    train_p,val_p = split_data(get_data(i))
    train.extend(train_p)
    val.extend(val_p)

write_file(train, TRAIN_TXT)
write_file(val, VAL_TXT)
