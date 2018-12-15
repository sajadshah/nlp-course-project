import io
import os

f = io.open(os.path.join('sst', 'd2v_train.txt'), mode='r', encoding='utf-8')
fw = io.open(os.path.join('sst', 'd2v_train_small.txt'), mode='w', encoding='utf-8')
count = 0
for l in f:
    fw.write(l)
    count += 1
    if count == 250000:
        break
f.close()
fw.close()
