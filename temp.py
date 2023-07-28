import os
import numpy as np

p=r"D:\tju_wpp\deep_regression\npy_fengyun"
p1=r"D:\tju_wpp\deep_regression\test.txt"
p2=r"D:\tju_wpp\deep_regression\testdata.txt"
with open(p1) as f:
    lines = f.readlines()
num = [x.split()[0] for x in lines]
list_dir=os.listdir(p)
list_dir.sort(key=lambda x: int(x.split("_")[0]))
for name in list_dir:
    for i in range(len(num)):
        if num[i]+"_IR1.npy"==name:
            with open(p2, 'a+') as f1:
                f1.write(lines[i])