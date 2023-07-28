from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import torch.optim as optim
from torchvision import datasets, transforms
from tropic_dataset import *
from tropic_model import model_3, model_wv, model_mask_deep, model_compare, model_fuse_add
import cv2 as cv
from sklearn.metrics import mean_squared_error, mean_absolute_error


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def eva():
    C_mask = cv.imread(r"C_mask.png", cv.IMREAD_GRAYSCALE)
    file_out = "./out/npy"
    C_mask = C_mask[9:, 9:]
    names = os.listdir(file_out)
    masks = np.where(C_mask != 0)
    aaa = []
    bbb = []
    for name_fy in tqdm(names, ascii=True):
        m = name_fy.split(".")[0][-1]
        if m != "1":
            continue
        fnpy = np.load(os.path.join(file_out, name_fy), allow_pickle=True)
        #
        a = fnpy[2]
        b = fnpy[3]
        a[C_mask == 0] = 0
        b[C_mask == 0] = 0
        masks = np.where((a > 0))
        aa = a[masks]
        bb = b[masks]
        aaa.append(aa)
        bbb.append(bb)
    aaa = np.concatenate(aaa, axis=0)
    bbb = np.concatenate(bbb, axis=0)
    for i in [1, 5, 10, 20, 30]:
        a1 = np.where(aaa > i)
        mae1 = mean_absolute_error(aaa[a1], bbb[a1])
        rmse1 = np.sqrt(mean_squared_error(aaa[a1], bbb[a1]))   
        print(f"in:{i} linear: mae:{mae1}, rmse:{rmse1}")



def rain_level(im_rain_level):
    l1 = np.where(im_rain_level >= 40)
    l2 = np.where((im_rain_level < 40) & (im_rain_level >= 20))
    l3 = np.where((im_rain_level < 20) & (im_rain_level >= 10))
    l4 = np.where((im_rain_level < 10) & (im_rain_level >= 8))
    l5 = np.where((im_rain_level < 8) & (im_rain_level >= 6))
    l6 = np.where((im_rain_level < 6) & (im_rain_level >= 5))
    l7 = np.where((im_rain_level < 5) & (im_rain_level >= 4))
    l8 = np.where((im_rain_level < 4) & (im_rain_level >= 3))
    l9 = np.where((im_rain_level < 3) & (im_rain_level >= 2))
    l10 = np.where((im_rain_level < 2) & (im_rain_level >= 1))
    l11 = np.where((im_rain_level < 1) & (im_rain_level >= 0.5))
    l12 = np.where((im_rain_level < 0.5) & (im_rain_level >= 0.1))
    l13 = np.where((im_rain_level < 0.1) & (im_rain_level > 0))
    im_rain = np.zeros(im_rain_level.shape, dtype=np.uint8)
    im_rain = cv.cvtColor(im_rain, cv.COLOR_GRAY2BGR)
    im_rain[l1] = [205, 0, 125]
    im_rain[l2] = [255, 0, 185]
    im_rain[l3] = [155, 0, 255]
    im_rain[l4] = [0, 0, 185]
    im_rain[l5] = [0, 40, 160]
    im_rain[l6] = [0, 120, 160]
    im_rain[l7] = [0, 205, 225]
    im_rain[l8] = [0, 255, 255]
    im_rain[l9] = [45, 255, 0]
    im_rain[l10] = [0, 185, 0]
    # [185, 165, 0]
    # im_rain[l11] = [0, 0, 0]
    # im_rain[l12] = [0, 0, 0]
    # im_rain[l13] = [0, 0, 0]
    im_rain[l11] = [185, 165, 0]
    # im_rain[l12] = [255, 255, 0]
    # im_rain[l13] = [255, 255, 255]
    return im_rain


C_mask = cv.imread(r"C_mask.png", cv.IMREAD_GRAYSCALE)
C_mask = C_mask[9:, 9:]
# C_mask = torch.from_numpy(C_mask).float().to(device).view(1, 192, 192)
# im1 = torch.repeat_interleave(C_mask, repeats=batch_size, dim=0)
im0 = np.where(C_mask == 0)


def loss_fn(output, target):
    weight = torch.ones_like(target)
    weight[target > 10] = 2
    weight[target > 20] = 5
    weight[target > 30] = 10
    weight[target > 50] = 20
    loss_all0 = torch.sum(weight * (output - target) ** 2, (1, 2, 3))
    sweight = torch.sum(weight, (1, 2, 3))
    a = torch.mean(loss_all0 / sweight)
    loss_all = torch.mean(a)
    return loss_all


# model = model_2().to(device)
model_net = torch.load(r"./model_pt/25_w3.pt", map_location='cuda').to(device)
model_net.eval()
batch_size = 1
mydata = dataset(
    r"D:/rain_est/data/exp_data/exp_data/test",
    r"D:/rain_est/data/exp_data/exp_data/rain",
)
test_loader = torch.utils.data.DataLoader(mydata, batch_size, shuffle=False)
testloss1 = 0
testloss2 = 0
for i, (name, data, dem_grid,  target) in tqdm(enumerate(test_loader), ascii=True):
    if name[0][-1]!='1':
        continue
    data = data.to(device)
    target = target.to(device)
    dem_grid = dem_grid.to(device)
    data0 = (data - 150) / (330 - 150)
    output = model_net(data0, dem_grid)
    loss = loss_fn(output, target)
    outimg = data.cpu().numpy().reshape(2, 2, 192, 192)
    im_fy0 = 255 - 2 * (outimg[1, 0, :, :] - 180)
    im_fy0[im_fy0 < 0] = 0
    im_fy0[im_fy0 > 255] = 255
    im_fy0 = cv.cvtColor(im_fy0, cv.COLOR_GRAY2BGR)
    im_fy1 = 255 - 2 * (outimg[1, 1, :, :] - 180)
    im_fy1[im_fy1 < 0] = 0
    im_fy1[im_fy1 > 255] = 255
    im_fy1 = cv.cvtColor(im_fy1, cv.COLOR_GRAY2BGR)
    eeee = target.cpu().numpy().reshape(192, 192)
    aaaa = output.detach().cpu().numpy().reshape(192, 192)
    # eeee = cv.GaussianBlur(eeee, (5, 5), 2)
    # aaaa[aaaa < 2] = 0
    # aaaa = cv.GaussianBlur(aaaa, (5, 5), 2)
    # aaaa[aaaa < 1] = 0
    # eeee[eeee < 1] = 0
    # nomasks[nomasks < 2] = 0
    # nomasks = cv.GaussianBlur(nomasks, (3, 3), 1)
    # nomasks[nomasks < 1] = 0
    outlabel = rain_level(eeee)
    outputs = rain_level(aaaa)
    # outnomask = rain_level(nomasks)
    # im_rain = np.zeros(masks.shape, dtype=np.uint8)
    # im_rain = cv.cvtColor(im_rain, cv.COLOR_GRAY2BGR)
    # im_rain[np.where(masks > 0)] = [0, 155, 0]
    im_fy0 = im_fy0.astype(np.uint8)
    # im_1 = cv.addWeighted(im_fy0, 0.7, im_rain, 0.3, 0)
    # imgs = np.hstack([im_fy0, im_fy1, outlabel, outputs, im_1, outnomask])
    imgs = np.hstack([im_fy0, im_fy1, outlabel, outputs])
    # print(i, loss)
    name_save = os.path.join("./out", name[0] + ".png")
    cv.imwrite(name_save, imgs)
    im = []

    im.append(outimg[0])
    im.append(outimg[1])
    im.append(eeee)
    im.append(aaaa)
    name_save = os.path.join("./out/npy", name[0] + ".npy")
    np.save(name_save, im)
eva()
