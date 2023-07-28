import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from tropic_dataset import dataset
from tropic_model import (
    model_3,
    model_wv,
    model_mask_deep,
    model_dem,
    model_compare,
    model_fuse_add,
)
import os
from matplotlib import pyplot as plt
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import cv2 as cv

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
p_dir = "runs/" + TIMESTAMP
writer = SummaryWriter(p_dir)

torch.manual_seed(1)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(1)
else:
    device = torch.device("cpu")

model = model_fuse_add().to(device)
# model = Net().to(device)
model.train()

epochs = 50
batch_size = 16


# C_mask = cv.imread(r"C_mask.png", cv.IMREAD_GRAYSCALE)
# C_mask = C_mask[9:, 9:]
# C_mask = torch.from_numpy(C_mask).float().to(device).view(1, 192, 192)
# # im1 = torch.repeat_interleave(C_mask, repeats=batch_size, dim=0)
# im0 = torch.where(C_mask == 0)


def loss_thres(output, target, mm, th1, th2):
    weight = torch.ones_like(target)
    # weight[target > th2] = 0
    loss_all = torch.sum(weight * mm, (1, 2, 3))
    sweight = torch.sum(weight, (1, 2, 3))
    loss = torch.mean(loss_all / (sweight + 0.0001))
    return loss


def loss_fn(output, target):
    weight = torch.ones_like(target)
    # weight[target < 0.1] = 2
    # weight[target > 1] = 2
    # weight[target > 5] = 5
    # weight[target > 10] = 10
    # weight[target > 20] = 20
    # weight[target > 30] = 30
    # w1
    # weight[target > 1] = 2
    # w2
    # weight[target > 1] = 2
    # weight[target > 5] = 5
    #w3
    weight[target > 1] = 2
    weight[target > 5] = 5
    weight[target > 10] = 10
    # for i in range(target.shape[0]):
    #     weight[i][im0] = 0
    loss_all0 = torch.sum(weight * (output - target) ** 2, (1, 2, 3))
    sweight = torch.sum(weight, (1, 2, 3))
    a = ssim(output, target, data_range=40)
    b = torch.mean(loss_all0 / sweight)
    loss_all = b + 0.1 * a
    mm = (output - target) ** 2
    loss_1 = loss_thres(output, target, mm, 1, 5)
    loss_5 = loss_thres(output, target, mm, 5, 10)
    loss_10 = loss_thres(output, target, mm, 10, 20)
    loss_20 = loss_thres(output, target, mm, 20, 30)
    loss_30 = loss_thres(output, target, mm, 30, 150)
    loss0 = torch.stack((loss_1, loss_5, loss_10, loss_20, loss_30))

    return loss_all, loss0


# 模型保存
path1 = r"model_pt"
# mydata = dataset(
#     r"/mnt/harddisk0/wpp_data/rain_data/sample/train_wv",
#     r"/mnt/harddisk0/wpp_data/rain_data/exp_data/train/rain",
# )
# mydataval = dataset(
#     r"/mnt/harddisk0/wpp_data/rain_data/sample/val_wv",
#     r"/mnt/harddisk0/wpp_data/rain_data/exp_data/val/rain",
# )
mydata = dataset(
    r"D:/rain_est/data/exp_data/exp_data/train",
    r"D:/rain_est/data/exp_data/exp_data/train",
)
mydataval = dataset(
    r"D:/rain_est/data/exp_data/exp_data/val",
    r"D:/rain_est/data/exp_data/exp_data/val",
)
train_loader = torch.utils.data.DataLoader(mydata, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mydataval, batch_size)
# lr = 0.0001
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr)
st = 5 * len(train_loader)
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=st, gamma=0.1)
# lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=4*st, eta_min=0.000001
# )

globe_i = 1
log_every_i = 500
train_loss_all = 0.0
train_loss0 = torch.zeros((5))
for epoch in range(epochs):
    if epoch % 5 == 0:
        path = os.path.join(path1, str(epoch) + "_w3.pt")
        torch.save(model, path)
    for batch_idx, (name, data, dem_grid, target) in enumerate(train_loader):
        model.train()
        globe_i += 1
        data = data.to(device)
        target = target.to(device)
        dem_grid = dem_grid.to(device)
        train_loss0 = train_loss0.to(device)
        data = (data - 150) / (330 - 150)
        optimizer.zero_grad()
        output = model(data, dem_grid)
        loss_all, loss0 = loss_fn(output, target)
        train_loss_all += loss_all.item()
        for i in range(5):
            train_loss0[i] += loss0[i].item()
        loss_all.backward()
        optimizer.step()
        lr_sch.step()
        optimizer.zero_grad()
        print(
            f"Iter:{batch_idx}, Len:{len(train_loader)}, Epoch:{epoch}, Training loss:{loss_all.item()}"
        )

        if globe_i % log_every_i == 1:
            with torch.no_grad():
                model.eval()
                valloss = 0.0
                train_valloss0 = torch.zeros((5))
                train_valloss0 = train_valloss0.to(device)
                for batch_idx, (name, data, dem_grid, target) in enumerate(val_loader):
                    data = data.cuda(device)
                    target = target.cuda(device)
                    dem_grid = dem_grid.to(device)
                    data = (data - 150) / (330 - 150)
                    output = model(data, dem_grid)
                    loss, loss0 = loss_fn(output, target)
                    valloss += loss.item()
                    for i in range(5):
                        train_valloss0[i] += loss0[i].item()
                    print(
                        f"Iter:{batch_idx}, Len:{len(val_loader)}, Validation loss:{loss.item()}"
                    )
                val_loss_val = valloss / len(val_loader)
                train_loss_all /= log_every_i

                train_loss0 /= log_every_i
                train_valloss0 = train_valloss0 / len(val_loader)
            writer.add_scalars(
                "Loss/train",
                {"train_loss": train_loss_all, "val_loss": val_loss_val},
                globe_i,
            )
            writer.add_scalars(
                "Loss/train_T1",
                {"train_T1": train_loss0[0], "val_T1": train_valloss0[0]},
                globe_i,
            )
            writer.add_scalars(
                "Loss/train_T5",
                {"train_T5": train_loss0[1], "val_T5": train_valloss0[1]},
                globe_i,
            )
            writer.add_scalars(
                "Loss/train_T10",
                {"train_T10": train_loss0[2], "val_T10": train_valloss0[2]},
                globe_i,
            )
            writer.add_scalars(
                "Loss/train_T20",
                {"train_T20": train_loss0[3], "val_T20": train_valloss0[3]},
                globe_i,
            )
            writer.add_scalars(
                "Loss/train_T30",
                {"train_T30": train_loss0[4], "val_T30": train_valloss0[4]},
                globe_i,
            )

            writer.add_scalar(
                "lr", optimizer.state_dict()["param_groups"][0]["lr"], globe_i
            )
            writer.add_scalar("epoch", epoch, globe_i)
            train_loss_all = 0.0
            train_loss0 = torch.zeros((5)).to(device)
