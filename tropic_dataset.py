import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

dem_1 = np.load(r"D:/rain_est/data/exp_data/dem_1.npy")
dem_1 = torch.from_numpy(dem_1).float().view(1, 192, 192)
dem_2 = np.load(r"D:/rain_est/data/exp_data/dem_2.npy")
dem_2 = torch.from_numpy(dem_2).float().view(1, 192, 192)
dem_3 = np.load(r"D:/rain_est/data/exp_data/dem_3.npy")
dem_3 = torch.from_numpy(dem_3).float().view(1, 192, 192)


class dataset(data.Dataset):
    def __init__(self, image_path, label_path, transform=None, size=192):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.size = size
        self.lines = os.listdir(image_path)

    def __getitem__(self, index):
        xyitems = self.lines[index]
        name = xyitems.split(".")[0]
        i1 = np.load(os.path.join(self.image_path, xyitems), allow_pickle=True)
        # i2 = np.load(os.path.join(self.label_path, xyitems)).astype(np.float32)
        img_name = np.array([i1[0].astype(np.float32), i1[1].astype(np.float32)])
        # img_name = img_name.astype(np.float32)
        label_name = i1[2]
        # img_name = i1[0:4]
        # label_name = i2[-1]
        label_name = cv2.GaussianBlur(label_name, (5, 5), 2)
        outimg0 = torch.from_numpy(img_name).float()
        outlabel = torch.from_numpy(label_name).float().view(1, self.size, self.size)
        if name[-1] == "1":
            dem = dem_1
        if name[-1] == "2":
            dem = dem_2
        if name[-1] == "3":
            dem = dem_3
        return name, outimg0, dem, outlabel

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":

    # trans = transforms.Compose([transforms.ToTensor()])
    dataset = dataset(r"I:\fy_rain_data\sample")
    outimg, label = dataset[1]
    img = outimg.numpy().astype(np.uint8).reshape(201, 201)
    # img[img == 103] = 0
    la = label.numpy().astype(np.uint8).reshape(201, 201)
    aaa = np.hstack([img, la])
    cv2.imshow("1", aaa)
    cv2.waitKey()
