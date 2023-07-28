import torch
import torch.nn as nn
import torch.nn.functional as F


class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 256, 3, 1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, padding=1),
        )
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x1)
        x3 = self.stn(x2)
        x4 = self.model3(x3)
        x5 = torch.cat([x1, x4], dim=1)
        x6 = self.model4(x5)
        return x6


class model_wv_nosnt(nn.Module):
    def __init__(self):
        super(model_wv, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(512, 512, 3, 1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            # nn.Conv2d(256, 256, 3, 1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, padding=1),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, padding=1),
            nn.Sigmoid(),
        )
        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x0):
        x1 = x0[:, 0:2, :, :]
        x2 = x0[:, 2:4, :, :]
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x_c = torch.cat((x1, x2), dim=1)
        x = self.model3(x_c)
        x = self.model4(x)
        return x


class model_wv(nn.Module):
    def __init__(self):
        super(model_wv, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 1, 3, 1, padding=1),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(64, 32, 1, 1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, 1, 1, padding=0),
        )
        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x0, x_dem):
        x1 = x0[:, 0, 0:1, :, :]
        x2 = x0[:, 1, 0:1, :, :]
        # x1 = x0[:, 0:2, :, :]
        # x2 = x0[:, 2:4, :, :]
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x_c = torch.cat((x1, x2), dim=1)
        # x = x1 + x2
        x = self.model3(x_c)
        x = self.stn(x)
        x = self.model4(x)
        x_d = self.model5(x_dem)
        x = torch.cat((x, x_d), dim=1)
        x = self.model6(x)

        return x


class model_mask_deep(nn.Module):
    def __init__(self):
        super(model_mask_deep, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, padding=1),
        )
        self.model5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1, 1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x0):
        x1 = x0[:, 0:2, :, :]
        x2 = x0[:, 2:4, :, :]
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x_c = torch.cat((x1, x2), dim=1)
        x = self.model3(x_c)
        x = self.model4(x)
        x_mask = self.model5(x2).round()

        x_out = x.mul(x_mask)
        return x_out, x_mask, x


class model_dem(nn.Module):
    def __init__(self):
        super(model_dem, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            l,
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 1, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1, 1, padding=0),
        )
        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x0, x_dem):
        x1 = x0[:, 0:2, :, :]
        x2 = x0[:, 2:4, :, :]
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x_c = torch.cat((x1, x2), dim=1)
        # x = x1 + x2
        x = self.model3(x_c)
        x = self.stn(x)
        x = self.model4(x)
        x_d = torch.repeat_interleave(x_dem, repeats=x1.shape[0], dim=0).view(
            x1.shape[0], 1, x_dem.shape[1], x_dem.shape[2]
        )
        x_d = self.model5(x_d)
        x = torch.cat((x, x_d), dim=1)
        x = self.model6(x)
        return x


class model_compare(nn.Module):
    def __init__(self):
        super(model_compare, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, 1, padding=1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0, x):
        x1 = x0[:, 1:2, 0, :, :]
        x2 = x0[:, 1:2, 1, :, :]
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x_c = torch.cat((x1, x2), dim=1)
        x = self.model2(x_c)
        return x


# 特征拼接方式为add
class model_fuse_add(nn.Module):
    def __init__(self):
        super(model_fuse_add, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 1, 3, 1, padding=1),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(64, 32, 1, 1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, 1, 1, padding=0),
        )
        # Spatial transformer localization-network
        self.localization1 = nn.Sequential(
            nn.Conv2d(256, 128, stride=1, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, stride=1, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 10, stride=1, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x0, x_dem):
        x1 = x0[:, 0, 0:1, :, :]
        x2 = x0[:, 1, 0:1, :, :]

        x1 = self.model1(x1)
        x2 = self.model1(x2)

        x_c = x1 + x2
        x = self.model3(x_c)
        x = self.stn(x)
        x = self.model4(x)
        x_d = self.model5(x_dem)
        x = x + x_d
        x = self.model6(x)

        return x
