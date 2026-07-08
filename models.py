import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.linear(x)


class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN2Layer(nn.Module):
    def __init__(self, in_channels, output_size, data_type, n_feature=6):
        super().__init__()
        self.n_feature = n_feature
        self.intermediate_size = 4 if data_type in ["mnist", "femnist"] else 5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature * self.intermediate_size * self.intermediate_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, self.n_feature * self.intermediate_size * self.intermediate_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout1(self.pool(F.relu(self.conv2(x))))
        x = self.dropout1(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return F.relu(x)


class MiniMobileNetFEMNIST(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.block1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.block2 = DepthwiseSeparableConv(64, 96, stride=2)
        self.block3 = DepthwiseSeparableConv(96, 128, stride=1)
        self.block4 = DepthwiseSeparableConv(128, 160, stride=2)
        self.block5 = DepthwiseSeparableConv(160, 192, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, output_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
