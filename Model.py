import torch
from torch import nn
import torch.nn.functional as F

channel_num = 26


class Spatiotemporal_Attention(nn.Module):
    def __init__(self):
        super(Spatiotemporal_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(channel_num, 9), bias=True)
        self.bn1 = nn.BatchNorm2d(9)

        self.conv2 = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=(channel_num, 1))
        self.bn2 = nn.BatchNorm2d(1)

        self.conv3 = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=(1, 9))
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.softmax(x1)
        x1 = x * x1  # (Batch, 4, 30, 30)

        x2 = F.relu(self.bn2(self.conv2(x)))
        x2 = F.softmax(x2)
        x2 = x * x2

        x3 = F.relu(self.bn3(self.conv3(x)))
        x3 = F.softmax(x3)
        x3 = x * x3

        return x1 + x2 + x3

class Separable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(9, 1), stride=1, padding=1, bias=True):
        super(Separable, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                         groups=in_channels, bias=bias)
										 
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.dtype)
        # x_raw = x.to(torch.float32)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 9), stride=1, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                         groups=in_channels, bias=bias)
										 
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.dtype)
        # x_raw = x.to(torch.float32)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


class Q_K_V(nn.Module):
    def __init__(self, hidden_dim):
        super(Q_K_V, self).__init__()
        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
 
        Q1 = self.Q_linear(x)
        K1 = self.K_linear(x).permute(0, 2, 1)  # 先进行一次转置
        V1 = self.V_linear(x)

        alpha1 = torch.matmul(Q1, K1)
        # 下面开始softmax
        alpha1 = F.softmax(alpha1, dim=2)
        out1 = torch.matmul(alpha1, V1)

        return out1


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.Spatiotemporal_Attention = Spatiotemporal_Attention()
        self.DW = Separable(in_channels=channel_num, out_channels=channel_num)
        self.DWConvd = SeparableConv2d(in_channels=channel_num, out_channels=channel_num)
        self.Q_K_V = Q_K_V(hidden_dim=81)
        self.fc1 = nn.Linear(channel_num * 81 + channel_num*9*18, 800)
        self.fc2 = nn.Linear(800, 100)
        self.fc3 = nn.Linear(100, 5)
        #
        # self.fc = nn.Linear(200, 5)
        # self.fc = nn.Linear(200, 5)

        # self.fc1 = nn.Linear(30*120, 100)
        # self.fc2 = nn.Linear(100, 5)
        self.dp = nn.Dropout(p=0.5)
        # self.s = nn.Sigmoid()
        # self.lstm = nn.LSTM(30*120, 100, 2, bidirectional=True)
        self.gru = nn.LSTM(9, 9, 9, bidirectional=True)

        # self.class_classifier = nn.Sequential(
        #     # nn.BatchNorm1d(248),
        #     nn.Linear(8019, channel_num),
        #     nn.BatchNorm1d(channel_num),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        y,_ = self.gru(x.contiguous().view(-1, channel_num*9, 9))

        x1 = self.DWConvd(x)

        x2 = self.DW(x)
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)

        x = torch.matmul(x1, x2)

        x = self.Spatiotemporal_Attention(x)
        x = x.permute(0, 2, 1, 3)

        # x = x.reshape(-1, channel_num, 4 * 30)
        # x = self.DW(x)
        x = x.reshape(-1, channel_num, 9 * 9)

        x = self.Q_K_V(x)
        # print(x.shape)

        x = x.view(-1, channel_num * 81)
        y = y.view(-1, channel_num*9*18)

        x = torch.cat([x,y],dim=1)

        # x1 = self.class_classifier(x)

        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = F.relu(self.fc2(x))
        x = self.dp(x)
        x = self.fc3(x)




        # x = F.tanh(x)
        # x = self.fc(x)

        # x = x.reshape(-1, 1, channel_num * 119)
        # x, y = self.gru(x)
        # x = x.view(-1, 800)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # x = self.fc(x)
        # x = F.softmax(x, dim=1)

        return x
