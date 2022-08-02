import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, device, **kwargs):
        super(TuckER, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device=device, requires_grad=True))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class HypER(torch.nn.Module):
    def __init__(self, d, d1, d2, device, **kwargs):
        super(HypER, self).__init__()
        self.in_channels = kwargs["in_channels"]  # 1
        self.out_channels = kwargs["out_channels"]  # 32
        self.filt_h = kwargs["filt_h"]  # 1
        self.filt_w = kwargs["filt_w"]  # 9

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)  # (40943,200)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)  # (11,200)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])  # 0.2
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])  # 0.3
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])  # 0.2
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)  # 对由3d数据足证的4d数据进行批标准化
        self.bn2 = torch.nn.BatchNorm1d(d1)  # 对2d或3d数据进行批标准化
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))  # self.b (40943)
        fc_length = (1 - self.filt_h + 1) * (d1 - self.filt_w + 1) * self.out_channels  # 会把每个特征图展平成一维向量 (1-1+1)*(200-9+1)*32
        self.fc = torch.nn.Linear(fc_length, d1)  # 一个全连接层，从fc_length到d1的映射   (6144,200)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w  # (1*32*1*9)
        self.fc1 = torch.nn.Linear(d2, fc1_length)  # (200,288)， hypernetwoek H

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))  # (128,1,1,200)
        r = self.R(r_idx)  # (128,200)
        x = self.bn0(e1)  # (128,1,1,200)
        x = self.inp_drop(x)  # (128,1,1,200)

        k = self.fc1(r)  # (128,288)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)  # (128,1,32,1,9)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)  # (4096,1,1,9)

        x = x.permute(1, 0, 2, 3)  # 将tensor的维度换位 (1,128,1,200)

        x = F.conv2d(x, k, groups=e1.size(0))  # (1,4096,1,192)
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_h + 1,
                   e1.size(3) - self.filt_w + 1)  # (128,1,32,1,192)
        x = x.permute(0, 3, 4, 1, 2)  # (128,1,192,1,32)
        x = torch.sum(x, dim=3)  # (128,1,192,32)
        x = x.permute(0, 3, 1, 2).contiguous()  # (128,32,1,192)

        x = self.bn1(x)  # (128,32,1,192)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)  # (128,6144)
        x = self.fc(x)  # (128,200)
        x = self.hidden_drop(x)
        x = self.bn2(x)  # (128,200)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))  # (128,200)*(200,40943)=(128,40943)
        x += self.b.expand_as(x)  # 将b扩展为和x同形的张量，添加到x中
        pred = torch.sigmoid(x)
        return pred


class HypE(torch.nn.Module):
    def __init__(self, d, d1, d2, device, **kwargs):
        super(HypE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        r_dim = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.R = torch.nn.Embedding(len(d.relations), r_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (10 - self.filt_h + 1) * (20 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 10, 20)
        r = self.R(r_idx)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = r.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 10 - self.filt_h + 1, 20 - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


class ConvE(torch.nn.Module):
    def __init__(self, d, d1, d2, device, **kwargs):
        super(ConvE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, (self.filt_h, self.filt_w), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (20 - self.filt_h + 1) * (20 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 10, 20)
        r = self.R(r_idx).view(-1, 1, 10, 20)
        x = torch.cat([e1, r], 2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


