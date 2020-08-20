import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstraction2,PointNetFeaturePropagation

class Generator(nn.Module):
    def __init__(self, num_points=2048,num_classes=40,bradius=1.0,up_ratio = 4,in_channel=259,in_channel1=168):
        super(Generator, self).__init__()
        self.num_points = num_points
        self.radius=bradius
        self.sa1 = PointNetSetAbstraction2(num_points, bradius*0.05, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction2(num_points//2, bradius*0.1, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction2(num_points//4, bradius*0.2, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction2(num_points//8, bradius*0.3, 32, 256+3, [256, 256, 512], False)
        self.fp3 = PointNetFeaturePropagation(512,[64])
        self.fp2 = PointNetFeaturePropagation(256,[64])
        self.fp1 = PointNetFeaturePropagation(128,[64])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.up_ratio=up_ratio
        for _ in range(up_ratio):
            last_channel = in_channel
            for out_channel in [256, 128]:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, (1,1)))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        last_channel=in_channel1
        for out_channel in [64, 3]:
            self.mlp_convs1.append(nn.Conv2d(last_channel, out_channel, (1,1)))
            self.mlp_bns1.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, xyz, labels_onehot,device):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        up_l4_points = self.fp3(xyz, l4_xyz, None, l4_points)
        up_l3_points = self.fp2(xyz, l3_xyz, None, l3_points)
        up_l2_points = self.fp1(xyz, l2_xyz, None, l2_points)
        new_points_list = []
        for i in range(self.up_ratio):
            concat_feat = torch.cat([up_l4_points, up_l3_points, up_l2_points, l1_points, xyz], dim=1)
            concat_feat = torch.unsqueeze(concat_feat, dim=2)

            conv = self.mlp_convs[2*i]
            bn = self.mlp_bns[2*i]
            concat_feat = F.relu(bn(conv(concat_feat)))
            conv = self.mlp_convs[2 * i+1]
            bn = self.mlp_bns[2 * i+1]
            new_points = F.relu(bn(conv(concat_feat)))
            new_points_list.append(new_points)
        net = torch.cat(new_points_list,dim=1).to(device)
        labels_onehot = torch.tensor(labels_onehot, dtype=torch.float32)
        labels_onehot = torch.unsqueeze(labels_onehot, 2)
        labels_onehot = torch.unsqueeze(labels_onehot, 2)
        labels_onehot = labels_onehot.repeat(1,1 , 1, self.num_points).to(device)
        net = torch.cat([net, labels_onehot], 1)

        bn = self.mlp_bns1[0]
        conv = self.mlp_convs1[0]
        coord = F.relu(bn(conv(net)))
        bn = self.mlp_bns1[1]
        conv = self.mlp_convs1[1]
        coord = F.relu(bn(conv(coord)))
        coord = torch.squeeze(coord, 2)  # B*(2N)*3
        return coord

if __name__== '__main__':
    net = Generator()
    for name, param in net.named_parameters():
        print(name, param.size())