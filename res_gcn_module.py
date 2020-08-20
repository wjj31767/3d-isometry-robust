import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.pointnet_util import farthest_point_sample,index_points,knn_point
def group(xyz, points, k, dilation=1, use_xyz=False):
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation]

    grouped_xyz = index_points(xyz, idx)  # (batch_size, npoint, k, 3)
    grouped_xyz -= torch.unsqueeze(xyz, 2)  # translation normalization
    if points is not None:
        grouped_points = index_points(points, idx)  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def pool(xyz, points, k, npoint):
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    _, idx = knn_point(k, xyz, new_xyz)
    new_points = torch.max(index_points(points, idx), dim=2).values

    return new_xyz, new_points

class pointcnn(nn.Module):
    def __init__(self,k,n_cout,n_blocks,activation=F.relu):
        super(pointcnn,self).__init__()
        self.k = k
        self.n_cout=n_cout
        self.n_blocks = n_blocks
        self.activation = activation
        self.conv1 = nn.Conv2d(3,n_cout,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(n_cout,n_cout,kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(n_cout)
    def forward(self,xyz):
        # grouped_points: knn points coordinates (normalized: minus centual points)
        xyz = xyz.permute(0,2,1)
        _, grouped_points, _ = group(xyz, None, self.k)
        grouped_points = grouped_points.permute(0,3,2,1)
        # print('n_blocks: ', n_blocks)
        # print('is_training: ', is_training)

        for idx in range(self.n_blocks):
            if idx==0:
                grouped_points = self.conv1(grouped_points)
            else:
                grouped_points = self.conv2(grouped_points)
            if idx == self.n_blocks - 1:
                return torch.max(grouped_points, 2).values
            else:
                grouped_points = self.activation(self.bn(grouped_points))

class res_gcn_d(nn.Module):
    def __init__(self, k, n_cout, n_blocks,indices=None):
        super(res_gcn_d,self).__init__()
        self.k=k
        self.n_blocks = n_blocks
        self.indices = indices
        self.convs = nn.ModuleList()
        self.n_cout = n_cout
        for i in range(n_blocks):
            self.convs.append(nn.Conv2d(n_cout,n_cout,kernel_size=(1,1)))
            self.convs.append(nn.Conv2d(n_cout,n_cout,kernel_size=(1,1)))

    def forward(self,xyz,points):
        for idx in range(self.n_blocks):
            shortcut = points
            # Center Features
            points = points.permute(0, 2, 1)
            xyz = xyz.permute(0, 2, 1)
            points = F.leaky_relu(points)
            # Neighbor Features
            if idx == 0 and self.indices is None:

                _, grouped_points, indices = group(xyz, points, self.k)
            else:
                grouped_points = index_points(points, self.indices)
            # Center Conv
            points = points.permute(0, 2, 1)
            xyz = xyz.permute(0, 2, 1)
            center_points = torch.unsqueeze(points, dim=2)
            points = self.convs[2*idx](center_points)
            # Neighbor Conv
            grouped_points = grouped_points.permute(0,3,2,1)
            grouped_points_nn = self.convs[2*idx+1](grouped_points)
            # CNN
            points = torch.mean(torch.cat([points, grouped_points_nn], dim=2), dim=2) + shortcut

        return points

class res_gcn_d_last(nn.Module):
    def __init__(self, n_cout,in_channel=64):
        super(res_gcn_d_last, self).__init__()
        self.conv = nn.Conv2d(in_channel,n_cout,kernel_size=(1,1))
    def forward(self,points):
        points = F.leaky_relu(points)
        center_points = torch.unsqueeze(points, dim=2)
        points = torch.squeeze(self.conv(center_points), dim=2)
        points = points.permute(0,2,1)
        return points

if __name__ == '__main__':
    xyz = torch.randn((8,2048,3))
    p = pointcnn(8,128,3,activation=nn.functional.leaky_relu)
    output = p(xyz)

