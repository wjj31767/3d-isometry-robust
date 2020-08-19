import math
import sys
sys.path.append('/home/wei/3d-isometry-robust')
from res_gcn_module import pointcnn, knn_point, index_points, pool, res_gcn_d, res_gcn_d_last
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self,activation,batch):
        super(Discriminator,self).__init__()
        self.activation = activation
        self.pointcnn = pointcnn(8,64,2,activation=self.activation)
        self.block_num = int(math.log2(batch / 64) / 2)
        self.res_gcn_d_list = nn.ModuleList()
        for i in range(self.block_num):
            self.res_gcn_d_list.append(res_gcn_d(8,64,4))
        self.res_gcn_d_last = res_gcn_d_last(1)
    def forward(self,xyz):
        points = self.pointcnn(xyz)

        for i in range(self.block_num):
            xyz, points = pool(xyz, points, 8, points.get_shape()[1].value // 4)
            points = self.res_gcn_d_list[i](xyz, points)
        points = self.res_gcn_d_last(points)


        return points
if __name__== '__main__':
    net = Discriminator(activation=nn.functional.leaky_relu)
    for name, param in net.named_parameters():
        print(name, param.size())