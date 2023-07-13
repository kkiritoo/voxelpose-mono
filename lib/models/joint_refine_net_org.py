# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_layer_jrn import ProjectLayerJrn

from pdb import set_trace as st

class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        # st()
        # (Pdb) x.shape
        # torch.Size([1, 15, 64, 64, 64])
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # (Pdb) x.reshape(batch_size, channel, -1, 1).shape
        # torch.Size([1, 15, 262144, 1])

        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)


        # grids = grids.unsqueeze(1)
        grids = grids.unsqueeze(0)

        # st()
        # (Pdb) grids.shape
        # torch.Size([1, 1, 262144, 3])
        # (Pdb) x.shape
        # torch.Size([1, 15, 262144, 1])
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x




# vis_time_cost_flag = True
vis_time_cost_flag = False
class JointRefineNet(nn.Module):
    def __init__(self, cfg):
        super(JointRefineNet, self).__init__()
        # self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        # self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.grid_size = [300,300,300]
        self.cube_size = [64,64,64]
        # self.cube_size = [30,30,30]

        self.project_layer = ProjectLayerJrn(cfg)

        self.f_weight_with_w = cfg.NETWORK.F_WEIGHT_WITH_W

        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS_WHOLEBODY, cfg.NETWORK.NUM_JOINTS_WHOLEBODY)

        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps, meta, grid_centers, all_cloud=None):
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)

        if vis_time_cost_flag:
            print('st1-project_layer')
            import time;time_start=time.time()

        
        cubes, grids, xy = self.project_layer(all_heatmaps, meta,
                                          self.grid_size, grid_centers, self.cube_size, all_cloud=all_cloud)


        if vis_time_cost_flag:
            time_end=time.time();print('cost1\n',time_end-time_start)
            print('st2-v2v_net')
            import time;time_start=time.time()


        #### 
        # print('这里需要改！！！')
        ### 只计算没有被nms过滤掉的
        # st()
        # index = grid_centers[:, 3] >= 0
        index = torch.ones(1, dtype=torch.bool)


        valid_cubes = self.v2v_net(cubes[index])

        # st()
        pred[index] = self.soft_argmax_layer(valid_cubes, grids)

        if vis_time_cost_flag:
            time_end=time.time();print('cost2\n',time_end-time_start)

        return pred
