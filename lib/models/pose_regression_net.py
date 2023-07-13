# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer

from pdb import set_trace as st

class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x




# vis_time_cost_flag = True
vis_time_cost_flag = False
class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, cfg.NETWORK.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps, meta, grid_centers, all_cloud=None):
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)

        if vis_time_cost_flag:
            print('st1-project_layer')
            import time;time_start=time.time()

        # st()
        # (Pdb) grid_centers.shape
        # torch.Size([1, 5])
        cubes, grids = self.project_layer(all_heatmaps, meta,
                                          self.grid_size, grid_centers, self.cube_size, all_cloud=all_cloud)


        if vis_time_cost_flag:
            time_end=time.time();print('cost1\n',time_end-time_start)
            print('st2-v2v_net')
            import time;time_start=time.time()


        index = grid_centers[:, 3] >= 0
        valid_cubes = self.v2v_net(cubes[index])
        pred[index] = self.soft_argmax_layer(valid_cubes, grids[index])
        
        if vis_time_cost_flag:
            time_end=time.time();print('cost2\n',time_end-time_start)

        return pred
