# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet, V2VNetLite
from models.project_layer_jrn import ProjectLayerJrn
from models.project_layer import ProjectLayer

from pdb import set_trace as st


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        # st()
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class SoftArgmaxLayerJrn(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayerJrn, self).__init__()
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
class JointRefineNetFace(nn.Module):
    def __init__(self, cfg):
        super(JointRefineNetFace, self).__init__()
        # self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        # self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        # self.grid_size = [300,300,300]
        self.grid_size = [600,600,600]
        self.cube_size = [64,64,64]

        self.v2v_channel_num = 68 ### face joints num

        self.project_layer_jrn = ProjectLayerJrn(cfg)
        self.project_layer = ProjectLayer(cfg)

        self.f_weight_with_w = cfg.NETWORK.F_WEIGHT_WITH_W

        # self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS_WHOLEBODY, cfg.NETWORK.NUM_JOINTS_WHOLEBODY)
        self.v2v_net = V2VNetLite(self.v2v_channel_num, self.v2v_channel_num)

        self.num_joints_wholebody = self.v2v_channel_num

        self.soft_argmax_layer_jrn = SoftArgmaxLayerJrn(cfg)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps_whole, meta, grid_centers, all_cloud=None):

        batch_size = all_heatmaps_whole[0].shape[0]
        # num_joints = all_heatmaps_whole[0].shape[1] ### 注意这里是hrnet wholebody的num_joints 
        num_joints = self.num_joints_wholebody

        device = all_heatmaps_whole[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)

        if vis_time_cost_flag:
            print('st1-project_layer')
            import time;time_start=time.time()

        # ### lwrist rwrist nose + hand face 经过ProjectLayer得到cube
        # ### 注意!!!索引的是133个joint的hm，不是你造的123个joint
        # body_joint_index_list = [5,6,7,8,11,12,13,14,15,16]


        face_joint_index_list = [i for i in range(23,23+68)]
        # lhand_joint_index_list = [9] + [i for i in range(23+68,23+68+21)]
        # rhand_joint_index_list = [10] + [i for i in range(23+68+21,23+68+21+21)]
        
        # # all_heatmaps_body 10
        # all_heatmaps_body = []
        # for whole_heatmaps in all_heatmaps_whole:
        #     all_heatmaps_body.append(whole_heatmaps[:, body_joint_index_list, :, :])

        all_heatmaps_face = []
        for whole_heatmaps in all_heatmaps_whole:
            all_heatmaps_face.append(whole_heatmaps[:, face_joint_index_list, :, :])

        # all_heatmaps_lhand = []
        # for whole_heatmaps in all_heatmaps_whole:
        #     all_heatmaps_lhand.append(whole_heatmaps[:, lhand_joint_index_list, :, :])

        # all_heatmaps_rhand = []
        # for whole_heatmaps in all_heatmaps_whole:
        #     all_heatmaps_rhand.append(whole_heatmaps[:, rhand_joint_index_list, :, :])
        
        # st()
        
        # cubes_body, grids_body, _ = self.project_layer_jrn(all_heatmaps_body, meta, self.grid_size, grid_centers, self.cube_size, all_cloud=all_cloud)
        
        ### 注意!!!索引grid_centers使用123标准
        cubes_face, grids_face = self.project_layer(all_heatmaps_face, meta, self.grid_size, grid_centers[0:1], self.cube_size, all_cloud=all_cloud)
        # cubes_lhand, grids_lhand = self.project_layer(all_heatmaps_lhand, meta, self.grid_size, grid_centers[5:6], self.cube_size, all_cloud=all_cloud)
        # cubes_rhand, grids_rhand = self.project_layer(all_heatmaps_rhand, meta, self.grid_size, grid_centers[6:7], self.cube_size, all_cloud=all_cloud)
        # st()
        # cubes = torch.cat((cubes_body, cubes_face, cubes_lhand, cubes_rhand), dim=1)
        cubes = cubes_face


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

        ### 注意!!!索引valid_cubes的是123标准
        # body_joint_index_list = [1,2,3,4,7,8,9,10,11,12]
        # face_joint_index_list = [0] + [i for i in range(1,1+68)]
        # lhand_joint_index_list = [5] + [i for i in range(13+68,13+68+21)]
        # rhand_joint_index_list = [6] + [i for i in range(13+68+21,13+68+21+21)]


        ### 分开做soft_argmax
        # pred_body = self.soft_argmax_layer_jrn(valid_cubes[:, body_joint_index_list, ...], grids_body)
        # st()

        # (Pdb) grids[index].shape
        # torch.Size([1, 262144, 3])
        # (Pdb) valid_cubes.shape
        # torch.Size([1, 15, 64, 64, 64])
        # (Pdb) pred[index].shape
        # torch.Size([1, 15, 3])

        pred_face = self.soft_argmax_layer(valid_cubes, grids_face)
        # pred_lhand = self.soft_argmax_layer(valid_cubes[:, lhand_joint_index_list, ...], grids_lhand)

        # st()
        # rhand_joint_index_list
        # pred_rhand = self.soft_argmax_layer(valid_cubes[:, rhand_joint_index_list, ...], grids_rhand)
        
        # pred[index] = torch.cat((pred_body, pred_face, pred_lhand, pred_rhand), dim=1)

        pred[index] = pred_face


        if vis_time_cost_flag:
            time_end=time.time();print('cost2\n',time_end-time_start)

        return pred
