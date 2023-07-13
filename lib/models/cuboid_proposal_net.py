# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from core.proposal import nms
from pdb import set_trace as st

class ProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)

            # loader中不能出来没有pose的item，否则这里报错
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1)

            # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the
            # unspecified dimension size -1 can be any value and is ambiguous


            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt ### cand match gt
            cand2gt[i][min_dist > 500.0] = -1.0 ### 和所有gt都距离过大的直接放弃

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta, all_net_training=True):
        batch_size = root_cubes.shape[0]
        ### 下面这个其实就是从单个3d heatmap中，提取多个峰的过程
        ### 巧妙地使用了max_pool，来缩小heatmap的响应，之后取topk的响应的位置
        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)

        ### 下面是把量化的80, 80, 20的坐标转化为真实坐标
        topk_unravel_index = self.get_real_loc(topk_unravel_index)
        
        ### topk好像是通过置信度来排序的，但是这个排序没有参考gt，怎么能和gt match上呢
        ### 不是好像，是就是
        ### 怎么能和gt match上呢->恭喜你发现了核心问题
        ### 在training的时候，grid_centers[:, :, 3]是cand2gt，match的方法是用filter_proposal计算的cand2gt，也就是pred root到gt root的距离
        ### testing的时候，grid_centers[:, :, 3]不是cand2gt，只是visibility，只取0或者-1，matching的方法是在数据集的eval方法中，直接和所有的gt计算距离来匹配

        # 看懂grid_centers各个维度的含义
        # grid_center[i][3] >= 0 才会match
        ### grid_center[i][4]就是heatmap的峰值，也就是置信度
        ### grid_center[i][3]就是是否选择这个grid_center，在project_layer forword中体现
        # (Pdb) grid_centers
        # tensor([[[ 6.5823e+02,  5.6329e+02,  8.5263e+02,  2.0000e+00,  9.8316e-01],
        #         [-6.5823e+02,  5.6329e+02,  8.5263e+02,  0.0000e+00,  9.7042e-01],
        #         [ 5.0633e+01, -3.4810e+02,  7.4737e+02,  1.0000e+00,  9.4866e-01],
        #         [ 1.4684e+03,  2.4873e+03,  1.0526e+01, -1.0000e+00,  8.9328e-03],
        #         [ 1.4684e+03,  2.6899e+03,  1.0526e+01, -1.0000e+00,  7.9846e-03],
        #         [ 3.0886e+03, -4.4304e+01,  8.5263e+02, -1.0000e+00,  7.5914e-03],
        #         [ 6.5823e+02,  6.6456e+02,  1.0526e+01, -1.0000e+00,  6.4420e-03],
        #         [ 8.6076e+02,  1.3734e+03,  3.2632e+02, -1.0000e+00,  6.2614e-03],
        #         [ 1.4684e+03,  2.4873e+03, -2.0000e+02, -1.0000e+00,  6.1072e-03],
        #         [ 1.9747e+03, -5.5063e+02, -2.0000e+02, -1.0000e+00,  5.9293e-03]]],
        #     device='cuda:0')
        # 为什么第三维会出现2，在训练的时候，第三维表示cand2gt，对应的gt的instance的index， 并且保证可见的>0
        # 测试的时候，所有的instance，可见的就是0，不可见的就是-1.0

        ### 根据confidence事先排序了
        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index # 前三维是坐标
        grid_centers[:, :, 4] = topk_values # 第四维是heatmap的值

        ### 这里继承自cpn的self.training
        
        
        # st()
        # match gt to filter those invalid proposals for training/validate PRN
        if all_net_training and ('roots_3d' in meta[0] and 'num_person' in meta[0]):
        # if self.training and ('roots_3d' in meta[0] and 'num_person' in meta[0]):
            gt_3d = meta[0]['roots_3d'].float()
            num_person = meta[0]['num_person']
            cand2gt = self.filter_proposal(topk_unravel_index, gt_3d, num_person)
            grid_centers[:, :, 3] = cand2gt
        else:
            grid_centers[:, :, 3] = (topk_values > self.threshold).float() - 1.0  # if ground-truths are not available.

        # nms
        # for b in range(batch_size):
        #     centers = copy.deepcopy(topk_unravel_index[b, :, :3])
        #     scores = copy.deepcopy(topk_values[b])
        #     keep = []
        #     keep_s = []
        #     while len(centers):
        #         keep.append(centers[0])
        #         keep_s.append(scores[0])
        #         dist = torch.sqrt(torch.sum((centers[0] - centers)**2, dim=-1))
        #         index = (dist > 500.0) & (scores > 0.1)
        #         centers = centers[index]
        #         scores = scores[index]
        #     grid_centers[b, :len(keep), :3] = torch.stack(keep, dim=0)
        #     grid_centers[b, :len(keep), 3] = 0.0
        #     grid_centers[b, :len(keep), 4] = torch.stack(keep_s, dim=0)

        return grid_centers


class CuboidProposalNet(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNet, self).__init__()
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER

        self.project_layer = ProjectLayer(cfg)
        # st()
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayer(cfg)

    def forward(self, all_heatmaps, meta, all_cloud=None, all_net_training=True):
        # st()
        # (Pdb) self.grid_center
        # [0.0, -500.0, 800.0]
        
        # 注意这里传进去的[self.grid_center]，说明每一个batch只创建一个cube
        initial_cubes, grids = self.project_layer(all_heatmaps, meta,
                                                  self.grid_size, [self.grid_center], self.cube_size, all_cloud=all_cloud)

        
        # lcc debugging
        # all_heatmaps完全一样， initial_cubes和grid完全一样
        # print('all_heatmaps', all_heatmaps[0])
        # print(all_heatmaps[0].min(), all_heatmaps[0].max(), all_heatmaps[0].mean())
        # print('initial_cubes', initial_cubes[0])
        # print(initial_cubes[0].min(), initial_cubes[0].max(), initial_cubes[0].mean())
        # print('grids', grids[0])
        # print(grids[0].min(), grids[0].max(), grids[0].mean())
        # st()

        #
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta, all_net_training=all_net_training)

        # lcc debugging
        # 这两个完全不一样，几乎可以确定是问题出在root_cubes = self.v2v_net(initial_cubes)这一行
        # print('root_cubes', root_cubes[0])
        # print(root_cubes[0].min(), root_cubes[0].max(), root_cubes[0].mean())
        # print('grid_centers', grid_centers[0])
        # print(grid_centers[0].min(), grid_centers[0].max(), grid_centers[0].mean())
        # st()

        # # 看一下这个grid center是否与root joint重合

        return root_cubes, grid_centers