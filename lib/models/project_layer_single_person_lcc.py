# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# 所以问题就是如何得到当前heatmap对应的3d keypoint的大概位置
# 如何用uv->3d clouds得到heatmap对应3d keypoint
# 1.直接integral或者argmax出来2d joint，然后映射为3d
# 2.找所有的点的对应的3d坐标，然后用heatmap值一起加权平均出来3d
# uv->3d 可以找最接近的，也可以试试torch的grid_sample函数，直接找最近的先
# 这是第一种思路，但是只能用来处理单人，多人就牵扯到多个峰的argmax

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform

from pdb import set_trace as st


import numpy as np
import math
import cv2

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# 这个东西超级慢，放弃了
# 找uv最近的有值的点
# https://stackoverflow.com/questions/52576498/find-nearest-neighbor-to-each-pixel-in-a-map/54459152
def bwdist(img, metric=cv2.DIST_L2, dist_mask=cv2.DIST_MASK_5, label_type=cv2.DIST_LABEL_CCOMP, ravel=False):
    """Mimics Matlab's bwdist function.

    Available metrics:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb
    Available distance masks:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaaa68392323ccf7fad87570e41259b497
    Available label types:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga3fe343d63844c40318ee627bd1c1c42f
    """
    # st()
    flip = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist, labeled = cv2.distanceTransformWithLabels(flip, metric, dist_mask)

    # return linear indices if ravel == True 
    if ravel:  
        idx = np.zeros(img.shape, dtype=np.intp)  # np.intp type is for indices
        for l in np.unique(labeled):
            mask = labeled == l
            # st()
            # print(np.flatnonzero(img * mask).shape)
            # print(np.flatnonzero(img * mask).shape)

            idx[mask] = np.flatnonzero(img * mask)
        return dist, idx

    # return two-channel indices if ravel == False (default)
    idx = np.zeros((*img.shape, 2), dtype=np.intp)  
    for l in np.unique(labeled):
        mask = labeled == l
        
        print(l)
        # print(np.squeeze(np.dstack(np.where(img * mask))))
        # print(np.squeeze(np.dstack(np.where(img * mask))).shape)
        # print(idx[mask].shape)

        label_to_index = np.squeeze(np.dstack(np.where(img * mask)))

        if len(label_to_index.shape) > 1:
            # print('shit\n', np.squeeze(np.dstack(np.where(img * mask))))
            label_to_index = label_to_index[0]
            
        idx[mask] = label_to_index

    return dist, idx

# joints_3d 是一个List len是num_people

def generate_3d_weightmap(space_size, space_center, initial_cube_size, joints_3d):
    space_size = space_size
    space_center = space_center
    cube_size = initial_cube_size

    try:
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
    except:
        # st()
        space_center = space_center.cpu().detach().numpy()
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        
        
    grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
    grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]
    target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
    cur_sigma = 200.0

    mu_x = joints_3d[0]
    mu_y = joints_3d[1]
    mu_z = joints_3d[2]

    i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
               np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
    i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
               np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
    i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
               np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
    
    if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
        print('也可以不生成这个的权重了，生成全0的权重？')
        raise

    gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
    g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
    target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

    # st()
    target = np.clip(target, 0, 1)
    return target

# 关于grid和cube的理解
# cube是整个的空间，是把世界空间离散化之后的，人可以活动的全部的范围
# 对于cpn和prn中的cube都是全部的空间，只不过离散的粒度不一样罢了
# grid是cube中的3d bbox，所以给一个center scale就能算出来
# grid表示这128000个点[80, 80, 20] 在cube空间中的位置

class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE

        self.f_weight = cfg.NETWORK.F_WEIGHT

        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
        

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps) # 就是num_views
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, n, device=device)
        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
        
        # batch_size好像就是batch_size原本的意思
        # st()

        for i in range(batch_size): 
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                # This part of the code can be optimized because the projection operation is time-consuming.
                # If the camera locations always keep the same, the grids and sample_grids are repeated across frames
                # and can be computed only one time.
                
                # 下面这句话，是为了区别cpn和prn阶段
                # 当batch_size==1的时候都没问题
                # test默认的batch_size==4, 但是cpn阶段给的batch中每一个grid_center都是一样的，
                # batch中所有的，直接取grid_center[0]即可
                # 但是prn就不一样了，需要区别
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid
                
                # (Pdb) grid.shape
                # torch.Size([128000, 3])
                # grid表示这128000个点[80, 80, 20] 在cube空间中的位置
                
                # 针对每一个view的heatmap进行投影
                # 注意这里是不断地把grid中的坐标变换到不同的相机参数下，进行heatmap投影
                for c in range(n):
                    center = meta[c]['center'][i]
                    scale = meta[c]['scale'][i]

                    width, height = center * 2 # center 的坐标是1/2长宽
                    trans = torch.as_tensor(
                        get_transform(center, scale, 0, self.img_size),
                        dtype=torch.float,
                        device=device)
                    cam = {}
                    for k, v in meta[c]['camera'].items():
                        cam[k] = v[i]
                    
                    xy = cameras.project_pose(grid, cam)

                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = do_transform(xy, trans)
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float, device=device) / torch.tensor(
                        self.img_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor(
                        [w - 1, h - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)

                    # st()
                    if self.f_weight:
                        # 关于grid_sample
                        # https://zhuanlan.zhihu.com/p/112030273
                        

                        # (Pdb) heatmaps[c][i:i + 1, :, :, :].shape
                        # torch.Size([1, 15, 128, 240])
                        # (Pdb) sample_grid.shape # 最后一个维度2对应到heatmap的坐标
                        # torch.Size([1, 1, 128000, 2])
                        # (Pdb) sample_grid
                        # tensor([[[[ 0.6043, -0.8170],
                        #         [ 0.6041, -0.7495],
                        #         [ 0.6039, -0.6821],
                        #         ...,
                        #         [ 0.2628, -0.1598],
                        #         [ 0.2629, -0.1873],
                        #         [ 0.2629, -0.2148]]]], device='cuda:0')

                        # (Pdb) sample_grid.max()
                        # tensor(1.1000, device='cuda:0')
                        # (Pdb) sample_grid.min()
                        # tensor(-1.0019, device='cuda:0')
                        # (Pdb) sample_grid.mean()
                        # tensor(0.0620, device='cuda:0')

                        # (Pdb) cubes[i:i + 1, :, :, :, c].shape
                        # torch.Size([1, 15, 1, 128000])
                        # (Pdb) F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True).shape
                        # torch.Size([1, 15, 1, 128000])

                        # sample_grid最后一个维度代表这个体素在heatmap上面的2d坐标，
                        # 但是由于投影不可能所有的点都正好找到位置，所以插值到所有的128000个体素都有概率值
                        # F.grid_sample输出的就是这个视角的概率值在cube上面的投影

                        # 学长的思路：计算sample_grid的时候投影这一步，让heatmap这个射线上面经过的所有的点的权重都一致了
                        # 因此不应该单纯的投影得到这12800个体素的概率值
                        # 而是应该通过12800个体素的坐标和3d keypoint的距离(通过depth得到)，
                        # 赋予一个权重(高斯)，权重的维度也应该是torch.Size([1, 15, 1, 128000])，然后直接乘上F.grid_sample即可
                        
                        # 所以问题就是如何得到当前heatmap对应的3d keypoint的大概位置
                        # 如何用uv->3d clouds得到heatmap对应3d keypoint
                        # 1.直接integral或者argmax出来2d joint，然后映射为3d
                        # 2.找所有的点的对应的3d坐标，然后用heatmap值一起加权平均出来3d
                        # uv->3d 可以找最接近的，也可以试试torch的grid_sample函数，直接找最近的先
                        # 第一种思路
                        # st()
                        # 
                        
                        # heatmap 128, 240
                        # network image 512, 960
                        # raw image 1080, 1920

                        # 有个大问题，这里如果有多个人的subject，heatmap直接get_max_preds会不会把其他的人的
                        # (Pdb) heatmaps[c][i:i + 1, :, :, :].shape
                        # torch.Size([1, 15, 128, 240])
                        # 下面这个argmax的方法只能用在单人的情况下把
                        # 多人的情况下，这个heatmap应该是有多个峰的，也就是说不能直接argmax
                        # 那就只能对于heatmap上面每一个点，找对应的point cloud，这样把2d的高斯变为3d的高斯
                        # 所以需要get_max_preds处理多个峰的情况

                        # 如果直接2d heatmap用所有有point的点去计算3d heatmap也可以，但是太稀疏
                        # 直接用这个点去影响12800中的周围的很多点，可能也可以，但是计算量超级大

                        # vis一下子

                        coords, maxvals = get_max_preds(heatmaps[c][i:i + 1, :, :, :].cpu().detach().numpy())
                        # 注意这个coords其实不是真实的heatmap的坐标，而是yx调换了，应该用yx索引而不是xy
                        # (Pdb) coords[0,0]
                        # array([108.,  46.], dtype=float32)
                        # (Pdb) heatmaps[c][i:i + 1, :, :, :].cpu().detach().numpy()[0,0,108,46]
                        # 6.563641e-05
                        # (Pdb) heatmaps[c][i:i + 1, :, :, :].cpu().detach().numpy()[0,0,46,108]
                        # 0.9671191


                        coords = coords[0]
                        network_coords = np.zeros_like(coords)
                        network_coords[:,0] = coords[:,0] * 960 / 240
                        network_coords[:,1] = coords[:,1] * 512 / 128

                        raw_coords = network_coords * 2
                        raw_coords = raw_coords.astype(np.uint32)

                        # 即使scale变化比例，也应该差不多，感觉差不多
                        # print(network_coords[:,:,1] / network_coords[:,:,0])
                        # print(meta[c]['joints'][i][0][:,1] / meta[c]['joints'][i][0][:,0])
                        # print(network_coords)
                        # print(meta[c]['joints'][i][0])


                        mask = np.multiply((meta[c]['cloud'][i][:,:,0].cpu().detach().numpy()==0), (meta[c]['cloud'][i][:,:,1].cpu().detach().numpy()==0))
                        mask = np.multiply(mask, (meta[c]['cloud'][i][:,:,2].cpu().detach().numpy()==0))

                        # (Pdb) (mask == True).sum()
                        # 1900712
                        # (Pdb) (mask == False).sum()
                        # 172888

                        # (Pdb) ((meta[c]['cloud'][i].sum(axis=2)==0)==True).sum()
                        # tensor(1900712, device='cuda:0')
                        # (Pdb) ((meta[c]['cloud'][i].sum(axis=2)==0)==False).sum()
                        # tensor(172888, device='cuda:0')

                        # st()
                        # import time

                        coords_to_cloud = np.zeros((coords.shape[0], 3))
                        for joint_i, raw_coord in enumerate(raw_coords):
                            # print('start timing...')
                            # time_start=time.time()
                            
                            # 下面这个raw_xy坐标可能反了
                            raw_x = raw_coord[1]
                            raw_y = raw_coord[0]

                            # if raw_y < 1920 and raw_x < 1080:
                            #     print('passed!')
                            # else:
                            #     print('forbiden!')

                            find_non_zero = False
                            too_far = False

                            # find a non-zero point index that is nearest to raw coord
                            for r in range(max(mask.shape[0], mask.shape[1])): # 一圈一圈找
                                
                                # 大部分r都能控制在5以内
                                # 5之外的joints拿不到深度就不加权了把，直接保留原状
                                if r > 5:
                                    too_far = True
                                    break

                                cur_x = raw_x - r
                                for delta_y in range(-r, r+1):
                                    cur_y = raw_y + delta_y # 卡住
                                    if cur_x < 0 or cur_x >= mask.shape[0] or cur_y < 0 or cur_y >= mask.shape[1]:continue
                                    if mask[cur_x, cur_y] == False:
                                        find_non_zero = True
                                        break
                                
                                if find_non_zero:
                                    break

                                cur_x = raw_x + r
                                for delta_y in range(-r, r+1):
                                    cur_y = raw_y + delta_y # 卡住
                                    if cur_x < 0 or cur_x >= mask.shape[0] or cur_y < 0 or cur_y >= mask.shape[1]:continue
                                    if mask[cur_x, cur_y] == False:
                                        break

                                if find_non_zero:
                                    break

                                cur_y = raw_y - r
                                for delta_x in range(-r, r+1):
                                    cur_x = raw_x + delta_x # 卡住
                                    if cur_x < 0 or cur_x >= mask.shape[0] or cur_y < 0 or cur_y >= mask.shape[1]:continue
                                    if mask[cur_x, cur_y] == False:
                                        break
                                
                                if find_non_zero:
                                    break

                                cur_y = raw_y + r
                                for delta_x in range(-r, r+1):
                                    cur_x = raw_x + delta_x # 卡住
                                    if cur_x < 0 or cur_x >= mask.shape[0] or cur_y < 0 or cur_y >= mask.shape[1]:continue
                                    if mask[cur_x, cur_y] == False:
                                        break
                                
                                if find_non_zero:
                                    break
                            
                            if too_far:
                                coords_to_cloud[joint_i] = np.array([-1.0, -1.0, -1.0])
                            else:
                                coords_to_cloud[joint_i] = meta[c]['cloud'][i][cur_x][cur_y].cpu().detach().numpy()
                            # print(f"r {r} | dist {(cur_x - raw_x) ** 2 + (cur_y - raw_y) ** 2}| raw_xy: ({raw_x}, {raw_y}) | cur_xy: ({cur_x}, {raw_y}) | {meta[c]['cloud'][i][cur_x][cur_y]}")
                        
                            # time_end=time.time()
                            # print(f'time cost for joint_i {joint_i}\n',time_end-time_start)


                        ### now 给没有找到depth的保留原样
                        weight_all = torch.ones(1, num_joints, 1, nbins, device=device)
                        # 需要找到joint_i的3d坐标到grid的映射，就能做一个高斯heatmap了
                        
                        # (Pdb) weight_all.shape
                        # torch.Size([1, 15, 1, 128000])
                        
                        for joint_i, coord_to_cloud in enumerate(coords_to_cloud):
                            if coord_to_cloud[0] > 0:
                                if len(grid_center) == 1:
                                    weightmap_i = generate_3d_weightmap(grid_size, grid_center[0], cube_size, coord_to_cloud)
                                else:
                                    weightmap_i = generate_3d_weightmap(grid_size, grid_center[i], cube_size, coord_to_cloud)
                                weight_all[:, joint_i, :, :] = torch.as_tensor(weightmap_i.reshape(1,1,1,-1), dtype=torch.float, device=device)
                        ### now 给没有找到depth的保留原样


                        # ### org 考虑所有的voxel
                        # # 需要找到joint_i的3d坐标到grid的映射，就能做一个高斯heatmap了
                        
                        # # get_voxel函数本身就是只生成一个voxel，一个整个空间，或者一个人的bbox，
                        # # 所以generate_3d_target中joints_3d只能是一个人
                        # weight_all = torch.zeros(1, num_joints, 1, nbins, device=device)
                        # # (Pdb) weight_all.shape
                        # # torch.Size([1, 15, 1, 128000])
                        
                        # for joint_i, coord_to_cloud in enumerate(coords_to_cloud):
                        #     weightmap_i = generate_3d_weightmap(grid_size, grid_center[i], cube_size, coord_to_cloud)
                        #     weight_all[:, joint_i, :, :] = torch.as_tensor(weightmap_i.reshape(1,1,1,-1), dtype=torch.float, device=device)
                        # ### org 考虑所有的voxel

                        # if pytorch version < 1.3.0, align_corners=True should be omitted.
                        cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) * weight_all
                    else:
                        cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True)

        # cubes = cubes.mean(dim=-1)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6)
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(batch_size, num_joints, cube_size[0], cube_size[1], cube_size[2])  ##
        return cubes, grids

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size):
        cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size)
        return cubes, grids