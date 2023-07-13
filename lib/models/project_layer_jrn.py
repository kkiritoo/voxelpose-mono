# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# 所以问题就是如何得到当前heatmap对应的3d keypoint的大概位置
# 如何用uv->3d clouds得到heatmap对应3d keypoint
# 1.直接integral或者argmax出来2d joint，然后映射为3d
# 2.找所有的点的对应的3d坐标，然后用heatmap值一起加权平均出来3d
# uv->3d 可以找最接近的，也可以试试torch的grid_sample函数，直接找最近的先
# 这是第二种思路
# 思路是把这12800个点本来不是投影到heatmap上面去吗
# 现在投影到你算好的uv->3d point cloud上面去，就得到了(128000,3) (射线上的应该的坐标)
# ，然后和12800个点原本的坐标(128000, 3)(grid里面的)
# 计算距离来算一个权重来加权即可

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
import copy

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


import cv2 as cv
# rgb内参
def project_pts(cloud, calib):
    K, d = calib.color_proj()
    proj_pts, _ = cv.projectPoints(cloud, np.zeros(3), np.zeros(3), K, d)
    proj_pts = proj_pts.reshape(-1, 2)
    return proj_pts

# 世界坐标系到相机坐标系
def joints_to_color(joints, calib):
    R, t = calib.joints_k_color()
    joints = joints[:, :3]
    joints = joints @ np.linalg.inv(R) + t.reshape(1, 3)
    return joints


def gaussian_np(dist, sigma=10):
    return math.e**(-dist**2/2/sigma**2)

def gaussian(dist, sigma=10):
    return math.e**(-dist**2/2/sigma**2)



# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# def show_np_image(np_image):
#     plt.figure()
#     # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
#     plt.imshow(np_image)
#     plt.show()

# def show_proj_cloud(proj_cloud, name=None):
#     # 原本的yz轴交换了，这边用proj_cloud[:, :, 1]
#     proj_cloud_show_z = proj_cloud[:, :, 1]
#     proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
#     proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
#     # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
#     proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)

#     cv.namedWindow(name,0)
#     cv.resizeWindow(name, 960, 540)
#     cv.imshow(name, proj_cloud_show_z_color1)
#     if 113 == cv.waitKey(100):
#         st()
    
#     # plt.figure()
#     # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
#     # plt.imshow(proj_cloud_show_z_color1)
#     # plt.show()

#     # plt.figure()
#     # # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
#     # plt.imshow(proj_cloud_show_z_color2)
#     # plt.show()



import cv2 as cv
show_h = 1080;show_w = 1920
# show_h = 540;show_w = 960
def show_view(view_to_show, name=None):
    view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)
    cv.namedWindow(name,0)
    cv.resizeWindow(name, show_w, show_h)
    cv.imshow(name, view_to_show)
    if 113 == cv.waitKey(100):
        st()




# 关于grid和cube的理解
# cube是整个的空间，是把世界空间离散化之后的，人可以活动的全部的范围
# 对于cpn和prn中的cube都是全部的空间，只不过离散的粒度不一样罢了
# grid是cube中的3d bbox，所以给一个center scale就能算出来
# grid表示这128000个点[80, 80, 20] 在cube空间中的位置




# vis_time_cost_flag = True
vis_time_cost_flag = False
# vis_xy_flag = True
vis_xy_flag = False

class ProjectLayerJrn(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayerJrn, self).__init__()

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE

        self.f_weight = cfg.NETWORK.F_WEIGHT

        self.f_weight_with_w = cfg.NETWORK.F_WEIGHT_WITH_W


        self.fill_cloud = cfg.DATASET.CLOUD_FILL
        self.f_weight_sigma = cfg.NETWORK.F_WEIGHT_SIGMA

        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
        self.dep_downsample = cfg.NETWORK.DEP_DOWNSAMPLE
        self.unet_dep15 = cfg.NETWORK.UNET_DEP15



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

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size, all_cloud=None):
        device = heatmaps[0].device



        num_joints = heatmaps[0].shape[1]



        ### 除了cube之外其余的batch_size的位置都变成15
        # batch_size = heatmaps[0].shape[0]
        batch_size = num_joints




        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps) # 就是num_views

        cubes = torch.zeros(num_joints, 1, 1, nbins, n, device=device)

        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
        
        # batch_size好像就是batch_size原本的意思
        # st()

        for i in range(batch_size): 
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0: ### 这句话的意思是最后两维不指定也可以，直接算是吗
                # This part of the code can be optimized because the projection operation is time-consuming.
                # If the camera locations always keep the same, the grids and sample_grids are repeated across frames
                # and can be computed only one time.
                
                if vis_time_cost_flag:
                    print('st1-compute_grid')
                    import time;time_start=time.time()

                # 下面这句话，是为了区别cpn和prn阶段
                # 当batch_size==1的时候都没问题
                # test默认的batch_size==4, 但是cpn阶段给的batch中每一个grid_center都是一样的，
                # batch中所有的，直接取grid_center[0]即可
                # 但是prn就不一样了，需要区别
                # st()
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid



                if vis_time_cost_flag:
                    time_end=time.time();print('cost1\n',time_end-time_start)
                    print('st2-project_pose others')
                    import time;time_start=time.time()


                # (Pdb) grid.shape
                # torch.Size([128000, 3])
                # grid表示这128000个点[80, 80, 20] 在cube空间中的位置
                
                # 针对每一个view的heatmap进行投影
                # 注意这里是不断地把grid中的坐标变换到不同的相机参数下，进行heatmap投影
                for c in range(n):
                    # st()
                    assert len(meta[c]['center']) == 1 ### batchsize必须等于1
                    center = meta[c]['center'][0]
                    scale = meta[c]['scale'][0]

                    width, height = center * 2 # center 的坐标是1/2长宽
                    

                    trans = torch.as_tensor(
                        get_transform(center, scale, 0, self.img_size),
                        dtype=torch.float,
                        device=device)
                    
                    cam = {}
                    # st()
                    for k, v in meta[c]['camera'].items():
                        cam[k] = v[0]
                    
                    ### 这个有可能是万恶之源
                    # (Pdb) xy.shape
                    # torch.Size([128000, 2])
                    # (Pdb) grid.shape
                    # torch.Size([128000, 3])

                    #1 org
                    xy = cameras.project_pose(grid, cam)

                    # st()


                    # print(f"1:xy={xy}")
                    # st1-compute_grid
                    # cost1
                    # 0.00035858154296875
                    # st2-project_pose others
                    # cost2
                    # 0.004947185516357422 ### 计算时间十分的珍贵


                    # #2 panoptic-dataset-tool投影不行 这是我转化为color之后再proj point，但是其实感觉结果上没有区别啊
                    # # 并且由于下面numpy的计算太占用cpu，还是直接用之前的吧╮(╯▽╰)╭
                    # # st1-compute_grid
                    # # cost1
                    # # 0.00041031837463378906
                    # # st2-project_pose others
                    # # cost2
                    # # 0.3301553726196289 ### 计算时间长的可怕

                    # import pickle
                    # with open('calib_list.pkl','rb') as calib_list_f:
                    #     calib_list = pickle.load(calib_list_f)

                    # M = np.array([[1.0, 0.0, 0.0],
                    #             [0.0, 0.0, -1.0],
                    #             [0.0, 1.0, 0.0]])
                    # # grid_tmp = grid.detach().cpu().numpy() * 0.0001
                    # grid_tmp = grid.detach().cpu().numpy() * 0.001
                    # # grid_tmp = grid.detach().cpu().numpy() * 0.1
                    # grid_tmp = grid_tmp.dot(np.linalg.inv(M))

                    # joints = joints_to_color(grid_tmp, calib_list[meta[c]['seq'][i]][meta[c]['camera_index'][i]])
                    
                    # # print(joints.shape)
                    # # print(joints)
                    # proj_joints = project_pts(joints, calib_list[meta[c]['seq'][i]][meta[c]['camera_index'][i]])
                    
                    # xy = torch.as_tensor(proj_joints, dtype=torch.float, device=device)
                    # # print(f"2:xy={xy}")


                    

                    #3
                    # import pickle
                    # with open('calib_list2.pkl','rb') as calib_list_f2:
                    #     calib_list2 = pickle.load(calib_list_f2)
                    
                    # calib2 = calib_list2[meta[c]['seq'][i]][(50, int(meta[c]['camera_index'][i])+1)]

                    # from utils.transforms import projectPoints
                    
                    # xy_3 = projectPoints((grid.detach().cpu().numpy()*0.1).transpose(), calib2['K'], calib2['R'],calib2['t'], calib2['distCoef']).transpose()[:, :2]
                    # st()
                    # 结果是xy_3完全等于xy，然后除了第一个batch之外其他的都是ok的纳尼



                    # st()
                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)

                    # st()
                    ### vis xy ###
                    if vis_xy_flag:
                        if not grid_size == [2000.0, 2000.0, 2000.0]:
                            st()
                            print(f'grid:{grid}')
                            print(f'cam:{cam}')
                            print(f'xy:{xy}')

                        xy_vis = xy.clone()
                        bounding_vis = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (xy[:, 1] < height)
                        bounding_vis = ~bounding_vis
                        xy_vis[bounding_vis] = 0.0
                        xy_vis = xy_vis.cpu().numpy().astype(np.int32)

                        if grid_size == [2000.0, 2000.0, 2000.0]:
                            print(f'prn:{xy_vis.mean(axis=0)}')
                        else:
                            # st()
                            print(f'cpn:{xy_vis.mean(axis=0)}')

                        
                        xy_toshow = np.zeros((1080, 1920))
                        for ii in range(xy_vis.shape[0]):
                            # st()
                            xy_toshow[xy_vis[ii][1], xy_vis[ii][0]] = 255
                        
                        if grid_size == [2000.0, 2000.0, 2000.0]:
                            show_view_name = 'prn'
                        else:
                            show_view_name = 'cpn'
                        show_view(xy_toshow.astype(np.uint8), name=show_view_name)
                        # st()s
                    
                    # (Pdb) xy_vis[:,0].max()
                    # tensor(1919.9944, device='cuda:0')
                    # (Pdb) xy_vis[:,0].min()
                    # tensor(0., device='cuda:0')
                    # (Pdb) xy_vis[:,1].min()
                    # tensor(0., device='cuda:0')
                    # (Pdb) xy_vis[:,1].max()
                    # tensor(1079.8984, device='cuda:0')
                    ### vis xy ###

                    


                    
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = do_transform(xy, trans)
                    # st()
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float, device=device) / torch.tensor(
                        self.img_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor(
                        [w - 1, h - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)


                    
                    # 这个sample_grid现在是-1.1到1.1的，已经归一化过的，所以应该可以用在heatmap，uv等等
                    # 投影到rgb uv上面比较方便毕竟之前算过了，如果投影到depth uv上面其实也行
                    # 先用第一种把

                    # 主要的区别在于heatmap uv只能索引到一个值，但是cloud可以索引到3个值，三个值分别sample之后合到一起吗
                    # 不需要，你观察一下，这个15相当于15个joint的map互不干预，插值
                    # 3个坐标其实也是互不干预插值啊
                    # (Pdb) meta[c]['cloud'].permute(0,3,1,2).cpu().detach().numpy().shape
                    # (1, 3, 1080, 1920)
                    # (Pdb) heatmaps[c][i:i + 1, :, :, :].shape
                    # torch.Size([1, 15, 128, 240])

                    # 我觉得这里可能有问题，因为cloud是稀疏的，先跑起来，如果有问题，就把cloud插值满之后再跑

                    # import time
                    # start = time.time()

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
                        # 这是第二种思路
                        # 思路是把这12800个点本来不是投影到heatmap上面去吗
                        # 现在投影到你算好的uv->3d point cloud上面去，就得到了(128000,3) (射线上的应该的坐标)
                        # ，然后和12800个点原本的坐标(128000, 3)(grid里面的)
                        # 计算距离来算一个权重来加权即可
                        
                        # 投影到你算好的uv->3d point cloud上面去
                        
                        # print('cloud 2 {}'.format(meta[c]['cloud'][i].mean()))
                        # print('cloud 2 {}'.format(meta[c]['cloud'][i][50,:,:].mean()))
                        # print('cloud 2 {}'.format(meta[c]['cloud'][i][100,:,:].mean()))
                        # print('cloud 2 {}'.format(meta[c]['cloud'][i][500,:,:].mean()))
                        # 感觉从meta中出来的变成tensor的cloud有一些坐标发生了变化，如果需要可以把用pickle来做
                        # st()

                        # 我记得之前考虑过这个问题
                        # 由于原本的cloud中的不存在的点在相机坐标系下表示为0,0,0，也就是说和gt的距离很远
                        # 因此即使变换为世界坐标系中，和gt的距离肯定也很远
                        # 按照学长的看法是把这些置为1表示用原本rgb中的信息，我觉得置为0？都试试把



                        if self.dep_downsample:
                            assert all_cloud is not None
                            # st()
                            cloud = all_cloud[c][0]
                            # torch.Size([1, 3, 1080, 2024])
                            # st()
                            # 测试refine之前的depth不需要unsqueeze
                            cloud = cloud.float() # 1 15 3 128 240
                            # cloud = cloud.unsqueeze(0).float() # 1 15 3 128 240
                        else:
                                
                            ### lcc debugging
                            ### 在显存中用deepcopy是不是会增大显存占用？
                            # cloud = copy.deepcopy(meta[c]['cloud'][i])
                            cloud = meta[c]['cloud'][i]
                            # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html?highlight=pad#torch.nn.functional.pad
                            # pad cloud
                            p2d_cloud = (0, 0, (2025-1920)//2, (2025-1920)//2)
                            cloud = F.pad(cloud, p2d_cloud, "constant", 0)
                            # st()
                            

                            if self.fill_cloud:
                                cloud_mask = meta[c]['cloud_mask'][i]
                                p2d_cloud_mask = ((2025-1920)//2, (2025-1920)//2)
                                cloud_mask = F.pad(cloud_mask, p2d_cloud_mask, "constant", 1) # 代表没有值
                                # st()
                                # cloud[cloud_mask] = cloud.min() - 1 # 最小值
        
                                # print(f'cloud.min() {cloud.min()}')
                                cloud[cloud_mask] = -1e10 # 最小值
                                
                                cloud = cloud.unsqueeze(0).permute(0,3,1,2).float()
                                # show_proj_cloud(cloud.squeeze(0).permute(1,2,0).detach().cpu().numpy(), '1')
                                
                                # from torch.nn import functional as F
                                # 这个操作非常耗时，其实按理来说应该可以先resize之后再max_pool，
                                # 反正之后要线性插值，可以省下几十倍的时间
                                cloud_max = F.max_pool2d(cloud, 3, stride=1, padding=1)
                                
                                # show_proj_cloud(cloud_max.squeeze(0).permute(1,2,0).detach().cpu().numpy(), '2')
                                cloud_fill = torch.where(cloud_mask, cloud_max, cloud)
                                # show_proj_cloud(cloud_fill.squeeze(0).permute(1,2,0).detach().cpu().numpy(), '3')
                                cloud = cloud_fill
                                # st()

                                # for ks in range(2, 10):
                                #     cloud_max = F.min_pool2d(cloud, ks, stride=1)
                                #     cloud_fill = torch.where(cloud_mask, cloud_max, cloud)
                                #     print((cloud_avg!=0).sum())
                                #     st()
                                #     show_proj_cloud(cloud_avg.squeeze(0).permute(1,2,0).detach().cpu().numpy())
                                
                                # st()
                                # import time
                                # start = time.time()
                                # for i in range(100):cloud_avg = F.max_pool2d(cloud_avg, 3, stride=1, padding=1);print((cloud_avg!=0).sum())
                                # print(f'cost {time.time() - start}')
                            else:
                                cloud = cloud.unsqueeze(0).permute(0,3,1,2).float()

                        # st()


                        # print(cloud[:,0,:,:].max(), cloud[:,1,:,:].max(), cloud[:,2,:,:].max())
                        # print(cloud[:,0,:,:].min(), cloud[:,1,:,:].min(), cloud[:,2,:,:].min())
                        # print('\n')


                        ########################### org cuda out of mem ###########################
                        if not self.unet_dep15:
                            # st()
                            grid_coord_from_depthmap = F.grid_sample(cloud, sample_grid, align_corners=True)
                        else:
                            sample_grid_repeat = sample_grid.repeat(cloud.shape[0], 1, 1, 1)
                            grid_coord_from_depthmap = F.grid_sample(cloud, sample_grid_repeat, align_corners=True)
                        
                        # grid_coord_from_depthmap = F.grid_sample(cloud, sample_grid, align_corners=True)
                        grid_reshape = grid.permute(1,0).reshape((1,3,1,nbins))
                        # st()
                        grid_res = grid_reshape - grid_coord_from_depthmap
                        grid_res_square = (grid_res * grid_res).sum(axis=1)
                        grid_res = grid_res_square.sqrt()

                        # # 下面这种根据距离来赋予权重的，即使里的很远，也能有一个很高的权重，不合适
                        # weight_all_dis = (1 - grid_res / grid_res.max()).reshape(1,1,1,nbins).repeat(1,num_joints,1,1)

                        # 这下面的测试的结果是发现
                        # sigma1000 > sigma100 > sigma10
                        # 万一sigma10000 > sigma1000
                        # 目前发现sigma2000有希望提升效果

                        # *weightmap几乎没有意义了，就说明这个加权没有卵用
                        # 高斯权重
                        # sigma = 10.0
                        # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        # print(weight_all_gau.mean())
                        # sigma = 100.0
                        # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        # print(weight_all_gau.mean())
                        # sigma = 1000.0
                        # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        # print(weight_all_gau.mean())

                        sigma = self.f_weight_sigma

                        weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        # print(weight_all_gau.mean())
                        # st()
                        weight_all = weight_all_gau

                        # st()
                        
                        # 感觉绝大多数的点都离得太远了，其实也合理，本身cloud中都只有1/10的位置有值
                        # 而且一条射线经过的点最短是边长，就算是80把，这80中只有可能有几个点在很近的地方
                        # 那就相当于128000/8000=16可不是就只有16个点很近吗

                        # if pytorch version < 1.3.0, align_corners=True should be omitted.
                        # st()
                        # cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) * weight_all
                        

                        # st()
                        cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][0:1, i:i + 1, :, :], sample_grid, align_corners=True) * weight_all




                        if self.f_weight_with_w == '':
                            cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][0:1, i:i + 1, :, :], sample_grid, align_corners=True) * weight_all
                        elif self.f_weight_with_w == 'add':
                            cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][0:1, i:i + 1, :, :] + 0.2, sample_grid, align_corners=True) * weight_all
                        else:
                            assert False

                        ### no dep15
                        # (Pdb) cubes[i:i + 1, :, :, :, c].shape
                        # torch.Size([1, 15, 1, 128000])
                        # (Pdb) (F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) * weight_all).shape
                        # torch.Size([1, 15, 1, 128000])
                        # (Pdb) weight_all.shape
                        # torch.Size([1, 1, 128000])
                        # (Pdb) F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True).shape
                        # torch.Size([1, 15, 1, 128000])



                        # (Pdb) cubes[i:i + 1, :, :, :, c].shape
                        # torch.Size([1, 15, 1, 128000])
                        # (Pdb) (F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) * weight_all).shape
                        # torch.Size([1, 15, 1, 128000])
                        # (Pdb) weight_all.shape
                        # torch.Size([15, 1, 128000])
                        # (Pdb) F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True).shape
                        # torch.Size([1, 15, 1, 128000])
                        ########################### org cuda out of mem ###########################

                        ########################### lcc ###########################
                        # for joint_i in range(15):
                        #     grid_coord_from_depthmap = F.grid_sample(cloud[joint_i:joint_i+1], sample_grid, align_corners=True)
                        #     grid_reshape = grid.permute(1,0).reshape((1,3,1,nbins))
                        #     # st()
                        #     grid_res = grid_reshape - grid_coord_from_depthmap
                        #     grid_res_square = (grid_res * grid_res).sum(axis=1)
                        #     grid_res = grid_res_square.sqrt()

                        #     # # 下面这种根据距离来赋予权重的，即使里的很远，也能有一个很高的权重，不合适
                        #     # weight_all_dis = (1 - grid_res / grid_res.max()).reshape(1,1,1,nbins).repeat(1,num_joints,1,1)

                        #     # 这下面的测试的结果是发现
                        #     # sigma1000 > sigma100 > sigma10
                        #     # 万一sigma10000 > sigma1000
                        #     # 目前发现sigma2000有希望提升效果

                        #     # *weightmap几乎没有意义了，就说明这个加权没有卵用
                        #     # 高斯权重
                        #     # sigma = 10.0
                        #     # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        #     # print(weight_all_gau.mean())
                        #     # sigma = 100.0
                        #     # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        #     # print(weight_all_gau.mean())
                        #     # sigma = 1000.0
                        #     # weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        #     # print(weight_all_gau.mean())
                        #     sigma = self.f_weight_sigma
                        #     weight_all_gau = (-grid_res**2/2/sigma**2).exp()
                        #     # print(weight_all_gau.mean())
                        #     # st()
                        #     weight_all = weight_all_gau

                        #     # st()
                            
                        #     # 感觉绝大多数的点都离得太远了，其实也合理，本身cloud中都只有1/10的位置有值
                        #     # 而且一条射线经过的点最短是边长，就算是80把，这80中只有可能有几个点在很近的地方
                        #     # 那就相当于128000/8000=16可不是就只有16个点很近吗

                        #     # if pytorch version < 1.3.0, align_corners=True should be omitted.
                        #     # st()
                        #     # cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) * weight_all
                        #     cubes[i:i + 1, joint_i:joint_i+1, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, joint_i:joint_i+1, :, :], sample_grid, align_corners=True) * weight_all
                        ########################### lcc ###########################
                    else:
                        cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True)

                    ## lcc debugging
                    # if grid_size == [2000.0, 2000.0, 2000.0]:
                    #     # (Pdb) xy.shape
                    #     # torch.Size([262144, 2])
                    #     # (Pdb) grid.shape
                    #     # torch.Size([262144, 3])
                    #     # (Pdb) cubes.shape
                    #     # torch.Size([1, 15, 1, 262144, 1])
                    #     # (Pdb) cubes[:,0,:,:,:].shape
                    #     # torch.Size([1, 1, 262144, 1])
                    #     # (Pdb) cubes[:,0,:,:,:].max()
                    #     # tensor(0.9346, device='cuda:0')
                    #     # (Pdb) cubes[:,0,:,:,:].min()
                    #     # tensor(-0.0024, device='cuda:0')
                    #     st()
                    #     pass


                    # print(f'f_weight cost {time.time() - start}')
                
                if vis_time_cost_flag:
                    time_end=time.time();print('cost2\n',time_end-time_start)

        # 一直都不对
        # # 验证一下投影的对不对，看看joint的位置cube值是否最大
        # # 这样算的不对，难道这两个坐标不是一样的吗
        # for jj in range(15):
        #     st()
        #     # 第jj个joint，下面输出多个表示多个人
        #     print(meta[0]['joints_3d'][i,:,jj,:])
        #     print(grid[cubes[:,jj,:,:,:].argmax(),:])
            
        # st()
        # cubes = cubes.mean(dim=-1)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6)
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(1, num_joints, cube_size[0], cube_size[1], cube_size[2])  ##

        ### 注意grids中包含batchsize和view_num
        ### xy是一个batch，一个view，但是由于我只做单视角这里简略了直接return出去
        return cubes, grids, xy

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size, all_cloud=None):
        cubes, grids, xy = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size, all_cloud=all_cloud)
        return cubes, grids, xy