# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_resnet
from models import pose_resnet_randsigma
from models import pose_hrnet_wholebody_mp
from models import pose_hrnet_wholebody
from models.cuboid_proposal_net import CuboidProposalNet
from models.pose_regression_net import PoseRegressionNet

from models.joint_refine_net import JointRefineNet
from models.joint_refine_net_face import JointRefineNetFace
from models.joint_refine_net_facehand import JointRefineNetFaceHand
from models.joint_refine_net_facehandc import JointRefineNetFaceHandC
from models.joint_refine_net_facehandsepv import JointRefineNetFaceHandSepV

from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss

from pdb import set_trace as st
import torch.nn.functional as F
import numpy as np
import os 
from os import path as osp
import gzip


vis_depth_flag = False
# vis_depth_flag = True

vis_flag = False

# vis_time_cost_flag = True
vis_time_cost_flag = False

show_h = 540;show_w = 960
# show_h = 270;show_w = 480



import cv2 as cv
def show_proj_cloud(proj_cloud, name=None):
    # 原本的yz轴交换了，这边用proj_cloud[:, :, 1]
    proj_cloud_show_z = proj_cloud[:, :, 0]
    # st()
    proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
    # proj_cloud_show_z = proj_cloud_show_z / 8.0

    proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
    # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
    proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)

    cv.namedWindow(name,0)
    cv.resizeWindow(name, show_w, show_h)
    cv.imshow(name, proj_cloud_show_z_color1)
    if 113 == cv.waitKey(100):
        st()
    
    # plt.figure()
    # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
    # plt.imshow(proj_cloud_show_z_color1)
    # plt.show()

    # plt.figure()
    # # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
    # plt.imshow(proj_cloud_show_z_color2)
#     # plt.show()



def tensor2im(input_image, imtype=np.uint8):
    """"
    Parameters:
        input_image (tensor) --  输入的tensor，维度为CHW，注意这里没有batch size的维度
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406] # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)


import cv2 as cv
def show_view(view, name=None):

    view_to_show = tensor2im(view[0])
    view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)

    cv.namedWindow(name,0)
    cv.resizeWindow(name, show_w, show_h)
    cv.imshow(name, view_to_show)
    if 113 == cv.waitKey(100):
        st()



def show_view_w_aligned_depth(view, proj_cloud, name=None):

    view_to_show = tensor2im(view[0])
    view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)


    # 原本的yz轴交换了，这边用proj_cloud[:, :, 1]
    proj_cloud_show_z = proj_cloud[:, :, 0]
    # st()
    proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
    # proj_cloud_show_z = proj_cloud_show_z / 8.0

    proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
    # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
    proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)

    view_to_show = (proj_cloud_show_z_color1 * 0.5 + view_to_show * 0.5).astype(np.uint8)


    cv.namedWindow(name,0)
    cv.resizeWindow(name, show_w, show_h)
    cv.imshow(name, view_to_show)
    if 113 == cv.waitKey(100):
        st()




# def depthmap_to_cloud_float(all_denoised_depth, meta, dep_downsample_size, code_to_seq, device):
#     all_cloud = []
#     # st()
#     for c, denoised_depth in enumerate(all_denoised_depth):
#         bsize = denoised_depth.shape[0]
#         cloud_batch_list = []
#         for i in range(bsize):
#             denoised_depth_batch = denoised_depth[i]
#             ### 关节点通道
#             proj_cloud_fromz_allj_list = []
#             for denoised_depth_index in range(len(denoised_depth_batch)):
                
#                 if show_depthmap_to_cloud_time_cost_flag:
#                     print('st1')
#                     import time;time_start=time.time()

#                 denoised_depth_i = denoised_depth_batch[denoised_depth_index] # depth一通道

                
#                 # calib = calib_list[meta[c]['seq'][i]][meta[c]['camera_index'][i]]
#                 calib = calib_list[code_to_seq[int(meta[c]['seq_code'][i])]][meta[c]['camera_index'][i]]

#                 # st()
#                 denoised_depth_i = denoised_depth_i * 1e3 * 1e1
#                 denoised_depth_i = torch.as_tensor(denoised_depth_i, dtype=torch.float, device=device)
#                 # X, Y = torch.meshgrid(torch.arange(denoised_depth_i.shape[1]), torch.arange(denoised_depth_i.shape[0]))
#                 X, Y = np.meshgrid(np.arange(denoised_depth_i.shape[1]), np.arange(denoised_depth_i.shape[0]))
#                 X, Y = torch.from_numpy(X), torch.from_numpy(Y)
#                 ### 我发现这里meshgrid的顺序torch和np好像不一样，改成numpy顺序即可
#                 if vis_flag:
#                     print(f'x {X}')
#                     print(f'y {Y}')


#                 ### lcc debugging
#                 ### 你可以测试一下不transform情况下的情况，直接用原分辨率来做
#                 # st()
#                 X = X.float()
#                 Y = Y.float()
#                 ### 注意！！这里在改depth分辨率的时候一定要一起改
#                 # w, h, w_resized, h_resized = 1920, 1080, 960, 512
#                 # w, h, w_resized, h_resized = 1920, 1080, 240, 128

#                 # st()
#                 w, h, w_resized, h_resized = 1920, 1080, dep_downsample_size[1], dep_downsample_size[0]

                
#                 w_pad = h / h_resized * w_resized
#                 h_pad = h
#                 # st()
#                 X = X * h / h_resized - (w_pad - w) / 2
#                 Y = Y * h / h_resized

#                 p2dd = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
#                 p2dd = torch.as_tensor(p2dd, dtype=torch.float, device=device)

#                 c_K, c_d = calib.color_proj()
#                 R_color, t_color, T_M_color = calib.lcc_M_color()

#                 Kdist = c_K
#                 Mdist = T_M_color[:3,:]
#                 # distCoeffs = c_d
#                 distCoeffs = c_d[:,0]

#                 Kdist = torch.as_tensor(Kdist, dtype=torch.float, device=device)
#                 Mdist = torch.as_tensor(Mdist, dtype=torch.float, device=device)
#                 distCoeffs = torch.as_tensor(distCoeffs, dtype=torch.float, device=device)

#                 lcc_p2d = torch.cat((p2dd, denoised_depth_i.reshape(-1, 1)), axis=1)

#                 if vis_flag:
#                     print(f'lcc_p2d.mean() {lcc_p2d.mean()}')
#                     print(f'denoised_depth_i.mean() {denoised_depth_i.mean()}')


                
#                 p3d = unproject2_float(lcc_p2d, Kdist, Mdist, distCoeffs, device)
#                 # st()
#                 # print(f'p3d.mean() {p3d.mean()}')

#                 # proj_cloud_color_xy_fromz = p3d.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)[:, :, :2] * 1e-1
#                 # if vis_flag:
#                 #     print(f'proj_cloud_color_xy_fromz[...,0].mean() {proj_cloud_color_xy_fromz[...,0].mean()} proj_cloud_color_xy_fromz[...,1].mean() {proj_cloud_color_xy_fromz[...,1].mean()}')
#                 #     print(f'(proj_cloud_color_xy_fromz[:,:,0]==0).sum() {(proj_cloud_color_xy_fromz[:,:,0]==0).sum()}')

#                 proj_cloud_color_xyz = p3d.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1]) * 1e-1


#                 ################################## 映射到世界坐标系下面的 #############################
#                 # print(f'p3d.mean() {p3d.mean()}')

#                 cloud_fromz = p3d * 1e-1
#                 cloud_local_mask = (cloud_fromz[:,2]==0)
                

#                 if show_depthmap_to_cloud_time_cost_flag:
#                     time_end=time.time();print('cost1\n',time_end-time_start)
#                     print('st2')
#                     import time;time_start=time.time()


#                 ################ <np implemented part> ################
#                 scale_kinoptic2panoptic = np.eye(4)
#                 scaleFactor = 100
#                 scale_kinoptic2panoptic[:3,:3] = scaleFactor*scale_kinoptic2panoptic[:3,:3]

#                 panoptic_calibData_R, panoptic_calibData_t = calib.lcc_panoptic_calibData()
#                 M = np.concatenate([panoptic_calibData_R, panoptic_calibData_t], axis=1)
#                 T_panopticWorld2KinectColor = np.row_stack([M, [0,0,0,1]])
#                 T_kinectColor2PanopticWorld = np.linalg.inv(T_panopticWorld2KinectColor)
            
#                 T_kinectColor2PanopticWorld = T_kinectColor2PanopticWorld.dot(scale_kinoptic2panoptic)
#                 T_kinectColor2PanopticWorld = torch.from_numpy(T_kinectColor2PanopticWorld)
#                 T_kinectColor2PanopticWorld = torch.as_tensor(T_kinectColor2PanopticWorld, dtype=torch.float, device=device)
#                 ################ <np implemented part> ################


#                 if show_depthmap_to_cloud_time_cost_flag:
#                     time_end=time.time();print('cost2\n',time_end-time_start)
#                     print('st3')
#                     import time;time_start=time.time()

#                 p3d_world_fromz = torch.mm(T_kinectColor2PanopticWorld, torch.cat([cloud_fromz.T, torch.ones((1, cloud_fromz.shape[0]), device=device).float()], axis=0))
#                 p3d_world_fromz = p3d_world_fromz[:3, :].T
                
#                 ### cloud_local原本z=0的位置不应该映射有xy
#                 # st()
#                 p3d_world_fromz[cloud_local_mask] = 0

#                 # st()
#                 ### 这两种reshape和reshape之后permute之后的结果居然不同，permute应该是对的把
#                 proj_cloud_fromz_1 = p3d_world_fromz.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)
#                 if vis_flag:
#                     print(f'proj_cloud_fromz_1[:,:,0].mean() {proj_cloud_fromz_1[:,:,0].mean()}')
#                     print(f'proj_cloud_fromz_1[:,:,1].mean() {proj_cloud_fromz_1[:,:,1].mean()}')
#                     print(f'proj_cloud_fromz_1[:,:,2].mean() {proj_cloud_fromz_1[:,:,2].mean()}')

#                 # proj_cloud_fromz_2 = p3d_world_fromz.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1])
#                 # if vis_flag:
#                 #     print(f'proj_cloud_fromz_2[0].mean() {proj_cloud_fromz_2[0].mean()}')
#                 #     print(f'proj_cloud_fromz_2[1].mean() {proj_cloud_fromz_2[1].mean()}')
#                 #     print(f'proj_cloud_fromz_2[2].mean() {proj_cloud_fromz_2[2].mean()}')

#                 ### 验证这个世界坐标一致性，注意cloud和depth都不能transform
#                 # proj_cloud = meta[0]['cloud'][0]
#                 # proj_cloud = proj_cloud.reshape(-1,3)
#                 # proj_cloud[cloud_local_mask] = 0
#                 # proj_cloud = proj_cloud.reshape(proj_cloud_fromz_1.shape)
                
#                 ### 世界坐标cloud的不一致性，应该也是transform导致的
#                 # 把cloud也transform之后即可

#                 ################################## 检验映射到世界坐标系下面的 #############################

#                 if vis_flag:
#                     print(f'proj_cloud_color_xyz[0].mean() {proj_cloud_color_xyz[0].mean()}')
#                     print(f'proj_cloud_color_xyz[1].mean() {proj_cloud_color_xyz[1].mean()}')
#                     print(f'proj_cloud_color_xyz[2].mean() {proj_cloud_color_xyz[2].mean()}')
                
#                 # 你可以测试一下不transform情况下的情况，直接用原分辨率来做
#                 ### transform
#                 M = np.array([[1.0, 0.0, 0.0],
#                             [0.0, 0.0, -1.0],
#                             [0.0, 1.0, 0.0]])
#                 M = torch.from_numpy(M)
#                 # st()
#                 M = torch.as_tensor(M, dtype=torch.float, device=device)
#                 proj_cloud_fromz_1 = torch.mm(proj_cloud_fromz_1.reshape(-1,3), M).reshape(proj_cloud_fromz_1.shape) * 10
#                 # 
#                 # st()
#                 # 我懂了，我之前测试的就是transform到540之前的坐标是没有问题的
#                 # (proj_cloud_fromz_1 - proj_cloud).max()
#                 # (Pdb) (proj_cloud_fromz_1 - proj_cloud).max()
#                 # tensor(12.1170, device='cuda:0', dtype=torch.float64)
#                 # (Pdb) (proj_cloud_fromz_1 - proj_cloud).min()
#                 # tensor(-11.8288, device='cuda:0', dtype=torch.float64)
#                 # 这证明整个流程是正确的
#                 proj_cloud_fromz = proj_cloud_fromz_1.permute((2,0,1))
#                 proj_cloud_fromz_allj_list.append(proj_cloud_fromz.unsqueeze(0))

#                 if show_depthmap_to_cloud_time_cost_flag:
#                     time_end=time.time();print('cost3\n',time_end-time_start)


#             proj_cloud_fromz_allj = torch.cat(proj_cloud_fromz_allj_list, axis=0)
#             cloud_batch_list.append(proj_cloud_fromz_allj.unsqueeze(0))

#         cloud_batch = torch.cat(cloud_batch_list, axis=0)
#         all_cloud.append(cloud_batch)
#     return all_cloud

# def unproject2_float(p2d, Kdist, Mdist, distCoeffs, device):
#     # st()
#     # pn2d = torch.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T
    
#     pn2d, _ = torch.solve(torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device).float()), axis=0), Kdist)
#     pn2d = pn2d.T

#     # pn2d = torch.linalg.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T

#     # print(f'pn2d.mean() {pn2d.mean()}')

#     # k = torch.cat([distCoeffs[:5], np.zeros(12-5)], axis=0)
#     k = torch.cat((distCoeffs[:len(distCoeffs)], torch.zeros(12-len(distCoeffs), device=device).float()), axis=0)
#     x0 = pn2d[:,0]
#     y0 = pn2d[:,1]
#     x = x0
#     y = y0

#     # print(f'x {x}')
#     # print(f'y {y}')

#     for iter in range(5):
#         r2 = x*x + y*y
#         # st()
#         icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
#         deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
#         deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
#         x = (x0 - deltaX)*icdist
#         y = (y0 - deltaY)*icdist
    
#     pn2d = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), axis=1)
    
#     if vis_flag:
#         print(f'pn2d.mean() {pn2d.mean()}')
#         print(f'p2d.mean() {p2d.mean()}')
#         print(f'(p2d[:,2:3]*0.001).mean() {(p2d[:,2:3]*0.001).mean()}')
#         print(f'(pn2d * (p2d[:,2:3]*0.001)).mean() {(pn2d * (p2d[:,2:3]*0.001)).mean()}')
#         print(f'(p2d[:,2:3]*0.001).mean() {(p2d[:,2:3]*0.001).mean()}')
#         print(f'pn2d {pn2d}')
    
#     p3d = torch.cat([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
#     # p3d = np.concatenate([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
    
#     if vis_flag:
#         print(f'p3d.mean() {p3d.mean()}')

#     # st()
    
#     # print(p3d.mean(axis=0))
#     # st()
#     # p3d = np.matmul(np.linalg.inv(np.row_stack([Mdist, [0,0,0,1]])), torch.cat([p3d, np.ones((p3d.shape[0], 1))], axis=1).T).T
#     # print(p3d.mean(axis=0))
#     p3d = p3d[:,:3]
#     # p3d = (inv([Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')'
#     # p3d = (inv([cam.Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')';
#     # p3d = [bsxfun(@times, pn2d, p2d(:,3)*0.001) p2d(:,3)*0.001];
#     ### 估计没用
#     # if pn2d.shape[1] == 1:
#     #     pn2d = pn2d.T
#     # st()
#     # pn2d = [x y]
#     # for iter=1:5
#     #     r2 = x.*x + y.*y;
#     #     icdist = (1 + ((k(1+7)*r2 + k(1+6)).*r2 + k(1+5)).*r2)./(1 + ((k(1+4)*r2 + k(1+1)).*r2 + k(1+0)).*r2);
#     #     deltaX = 2*k(1+2)*x.*y + k(1+3)*(r2 + 2*x.*x)+ k(1+8)*r2+k(1+9)*r2.*r2;
#     #     deltaY = k(1+2)*(r2 + 2*y.*y) + 2*k(1+3)*x.*y+ k(1+10)*r2+k(1+11)*r2.*r2;
#     #     x = (x0 - deltaX).*icdist;
#     #     y = (y0 - deltaY).*icdist;
#     # end
#     # k = [reshape(cam.distCoeffs(1:5),[],1); zeros(12-5,1)];
#     # pn2d = ( cam.Kdist \ [p2d(:,1:2)'; ones(1, size(p2d,1))] )';
#     return p3d

    

show_depthmap_to_cloud_time_cost_flag = False
# show_depthmap_to_cloud_time_cost_flag = True
def depthmap_to_cloud(all_denoised_depth, meta, dep_downsample_size, code_to_seq, calib_list, device):
    all_cloud = []
    # st()
    for c, denoised_depth in enumerate(all_denoised_depth):
        bsize = denoised_depth.shape[0]
        cloud_batch_list = []
        for i in range(bsize):
            denoised_depth_batch = denoised_depth[i]
            ### 关节点通道
            proj_cloud_fromz_allj_list = []
            for denoised_depth_index in range(len(denoised_depth_batch)):
                
                if show_depthmap_to_cloud_time_cost_flag:
                    print('st1')
                    import time;time_start=time.time()

                denoised_depth_i = denoised_depth_batch[denoised_depth_index] # depth一通道

                try:
                    # calib = calib_list[meta[c]['seq'][i]][meta[c]['camera_index'][i]]
                    calib = calib_list[code_to_seq[int(meta[c]['seq_code'][i])]][meta[c]['camera_index'][i]]
                except:
                    st()

                # st()
                denoised_depth_i = denoised_depth_i * 1e3 * 1e1
                denoised_depth_i = torch.as_tensor(denoised_depth_i, dtype=torch.double, device=device)
                # X, Y = torch.meshgrid(torch.arange(denoised_depth_i.shape[1]), torch.arange(denoised_depth_i.shape[0]))
                X, Y = np.meshgrid(np.arange(denoised_depth_i.shape[1]), np.arange(denoised_depth_i.shape[0]))
                X, Y = torch.from_numpy(X), torch.from_numpy(Y)
                ### 我发现这里meshgrid的顺序torch和np好像不一样，改成numpy顺序即可
                if vis_flag:
                    print(f'x {X}')
                    print(f'y {Y}')


                ### lcc debugging
                ### 你可以测试一下不transform情况下的情况，直接用原分辨率来做
                # st()
                X = X.double()
                Y = Y.double()
                ### 注意！！这里在改depth分辨率的时候一定要一起改
                # w, h, w_resized, h_resized = 1920, 1080, 960, 512
                # w, h, w_resized, h_resized = 1920, 1080, 240, 128

                # st()
                w, h, w_resized, h_resized = 1920, 1080, dep_downsample_size[1], dep_downsample_size[0]

                
                w_pad = h / h_resized * w_resized
                h_pad = h
                # st()
                X = X * h / h_resized - (w_pad - w) / 2
                Y = Y * h / h_resized

                p2dd = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
                p2dd = torch.as_tensor(p2dd, dtype=torch.double, device=device)

                c_K, c_d = calib.color_proj()
                R_color, t_color, T_M_color = calib.lcc_M_color()

                Kdist = c_K
                Mdist = T_M_color[:3,:]
                # distCoeffs = c_d
                distCoeffs = c_d[:,0]

                Kdist = torch.as_tensor(Kdist, dtype=torch.double, device=device)
                Mdist = torch.as_tensor(Mdist, dtype=torch.double, device=device)
                distCoeffs = torch.as_tensor(distCoeffs, dtype=torch.double, device=device)

                lcc_p2d = torch.cat((p2dd, denoised_depth_i.reshape(-1, 1)), axis=1)

                if vis_flag:
                    print(f'lcc_p2d.mean() {lcc_p2d.mean()}')
                    print(f'denoised_depth_i.mean() {denoised_depth_i.mean()}')


                
                p3d = unproject2(lcc_p2d, Kdist, Mdist, distCoeffs, device)
                # st()
                # print(f'p3d.mean() {p3d.mean()}')

                # proj_cloud_color_xy_fromz = p3d.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)[:, :, :2] * 1e-1
                # if vis_flag:
                #     print(f'proj_cloud_color_xy_fromz[...,0].mean() {proj_cloud_color_xy_fromz[...,0].mean()} proj_cloud_color_xy_fromz[...,1].mean() {proj_cloud_color_xy_fromz[...,1].mean()}')
                #     print(f'(proj_cloud_color_xy_fromz[:,:,0]==0).sum() {(proj_cloud_color_xy_fromz[:,:,0]==0).sum()}')

                proj_cloud_color_xyz = p3d.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1]) * 1e-1


                ################################## 映射到世界坐标系下面的 #############################
                # print(f'p3d.mean() {p3d.mean()}')

                cloud_fromz = p3d * 1e-1
                cloud_local_mask = (cloud_fromz[:,2]==0)
                

                if show_depthmap_to_cloud_time_cost_flag:
                    time_end=time.time();print('cost1\n',time_end-time_start)
                    print('st2')
                    import time;time_start=time.time()


                ################ <np implemented part> ################
                scale_kinoptic2panoptic = np.eye(4)
                scaleFactor = 100
                scale_kinoptic2panoptic[:3,:3] = scaleFactor*scale_kinoptic2panoptic[:3,:3]

                panoptic_calibData_R, panoptic_calibData_t = calib.lcc_panoptic_calibData()
                M = np.concatenate([panoptic_calibData_R, panoptic_calibData_t], axis=1)
                T_panopticWorld2KinectColor = np.row_stack([M, [0,0,0,1]])
                T_kinectColor2PanopticWorld = np.linalg.inv(T_panopticWorld2KinectColor)
            
                T_kinectColor2PanopticWorld = T_kinectColor2PanopticWorld.dot(scale_kinoptic2panoptic)
                T_kinectColor2PanopticWorld = torch.from_numpy(T_kinectColor2PanopticWorld)
                T_kinectColor2PanopticWorld = torch.as_tensor(T_kinectColor2PanopticWorld, dtype=torch.double, device=device)
                ################ <np implemented part> ################


                if show_depthmap_to_cloud_time_cost_flag:
                    time_end=time.time();print('cost2\n',time_end-time_start)
                    print('st3')
                    import time;time_start=time.time()

                p3d_world_fromz = torch.mm(T_kinectColor2PanopticWorld, torch.cat([cloud_fromz.T, torch.ones((1, cloud_fromz.shape[0]), device=device).double()], axis=0))
                p3d_world_fromz = p3d_world_fromz[:3, :].T
                
                ### cloud_local原本z=0的位置不应该映射有xy
                # st()
                p3d_world_fromz[cloud_local_mask] = 0

                # st()
                ### 这两种reshape和reshape之后permute之后的结果居然不同，permute应该是对的把
                proj_cloud_fromz_1 = p3d_world_fromz.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)
                if vis_flag:
                    print(f'proj_cloud_fromz_1[:,:,0].mean() {proj_cloud_fromz_1[:,:,0].mean()}')
                    print(f'proj_cloud_fromz_1[:,:,1].mean() {proj_cloud_fromz_1[:,:,1].mean()}')
                    print(f'proj_cloud_fromz_1[:,:,2].mean() {proj_cloud_fromz_1[:,:,2].mean()}')

                # proj_cloud_fromz_2 = p3d_world_fromz.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1])
                # if vis_flag:
                #     print(f'proj_cloud_fromz_2[0].mean() {proj_cloud_fromz_2[0].mean()}')
                #     print(f'proj_cloud_fromz_2[1].mean() {proj_cloud_fromz_2[1].mean()}')
                #     print(f'proj_cloud_fromz_2[2].mean() {proj_cloud_fromz_2[2].mean()}')

                ### 验证这个世界坐标一致性，注意cloud和depth都不能transform
                # proj_cloud = meta[0]['cloud'][0]
                # proj_cloud = proj_cloud.reshape(-1,3)
                # proj_cloud[cloud_local_mask] = 0
                # proj_cloud = proj_cloud.reshape(proj_cloud_fromz_1.shape)
                
                ### 世界坐标cloud的不一致性，应该也是transform导致的
                # 把cloud也transform之后即可

                ################################## 检验映射到世界坐标系下面的 #############################

                if vis_flag:
                    print(f'proj_cloud_color_xyz[0].mean() {proj_cloud_color_xyz[0].mean()}')
                    print(f'proj_cloud_color_xyz[1].mean() {proj_cloud_color_xyz[1].mean()}')
                    print(f'proj_cloud_color_xyz[2].mean() {proj_cloud_color_xyz[2].mean()}')
                
                # 你可以测试一下不transform情况下的情况，直接用原分辨率来做
                ### transform
                M = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])
                M = torch.from_numpy(M)
                # st()
                M = torch.as_tensor(M, dtype=torch.double, device=device)
                proj_cloud_fromz_1 = torch.mm(proj_cloud_fromz_1.reshape(-1,3), M).reshape(proj_cloud_fromz_1.shape) * 10
                # 
                # st()
                # 我懂了，我之前测试的就是transform到540之前的坐标是没有问题的
                # (proj_cloud_fromz_1 - proj_cloud).max()
                # (Pdb) (proj_cloud_fromz_1 - proj_cloud).max()
                # tensor(12.1170, device='cuda:0', dtype=torch.float64)
                # (Pdb) (proj_cloud_fromz_1 - proj_cloud).min()
                # tensor(-11.8288, device='cuda:0', dtype=torch.float64)
                # 这证明整个流程是正确的
                proj_cloud_fromz = proj_cloud_fromz_1.permute((2,0,1))
                proj_cloud_fromz_allj_list.append(proj_cloud_fromz.unsqueeze(0))

                if show_depthmap_to_cloud_time_cost_flag:
                    time_end=time.time();print('cost3\n',time_end-time_start)


            proj_cloud_fromz_allj = torch.cat(proj_cloud_fromz_allj_list, axis=0)
            cloud_batch_list.append(proj_cloud_fromz_allj.unsqueeze(0))

        cloud_batch = torch.cat(cloud_batch_list, axis=0)
        all_cloud.append(cloud_batch)
    return all_cloud

def depthmap_to_cloud_one(denoised_depth_batch, meta, dep_downsample_size, code_to_seq, calib_list, view_index, batch_index, device):
    ### 关节点通道
    proj_cloud_fromz_allj_list = []
    for denoised_depth_index in range(len(denoised_depth_batch)):
        
        if show_depthmap_to_cloud_time_cost_flag:
            print('st1')
            import time;time_start=time.time()

        denoised_depth_i = denoised_depth_batch[denoised_depth_index] # depth一通道

        try:
            # calib = calib_list[meta[c]['seq'][i]][meta[c]['camera_index'][i]]
            calib = calib_list[code_to_seq[int(meta[view_index]['seq_code'][batch_index])]][meta[view_index]['camera_index'][batch_index]]
        except:
            st()

        # st()
        denoised_depth_i = denoised_depth_i * 1e3 * 1e1
        denoised_depth_i = torch.as_tensor(denoised_depth_i, dtype=torch.double, device=device)
        # X, Y = torch.meshgrid(torch.arange(denoised_depth_i.shape[1]), torch.arange(denoised_depth_i.shape[0]))
        X, Y = np.meshgrid(np.arange(denoised_depth_i.shape[1]), np.arange(denoised_depth_i.shape[0]))
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        ### 我发现这里meshgrid的顺序torch和np好像不一样，改成numpy顺序即可
        if vis_flag:
            print(f'x {X}')
            print(f'y {Y}')


        ### lcc debugging
        ### 你可以测试一下不transform情况下的情况，直接用原分辨率来做
        # st()
        X = X.double()
        Y = Y.double()
        ### 注意！！这里在改depth分辨率的时候一定要一起改
        # w, h, w_resized, h_resized = 1920, 1080, 960, 512
        # w, h, w_resized, h_resized = 1920, 1080, 240, 128

        # st()
        w, h, w_resized, h_resized = 1920, 1080, dep_downsample_size[1], dep_downsample_size[0]

        
        w_pad = h / h_resized * w_resized
        h_pad = h
        # st()
        X = X * h / h_resized - (w_pad - w) / 2
        Y = Y * h / h_resized

        p2dd = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        p2dd = torch.as_tensor(p2dd, dtype=torch.double, device=device)

        c_K, c_d = calib.color_proj()
        R_color, t_color, T_M_color = calib.lcc_M_color()

        Kdist = c_K
        Mdist = T_M_color[:3,:]
        # distCoeffs = c_d
        distCoeffs = c_d[:,0]

        Kdist = torch.as_tensor(Kdist, dtype=torch.double, device=device)
        Mdist = torch.as_tensor(Mdist, dtype=torch.double, device=device)
        distCoeffs = torch.as_tensor(distCoeffs, dtype=torch.double, device=device)

        lcc_p2d = torch.cat((p2dd, denoised_depth_i.reshape(-1, 1)), axis=1)

        if vis_flag:
            print(f'lcc_p2d.mean() {lcc_p2d.mean()}')
            print(f'denoised_depth_i.mean() {denoised_depth_i.mean()}')


        
        p3d = unproject2(lcc_p2d, Kdist, Mdist, distCoeffs, device)
        # st()
        # print(f'p3d.mean() {p3d.mean()}')

        # proj_cloud_color_xy_fromz = p3d.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)[:, :, :2] * 1e-1
        # if vis_flag:
        #     print(f'proj_cloud_color_xy_fromz[...,0].mean() {proj_cloud_color_xy_fromz[...,0].mean()} proj_cloud_color_xy_fromz[...,1].mean() {proj_cloud_color_xy_fromz[...,1].mean()}')
        #     print(f'(proj_cloud_color_xy_fromz[:,:,0]==0).sum() {(proj_cloud_color_xy_fromz[:,:,0]==0).sum()}')

        proj_cloud_color_xyz = p3d.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1]) * 1e-1


        ################################## 映射到世界坐标系下面的 #############################
        # print(f'p3d.mean() {p3d.mean()}')

        cloud_fromz = p3d * 1e-1
        cloud_local_mask = (cloud_fromz[:,2]==0)
        

        if show_depthmap_to_cloud_time_cost_flag:
            time_end=time.time();print('cost1\n',time_end-time_start)
            print('st2')
            import time;time_start=time.time()


        ################ <np implemented part> ################
        scale_kinoptic2panoptic = np.eye(4)
        scaleFactor = 100
        scale_kinoptic2panoptic[:3,:3] = scaleFactor*scale_kinoptic2panoptic[:3,:3]

        panoptic_calibData_R, panoptic_calibData_t = calib.lcc_panoptic_calibData()
        M = np.concatenate([panoptic_calibData_R, panoptic_calibData_t], axis=1)
        T_panopticWorld2KinectColor = np.row_stack([M, [0,0,0,1]])
        T_kinectColor2PanopticWorld = np.linalg.inv(T_panopticWorld2KinectColor)
    
        T_kinectColor2PanopticWorld = T_kinectColor2PanopticWorld.dot(scale_kinoptic2panoptic)
        T_kinectColor2PanopticWorld = torch.from_numpy(T_kinectColor2PanopticWorld)
        T_kinectColor2PanopticWorld = torch.as_tensor(T_kinectColor2PanopticWorld, dtype=torch.double, device=device)
        ################ <np implemented part> ################


        if show_depthmap_to_cloud_time_cost_flag:
            time_end=time.time();print('cost2\n',time_end-time_start)
            print('st3')
            import time;time_start=time.time()

        p3d_world_fromz = torch.mm(T_kinectColor2PanopticWorld, torch.cat([cloud_fromz.T, torch.ones((1, cloud_fromz.shape[0]), device=device).double()], axis=0))
        p3d_world_fromz = p3d_world_fromz[:3, :].T
        
        ### cloud_local原本z=0的位置不应该映射有xy
        # st()
        p3d_world_fromz[cloud_local_mask] = 0

        # st()
        ### 这两种reshape和reshape之后permute之后的结果居然不同，permute应该是对的把
        proj_cloud_fromz_1 = p3d_world_fromz.reshape(denoised_depth_i.shape[0], denoised_depth_i.shape[1], 3)
        if vis_flag:
            print(f'proj_cloud_fromz_1[:,:,0].mean() {proj_cloud_fromz_1[:,:,0].mean()}')
            print(f'proj_cloud_fromz_1[:,:,1].mean() {proj_cloud_fromz_1[:,:,1].mean()}')
            print(f'proj_cloud_fromz_1[:,:,2].mean() {proj_cloud_fromz_1[:,:,2].mean()}')

        # proj_cloud_fromz_2 = p3d_world_fromz.reshape(3, denoised_depth_i.shape[0], denoised_depth_i.shape[1])
        # if vis_flag:
        #     print(f'proj_cloud_fromz_2[0].mean() {proj_cloud_fromz_2[0].mean()}')
        #     print(f'proj_cloud_fromz_2[1].mean() {proj_cloud_fromz_2[1].mean()}')
        #     print(f'proj_cloud_fromz_2[2].mean() {proj_cloud_fromz_2[2].mean()}')

        ### 验证这个世界坐标一致性，注意cloud和depth都不能transform
        # proj_cloud = meta[0]['cloud'][0]
        # proj_cloud = proj_cloud.reshape(-1,3)
        # proj_cloud[cloud_local_mask] = 0
        # proj_cloud = proj_cloud.reshape(proj_cloud_fromz_1.shape)
        
        ### 世界坐标cloud的不一致性，应该也是transform导致的
        # 把cloud也transform之后即可

        ################################## 检验映射到世界坐标系下面的 #############################

        if vis_flag:
            print(f'proj_cloud_color_xyz[0].mean() {proj_cloud_color_xyz[0].mean()}')
            print(f'proj_cloud_color_xyz[1].mean() {proj_cloud_color_xyz[1].mean()}')
            print(f'proj_cloud_color_xyz[2].mean() {proj_cloud_color_xyz[2].mean()}')
        
        # 你可以测试一下不transform情况下的情况，直接用原分辨率来做
        ### transform
        M = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]])
        M = torch.from_numpy(M)
        # st()
        M = torch.as_tensor(M, dtype=torch.double, device=device)
        proj_cloud_fromz_1 = torch.mm(proj_cloud_fromz_1.reshape(-1,3), M).reshape(proj_cloud_fromz_1.shape) * 10
        # 
        # st()
        # 我懂了，我之前测试的就是transform到540之前的坐标是没有问题的
        # (proj_cloud_fromz_1 - proj_cloud).max()
        # (Pdb) (proj_cloud_fromz_1 - proj_cloud).max()
        # tensor(12.1170, device='cuda:0', dtype=torch.float64)
        # (Pdb) (proj_cloud_fromz_1 - proj_cloud).min()
        # tensor(-11.8288, device='cuda:0', dtype=torch.float64)
        # 这证明整个流程是正确的
        proj_cloud_fromz = proj_cloud_fromz_1.permute((2,0,1))
        proj_cloud_fromz_allj_list.append(proj_cloud_fromz.unsqueeze(0))

        if show_depthmap_to_cloud_time_cost_flag:
            time_end=time.time();print('cost3\n',time_end-time_start)


    proj_cloud_fromz_allj = torch.cat(proj_cloud_fromz_allj_list, axis=0)
    return proj_cloud_fromz_allj.unsqueeze(0) ### 保存

def unproject2(p2d, Kdist, Mdist, distCoeffs, device):
    # st()
    # pn2d = torch.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T
    
    pn2d, _ = torch.solve(torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device).double()), axis=0), Kdist)
    pn2d = pn2d.T

    # pn2d = torch.linalg.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T

    # print(f'pn2d.mean() {pn2d.mean()}')

    # k = torch.cat([distCoeffs[:5], np.zeros(12-5)], axis=0)
    k = torch.cat((distCoeffs[:len(distCoeffs)], torch.zeros(12-len(distCoeffs), device=device).double()), axis=0)
    x0 = pn2d[:,0]
    y0 = pn2d[:,1]
    x = x0
    y = y0

    # print(f'x {x}')
    # print(f'y {y}')

    for iter in range(5):
        r2 = x*x + y*y
        # st()
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    
    pn2d = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), axis=1)
    
    if vis_flag:
        print(f'pn2d.mean() {pn2d.mean()}')
        print(f'p2d.mean() {p2d.mean()}')
        print(f'(p2d[:,2:3]*0.001).mean() {(p2d[:,2:3]*0.001).mean()}')
        print(f'(pn2d * (p2d[:,2:3]*0.001)).mean() {(pn2d * (p2d[:,2:3]*0.001)).mean()}')
        print(f'(p2d[:,2:3]*0.001).mean() {(p2d[:,2:3]*0.001).mean()}')
        print(f'pn2d {pn2d}')
    
    p3d = torch.cat([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
    # p3d = np.concatenate([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
    
    if vis_flag:
        print(f'p3d.mean() {p3d.mean()}')

    # st()
    
    # print(p3d.mean(axis=0))
    # st()
    # p3d = np.matmul(np.linalg.inv(np.row_stack([Mdist, [0,0,0,1]])), torch.cat([p3d, np.ones((p3d.shape[0], 1))], axis=1).T).T
    # print(p3d.mean(axis=0))
    p3d = p3d[:,:3]
    # p3d = (inv([Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')'
    # p3d = (inv([cam.Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')';
    # p3d = [bsxfun(@times, pn2d, p2d(:,3)*0.001) p2d(:,3)*0.001];
    ### 估计没用
    # if pn2d.shape[1] == 1:
    #     pn2d = pn2d.T
    # st()
    # pn2d = [x y]
    # for iter=1:5
    #     r2 = x.*x + y.*y;
    #     icdist = (1 + ((k(1+7)*r2 + k(1+6)).*r2 + k(1+5)).*r2)./(1 + ((k(1+4)*r2 + k(1+1)).*r2 + k(1+0)).*r2);
    #     deltaX = 2*k(1+2)*x.*y + k(1+3)*(r2 + 2*x.*x)+ k(1+8)*r2+k(1+9)*r2.*r2;
    #     deltaY = k(1+2)*(r2 + 2*y.*y) + 2*k(1+3)*x.*y+ k(1+10)*r2+k(1+11)*r2.*r2;
    #     x = (x0 - deltaX).*icdist;
    #     y = (y0 - deltaY).*icdist;
    # end
    # k = [reshape(cam.distCoeffs(1:5),[],1); zeros(12-5,1)];
    # pn2d = ( cam.Kdist \ [p2d(:,1:2)'; ones(1, size(p2d,1))] )';
    return p3d



vis_check_flag = False

def _filter_pts_cuda(img, cloud, shape):
    img_pts, cloud_pts = img, cloud
    h, w = shape
    for _idx in [
            lambda pts: 0 < pts[:, 0],
            lambda pts: pts[:, 0] < w,
            lambda pts: 0 < pts[:, 1],
            lambda pts: pts[:, 1] < h
            ]:
        # st()
        # (Pdb) idx.shape
        # torch.Size([217088, 2])
        # (Pdb) img_pts.shape
        # torch.Size([217088, 2, 2])
        idx = _idx(img_pts)
        img_pts = img_pts[idx]
        cloud_pts = cloud_pts[idx]
    return img_pts, cloud_pts

def matlab_poseproject2d_cuda(pts, R_color, t_color, c_K, c_d, bApplyDistort=True):
    # st()
    ### <org>
    # x = R_color.dot(pts.T) + t_color
    # xp = x[:2,:] / x[2,:]
    # # xp.mean(axis=1)
    ### <lcc>
    x = torch.matmul(R_color, pts.T) + t_color
    xp = x[:2,:] / x[2,:]
    if vis_check_flag:
        print(f'xp1 {xp}')
        print(f'xp1.mean() {xp.mean()}')
        print(f'xp1.shape {xp.shape}')
    if bApplyDistort:
        ### xp[0:1,:]和xp[0,:]的区别就是前者多了一个维度
        X2 = xp[0:1,:] * xp[0:1,:]
        Y2 = xp[1:2,:] * xp[1:2,:]
        XY = xp[0:1,:] * xp[1:2,:]
        r2 = X2 + Y2
        r4 = r2 * r2
        r6 = r2 * r4
        Kp = c_d
        # st()
        # radial       = 1.0 + Kp[0].dot(r2) + Kp[1].dot(r4) + Kp[4].dot(r6)
        # tangential_x = 2.0*Kp[2].dot(XY) + Kp[3].dot(r2 + 2.0*X2)
        # tangential_y = 2.0*Kp[3].dot(XY) + Kp[2].dot(r2 + 2.0*Y2)
        # np.stack 默认的新的axis=0和matlab一致
        # xp = np.stack([radial, radial]) * xp[:2,:] + np.stack([tangential_x, tangential_y])
        # st()
        radial       = 1.0 + torch.matmul(Kp[0], r2) + torch.matmul(Kp[1], r4) + torch.matmul(Kp[4], r6)
        tangential_x = 2.0 * torch.matmul(Kp[2], XY) + torch.matmul(Kp[3], r2 + 2.0*X2)
        tangential_y = 2.0 * torch.matmul(Kp[3], XY) + torch.matmul(Kp[2], r2 + 2.0*Y2)
        # radial       = 1.0 + Kp[0].dot(r2) + Kp[1].dot(r4) + Kp[4].dot(r6)
        # tangential_x = 2.0*Kp[2].dot(XY) + Kp[3].dot(r2 + 2.0*X2)
        # tangential_y = 2.0*Kp[3].dot(XY) + Kp[2].dot(r2 + 2.0*Y2)
        # radial       = 1.0 + Kp[0]*r2 + Kp[1]*r4 + Kp[4]*r6
        # tangential_x = 2.0 + Kp[2]*XY + Kp[3]*(r2 + 2.0*X2)
        # tangential_y = 2.0 + Kp[3]*XY + Kp[2]*(r2 + 2.0*Y2)
        # (Pdb) radial.shape
        # torch.Size([1, 217088])
        # (Pdb) np.stack([radial, radial]).shape
        # *** TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        # (Pdb) torch.stack([radial, radial]).shape
        # torch.Size([2, 1, 217088])
        # (Pdb) xp[:2,:].shape
        # torch.Size([2, 217088])
        if vis_check_flag:
            print(f'X2 {X2}')
            print(f'Y2 {Y2}')
            print(f'XY {XY}')
            print(f'r2 {r2}')
            print(f'r4 {r4}')
            print(f'r6 {r6}')
            print(f'Kp {Kp}')
            print(f'radial {radial}')
            print(f'tangential_x {tangential_x}')
        
        xp = torch.stack([radial, radial]) * xp[:2,:] + torch.stack([tangential_x, tangential_y])
        # st()
    
    
    if vis_check_flag:
        print(f'c_K[:2, :2].shape {c_K[:2, :2].shape}')
        print(f'xp2.shape {xp.shape}')
        print(f'c_K[:2, 2:3].shape {c_K[:2, 2:3].shape}')
    # pt = (c_K[:2, :2].dot(xp) + c_K[:2, 2:3]).T
    pt = (torch.matmul(c_K[:2, :2], xp) + c_K[:2, 2:3]).T
    
    if vis_check_flag:
        print(f'xp2 {xp}')
        print(f'xp2.mean() {xp.mean()}')
        print(f'pt {pt}')
        print(f'pt.mean() {pt.mean()}')
    # st()
    return pt
   
def unproject2_aligned_depth(p2d, Kdist, Mdist, distCoeffs):
    # st()
    # pn2d = torch.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T
    
    pn2d, _ = torch.solve(torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0])).cuda().double()), axis=0), Kdist)
    pn2d = pn2d.T
    if vis_check_flag:
        print(f'pn2d.mean() {pn2d.mean()}')
    # pn2d = torch.linalg.solve(Kdist, torch.cat((p2d[:,:2].T, torch.ones((1, p2d.shape[0]), device=device)), axis=0)).T
    # print(f'pn2d.mean() {pn2d.mean()}')
    # st()
    # k = torch.cat([distCoeffs[:5], np.zeros(12-5)], axis=0)
    k = torch.cat((distCoeffs[:5], torch.zeros(12-5).cuda().double()), axis=0)
    # k = torch.cat((distCoeffs[:len(distCoeffs)], torch.zeros(12-len(distCoeffs)).cuda().double()), axis=0)
    x0 = pn2d[:,0]
    y0 = pn2d[:,1]
    x = x0
    y = y0
    # print(f'x {x}')
    # print(f'y {y}')
    for iter in range(5):
        r2 = x*x + y*y
        # st()
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    
    pn2d = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), axis=1)
    
    if vis_check_flag:
        print(f'pn2d.mean() {pn2d.mean()}')
        print(f'p2d.mean() {p2d.mean()}')
        print(f'(p2d[:,2:3]*0.001).mean() {(p2d[:,2:3]*0.001).mean()}')
        print(f'(pn2d * (p2d[:,2:3]*0.001)).mean() {(pn2d * (p2d[:,2:3]*0.001)).mean()}')
        print(f'pn2d {pn2d}')
    
    p3d = torch.cat([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
    # p3d = np.concatenate([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
    
    # p3d = np.matmul(np.linalg.inv(np.row_stack([Mdist, [0,0,0,1]])), np.concatenate([p3d, np.ones((p3d.shape[0], 1))], axis=1).T).T
    
    p3d = torch.matmul(torch.inverse(torch.cat([Mdist, torch.DoubleTensor([[0,0,0,1]]).cuda()], dim=0)), torch.cat((p3d, torch.ones((p3d.shape[0], 1)).cuda().double()), axis=1).T).T
    # if vis_check_flag:
    #     aaa = torch.inverse(torch.cat([Mdist, torch.DoubleTensor([[0,0,0,1]]).cuda()], dim=0))
    #     bbb = torch.cat((p3d, torch.ones((p3d.shape[0], 1)).cuda().double()), axis=1).T
    #     ccc = torch.matmul(aaa, bbb).T
    #     print(f'aaa {aaa}')
    #     print(f'bbb {bbb}')
    #     print(f'ccc {ccc}')
    # st()
    if vis_check_flag:
        print(f'p3d.mean() {p3d.mean()}')
    # st()
    
    # print(p3d.mean(axis=0))
    # st()
    # p3d = np.matmul(np.linalg.inv(np.row_stack([Mdist, [0,0,0,1]])), torch.cat([p3d, np.ones((p3d.shape[0], 1))], axis=1).T).T
    # print(p3d.mean(axis=0))
    p3d = p3d[:,:3]
    # p3d = (inv([Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')'
    # p3d = (inv([cam.Mdist;[0 0 0 1]])*[p3d ones(size(p3d,1),1)]')';
    # p3d = [bsxfun(@times, pn2d, p2d(:,3)*0.001) p2d(:,3)*0.001];
    ### 估计没用
    # if pn2d.shape[1] == 1:
    #     pn2d = pn2d.T
    # st()
    # pn2d = [x y]
    # for iter=1:5
    #     r2 = x.*x + y.*y;
    #     icdist = (1 + ((k(1+7)*r2 + k(1+6)).*r2 + k(1+5)).*r2)./(1 + ((k(1+4)*r2 + k(1+1)).*r2 + k(1+0)).*r2);
    #     deltaX = 2*k(1+2)*x.*y + k(1+3)*(r2 + 2*x.*x)+ k(1+8)*r2+k(1+9)*r2.*r2;
    #     deltaY = k(1+2)*(r2 + 2*y.*y) + 2*k(1+3)*x.*y+ k(1+10)*r2+k(1+11)*r2.*r2;
    #     x = (x0 - deltaX).*icdist;
    #     y = (y0 - deltaY).*icdist;
    # end
    # k = [reshape(cam.distCoeffs(1:5),[],1); zeros(12-5,1)];
    # pn2d = ( cam.Kdist \ [p2d(:,1:2)'; ones(1, size(p2d,1))] )';
    return p3d

def depth_to_aligned_depth(depth, shape, calib):
    # show_time_cost_flag = True
    show_time_cost_flag = False

    if show_time_cost_flag:
        print('st1-lcc')
        import time;time_start=time.time()
    ###### <part 1-torch> ######
    # depth_cuda = torch.as_tensor(depth.copy().astype(np.float64), dtype=torch.double).cuda()
    depth_cuda = torch.as_tensor(depth, dtype=torch.double).cuda()


    X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    p2dd = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    p2dd = torch.as_tensor(p2dd, dtype=torch.double).cuda()
    d_K, d_d = calib.depth_proj()
    R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
    Kdist = d_K
    Mdist = T_M_depth[:3,:]
    distCoeffs = d_d
    Kdist = torch.as_tensor(Kdist, dtype=torch.double).cuda()
    Mdist = torch.as_tensor(Mdist, dtype=torch.double).cuda()
    distCoeffs = torch.as_tensor(distCoeffs, dtype=torch.double).cuda()
    lcc_p2d = torch.cat((p2dd, depth_cuda.reshape(-1, 1)), axis=1)
    # st()
    # (Pdb) lcc_p2d
    # tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
    #         [1.0000e+00, 0.0000e+00, 0.0000e+00],
    #         [2.0000e+00, 0.0000e+00, 0.0000e+00],
    #         ...,
    #         [5.0900e+02, 4.2300e+02, 0.0000e+00],
    #         [5.1000e+02, 4.2300e+02, 2.2230e+03],
    #         [5.1100e+02, 4.2300e+02, 0.0000e+00]], device='cuda:0',
    #     dtype=torch.float64)
    # (Pdb) Kdist
    # tensor([[364.8531,   0.0000, 249.2050],
    #         [  0.0000, 364.8531, 202.5364],
    #         [  0.0000,   0.0000,   1.0000]], device='cuda:0', dtype=torch.float64)
    # (Pdb) Mdist
    # tensor([[-1.0000e+00, -1.2246e-16,  0.0000e+00,  0.0000e+00],
    #         [ 1.2246e-16, -1.0000e+00,  0.0000e+00,  0.0000e+00],
    #         [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00]], device='cuda:0',
    #     dtype=torch.float64)
    # (Pdb) distCoeffs
    # tensor([ 9.2577e-02, -2.6353e-01,  1.1089e-04,  2.2360e-04,  8.4514e-02,
    #         1.0000e+03,  0.0000e+00], device='cuda:0', dtype=torch.float64)
    # (Pdb) cloud_local
    # tensor([[-0.0000e+00, -0.0000e+00,  0.0000e+00],
    #         [-0.0000e+00, -0.0000e+00,  0.0000e+00],
    #         [-0.0000e+00, -0.0000e+00,  0.0000e+00],
    #         ...,
    #         [-0.0000e+00, -0.0000e+00,  0.0000e+00],
    #         [-2.0286e-05, -1.4476e-05,  2.2230e+00],
    #         [-0.0000e+00, -0.0000e+00,  0.0000e+00]], device='cuda:0',
    #     dtype=torch.float64)
    if vis_check_flag:
        print("\n")
        print("checking unproject2")
    
    p3d = unproject2_aligned_depth(lcc_p2d, Kdist, Mdist, distCoeffs)

    cloud_local = p3d
    if vis_check_flag:
        print(f'lcc_p2d.mean() {lcc_p2d.mean()}')
        print(f'cloud_local.mean() {cloud_local.mean()}')
    # st()
    ###### </part 1-torch> ######
    if show_time_cost_flag:
        time_end=time.time();print('cost1-lcc\n',time_end-time_start)
        print('st2-lcc')
        import time;time_start=time.time()
    ###### <part 2-torch> ######
    R_color, t_color, T_M_color = calib.lcc_M_color_all()
    c_K, c_d = calib.color_proj()
    R_color = torch.as_tensor(R_color, dtype=torch.double).cuda()
    t_color = torch.as_tensor(t_color, dtype=torch.double).cuda()
    c_K = torch.as_tensor(c_K, dtype=torch.double).cuda()
    c_d = torch.as_tensor(c_d, dtype=torch.double).cuda()
    img_pts = matlab_poseproject2d_cuda(cloud_local, R_color, t_color, c_K, c_d)
    img_pts = torch.as_tensor(img_pts + 0.5, dtype=torch.int32).cuda()
    if vis_check_flag:
        print(f'img_pts {img_pts}')
        # print(f'img_pts.mean() {img_pts.mean()}') # torch.int无法计算mean
    # st()
    ###### </part 2-torch> ######
    
    if show_time_cost_flag:
        time_end=time.time();print('cost2-lcc\n',time_end-time_start)
        print('st3-lcc')
        import time;time_start=time.time()
    ###### <part 3-common> ######
    # st()
    panoptic_calibData_R, panoptic_calibData_t = calib.lcc_panoptic_calibData()
    M = np.concatenate([panoptic_calibData_R, panoptic_calibData_t], axis=1)
    T_panopticWorld2KinectColor = np.row_stack([M, [0,0,0,1]])
    T_kinectColor2PanopticWorld = np.linalg.inv(T_panopticWorld2KinectColor)
    scale_kinoptic2panoptic = np.eye(4)
    scaleFactor = 100
    scale_kinoptic2panoptic[:3,:3] = scaleFactor*scale_kinoptic2panoptic[:3,:3]
    _, _, T_M_color = calib.lcc_M_color()
    T_kinectColor2KinectLocal = T_M_color
    T_kinectLocal2KinectColor = np.linalg.inv(T_kinectColor2KinectLocal) 
      
    T_kinectLocal2PanopticWorld =  T_kinectColor2PanopticWorld.dot(scale_kinoptic2panoptic).dot(T_kinectLocal2KinectColor)
    ###### </part 3-common> ######
    if show_time_cost_flag:
        time_end=time.time();print('cost3-lcc\n',time_end-time_start)
        print('st4-lcc')
        import time;time_start=time.time()
    ###### <part 4-torch> ######
    T_kinectLocal2KinectColor = torch.from_numpy(T_kinectLocal2KinectColor)
    T_kinectLocal2KinectColor = torch.as_tensor(T_kinectLocal2KinectColor, dtype=torch.double).cuda()
    point3d_kinectcolor = torch.matmul(T_kinectLocal2KinectColor, torch.cat([cloud_local.T, torch.ones((1, cloud_local.shape[0])).cuda().double()], axis=0))
    
    point3d_kinectcolor = point3d_kinectcolor[:3, :].T
    cloud_color = point3d_kinectcolor
    # st()
    img_pts_filter_color, cloud_filter_color = _filter_pts_cuda(img_pts, cloud_color, shape[:2])
    
    # st()
    img_pts_filter_color = torch.as_tensor(img_pts_filter_color, dtype=torch.long).cuda()
    proj_cloud_color = torch.as_tensor(torch.zeros(shape), dtype=torch.double).cuda()
    proj_cloud_color[img_pts_filter_color[:, 1], img_pts_filter_color[:, 0], :] = cloud_filter_color
    # proj_cloud_color = np.flip(proj_cloud_color, axis=0)
    aligned_depth = proj_cloud_color[:, :, 2]
    
    
    ###### <part 4-torch> ######
    if show_time_cost_flag:
        time_end=time.time();print('cost4-lcc\n',time_end-time_start)

    return aligned_depth


def smooth_depth_image(depth_image, max_hole_size=10, hole_value=0):
    """Smoothes depth image by filling the holes using inpainting method
        Parameters:
        depth_image(Image): Original depth image
        max_hole_size(int): Maximum size of hole to fill
            
        Returns:
        Image: Smoothed depth image
        
        Remarks:
        Bigger maximum hole size will try to fill bigger holes but requires longer time
        """
    mask = np.zeros(depth_image.shape,dtype=np.uint8)
    mask[depth_image==hole_value] = 1

    # Do not include in the mask the holes bigger than the maximum hole size
    kernel = np.ones((max_hole_size,max_hole_size),np.uint8)
    erosion = cv.erode(mask,kernel,iterations = 1)
    mask = mask - erosion
    smoothed_depth_image = cv.inpaint(depth_image.astype(np.uint16),mask,max_hole_size,cv.INPAINT_NS)
    # smoothed_depth_image = cv.inpaint(depth_image.astype(np.uint16),mask,max_hole_size,cv.INPAINT_TELEA)
    return smoothed_depth_image

class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_joints_wholebody = cfg.NETWORK.NUM_JOINTS_WHOLEBODY

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)

        self.pose_net = PoseRegressionNet(cfg)

        self.use_jrn = cfg.NETWORK.USE_JRN
        self.jrn_type = cfg.NETWORK.JRN_TYPE

        self.use_nojrn_wholebody = cfg.NETWORK.USE_NOJRN_WHOLEBODY
        if self.use_jrn:
            if cfg.NETWORK.JRN_TYPE == 'all':
                self.joint_refine_net = JointRefineNet(cfg)
            elif cfg.NETWORK.JRN_TYPE == 'face':
                self.joint_refine_net = JointRefineNetFace(cfg)
            elif cfg.NETWORK.JRN_TYPE == 'facehand':
                self.joint_refine_net = JointRefineNetFaceHand(cfg)
            elif cfg.NETWORK.JRN_TYPE == 'facehandc':
                self.joint_refine_net = JointRefineNetFaceHandC(cfg)
            elif cfg.NETWORK.JRN_TYPE == 'facehandsepv':
                self.joint_refine_net = JointRefineNetFaceHandSepV(cfg)
        self.USE_GT = cfg.NETWORK.USE_GT
        self.USE_GT_HM = cfg.NETWORK.USE_GT_HM
        
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.dep_downsample = cfg.NETWORK.DEP_DOWNSAMPLE

        self.dep_downsample_size = cfg.NETWORK.DEP_DOWNSAMPLE_SIZE

        self.use_unet = cfg.NETWORK.USE_UNET
        self.unet_res_out = True

        self.unet_type = cfg.NETWORK.UNET_TYPE

        self.unet_dep15 = cfg.NETWORK.UNET_DEP15

        self.pseudep = cfg.NETWORK.PSEUDEP

        self.namedataset = cfg.DATASET.NAMEDATASET

        self.f_weight = cfg.NETWORK.F_WEIGHT

        self.focus_lrecon = cfg.LOSS.USE_FOCUS_DEPTH_RECON_LOSS

        self.depth_maxpool = cfg.NETWORK.DEPTH_MAXPOOL
        print(f'depth_maxpool:{self.depth_maxpool}')

        self.depth_inpaint = cfg.NETWORK.DEPTH_INPAINT
        print(f'depth_inpaint:{self.depth_inpaint}')
        
        self.depth_inpaint_method = cfg.NETWORK.DEPTH_INPAINT_METHOD
        print(f'depth_inpaint_method:{self.depth_inpaint_method}')

        self.interval = cfg.DATASET.INTERVAL if cfg.DATASET.INTERVAL > 0 else 6
        if cfg.USE_SMALL_DATASET:
            self.interval = 3*10

        if self.namedataset == 'large' or self.namedataset == 'large_interp25j':
            self.code_to_seq = {
                    0:'160906_band3',
                    1:'160906_band2',
                    2:'160422_ultimatum1',
                    3:'160226_haggling1',
                    4:'160224_haggling1',
                    5:'170307_dance5',
                    6:'160906_ian3',
                    7:'160906_ian2',
                    8:'160906_ian1',
                    9:'170915_office1', # 虽然单人，但是遮挡多
                    10:'161202_haggling1',
                    11:'160906_pizza1', 
                    12:'160422_haggling1', 
                    13:'160906_ian5', 
                    14:'160906_band1',
                    15:'170407_office2',
            }

            # self.code_to_seq = {
            #         0:'160422_ultimatum1',
            #         1:'160224_haggling1',
            #         2:'160226_haggling1',
            #         3:'161202_haggling1',
            #         4:'160906_ian1',
            #         5:'160906_ian2',
            #         6:'160906_ian3',
            #         7:'160906_band1',
            #         8:'160906_band2',
            #         9:'160906_band3',
            #         10:'160906_pizza1',
            #         11:'160422_haggling1',
            #         12:'160906_ian5',
            # }
        
        elif cfg['OUT_3D_POSE_VID_NAMEDSET'] and self.namedataset == 'OUT_3D_POSE_VID':
            self.code_to_seq = {
                    0:'160224_haggling1'
            } 
        elif self.namedataset == 'all':
            self.code_to_seq = {
                    0:'160422_ultimatum1',
                    1:'160224_haggling1',
                    2:'160226_haggling1',
                    3:'161202_haggling1',
                    4:'160906_ian1',
                    5:'160906_ian2',
                    6:'160906_ian3',
                    7:'160906_band1',
                    8:'160906_band2',
                    9:'160906_band3',
                    10:'160906_pizza1',
                    11:'160422_haggling1',
                    12:'160906_ian5',
            }
        elif self.namedataset == 'large_wholebody':
            # self.code_to_seq = {
            #         0:'170915_office1',
            #         1:'170407_office2'
            # }
            # self.code_to_seq = {
            #         0:'171204_pose1',
            #         1:'171204_pose2',
            #         2:'171026_pose1',
            #         3:'171026_pose2',
            #         4:'170915_office1',
            #         5:'171204_pose3',
            #         6:'171026_pose3',
            #         7:'170407_office2',
            # }
            self.code_to_seq = {
                    0:'171204_pose1',
                    1:'171204_pose2',
                    2:'171204_pose3',
            }
        elif self.namedataset == 'large_wholebody_mp':
            self.code_to_seq = {
                    0:'160226_haggling1',
                    1:'160224_haggling1',
                    2:'160422_haggling1',
            }
            # self.code_to_seq = {
            #         0:'160906_ian1',
            #         1:'160906_ian1',
            # }


        if self.use_unet:
            from models.unet import UNet, UNet_LCC, UNet_LCC_MSH, UNet_LCC_MSH_1920x1024, UNet_LCC_1920x1024
            if self.unet_dep15: # 输出15个depth，每一个对应一个joint，显存不够
                n_classes = 15
            else:
                n_classes = 1

            # n_classes = 1

            ### n_channels is in_channels rgb3 depth1
            ### n_classes is out_channels 
            # self.unet = UNet(in_channels=4, n_classes=1, bilinear=True)
            if self.unet_type == 'dmsh':
                # self.unet = UNet_LCC_MSH(unet_type=self.unet_type, n_classes=n_classes, bilinear=True)
                self.unet = UNet_LCC_MSH_1920x1024(unet_type=self.unet_type, n_classes=n_classes, bilinear=True)
            else:
                # self.unet = UNet_LCC(unet_type=self.unet_type, n_classes=n_classes, bilinear=True)
                self.unet = UNet_LCC_1920x1024(unet_type=self.unet_type, n_classes=n_classes, bilinear=True)
                

            ### lcc debugging:half
            # self.unet.half()


        ### 需要保证在数据集初始化完成之后读取calib_list.pkl
        import pickle
        with open('train_calib_list.pkl','rb') as train_calib_list_f:
            self.train_calib_list = pickle.load(train_calib_list_f)

        with open('val_calib_list.pkl','rb') as val_calib_list_f:
            self.val_calib_list = pickle.load(val_calib_list_f)

        

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None, unet_turn=True, jrn_turn=True):
        # ### lcc debugging:j30
        # for view_index in range(len(targets_2d)):
        #     targets_2d[view_index] = targets_2d[view_index].repeat([1,2,1,1])
        #     weights_2d[view_index] = weights_2d[view_index].repeat([1,2,1])
        #     meta[view_index]['joints_3d'] = meta[view_index]['joints_3d'].repeat([1,1,2,1])
        #     meta[view_index]['joints_3d_vis'] = meta[view_index]['joints_3d_vis'].repeat([1,1,2,1])
        #     meta[view_index]['joints'] = meta[view_index]['joints'].repeat([1,1,2,1])
        #     meta[view_index]['joints_vis'] = meta[view_index]['joints_vis'].repeat([1,1,2,1])
    

        if vis_time_cost_flag:
            # st()
            print('st1')
            import time;time_start=time.time()

        # st()
        calib_list = self.train_calib_list if self.training else self.val_calib_list
        # calib_list = self.val_calib_list
        
        all_cloud = None

        # print(f"self.use_unet :{self.use_unet}")

        if views is not None:
            all_heatmaps = []
            all_heatmaps_whole = []
            
            # ######## <org get heatmap from backbone> ########
            # for view in views:
            #     # st()
            #     ### 注意这里只能接受batch_size=1
            #     assert view.shape[0] == 1
            #     heatmaps = self.backbone(view[0].cpu().numpy())
            #     # (Pdb) heatmaps.shape
            #     # torch.Size([1, 15, 128, 240])
            #     # st()
            #     # 这里能出来多个峰的heatmap吗

            #     device = meta[0]['aligned_depth'][0].device
            #     heatmaps = heatmaps.to(device=device)

            #     ### incorrect
            #     # all_heatmaps.append(heatmaps[:, :13, ...]) ### 注意这里是133的顺序，千万不能直接:13

            #     ### correct
            #     body_joint_index_list_13 = [0,5,6,7,8,9,10,11,12,13,14,15,16]
            #     all_heatmaps.append(heatmaps[:, body_joint_index_list_13, ...])
            #     # st()

            #     all_heatmaps_whole.append(heatmaps)
            # ######## </org get heatmap from backbone> ########

            ######## <lcc get heatmap from pre-computed db> ########
            check_flag = False
            for view in views:
                # heatmaps = self.backbone(view)
                if self.namedataset == 'large_wholebody':
                    save_dir = './heatmap_db_wholebody'
                elif self.namedataset == 'large_wholebody_mp':
                    save_dir = './heatmap_db_wholebody_mp'
                else:
                    save_dir = './heatmap_db_tmp'

                os.makedirs(save_dir, exist_ok=True)

                batch_size = meta[0]['joints_3d'].shape[0]
                heatmaps_list = []

                cal_heatmaps_flag = False
                for batch_index in range(batch_size):

                    # seq = meta[0]['seq'][batch_index]
                    seq = self.code_to_seq[int(meta[0]['seq_code'][batch_index])]
                    
                    camera_index = meta[0]['camera_index'][batch_index]
                    color_index = meta[0]['color_index'][batch_index]
                    heatmap_key = f"{seq}_{camera_index}_{color_index}"
                    
                    save_path = osp.join(save_dir, f'{heatmap_key}.npy')

                    save_path_no_out = save_path.replace('.npy', '.nonpy')
                    if osp.exists(save_path_no_out):
                        raise Exception('save_path_no_out')

                        # t0 = torch.FloatTensor([0]).cuda() ### support multi-gpu
                        # return t0, t0, t0, t0, t0, t0, t0, t0 ### return all 0 -> continue
                        # # return 0, 0, 0, 0, 0, 0, 0, 0

                    save_path_loaded_flag = True
                    # if False:
                    if osp.exists(save_path):
                        try:
                            f = gzip.GzipFile(save_path, "r")

                            # heatmap_np_load = np.load(f) ### 有时会报alllow_pickle=False
                            heatmap_np_load = np.load(f, allow_pickle=True) ### 有时会报错zlib.error: Error -3 while decompressing data: invalid distance too far back

                            # heatmap_np_load = ((heatmap_np_load / 1e4) - 1.0).astype(np.float32) ### 压缩保存1
                            # heatmap_np_load = heatmap_np_load.astype(np.float32) ### 压缩保存2
                            pass ### 不压缩保存3

                            f.close()
                            heatmap = torch.from_numpy(heatmap_np_load).cuda()
                        except:
                            save_path_loaded_flag = False
                    else:
                        save_path_loaded_flag = False

                    if vis_time_cost_flag:
                        # save_path_loaded_flag True
                        # cost1
                        # 0.061727285385131836
                        # save_path_loaded_flag False
                        # cost1
                        # 1.1405243873596191
                        print(f'save_path_loaded_flag {save_path_loaded_flag}')
                    
                    if not save_path_loaded_flag:
                        if not cal_heatmaps_flag:
                            # heatmaps = self.backbone(view)

                            assert view.shape[0] == 1

                            try:
                                heatmaps = self.backbone(view[0].cpu().numpy())
                            except:
                                ### backbone no output
                                save_path = save_path.replace('.npy', '.nonpy')
                                print(f'backbone no output:{save_path}')
                                heatmap_np_save = np.array([])
                                f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                                np.save(f, heatmap_np_save) 
                                f.close()

                                raise Exception('save_path_no_out')

                                # t0 = torch.FloatTensor([0]).cuda() ### support multi-gpu
                                # return t0, t0, t0, t0, t0, t0, t0, t0 ### return all 0 -> continue
                                # # return 0, 0, 0, 0, 0, 0, 0, 0
                                

                            device = meta[0]['aligned_depth'][0].device
                            heatmaps = heatmaps.to(device=device)

                            cal_heatmaps_flag = True

                        heatmap = heatmaps[batch_index]
                        heatmap_np_save = heatmap.detach().cpu().numpy()

                        if check_flag:
                            print(heatmap_np_save.min())
                            print(heatmap_np_save.max())
                            print(heatmap_np_save.mean())

                        # st()
                        ### 保存的规则是+1.0之后*1e4
                        # heatmap_np_save = ((heatmap_np_save.copy() + 1.0) * 1e4).astype(np.uint16) ### 压缩保存1
                        # heatmap_np_save = heatmap_np_save.copy().astype(np.float16) ### 压缩保存2
                        pass ### 不压缩保存3

                        f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                        np.save(f, heatmap_np_save) 
                        f.close()

                        ### check
                        if check_flag:
                            f = gzip.GzipFile(save_path, "r")
                            heatmap_np_load = np.load(f)
                            # heatmap_np_load = ((heatmap_np_load / 1e4) - 1.0).astype(np.float32)
                            heatmap_np_load = heatmap_np_load.astype(np.float32)
                            f.close()

                            print(heatmap_np_load.min())
                            print(heatmap_np_load.max())
                            print(heatmap_np_load.mean())
                    
                    heatmaps_list.append(heatmap.unsqueeze(0))
                
                # st()
                heatmaps = torch.cat(heatmaps_list, dim=0)
                device = meta[0]['aligned_depth'][0].device
                heatmaps = torch.as_tensor(heatmaps, dtype=torch.float, device=device)
                # st()

                body_joint_index_list_13 = [0,5,6,7,8,9,10,11,12,13,14,15,16]
                all_heatmaps.append(heatmaps[:, body_joint_index_list_13, ...])
                all_heatmaps_whole.append(heatmaps)

            # return 0, 0, 0, 0, 0, 0
            ######## </lcc get heatmap from pre-computed db> ########

        else:
            all_heatmaps = input_heatmaps


        if self.use_nojrn_wholebody:
            ### 更改all_heatmaps_whole的顺序
            ### lwrist rwrist nose + hand face 经过ProjectLayer得到cube
            ### 注意!!!索引的是133个joint的hm，不是你造的123个joint
            body_joint_index_list = [5,6,7,8,11,12,13,14,15,16]


            face_joint_index_list = [0] + [i for i in range(23,23+68)]
            lhand_joint_index_list = [9] + [i for i in range(23+68,23+68+21)]
            rhand_joint_index_list = [10] + [i for i in range(23+68+21,23+68+21+21)]
            
            # all_heatmaps_body 10

            new_all_heatmaps_whole = []
            for whole_heatmaps in all_heatmaps_whole:
                new_all_heatmaps_whole.append(torch.cat((whole_heatmaps[:, body_joint_index_list, :, :], \
                    whole_heatmaps[:, face_joint_index_list, :, :], \
                        whole_heatmaps[:, lhand_joint_index_list, :, :], \
                            whole_heatmaps[:, rhand_joint_index_list, :, :]), dim=1))

            all_heatmaps_whole = new_all_heatmaps_whole
            # st()



        if vis_time_cost_flag:
            time_end=time.time();print('cost1\n',time_end-time_start)
            # st()
            print('st2')
            import time;time_start=time.time()


        ## 刚刚缩进有问题好像20220228，已确认
        if self.f_weight and self.dep_downsample:
            all_aligned_depth_fill = []
            # len(all_heatmaps) is num_views
            # depthmap max pooling + down sample(in unet)
            for view_index in range(len(meta)):

                # ########################### <gpu precompute> ###########################
                # aligned_depth_list = []
                # batch_size = meta[0]['joints_3d'].shape[0]
                # for batch_index in range(batch_size):
                    
                #     aligned_depth_one_read_flag = meta[view_index]['aligned_depth_read_flag'][batch_index]
                #     # st()

                #     # seq = meta[0]['seq'][batch_index]
                #     seq = self.code_to_seq[int(meta[0]['seq_code'][batch_index])]

                #     camera_index = meta[0]['camera_index'][batch_index]
                #     depth_index = meta[0]['depth_index'][batch_index]
                #     aligned_depth_key = f"{seq}_{camera_index}_{depth_index}"
                    

                #     if self.namedataset == 'large':
                #         # save_dir = './aligned_depth_db_node1' ### cam5
                #         save_dir = './aligned_depth_db_cam2'
                #     elif self.namedataset == 'OUT_3D_POSE_VID':
                #         save_dir = './aligned_depth_db_OUT_3D_POSE_VID'
                #     else:
                #         save_dir = './aligned_depth_db'
                    
                #     os.makedirs(save_dir, exist_ok=True)
                #     save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')

                #     if not aligned_depth_one_read_flag:
                #         ### load失败，预计算
                #         # assert meta[view_index]['depth_data_numpy'][batch_index].shape != torch.Size([])
                #         # st()
                #         # aligned_depth_one = depth_to_aligned_depth(meta[view_index]['depth_data_numpy'][batch_index], (1080, 1920, 3), calib_list[meta[view_index]['seq'][batch_index]][meta[view_index]['camera_index'][batch_index]])
                #         aligned_depth_one = depth_to_aligned_depth(meta[view_index]['depth_data_numpy'][batch_index], (1080, 1920, 3), calib_list[self.code_to_seq[int(meta[view_index]['seq_code'][batch_index])]][meta[view_index]['camera_index'][batch_index]])

                #         aligned_depth_save = aligned_depth_one.cpu().numpy()

                #         # print('before save')
                #         f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                #         np.save(f, aligned_depth_save) 
                #         f.close()

                #         aligned_depth_one = aligned_depth_one.unsqueeze(0)
                #     else:
                #         aligned_depth_one = meta[view_index]['aligned_depth'][batch_index]
                    
                #     aligned_depth_list.append(aligned_depth_one.unsqueeze(0))
                
                # # st()
                # aligned_depth = torch.cat(aligned_depth_list, dim=0)
                # # st()
                # ########################### </gpu precompute> ###########################

                ########################### <cpu precompute> ###########################
                aligned_depth = meta[view_index]['aligned_depth']
                ########################### </cpu precompute> ###########################

                # 验证torch版本的unproject代码->一致
                device = aligned_depth[0].device

                # print(f'aligned_depth.max() {aligned_depth.max()}')
                # depthmap_to_cloud([torch.as_tensor(aligned_depth, dtype=torch.double, device=device)], meta, device)
                

                # 先插值，再pad，再resize
                if vis_depth_flag:show_proj_cloud(aligned_depth[0].permute(1,2,0).detach().cpu().numpy(), '1')
                

                if self.depth_maxpool:
                    if not self.depth_inpaint:
                        ## org fill 
                        aligned_depth_fill = F.max_pool2d(aligned_depth, 3, stride=1, padding=1)
                    else:
                        # ### try1
                        # ### new fill:1.maxpool with mask 2.minpool 3.inpaint with predefined bbox
                        # # 1.maxpool with mask
                        # # aligned_depth_fill_tmp = F.max_pool2d(aligned_depth, 3, stride=1, padding=1)
                        # aligned_depth_fill_tmp = -F.max_pool2d(-aligned_depth, 3, stride=1, padding=1)

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-1')

                        # proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask']
                        # aligned_depth_fill_tmp = torch.where(proj_cloud_color_mask<1, aligned_depth_fill_tmp, aligned_depth)

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-2')
                        
                        # # 2.minpool
                        # aligned_depth_fill_tmp = -F.max_pool2d(-aligned_depth_fill_tmp, 3, stride=1, padding=1)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-3')
                        
                        # # rgb上面有投影的区域为0-1080:1-1079 0-1920:150-1700
                        # # # 3.inpaint with predefined bbox
                        # # # st()
                        # # x_min = 180
                        # # x_max = 1000
                        # # y_min = 150
                        # # y_max = 1700
                        # # aligned_depth_fill_tmp_clip = smooth_depth_image((aligned_depth_fill_tmp.squeeze().cpu().numpy()[x_min:x_max, y_min:y_max] * 1000).astype(np.uint16), max_hole_size=20)
                        # # aligned_depth_fill_tmp[0,0,x_min:x_max, y_min:y_max] = torch.from_numpy(aligned_depth_fill_tmp_clip / 1000)
                        # # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-4')
                        # # st()
                        # # st()

                        # ### try 1-1 max pool+inpaint wo mask 效果不好，展示
                        # aligned_depth_fill_list = []
                        # batch_size = meta[0]['joints_3d'].shape[0]
                        # for batch_index in range(batch_size):
                        #     ### try4 maxpool+inpaint
                        #     aligned_depth_one = aligned_depth[batch_index:batch_index+1]
                        #     # st()
                        #     aligned_depth_fill_one = F.max_pool2d(aligned_depth_one, 3, stride=1, padding=1)
                        #     if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '2')

                        #     max_hole_size = 10
                        #     # # max_hole_size = 20

                        #     # mask = torch.zeros_like(aligned_depth_fill_one)
                        #     # mask[aligned_depth_fill_one == 0] = 1
                            
                        #     # # aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                        #     # aligned_depth_fill_down = -F.max_pool2d(-aligned_depth_fill_one, 3, stride=8, padding=1)


                        #     selected_area = aligned_depth_fill_one.squeeze().cpu().numpy()
                            
                        #     aligned_depth_fill_one = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=0)

                        #     aligned_depth_fill_one = torch.from_numpy(aligned_depth_fill_one / 1000).cuda().unsqueeze(0).unsqueeze(0)

                        #     if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '4')
                            

                        #     aligned_depth_fill_list.append(aligned_depth_fill_one)
                        # aligned_depth_fill = torch.cat(aligned_depth_fill_list, dim=0)

                        # # ### try2 用前景的部分影响
                        # # # ap@25: 0.0000   ap@50: 0.0336   ap@75: 0.3670   ap@100: 0.6709  ap@125: 0.7882  ap@150: 0.8421  recall@500mm: 0.9184    mpjpe@500mm: 83.412
                        # ### new fill:1.maxpool with mask 2.minpool 3.inpaint with predefined bbox
                        # proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask']
                        # aligned_depth_fill_tmp = torch.where(proj_cloud_color_mask<1, torch.as_tensor(torch.ones(aligned_depth.shape) * 8.0, dtype=torch.double).cuda(), aligned_depth)

                        # # 1.maxpool with mask
                        # # aligned_depth_fill_tmp = F.max_pool2d(aligned_depth, 3, stride=1, padding=1)
                        # aligned_depth_fill_tmp = -F.max_pool2d(-aligned_depth_fill_tmp, 3, stride=1, padding=1)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-1')

                        # aligned_depth_fill_tmp = torch.where(proj_cloud_color_mask<1, aligned_depth_fill_tmp, aligned_depth)

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp[0].permute(1,2,0).detach().cpu().numpy(), '2-2')
                        


                        # # 2.minpool
                        # ###1
                        # # aligned_depth_fill_tmp_1 = -F.max_pool2d(-aligned_depth_fill_tmp, 3, stride=1, padding=1)
                        # # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp_1[0].permute(1,2,0).detach().cpu().numpy(), '2-3-1')
                        
                        # # # aligned_depth_fill_tmp_2 = -F.max_pool2d(-aligned_depth_fill_tmp, 5, stride=1, padding=2)
                        # # # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp_2[0].permute(1,2,0).detach().cpu().numpy(), '2-3-2')

                        # # aligned_depth_fill_tmp_1 = F.interpolate(aligned_depth_fill_tmp_1, size=self.dep_downsample_size, mode='bilinear')

                        # ### 2
                        # stride = 4
                        # aligned_depth_fill_tmp_1 = -F.max_pool2d(-aligned_depth_fill_tmp, 5, stride=stride, padding=2)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp_1[0].permute(1,2,0).detach().cpu().numpy(), '2-3-1')

                        # x_min = 0 // stride
                        # x_max = 1000 // stride
                        # y_min = 150 // stride
                        # y_max = 1700 // stride
                        # max_hole_size = 10
                        # # max_hole_size = 20

                        # selected_area = aligned_depth_fill_tmp_1.squeeze().cpu().numpy()[x_min:x_max, y_min:y_max]

                        # # import time;time_start=time.time()
                        # aligned_depth_fill_tmp_clip = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size)
                        # # time_end=time.time();print('cost smooth_depth_image\n',time_end-time_start)


                        # aligned_depth_fill_tmp_1[0,0,x_min:x_max, y_min:y_max] = torch.from_numpy(aligned_depth_fill_tmp_clip / 1000)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_tmp_1[0].permute(1,2,0).detach().cpu().numpy(), '2-4')

                        # aligned_depth_fill = F.interpolate(aligned_depth_fill_tmp_1, size=aligned_depth_fill_tmp.shape[-2:], mode='bilinear')


                        # ### try3 只用min pool
                        # proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask']
                        # aligned_depth_fill_tmp = torch.where(proj_cloud_color_mask<1, torch.as_tensor(torch.ones(aligned_depth.shape) * 8.0, dtype=torch.double).cuda(), aligned_depth)
                        # # 1.maxpool with mask
                        # # aligned_depth_fill_tmp = F.max_pool2d(aligned_depth, 3, stride=1, padding=1)
                        # aligned_depth_fill = -F.max_pool2d(-aligned_depth_fill_tmp, 3, stride=1, padding=1)


                        ### try-final 4,5
                        aligned_depth_fill_list = []
                        batch_size = meta[0]['joints_3d'].shape[0]
                        for batch_index in range(batch_size):
                            # st()
                            image_set = 'train' if self.training else 'val'
                            # image_set = 'val'

                            seq = self.code_to_seq[int(meta[0]['seq_code'][batch_index])]

                            camera_index = meta[0]['camera_index'][batch_index]
                            depth_index = meta[0]['depth_index'][batch_index]
                            aligned_depth_inp_key = f"{seq}_{camera_index}_{depth_index}"
                            
                            num_views = len(calib_list[list(calib_list.keys())[0]])

                            if self.namedataset == 'large' or \
                                self.namedataset == 'large_interp25j' or \
                                self.namedataset == 'large_wholebody' or \
                                self.namedataset == 'large_wholebody_mp':
                                save_dir = f'./aligned_depth_db_{image_set}_{num_views}cam_itv{self.interval}'

                                ### lcc debugging test
                                # save_dir = f'./aligned_depth_db_{image_set}_{num_views}cam_itv{3*1000}'
                            elif self.namedataset == 'OUT_3D_POSE_VID':
                                save_dir = './aligned_depth_db_OUT_3D_POSE_VID'
                            else:
                                save_dir = './aligned_depth_db'
                            save_dir += f'_inp{self.depth_inpaint_method}'

                            os.makedirs(save_dir, exist_ok=True)
                            save_path = osp.join(save_dir, f'{aligned_depth_inp_key}.npy')


                            if osp.exists(save_path):
                                # st()
                                # print('before load')

                                f = gzip.GzipFile(save_path, "r")
                                aligned_depth_save = np.load(f) ### 有时会报alllow_pickle=False
                                # aligned_depth = np.load(f, allow_pickle=True)
                                f.close()

                                # st()
                                # aligned_depth = (aligned_depth / 1000).astype(np.float64) ### 压缩保存1
                                pass ### 不压缩保存2，啥也不用干
                                
                                aligned_depth_fill_one = torch.from_numpy(aligned_depth_save).cuda() 
                            else:
                                if self.depth_inpaint_method == 4:
                                    ### try4 maxpool+inpaint
                                    aligned_depth_one = aligned_depth[batch_index:batch_index+1]
                                    # st()
                                    aligned_depth_fill_one = F.max_pool2d(aligned_depth_one, 3, stride=1, padding=1)
                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '2')

                                    max_hole_size = 10
                                    # max_hole_size = 20

                                    mask = torch.zeros_like(aligned_depth_fill_one)
                                    mask[aligned_depth_fill_one == 0] = 1
                                    
                                    # aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                                    aligned_depth_fill_down = -F.max_pool2d(-aligned_depth_fill_one, 3, stride=8, padding=1)


                                    selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                                    
                                    aligned_depth_fill_down = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=0)

                                    aligned_depth_fill_down = torch.from_numpy(aligned_depth_fill_down / 1000).cuda().unsqueeze(0).unsqueeze(0)

                                    aligned_depth_fill_up = F.interpolate(aligned_depth_fill_down, size=[1080, 1920], mode='bilinear')

                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_up[0].permute(1,2,0).detach().cpu().numpy(), '3')
                                    # st()
                                    aligned_depth_fill_one = torch.where(mask > 0, aligned_depth_fill_up, aligned_depth_fill_one)

                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '4')
                                elif self.depth_inpaint_method == 5:

                                    ### try5 minpool+inpaint
                                    aligned_depth_one = aligned_depth[batch_index:batch_index+1]
                                    # st()
                                    proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask'][batch_index:batch_index+1]
                                    aligned_depth_fill_one = torch.where(proj_cloud_color_mask<1, torch.as_tensor(torch.ones(aligned_depth_one.shape) * 8.0, dtype=torch.double).cuda(), aligned_depth_one)

                                    aligned_depth_fill_one = -F.max_pool2d(-aligned_depth_fill_one, 3, stride=1, padding=1)
                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '2')

                                    max_hole_size = 10
                                    # max_hole_size = 20

                                    mask = torch.zeros_like(aligned_depth_fill_one)
                                    mask[aligned_depth_fill_one == 8.0] = 1
                                    
                                    # aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                                    aligned_depth_fill_down = F.max_pool2d(aligned_depth_fill_one, 3, stride=8, padding=1)

                                    selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                                    
                                    aligned_depth_fill_down = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=8000)

                                    aligned_depth_fill_down = torch.from_numpy(aligned_depth_fill_down / 1000).cuda().unsqueeze(0).unsqueeze(0)

                                    aligned_depth_fill_up = F.interpolate(aligned_depth_fill_down, size=[1080, 1920], mode='bilinear')

                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_up[0].permute(1,2,0).detach().cpu().numpy(), '3')
                                    # st()
                                    aligned_depth_fill_one = torch.where(mask > 0, aligned_depth_fill_up, aligned_depth_fill_one)

                                    if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '4')


                                aligned_depth_save = aligned_depth_fill_one.cpu().numpy()

                                # print('before save')
                                f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                                np.save(f, aligned_depth_save) 
                                f.close()
                            aligned_depth_fill_list.append(aligned_depth_fill_one)
                        aligned_depth_fill = torch.cat(aligned_depth_fill_list, dim=0)

                        # ### try-final 4,5 prepare data for inpaint network
                        # # print('prepare data for inpaint network')
                        # aligned_depth_fill_list = []
                        # batch_size = meta[0]['joints_3d'].shape[0]
                        # for batch_index in range(batch_size):
                        #     # st()
                        #     image_set = 'train' if self.training else 'val'

                        #     seq = self.code_to_seq[int(meta[0]['seq_code'][batch_index])]

                        #     camera_index = meta[0]['camera_index'][batch_index]
                        #     depth_index = meta[0]['depth_index'][batch_index]
                        #     aligned_depth_inp_key = f"{seq}_{camera_index}_{depth_index}"
                            
                        #     num_views = len(calib_list[list(calib_list.keys())[0]])

                        #     if self.namedataset == 'large':
                        #         save_dir = f'./aligned_depth_db_{image_set}_{num_views}cam'
                        #     elif self.namedataset == 'OUT_3D_POSE_VID':
                        #         save_dir = './aligned_depth_db_OUT_3D_POSE_VID'
                        #     else:
                        #         save_dir = './aligned_depth_db'

                        #     ### prepare
                        #     save_dir += f'_inp{self.depth_inpaint_method}_pp'
                        #     save_dir_mask = save_dir + '_mask'

                        #     os.makedirs(save_dir, exist_ok=True)
                        #     os.makedirs(save_dir_mask, exist_ok=True)

                        #     save_path = osp.join(save_dir, f'{aligned_depth_inp_key}.png')
                        #     save_path_mask = osp.join(save_dir_mask, f'{aligned_depth_inp_key}.png')

                        #     if False:
                        #         pass 
                        #     else:
                        #         if self.depth_inpaint_method == 4:
                        #             pass
                        #         elif self.depth_inpaint_method == 5:
                        #             ### try5 minpool+inpaint
                        #             aligned_depth_one = aligned_depth[batch_index:batch_index+1]
                        #             # st()
                        #             proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask'][batch_index:batch_index+1]
                        #             aligned_depth_fill_one = torch.where(proj_cloud_color_mask<1, torch.as_tensor(torch.ones(aligned_depth_one.shape) * 8.0, dtype=torch.double).cuda(), aligned_depth_one)

                        #             aligned_depth_fill_one = -F.max_pool2d(-aligned_depth_fill_one, 3, stride=1, padding=1)
                        #             if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '2')


                        #             # ### 第一个max pool之后就直接pad，成为可以直接resize到
                        #             # p2d_cloud = ((2025-1920)//2, (2025-1920)//2)
                        #             # aligned_depth_fill_one = F.pad(aligned_depth_fill_one, p2d_cloud, "constant", 8.0)


                        #             max_hole_size = 10
                        #             # max_hole_size = 20

                        #             mask = torch.zeros_like(aligned_depth_fill_one)
                        #             mask[aligned_depth_fill_one == 8.0] = 1
                                    

                        #             # ### 1org
                        #             # ### 经过测试，区别不大，那用interpolate吧
                        #             # # aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                        #             # # aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[1080 // 4, 1920 // 4], mode='nearest') ### 原本0位置被填，有斑点
                        #             # aligned_depth_fill_down = F.max_pool2d(aligned_depth_fill_one, 3, stride=8, padding=1)

                        #             ### 1prepare data for inpaint network
                        #             # st()
                        #             ### todo
                        #             # resize mask and aligned_depth_fill_down to 512x512
                        #             aligned_depth_fill_down = F.interpolate(aligned_depth_fill_one, size=[512, 512], mode='nearest')
                        #             if vis_depth_flag:show_proj_cloud(aligned_depth_fill_down[0].permute(1,2,0).detach().cpu().numpy(), '2-1')
                        #             selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                                    
                        #             selected_area_255 = (selected_area / 8.0 * 255).astype(np.uint8)
                        #             selected_area_255 = selected_area_255[..., np.newaxis]
                        #             selected_area_255 = np.repeat(selected_area_255, 3, axis=2)
                        #             if vis_depth_flag:
                        #                 name = '2-2'
                        #                 cv.namedWindow(name,0)
                        #                 cv.resizeWindow(name, show_w, show_h)
                        #                 cv.imshow(name, selected_area_255)
                        #                 if 113 == cv.waitKey(100):
                        #                     st()
                                    
                        #             # st()
                        #             mask_255 = np.zeros(selected_area.shape,dtype=np.uint8)
                        #             mask_255[selected_area==8.0] = 255
                        #             mask_255 = mask_255[..., np.newaxis]
                        #             mask_255 = np.repeat(mask_255, 3, axis=2)

                        #             if vis_depth_flag:
                        #                 name = '2-3'
                        #                 cv.namedWindow(name,0)
                        #                 cv.resizeWindow(name, show_w, show_h)
                        #                 cv.imshow(name, mask_255)
                        #                 if 113 == cv.waitKey(100):
                        #                     st()
                                    
                        #             # save 
                        #             cv.imwrite(save_path, selected_area_255)
                        #             cv.imwrite(save_path_mask, mask_255)
                                    

                        #             selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                                    
                        #             aligned_depth_fill_down = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=8000)

                        #             aligned_depth_fill_down = torch.from_numpy(aligned_depth_fill_down / 1000).cuda().unsqueeze(0).unsqueeze(0)

                        #             aligned_depth_fill_up = F.interpolate(aligned_depth_fill_down, size=[1080, 1920], mode='bilinear')

                        #             if vis_depth_flag:show_proj_cloud(aligned_depth_fill_up[0].permute(1,2,0).detach().cpu().numpy(), '3')
                        #             # st()
                        #             aligned_depth_fill_one = torch.where(mask > 0, aligned_depth_fill_up, aligned_depth_fill_one)

                        #             if vis_depth_flag:show_proj_cloud(aligned_depth_fill_one[0].permute(1,2,0).detach().cpu().numpy(), '4')
                                
                        #         # aligned_depth_save = aligned_depth_fill_one.cpu().numpy()
                        #         # # print('before save')
                        #         # f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                        #         # np.save(f, aligned_depth_save) 
                        #         # f.close()
                        #     aligned_depth_fill_list.append(aligned_depth_fill_one)
                        # aligned_depth_fill = torch.cat(aligned_depth_fill_list, dim=0)


                        # ### try4 maxpool+inpaint
                        # # st()
                        # aligned_depth_fill = F.max_pool2d(aligned_depth, 3, stride=1, padding=1)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '2')

                        # max_hole_size = 10
                        # # max_hole_size = 20

                        # mask = torch.zeros_like(aligned_depth_fill)
                        # mask[aligned_depth_fill == 0] = 1
                        
                        # # aligned_depth_fill_down = F.interpolate(aligned_depth_fill, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                        # aligned_depth_fill_down = -F.max_pool2d(-aligned_depth_fill, 3, stride=8, padding=1)


                        # selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                        
                        # aligned_depth_fill_down = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=0)

                        # aligned_depth_fill_down = torch.from_numpy(aligned_depth_fill_down / 1000).cuda().unsqueeze(0).unsqueeze(0)

                        # aligned_depth_fill_up = F.interpolate(aligned_depth_fill_down, size=[1080, 1920], mode='bilinear')

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_up[0].permute(1,2,0).detach().cpu().numpy(), '3')
                        # # st()
                        # aligned_depth_fill = torch.where(mask > 0, aligned_depth_fill_up, aligned_depth_fill)

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '4')


                        # ### try5 minpool+inpaint
                        # # st()
                        # proj_cloud_color_mask = meta[view_index]['proj_cloud_color_mask']
                        # aligned_depth_fill = torch.where(proj_cloud_color_mask<1, torch.as_tensor(torch.ones(aligned_depth.shape) * 8.0, dtype=torch.double).cuda(), aligned_depth)

                        # aligned_depth_fill = -F.max_pool2d(-aligned_depth_fill, 3, stride=1, padding=1)
                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '2')

                        # max_hole_size = 10
                        # # max_hole_size = 20

                        # mask = torch.zeros_like(aligned_depth_fill)
                        # mask[aligned_depth_fill == 8.0] = 1
                        
                        # # aligned_depth_fill_down = F.interpolate(aligned_depth_fill, size=[1080 // 4, 1920 // 4], mode='bilinear') ### 原本0位置被填，有斑点
                        # aligned_depth_fill_down = F.max_pool2d(aligned_depth_fill, 3, stride=8, padding=1)


                        # selected_area = aligned_depth_fill_down.squeeze().cpu().numpy()
                        
                        # aligned_depth_fill_down = smooth_depth_image((selected_area * 1000).astype(np.uint16), max_hole_size=max_hole_size, hole_value=8000)

                        # aligned_depth_fill_down = torch.from_numpy(aligned_depth_fill_down / 1000).cuda().unsqueeze(0).unsqueeze(0)

                        # aligned_depth_fill_up = F.interpolate(aligned_depth_fill_down, size=[1080, 1920], mode='bilinear')

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill_up[0].permute(1,2,0).detach().cpu().numpy(), '3')
                        # # st()
                        # aligned_depth_fill = torch.where(mask > 0, aligned_depth_fill_up, aligned_depth_fill)

                        # if vis_depth_flag:show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '4')
                else:
                    aligned_depth_fill = aligned_depth
                # show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '2')
                
                p2d_cloud = ((2025-1920)//2, (2025-1920)//2)
                aligned_depth_fill = F.pad(aligned_depth_fill, p2d_cloud, "constant", 0)

                # show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '3')
                # st()
                # 和网络imagesize对齐的分辨率
                # aligned_depth_fill = F.interpolate(aligned_depth_fill, size=[512, 960], mode='bilinear')

                # 和heatmap对齐的分辨率
                
                try:
                    aligned_depth_fill = F.interpolate(aligned_depth_fill, size=self.dep_downsample_size, mode='bilinear')
                except:
                    st()
                # aligned_depth_fill = F.interpolate(aligned_depth_fill, size=[512, 960], mode='bilinear')
                # aligned_depth_fill = F.interpolate(aligned_depth_fill, size=[1024, 1920], mode='bilinear')
                # aligned_depth_fill = F.interpolate(aligned_depth_fill, size=[128, 240], mode='bilinear')

                if vis_depth_flag:show_proj_cloud(aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '4')
                

                # if vis_depth_flag:show_view_w_aligned_depth(views[0], aligned_depth_fill[0].permute(1,2,0).detach().cpu().numpy(), '4_view_w_aligned_depth')


                all_aligned_depth_fill.append(aligned_depth_fill)

            if self.pseudep:
                all_limb_depth = []
                for view_index in range(len(meta)):
                    limb_depth = meta[view_index]['limb_depth']
                    all_limb_depth.append(limb_depth)

            ### 不用unet也可以直接用pseudep作为测试
            if self.pseudep:
                all_depth_recon_target = []
                for aligned_depth_fill, limb_depth in zip(all_aligned_depth_fill, all_limb_depth):
                    depth_recon_target = torch.where(limb_depth>0, limb_depth, aligned_depth_fill)
                    # st()
                    if vis_depth_flag:show_proj_cloud(limb_depth[0].permute(1,2,0).detach().cpu().numpy(), '7')
                    if vis_depth_flag:show_proj_cloud(depth_recon_target[0].permute(1,2,0).detach().cpu().numpy(), '8')

                    all_depth_recon_target.append(depth_recon_target)
            else:
                all_depth_recon_target = all_aligned_depth_fill

        ### lcc: 直接用refine之前的depth，这里仅仅为了比较对depth downsample对于结果的影响，
        if self.f_weight and (not self.use_unet):
            # st()
            all_refined_depth = all_depth_recon_target

            ### <org> ###
            # all_cloud = depthmap_to_cloud(all_refined_depth, meta, self.dep_downsample_size, self.code_to_seq, calib_list, device)

            ### <precompute cloud> ###
            # st()
            ### save time
            all_cloud = []
            # len(all_heatmaps) is num_views
            # depthmap max pooling + down sample(in unet)
            for view_index, denoised_depth in enumerate(all_refined_depth):
                cloud_batch_list = []
                batch_size = denoised_depth.shape[0]
                for batch_index in range(batch_size):
                    denoised_depth_batch = denoised_depth[batch_index]
                    # st()
                    image_set = 'train' if self.training else 'val'
                    # image_set = 'val'

                    seq = self.code_to_seq[int(meta[0]['seq_code'][batch_index])]

                    camera_index = meta[0]['camera_index'][batch_index]
                    depth_index = meta[0]['depth_index'][batch_index]
                    aligned_depth_inp_key = f"{seq}_{camera_index}_{depth_index}"
                    
                    num_views = len(calib_list[list(calib_list.keys())[0]])

                    if self.namedataset == 'large' or \
                        self.namedataset == 'large_interp25j' or \
                        self.namedataset == 'large_wholebody' or \
                        self.namedataset == 'large_wholebody_mp':
                        save_dir = f'./aligned_depth_db_{image_set}_{num_views}cam_itv{self.interval}'

                        ### lcc debugging test
                        # save_dir = f'./aligned_depth_db_{image_set}_{num_views}cam_itv{3*1000}'
                    elif self.namedataset == 'OUT_3D_POSE_VID':
                        save_dir = './aligned_depth_db_OUT_3D_POSE_VID'
                    else:
                        save_dir = './aligned_depth_db'
                    
                    # dtype_ext = 'fl64'
                    dtype_ext = 'fl16'

                    save_dir += f'_inp{self.depth_inpaint_method}_cloud_{dtype_ext}'

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = osp.join(save_dir, f'{aligned_depth_inp_key}.npy')

                    if osp.exists(save_path):
                        # st()
                        # print('before load')

                        f = gzip.GzipFile(save_path, "r")
                        aligned_cloud_save = np.load(f)

                        if dtype_ext == 'fl16':
                            aligned_cloud_save = aligned_cloud_save.copy().astype(np.float64) ### 压缩保存1
                        elif dtype_ext == 'fl64':
                            pass ### 不压缩保存

                        f.close()

                        
                        cloud_one = torch.from_numpy(aligned_cloud_save).cuda() 
                    else:
                        cloud_one = depthmap_to_cloud_one(denoised_depth_batch, meta, self.dep_downsample_size, self.code_to_seq, calib_list, view_index, batch_index, device)

                        # st()
                        aligned_cloud_save = cloud_one.cpu().numpy()

                        if dtype_ext == 'fl16':
                            aligned_cloud_save = aligned_cloud_save.copy().astype(np.float16) ### 压缩保存1
                        elif dtype_ext == 'fl64':
                            pass ### 不压缩保存
                        
                        # print('before save')
                        f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                        np.save(f, aligned_cloud_save) 
                        f.close()
                    
                    cloud_batch_list.append(cloud_one)  ### 保存

                cloud_batch = torch.cat(cloud_batch_list, axis=0)
                all_cloud.append(cloud_batch)    


        # unet refine for depth map
        # input rgb depth 
        # output depth offset
        # st()
        if self.f_weight and self.use_unet:
            all_refined_depth = []
            # 对于每一个view输出denoise之后的depth
            for heatmaps, aligned_depth_fill, view in zip(all_heatmaps, all_aligned_depth_fill, views):
                aligned_depth_fill = torch.as_tensor(aligned_depth_fill, dtype=torch.float, device=device)    
                
                
                # ### org 
                refined_depth_offset = self.unet(heatmaps, aligned_depth_fill, view)

                # st()
                # # ### lcc debugging:half cudnn报错
                # refined_depth_offset = self.unet(heatmaps.half(), aligned_depth_fill.half(), view.half())
                # refined_depth_offset = refined_depth_offset.float()
                
                if vis_depth_flag:show_view(view, '5_view')

                # st()
                if self.unet_res_out:
                    refined_depth = refined_depth_offset + aligned_depth_fill
                    
                if vis_depth_flag:show_proj_cloud(refined_depth[0].permute(1,2,0).detach().cpu().numpy(), '5')

                if vis_depth_flag:show_proj_cloud(refined_depth_offset[0].permute(1,2,0).detach().cpu().numpy(), '6')


                all_refined_depth.append(refined_depth)

            all_cloud = depthmap_to_cloud(all_refined_depth, meta, self.dep_downsample_size, self.code_to_seq, calib_list, device)

            # st()

            # 生成的all_denoised_depth，挖洞的地方应该盖住heatmap的响应的区域比较好，所以在造gt的时候洞可以挖的大一些
            # 这里有一个all_denoised_depth的2d loss
            # TODO
            criterion_depth_recon = nn.MSELoss(reduction='mean').cuda()
            criterion_depth_recon_focus = nn.MSELoss(reduce=False).cuda()
            loss_depth_recon = criterion_depth_recon(torch.zeros(1, device=device), torch.zeros(1, device=device))
            for t, o, h in zip(all_depth_recon_target, all_refined_depth, all_heatmaps):
                if self.focus_lrecon:
                    # st()
                    heatmaps_max, _ = torch.max(F.interpolate(h, size=self.dep_downsample_size, mode='bilinear'), dim=1, keepdim=True)
                    heatmaps_max = torch.clamp(heatmaps_max, min=0, max=1.0)
                    loss_depth_recon += torch.mean(criterion_depth_recon_focus(o.double(), t) * (1 - heatmaps_max))
                else:
                    loss_depth_recon += criterion_depth_recon(o.double(), t)
            loss_depth_recon /= len(all_depth_recon_target)
            
            ### lcc debugging:直接测试使用我构造的label的效果
            # all_refined_depth = all_depth_recon_target
            
            # denoised depth map -> cloud 
            ### 下面本来又计算了一遍md
            # all_cloud = depthmap_to_cloud(all_refined_depth, meta, self.dep_downsample_size, self.code_to_seq, calib_list, device)
            # st()
            ## st() 检查dep15爆炸原因，果然是这里爆炸
            # all_cloud = [cloud[:,0:1,:,:,:] for cloud in all_cloud]


            ### lcc 还是不行，循环不能避免爆炸
            # all_cloud_list = []
            # for joint_i in range(all_refined_depth[0].shape[1]):
            #     one_cloud = depthmap_to_cloud([all_refined_depth[0][:,joint_i:joint_i+1,:,:]], meta, self.dep_downsample_size, device)
            #     all_cloud_list.append(one_cloud[0])
            # # st()
            # all_cloud = [torch.cat(all_cloud_list, dim=1)]
            # # st()
            # all_cloud = [cloud[:,0:1,:,:,:] for cloud in all_cloud]




        if vis_time_cost_flag:
            time_end=time.time();print('cost2\n',time_end-time_start)
            # st()
            print('st3')
            import time;time_start=time.time()

        # 下面这个注释指的是使用gt 
        ### lcc debugging 使用gt heatmap
        # all_heatmaps = targets_2d
        if self.USE_GT_HM:
            all_heatmaps = targets_2d
        
        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        # 只是算算，不一定用来优化2d网络
        # calculate 2D heatmap loss
        criterion = PerJointMSELoss().cuda()

        loss_2d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))

        # if targets_2d is not None:
        if False:
            for t, w, o in zip(targets_2d, weights_2d, all_heatmaps):
                # st()
                loss_2d += criterion(o, t, True, w)
            loss_2d /= len(all_heatmaps)

        loss_3d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
            
        # 用gt proposal
        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            ### lcc debugging: 比较train unet前后的root_net输入输出
            ### 结论输入一样，输出不同
            ### train unet之后
            # (Pdb) all_cloud[0].mean()
            # tensor(408.1554, device='cuda:0', dtype=torch.float64)
            # (Pdb) all_cloud[0].min()                                                   
            # tensor(-6388.3353, device='cuda:0', dtype=torch.float64)
            # (Pdb) all_cloud[0].max()                                                   
            # tensor(5616.4895, device='cuda:0', dtype=torch.float64)

            # (Pdb) root_cubes.mean()                                                    
            # tensor(0.0034, device='cuda:0')                                            
            # (Pdb) root_cubes.max()                                                     
            # tensor(0.9851, device='cuda:0')                                            
            # (Pdb) root_cubes.min()                                                     
            # tensor(-0.0032, device='cuda:0')

            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].mean()
            # tensor(0.0052, device='cuda:0')
            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].max()
            # tensor(0.0781, device='cuda:0')
            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].min()
            # tensor(-0.0431, device='cuda:0')


            ### train unet之前
            # (Pdb) all_cloud[0].mean()
            # tensor(408.1554, device='cuda:0', dtype=torch.float64)
            # (Pdb) all_cloud[0].max()
            # tensor(5616.4895, device='cuda:0', dtype=torch.float64)
            # (Pdb) all_cloud[0].min()
            # tensor(-6388.3353, device='cuda:0', dtype=torch.float64)
            # (Pdb) root_cubes.mean()
            # tensor(0.0034, device='cuda:0')
            # (Pdb) root_cubes.max()
            # tensor(0.9939, device='cuda:0')
            # (Pdb) root_cubes.min()
            # tensor(-0.0034, device='cuda:0')
            
            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].mean()
            # tensor(0.0052, device='cuda:0')
            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].max()
            # tensor(0.0781, device='cuda:0')
            # (Pdb) self.root_net.state_dict()['v2v_net.output_layer.weight'].min()
            # tensor(-0.0431, device='cuda:0')

            ### lcc debugging:root_net的参数打印
            ### 结论发现weight和bias并没有改变，但是runing mean和runing var改变了
            # root_net_state_dict = self.root_net.state_dict()
            # for key in root_net_state_dict:
            #     try:
            #         print(f'key:{key}, mean:{root_net_state_dict[key].mean()}')
            #     except:
            #         continue
            
            ### lcc debugging 在这里detach 3d loss不更新unet
            ### 经过小数据集比较，detach 3d loss的效果比detach joint的效果好 
            if self.f_weight and self.use_unet:
                all_cloud_detach = [cloud.detach() for cloud in all_cloud]
                root_cubes, grid_centers = self.root_net(all_heatmaps, meta, all_cloud=all_cloud_detach)
            else:
                if not self.use_nojrn_wholebody:
                    root_cubes, grid_centers = self.root_net(all_heatmaps, meta, all_cloud=all_cloud, all_net_training=self.training)
                else:
                    root_cubes, grid_centers = self.root_net(all_heatmaps_whole, meta, all_cloud=all_cloud, all_net_training=self.training)
            
            # st()
            # # lcc debugging
            # # all_heatmaps几乎完全一样， root_cubes和grid_centers都不一样
            # print('all_heatmaps', all_heatmaps[0])
            # print(all_heatmaps[0].min(), all_heatmaps[0].max(), all_heatmaps[0].mean())
            # print('root_cubes', root_cubes[0])
            # print(root_cubes[0].min(), root_cubes[0].max(), root_cubes[0].mean())
            # print('grid_centers', grid_centers[0])
            # print(grid_centers[0].min(), grid_centers[0].max(), grid_centers[0].mean())
            # st()
            
            # calculate 3D heatmap loss
            if targets_3d is not None:
                # st()
                loss_3d = criterion(root_cubes, targets_3d)
            del root_cubes

        if vis_time_cost_flag:
            time_end=time.time();print('cost3\n',time_end-time_start)
            # st()
            print('st4')
            import time;time_start=time.time()

        # st()
        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

        pred_wholebody = torch.zeros(batch_size, self.num_cand, self.num_joints_wholebody, 5, device=device)
        pred_wholebody[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt
        
        
        ### 对于同一个人的每一个关节点赋予了相同的confidence和cand2gt
        # (Pdb) pred[:, :, :, 3:].shape
        # torch.Size([1, 10, 15, 2])
        # (Pdb) grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)
        # tensor([[[[ 1.0000,  0.9019]],

        #         [[ 2.0000,  0.8904]],

        #         [[ 0.0000,  0.8700]],

        #         [[-1.0000,  0.0535]],

        #         [[-1.0000,  0.0378]],

        #         [[-1.0000,  0.0340]],

        #         [[-1.0000,  0.0263]],

        #         [[-1.0000,  0.0245]],

        #         [[-1.0000,  0.0236]],

        #         [[-1.0000,  0.0233]]]], device='cuda:0')
        # (Pdb) grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2).shape
        # torch.Size([1, 10, 1, 2])
        
        
        
        # st()
        ### 注意从这里可以看出来pred的后两维是center的坐标，用来match gt的，但是center的坐标为什么是2维呢
        ### grid_centers的前三维才是坐标，后两维应该分别是是否选择和置信度
        # 看懂grid_centers各个维度的含义
        # grid_center[i][3] >= 0 才会match
        ### grid_center[i][4]就是heatmap的峰值，也就是置信度
        ### grid_center[i][3]就是是否选择这个grid_center，在project_layer forword中体现


        loss_cord = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        loss_cord_refine = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))

        criterion_cord = PerJointL1Loss().cuda()
        count = 0

        # st()

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:

                # st()
                # posenet居然每次只处理一个人，这跟我想的不一样woc
                # all_cloud_detach = [cloud.detach() for cloud in all_cloud]
                # single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], all_cloud=all_cloud_detach)

                ### cloud.detach()只是说这里的梯度不去经过cloud去优化unet了，但是v2v还是会被优化的
                ### 注意在model中不传入unet_turn的时候unet_turn=True其实就相当于原始的不加unet_turn的效果，不需要用config判断了

                # st()
                if unet_turn:
                    if not self.use_nojrn_wholebody:
                        single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], all_cloud=all_cloud)
                    else:
                        single_pose = self.pose_net(all_heatmaps_whole, meta, grid_centers[:, n], all_cloud=all_cloud)
                
                    # single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], all_cloud=all_cloud)
                else:
                    ### lcc debugging yici: 
                    all_cloud_detach = [cloud.detach() for cloud in all_cloud]
                    single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], all_cloud=all_cloud_detach)

                ### org
                # single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], all_cloud=all_cloud)
                # st()
                pred[:, n, :, 0:3] = single_pose.detach()

                ### debugging
                # print(f'now jrn turn {jrn_turn}')
                # jrn_turn = False
                # jrn_turn = True
                
                ### 注意test默认jrn_turn=True，但是epoch=0的时候jrn并没有被训练
                ### 所以在epoch=0的时候不valid得了，或者就jrn和前面的cpn prn一起训练
                # st()
                if self.use_jrn and jrn_turn:
                    for i in range(batch_size):
                        gc = single_pose[i].detach().clone()

                        if self.jrn_type == 'all':
                            # st()
                            # (Pdb) single_pose.shape
                            # torch.Size([1, 13, 3])

                            ### 这里13个gc中有三个要特殊处理，face,lhand,rhand
                            ### 最后出来的single_pose_refine应该是123个joints
                            single_pose_refine = self.joint_refine_net(all_heatmaps_whole, meta, gc, all_cloud=all_cloud)                        
                            ### todo pred
                            # st()
                            pred_wholebody[i:i+1, n, :, 0:3] = single_pose_refine 
                        elif self.jrn_type == 'face':
                            
                            pred_wholebody[i:i+1, n, :pred.shape[2], 0:3] = pred[i:i+1, n, :, 0:3]
                            # st()
                            single_pose_refine = self.joint_refine_net(all_heatmaps_whole, meta, gc, all_cloud=all_cloud)
                            pred_wholebody[i:i+1, n, 13:13+68, 0:3] = single_pose_refine
                        elif self.jrn_type == 'facehandsepv':
                            pred_wholebody[i:i+1, n, :pred.shape[2], 0:3] = pred[i:i+1, n, :, 0:3]
                            # st()
                            single_pose_refine = self.joint_refine_net(all_heatmaps_whole, meta, gc, all_cloud=all_cloud)
                            # st()
                            pred_wholebody[i:i+1, n, 13:, 0:3] = single_pose_refine

                ### 注意分清楚train和eval，eval的时候不会跑这个
                # calculate 3D pose loss
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                # if 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0: # 过滤不可见的
                            # st()
                            targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()] ### 除了不可见的，其他的pred第三维都是cand2gt
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                            count += 1

                            # st()
                            loss_cord = (loss_cord * (count - 1) +
                                        criterion_cord(single_pose[i:i + 1], targets[:,:self.num_joints,:], True, weights_3d[:,:self.num_joints,:])) / count
                
                            ### joint refinenet
                            ### 可以尝试一下grid center只用三维，并且尝试一下15个joint 同时过JRN
                            # st()
                            if self.use_jrn and jrn_turn:
                                if self.jrn_type == 'all':
                                    loss_cord_refine = (loss_cord_refine * (count - 1) +
                                                criterion_cord(single_pose_refine[i:i + 1], targets, True, weights_3d)) / count
                                elif self.jrn_type == 'face':
                                    # st()
                                    loss_cord_refine = (loss_cord_refine * (count - 1) +
                                                criterion_cord(single_pose_refine[i:i + 1], targets[:,13:13+68,:], True, weights_3d[:,13:13+68,:])) / count
                                elif self.jrn_type == 'facehandsepv':
                                    # st()
                                    loss_cord_refine = (loss_cord_refine * (count - 1) +
                                                criterion_cord(single_pose_refine[i:i + 1], targets[:,13:,:], True, weights_3d[:,13:,:])) / count

                del single_pose
                if self.use_jrn and jrn_turn:del single_pose_refine

        ### org
        # return pred, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord

        if vis_time_cost_flag:
            time_end=time.time();print('cost4\n',time_end-time_start)
            # st()
            # print('st5')


        if not self.use_nojrn_wholebody:
            pass
        else:
            pred_wholebody = pred.clone()
            pred = pred_wholebody[:, :, :13, :]

        # st()
        if self.f_weight and self.use_unet:
            return pred, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon
        else:
            return pred, pred_wholebody, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_cord_refine
        


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.NETWORK.USE_WHOLEBODY_BACKBONE:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')()
    else:
        if (not cfg.NETWORK.USE_PRECOMPUTED_HM) and cfg.BACKBONE_MODEL:
            backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
        else:
            backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
