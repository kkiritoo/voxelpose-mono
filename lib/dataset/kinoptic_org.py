# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp

from torch._C import BenchmarkConfig
import numpy as np
import json_tricks
import json
import pickle
import logging
import os
import copy

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints

from pdb import set_trace as st

# lcc import
from os import lseek, path
import cv2 as cv


vis_flag = False

logger = logging.getLogger(__name__)


JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]




# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# def show_np_image(np_image):
#     plt.figure()
#     # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
#     plt.imshow(np_image)
#     plt.show()

class Kinoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
             
        ### lcc debugging
        # CAMS = [(50, 2)]

        if cfg.DATASET.CAMERA_NUM == 1 and cfg.DATASET.NODE != -1:
            self.CAMS = [(50, cfg.DATASET.NODE)]
            print(f'runing single camera with node {cfg.DATASET.NODE}')
        else:
            self.CAMS = [(50, 1),
                    (50, 2),
                    (50, 3),
                    (50, 4),
                    (50, 5),
                    (50, 6),
                    (50, 7),
                    (50, 8),
                    (50, 9),
                    (50, 10)]

        if cfg.DATASET.NAMEDATASET == 'all':
            TRAIN_LIST = [
                '160422_ultimatum1',
                '160224_haggling1',
                '160226_haggling1',
                '161202_haggling1',
                '160906_ian1',
                '160906_ian2',
                '160906_ian3',
                '160906_band1',
                '160906_band2',
                '160906_band3',
            ]
            VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5']
            # VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5']
        elif cfg.DATASET.NAMEDATASET == 'no_occlu':
            TRAIN_LIST = [
                '171204_pose1'
            ]
            VAL_LIST = ['171204_pose3']
        elif cfg.DATASET.NAMEDATASET == 'debugging':
            TRAIN_LIST = [
                '160906_band1'
            ]
            VAL_LIST = ['160906_band1']


        self.pixel_std = 200.0 # 没有用过
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)

        ### lcc debugging
        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            # self._interval = 1
            self._interval = 3
            
            ### lcc debugging: smaller dataset training
            # self._interval = 3*10
            self.cam_list = self.CAMS[:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            # self._interval = 3
            self._interval = 12

            ### lcc debugging: smaller dataset training
            self._interval = 12*10
            # self._interval = 12*30
            # self._interval = 12*30*6

            self.cam_list = self.CAMS[:self.num_views]
            self.num_views = len(self.cam_list)

        ### 创建reader_list
        # 这里最用img读取，并且把depth.dat拆分一下，否则内存肯定会爆炸，应该不会吧，open不是全部读到内存 read才是内存复用就行了
        self._color_reader_list = {}
        self._depth_reader_list = {}

        for seq in self.sequence_list:
            selected_cams = self.CAMS[:self.num_views]
            node_id_list = [cam[1] for cam in selected_cams]
            
            seq_root = osp.join(self.dataset_root, seq)
            
            _color_reader_seq_list = []
            _depth_reader_seq_list = []

            for node_id in node_id_list:
                # _color_reader_seq_list.append(ColorReader(path.join(seq_root, 'kinectVideos', f'kinect_50_{node_id:02d}.mp4')))

                _color_reader_seq_list.append(ColorImgReader(path.join(seq_root, 'kinectImgs', f'50_{node_id:02d}')))
                _depth_reader_seq_list.append(DepthReader(path.join(seq_root, 'kinect_shared_depth', f'KINECTNODE{node_id}', 'depthdata.dat')))
            
            self._color_reader_list[seq] = _color_reader_seq_list
            self._depth_reader_list[seq] = _depth_reader_seq_list
        
        # 经过验证和拆出来的文件的数量一致的，可以直接索引拆出来的文件
        # (Pdb) self._color_reader_list['171204_pose1'][0].__len__()
        # 29476
        # (Pdb) self._color_reader_list['171204_pose1'][1].__len__()
        # 29347

        ### 相机参数
        self._calib_list = {}
        for seq in self.sequence_list:
            selected_cams = self.CAMS[:self.num_views]
            node_id_list = [cam[1] for cam in selected_cams]
            
            seq_root = osp.join(self.dataset_root, seq)
            
            _calib_seq_list = []

            for node_id in node_id_list:
                _calib_seq_list.append(CalibReader(path.join(seq_root, f'calibration_{seq}.json'), path.join(seq_root, f'kcalibration_{seq}.json'), node_id))

            self._calib_list[seq] = _calib_seq_list
        
        # save self._calib_list for latter use 
        if osp.exists('calib_list.pkl'):
            with open('calib_list.pkl','rb') as calib_list_f:
                calib_list = pickle.load(calib_list_f)
            calib_list.update(self._calib_list)
        else:
            calib_list = self._calib_list
        with open('calib_list.pkl','wb') as calib_list_f:
            pickle.dump(calib_list, calib_list_f)

        calib_list2_tmp = {}
        for seq in self.sequence_list:
            # 选出需要的相机的参数
            calib_list2_tmp[seq] = self._get_cam(seq)
        # st()
        # save self._calib_list for latter use 
        if osp.exists('calib_list2.pkl'):
            with open('calib_list2.pkl','rb') as calib_list_f2:
                calib_list2 = pickle.load(calib_list_f2)
            calib_list2.update(calib_list2_tmp)
        else:
            calib_list2 = calib_list2_tmp

        with open('calib_list2.pkl','wb') as calib_list_f2:
            pickle.dump(calib_list2, calib_list_f2)

        # st()

        ### 读取db
        # self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = 'group_{}_cam{}_dname{}_interval{}.pkl'.format(self.image_set, self.num_views, cfg.DATASET.NAMEDATASET, self._interval)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        # for now
        # 这个如果用以前的pose可能会全乱掉
        if False:
            # if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            
            ### lcc debugging: smaller dataset training
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        # st()
        self.db_size = len(self.db)

    def unproject(self, p2d, Kdist, Mdist, distCoeffs):
        # st()
        pn2d = np.linalg.solve(Kdist, np.concatenate([p2d[:,:2].T, np.ones((1, p2d.shape[0]))], axis=0)).T
        k = np.concatenate([distCoeffs[:5], np.zeros(12-5)], axis=0)
        x0 = pn2d[:,0]
        y0 = pn2d[:,1]
        x = x0
        y = y0

        for iter in range(5):
            r2 = x*x + y*y
            # st()
            icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
            deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
            deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
            x = (x0 - deltaX)*icdist
            y = (y0 - deltaY)*icdist
        
        pn2d = np.concatenate([x[:,np.newaxis], y[:,np.newaxis]], axis=1)
        
        p3d = np.concatenate([pn2d * (p2d[:,2:3]*0.001), p2d[:,2:3]*0.001], axis=1)
        
        # print(p3d.mean(axis=0))
        # st()
        p3d = np.matmul(np.linalg.inv(np.row_stack([Mdist, [0,0,0,1]])), np.concatenate([p3d, np.ones((p3d.shape[0], 1))], axis=1).T).T
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
        
    def matlab_poseproject2d(self, pts, R_color, t_color, c_K, c_d, bApplyDistort=True):
        # st()
        x = R_color.dot(pts.T) + t_color
        xp = x[:2,:] / x[2,:]
        # xp.mean(axis=1)

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
            radial       = 1.0 + Kp[0].dot(r2) + Kp[1].dot(r4) + Kp[4].dot(r6)
            tangential_x = 2.0*Kp[2].dot(XY) + Kp[3].dot(r2 + 2.0*X2)
            tangential_y = 2.0*Kp[3].dot(XY) + Kp[2].dot(r2 + 2.0*Y2)
            # np.stack 默认的新的axis=0和matlab一致
            xp = np.stack([radial, radial]) * xp[:2,:] + np.stack([tangential_x, tangential_y])
        
            
        # st()
        pt = (c_K[:2, :2].dot(xp) + c_K[:2, 2:3]).T

        return pt



    def depth_to_color_cloud(self, depth, shape, calib):

        X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        d_K, d_d = calib.depth_proj()
        R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        Kdist = d_K
        Mdist = T_M_depth[:3,:]
        distCoeffs = d_d
        lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        p3d = self.unproject(lcc_p2d, Kdist, Mdist, distCoeffs)
        cloud_local = p3d


        # 其实两种投影的效果，在coorddist的体现差不多
        # true 154 166.68120042242973
        # false 147 112.69317453625102
        
        # true 
        # coordist_sum_1 73431.36992994243 coordist_cnt_1 965 coordist_avg_1 76.09468386522532
        # coordist_sum_2 66934.35505974585 coordist_cnt_2 36 coordist_avg_2 1859.2876405484958
        # coordist_sum_1 75033.15063969753 coordist_cnt_1 951 coordist_avg_1 78.89921202912464
        # coordist_sum_2 124363.17691906726 coordist_cnt_2 50 coordist_avg_2 2487.2635383813454

        # thres=100
        # coordist_sum_1 17833.197513940264 coordist_cnt_1 348 coordist_avg_1 51.24482044235708
        # coordist_sum_2 44798.955457799835 coordist_cnt_2 153 coordist_avg_2 292.80363044313617
        # coordist_sum_1 19226.577779208976 coordist_cnt_1 369 coordist_avg_1 52.104546827124594
        # coordist_sum_2 44732.81980330921 coordist_cnt_2 132 coordist_avg_2 338.88499850991826

        # False 
        # coordist_sum_1 74938.09727888057 coordist_cnt_1 960 coordist_avg_1 78.06051799883393
        # coordist_sum_2 70634.86562584229 coordist_cnt_2 41 coordist_avg_2 1722.8016006302996
        # coordist_sum_1 75310.92724590703 coordist_cnt_1 960 coordist_avg_1 78.44888254781982
        # coordist_sum_2 91040.57444319566 coordist_cnt_2 41 coordist_avg_2 2220.501815687699

        # thres=100
        # coordist_sum_1 18199.110636028643 coordist_cnt_1 352 coordist_avg_1 51.7020188523541
        # coordist_sum_2 50745.75724522605 coordist_cnt_2 149 coordist_avg_2 340.5755519813829
        # coordist_sum_1 19497.826259839887 coordist_cnt_1 365 coordist_avg_1 53.41870208175312
        # coordist_sum_2 54418.245831116204 coordist_cnt_2 136 coordist_avg_2 400.13416052291325

        org_project_flag = False
        if org_project_flag:
            ######## org project_pts
            R, t = calib.k_depth_color()
            ## 居然是原本的kinoptic-dataset-tools这里R_color进行了改动造成的问题，md
            ## 看来更有必要还原原汁原味的matlab代码了
            R_color, t_color, T_M_color = calib.lcc_M_color_all()
            cloud_forproj = cloud_local @ np.linalg.inv(R_color) + t_color.reshape(1, 3)
            # cloud_world = self.lcc_camera_to_world(cloud_local, calib)
            # st()
            print(cloud_local.mean(axis=0))
            print(cloud_forproj.mean(axis=0))

            # ###### 3d point cloud_local -> rgb image
            img_pts = self.project_pts(cloud_forproj, calib)
            # 一直到这里都是一致的
            # print(img_pts.mean(axis=0))
            img_pts = np.array(img_pts + 0.5).astype(np.int32)
        else:
            ####### matlab project points
            R_color, t_color, T_M_color = calib.lcc_M_color_all()
            c_K, c_d = calib.color_proj()
            img_pts = self.matlab_poseproject2d(cloud_local, R_color, t_color, c_K, c_d)
            img_pts = np.array(img_pts + 0.5).astype(np.int32)
            
        

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
        


        point3d_panopticWorld = T_kinectLocal2PanopticWorld.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_panopticWorld = point3d_panopticWorld[:3, :].T

        cloud_world = point3d_panopticWorld


        # delete points out of image space
        # st()
        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud_world, shape[:2])
        # print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        # (Pdb) cloud_filter.shape
        # (30445, 3)
        # (Pdb) img_pts_filter.shape
        # (30445, 2)
        # st()
        # proj_cloud其实就是image u,v 到 3d point xyz的映射
        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter

        # st()
        # if flag:
        #     proj_cloud = np.flip(proj_cloud, axis=0)

        mask = np.multiply((proj_cloud[:,:,0]==0), (proj_cloud[:,:,1]==0))
        mask = np.multiply(mask, (proj_cloud[:,:,2]==0))
        # st()

        # 相机坐标系到世界坐标系
        # proj_cloud = self.lcc_camera_to_world(proj_cloud.reshape(-1,3), calib).reshape(proj_cloud.shape)
        # proj_cloud_2 = self.joints_to_color(proj_cloud_1.reshape(-1,3), calib).reshape(proj_cloud_1.shape)

        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
        # show_np_image(proj_cloud_show_z_color1)
        # show_np_image(proj_cloud_show_z_color2)


        ### lcc add:cloud_color
        # point3d_kinectcolor = T_kinectLocal2KinectColor.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        # point3d_kinectcolor = point3d_kinectcolor[:3, :].T
        

        # cloud_color = point3d_kinectcolor

        # img_pts_filter_color, cloud_filter_color = self._filter_pts(img_pts, cloud_color, shape[:2])
        # proj_cloud_color = np.zeros(shape=shape)
        # proj_cloud_color[img_pts_filter_color[:, 1], img_pts_filter_color[:, 0], :] = cloud_filter_color

        # proj_cloud_color = np.flip(proj_cloud_color, axis=0)

        ### lcc debugging
        # proj_cloud_color[:, :, 2] = proj_cloud_color[:, :, 2] * 1e3
        # depth = proj_cloud_color[:, :, 2] * 1e1

        #### z -> xy
        # proj_cloud_color_xy = proj_cloud_color[:, :, :2]
        # proj_cloud_color_xy_fromz = np.zeros(shape=proj_cloud_color[:, :, :2].shape)


        # ### 感觉区别就是matlab版本的误差是噪声的，python版本是平滑的，感觉还是matlab可能靠谱一些
        # ### 用matlab原汁原味
        # ### 我突然在想在做这个下面之前depth是不是应该是左右调换？
        # # depth = depth * 1e4
        # # print(f'2depth.mean(){depth.mean()}')
        # X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        # p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        # # color相机的内参
        # c_K, c_d = calib.color_proj()
        # R_color, t_color, T_M_color = calib.lcc_M_color()

        # # d_K, d_d = calib.depth_proj()
        # # R, t = calib.k_depth_color()
        # # R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        # # st()
        # Kdist = c_K
        # Mdist = T_M_color[:3,:]
        # # distCoeffs = c_d
        # distCoeffs = c_d[:,0]
        # lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        # p3d = self.unproject2(lcc_p2d, Kdist, Mdist, distCoeffs)
        # proj_cloud_color_xy_fromz = p3d.reshape(depth.shape[0], depth.shape[1], 3)[:, :, :2] * 1e-1
        
        return proj_cloud, mask

    def depth_to_color_cloud_w_aligned_depth(self, depth, shape, calib):

        X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        d_K, d_d = calib.depth_proj()
        R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        Kdist = d_K
        Mdist = T_M_depth[:3,:]
        distCoeffs = d_d
        lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        p3d = self.unproject(lcc_p2d, Kdist, Mdist, distCoeffs)
        cloud_local = p3d

        ####### matlab project points
        R_color, t_color, T_M_color = calib.lcc_M_color_all()
        c_K, c_d = calib.color_proj()
        img_pts = self.matlab_poseproject2d(cloud_local, R_color, t_color, c_K, c_d)
        img_pts = np.array(img_pts + 0.5).astype(np.int32)
        

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
        


        point3d_panopticWorld = T_kinectLocal2PanopticWorld.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_panopticWorld = point3d_panopticWorld[:3, :].T

        cloud_world = point3d_panopticWorld


        # delete points out of image space
        # st()
        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud_world, shape[:2])
        # print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        # (Pdb) cloud_filter.shape
        # (30445, 3)
        # (Pdb) img_pts_filter.shape
        # (30445, 2)
        # st()
        # proj_cloud其实就是image u,v 到 3d point xyz的映射
        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter

        # st()
        # proj_cloud = np.flip(proj_cloud, axis=0)

        mask = np.multiply((proj_cloud[:,:,0]==0), (proj_cloud[:,:,1]==0))
        mask = np.multiply(mask, (proj_cloud[:,:,2]==0))
        # st()

        # 相机坐标系到世界坐标系
        # proj_cloud = self.lcc_camera_to_world(proj_cloud.reshape(-1,3), calib).reshape(proj_cloud.shape)
        # proj_cloud_2 = self.joints_to_color(proj_cloud_1.reshape(-1,3), calib).reshape(proj_cloud_1.shape)

        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
        # show_np_image(proj_cloud_show_z_color1)
        # show_np_image(proj_cloud_show_z_color2)


        ### lcc add:cloud_color
        point3d_kinectcolor = T_kinectLocal2KinectColor.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_kinectcolor = point3d_kinectcolor[:3, :].T
        

        cloud_color = point3d_kinectcolor

        img_pts_filter_color, cloud_filter_color = self._filter_pts(img_pts, cloud_color, shape[:2])
        proj_cloud_color = np.zeros(shape=shape)
        proj_cloud_color[img_pts_filter_color[:, 1], img_pts_filter_color[:, 0], :] = cloud_filter_color

        # proj_cloud_color = np.flip(proj_cloud_color, axis=0)


        aligned_depth = proj_cloud_color[:, :, 2]


        ########################################### 下面就是反投影的部分 ###########################################
        
        ### lcc debugging 注意！！！！
        proj_cloud_color[:, :, 2] = proj_cloud_color[:, :, 2] * 1e3
        depth = proj_cloud_color[:, :, 2] * 1e1

        #### z -> xy
        proj_cloud_color_xy = proj_cloud_color[:, :, :2]
        proj_cloud_color_xy_fromz = np.zeros(shape=proj_cloud_color[:, :, :2].shape)


        ### 感觉区别就是matlab版本的误差是噪声的，python版本是平滑的，感觉还是matlab可能靠谱一些
        ### 用matlab原汁原味
        ### 我突然在想在做这个下面之前depth是不是应该是左右调换？
        # depth = depth * 1e4
        # print(f'2depth.mean(){depth.mean()}')
        X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        # color相机的内参
        c_K, c_d = calib.color_proj()
        R_color, t_color, T_M_color = calib.lcc_M_color()

        # d_K, d_d = calib.depth_proj()
        # R, t = calib.k_depth_color()
        # R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        # st()
        Kdist = c_K
        Mdist = T_M_color[:3,:]
        # distCoeffs = c_d
        distCoeffs = c_d[:,0]
        lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        p3d = self.unproject2(lcc_p2d, Kdist, Mdist, distCoeffs)
        proj_cloud_color_xy_fromz = p3d.reshape(depth.shape[0], depth.shape[1], 3)[:, :, :2] * 1e-1
        
        
        if vis_flag:
            print(f'proj_cloud_color_xy_fromz[...,0].mean() {proj_cloud_color_xy_fromz[...,0].mean()} | proj_cloud_color_xy[...,0].mean() {proj_cloud_color_xy[...,0].mean()} proj_cloud_color_xy_fromz[...,1].mean() {proj_cloud_color_xy_fromz[...,1].mean()} | proj_cloud_color_xy[...,1].mean() {proj_cloud_color_xy[...,1].mean()} ')
            # st()
            print(f'(proj_cloud_color_xy_fromz[:,:,0]==0).sum() {(proj_cloud_color_xy_fromz[:,:,0]==0).sum()} | (proj_cloud_color_xy[:,:,0]==0).sum() {(proj_cloud_color_xy[:,:,0]==0).sum()}')
            

            proj_cloud_show_x1 = proj_cloud_color_xy[:, :, 0]
            proj_cloud_show_x1_norm = proj_cloud_show_x1 / proj_cloud_show_x1.max()
            proj_cloud_show_x1_255 = (proj_cloud_show_x1_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x1_color =  cv.applyColorMap(proj_cloud_show_x1_255, cv.COLORMAP_JET)
            # proj_cloud_show_x_color2 =  cv.applyColorMap(proj_cloud_show_x, cv.COLORMAP_PARULA)

            cv.namedWindow("proj_cloud_show_x_color1",0)
            cv.resizeWindow("proj_cloud_show_x_color1", 960, 540)
            cv.imshow('proj_cloud_show_x_color1', proj_cloud_show_x1_color)

            if 113 == cv.waitKey(100):
                st()
            
            proj_cloud_show_x2 = proj_cloud_color_xy_fromz[:, :, 0]
            proj_cloud_show_x2_norm = proj_cloud_show_x2 / proj_cloud_show_x2.max()
            proj_cloud_show_x2_255 = (proj_cloud_show_x2_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x2_color =  cv.applyColorMap(proj_cloud_show_x2_255, cv.COLORMAP_JET)

            cv.namedWindow("proj_cloud_show_x_color2",0)
            cv.resizeWindow("proj_cloud_show_x_color2", 960, 540)
            cv.imshow('proj_cloud_show_x_color2', proj_cloud_show_x2_color)
            if 113 == cv.waitKey(100):
                st()
            
            # st()
            mask1 = (proj_cloud_show_x1==0)
            mask2 = (proj_cloud_show_x2==0)
            print(mask1==mask2)


            proj_cloud_show_x3 = proj_cloud_color_xy[:, :, 0] - proj_cloud_color_xy_fromz[:, :, 0]
            print(f'proj_cloud_show_x3.mean()  {proj_cloud_show_x3.mean()}')

            proj_cloud_show_x3_norm = proj_cloud_show_x3 / proj_cloud_show_x3.max()
            proj_cloud_show_x3_255 = (proj_cloud_show_x3_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x3_color =  cv.applyColorMap(proj_cloud_show_x3_255, cv.COLORMAP_JET)

            cv.namedWindow("proj_cloud_show_x_color3",0)
            cv.resizeWindow("proj_cloud_show_x_color3", 960, 540)
            cv.imshow('proj_cloud_show_x_color3', proj_cloud_show_x3_color)
            if 113 == cv.waitKey(100):
                st()

            if abs(proj_cloud_show_x3).mean() > 0.001:
                print('dist too large')
            # 我想知道这个差距的原因是因为每一个点还是因为某个别的noise
            # if abs(proj_cloud_show_x3).mean() > 0.001:
            #     # (proj_cloud_show_x3 > 0.01).sum()
            #     # (proj_cloud_show_x3 > -100000).sum()

            #     print(f'(mask1==False).sum() {(mask1==False).sum()}')
            #     cnt = 0 
            #     for i in range(mask1.shape[0]):
            #         for j in range(mask1.shape[1]):
            #             if not mask1[i, j]:
            #                 print(f'x1 {proj_cloud_show_x1[i, j]} x2 {proj_cloud_show_x2[i, j]} abs(x1-x2) {abs(proj_cloud_show_x1[i, j] - proj_cloud_show_x2[i, j])}')
            #                 cnt += 1
            #                 if cnt > 1000:
            #                     st()

        # aligned_depth = proj_cloud_color[:, :, 2]
        # print(f'{aligned_depth.max()} {aligned_depth.mean()} {aligned_depth.min()}')
        # 8.008048494015108 0.31072447941348585 0.0
        # 基本上最远的depth为8

        return proj_cloud, mask, aligned_depth

     ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
     ### kinect相机坐标系 * rgb参数 = rgb相机空间
     ### kinect相机坐标系 * depth参数 = depth相机空间
     ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
    def depth_to_color_cloud_w_aligned_depth_w_cloud_color(self, depth, shape, calib):

        X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        d_K, d_d = calib.depth_proj()
        R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        Kdist = d_K
        Mdist = T_M_depth[:3,:]
        distCoeffs = d_d
        lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        p3d = self.unproject(lcc_p2d, Kdist, Mdist, distCoeffs)
        cloud_local = p3d

        ####### matlab project points
        R_color, t_color, T_M_color = calib.lcc_M_color_all()
        c_K, c_d = calib.color_proj()
        img_pts = self.matlab_poseproject2d(cloud_local, R_color, t_color, c_K, c_d)
        img_pts = np.array(img_pts + 0.5).astype(np.int32)
        

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
        


        point3d_panopticWorld = T_kinectLocal2PanopticWorld.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_panopticWorld = point3d_panopticWorld[:3, :].T

        cloud_world = point3d_panopticWorld


        # delete points out of image space
        # st()
        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud_world, shape[:2])
        # print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        # (Pdb) cloud_filter.shape
        # (30445, 3)
        # (Pdb) img_pts_filter.shape
        # (30445, 2)
        # st()
        # proj_cloud其实就是image u,v 到 3d point xyz的映射
        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter

        # st()
        # proj_cloud = np.flip(proj_cloud, axis=0)

        mask = np.multiply((proj_cloud[:,:,0]==0), (proj_cloud[:,:,1]==0))
        mask = np.multiply(mask, (proj_cloud[:,:,2]==0))
        
        # st()
        # print(f'proj_cloud[:,:,0].mean() {proj_cloud[:,:,0].mean()}')
        # print(f'proj_cloud[:,:,1].mean() {proj_cloud[:,:,1].mean()}')
        # print(f'proj_cloud[:,:,2].mean() {proj_cloud[:,:,2].mean()}')


        # 相机坐标系到世界坐标系
        # proj_cloud = self.lcc_camera_to_world(proj_cloud.reshape(-1,3), calib).reshape(proj_cloud.shape)
        # proj_cloud_2 = self.joints_to_color(proj_cloud_1.reshape(-1,3), calib).reshape(proj_cloud_1.shape)

        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
        # show_np_image(proj_cloud_show_z_color1)
        # show_np_image(proj_cloud_show_z_color2)


        ### lcc add:cloud_color
        point3d_kinectcolor = T_kinectLocal2KinectColor.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_kinectcolor = point3d_kinectcolor[:3, :].T
        

        cloud_color = point3d_kinectcolor

        img_pts_filter_color, cloud_filter_color = self._filter_pts(img_pts, cloud_color, shape[:2])
        proj_cloud_color = np.zeros(shape=shape)
        proj_cloud_color[img_pts_filter_color[:, 1], img_pts_filter_color[:, 0], :] = cloud_filter_color

        # proj_cloud_color = np.flip(proj_cloud_color, axis=0)


        aligned_depth = proj_cloud_color[:, :, 2]


        ########################################### 下面就是反投影的部分 ###########################################
        
        # ### lcc debugging 注意！！！！
        # proj_cloud_color[:, :, 2] = proj_cloud_color[:, :, 2] * 1e3
        # depth = proj_cloud_color[:, :, 2] * 1e1

        # #### z -> xy
        # proj_cloud_color_xy = proj_cloud_color[:, :, :2]
        # proj_cloud_color_xy_fromz = np.zeros(shape=proj_cloud_color[:, :, :2].shape)


        # ### 感觉区别就是matlab版本的误差是噪声的，python版本是平滑的，感觉还是matlab可能靠谱一些
        # ### 用matlab原汁原味
        # ### 我突然在想在做这个下面之前depth是不是应该是左右调换？
        # # depth = depth * 1e4
        # # print(f'2depth.mean(){depth.mean()}')
        # X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        # p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        # # color相机的内参
        # c_K, c_d = calib.color_proj()
        # R_color, t_color, T_M_color = calib.lcc_M_color()

        # # d_K, d_d = calib.depth_proj()
        # # R, t = calib.k_depth_color()
        # # R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        # # st()
        # Kdist = c_K
        # Mdist = T_M_color[:3,:]
        # # distCoeffs = c_d
        # distCoeffs = c_d[:,0]
        # lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        # p3d = self.unproject2(lcc_p2d, Kdist, Mdist, distCoeffs)
        # proj_cloud_color_xy_fromz = p3d.reshape(depth.shape[0], depth.shape[1], 3)[:, :, :2] * 1e-1
        
        
        if vis_flag:
            print(f'proj_cloud_color_xy_fromz[...,0].mean() {proj_cloud_color_xy_fromz[...,0].mean()} | proj_cloud_color_xy[...,0].mean() {proj_cloud_color_xy[...,0].mean()} proj_cloud_color_xy_fromz[...,1].mean() {proj_cloud_color_xy_fromz[...,1].mean()} | proj_cloud_color_xy[...,1].mean() {proj_cloud_color_xy[...,1].mean()} ')
            # st()
            print(f'(proj_cloud_color_xy_fromz[:,:,0]==0).sum() {(proj_cloud_color_xy_fromz[:,:,0]==0).sum()} | (proj_cloud_color_xy[:,:,0]==0).sum() {(proj_cloud_color_xy[:,:,0]==0).sum()}')
            

            proj_cloud_show_x1 = proj_cloud_color_xy[:, :, 0]
            proj_cloud_show_x1_norm = proj_cloud_show_x1 / proj_cloud_show_x1.max()
            proj_cloud_show_x1_255 = (proj_cloud_show_x1_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x1_color =  cv.applyColorMap(proj_cloud_show_x1_255, cv.COLORMAP_JET)
            # proj_cloud_show_x_color2 =  cv.applyColorMap(proj_cloud_show_x, cv.COLORMAP_PARULA)

            cv.namedWindow("proj_cloud_show_x_color1",0)
            cv.resizeWindow("proj_cloud_show_x_color1", 960, 540)
            cv.imshow('proj_cloud_show_x_color1', proj_cloud_show_x1_color)

            if 113 == cv.waitKey(100):
                st()
            
            proj_cloud_show_x2 = proj_cloud_color_xy_fromz[:, :, 0]
            proj_cloud_show_x2_norm = proj_cloud_show_x2 / proj_cloud_show_x2.max()
            proj_cloud_show_x2_255 = (proj_cloud_show_x2_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x2_color =  cv.applyColorMap(proj_cloud_show_x2_255, cv.COLORMAP_JET)

            cv.namedWindow("proj_cloud_show_x_color2",0)
            cv.resizeWindow("proj_cloud_show_x_color2", 960, 540)
            cv.imshow('proj_cloud_show_x_color2', proj_cloud_show_x2_color)
            if 113 == cv.waitKey(100):
                st()
            
            # st()
            mask1 = (proj_cloud_show_x1==0)
            mask2 = (proj_cloud_show_x2==0)
            print(mask1==mask2)


            proj_cloud_show_x3 = proj_cloud_color_xy[:, :, 0] - proj_cloud_color_xy_fromz[:, :, 0]
            print(f'proj_cloud_show_x3.mean()  {proj_cloud_show_x3.mean()}')

            proj_cloud_show_x3_norm = proj_cloud_show_x3 / proj_cloud_show_x3.max()
            proj_cloud_show_x3_255 = (proj_cloud_show_x3_norm * 255.0).astype(np.uint8) 
            proj_cloud_show_x3_color =  cv.applyColorMap(proj_cloud_show_x3_255, cv.COLORMAP_JET)

            cv.namedWindow("proj_cloud_show_x_color3",0)
            cv.resizeWindow("proj_cloud_show_x_color3", 960, 540)
            cv.imshow('proj_cloud_show_x_color3', proj_cloud_show_x3_color)
            if 113 == cv.waitKey(100):
                st()

            if abs(proj_cloud_show_x3).mean() > 0.001:
                print('dist too large')
            # 我想知道这个差距的原因是因为每一个点还是因为某个别的noise
            # if abs(proj_cloud_show_x3).mean() > 0.001:
            #     # (proj_cloud_show_x3 > 0.01).sum()
            #     # (proj_cloud_show_x3 > -100000).sum()

            #     print(f'(mask1==False).sum() {(mask1==False).sum()}')
            #     cnt = 0 
            #     for i in range(mask1.shape[0]):
            #         for j in range(mask1.shape[1]):
            #             if not mask1[i, j]:
            #                 print(f'x1 {proj_cloud_show_x1[i, j]} x2 {proj_cloud_show_x2[i, j]} abs(x1-x2) {abs(proj_cloud_show_x1[i, j] - proj_cloud_show_x2[i, j])}')
            #                 cnt += 1
            #                 if cnt > 1000:
            #                     st()

        # aligned_depth = proj_cloud_color[:, :, 2]
        # print(f'{aligned_depth.max()} {aligned_depth.mean()} {aligned_depth.min()}')
        # 8.008048494015108 0.31072447941348585 0.0
        # 基本上最远的depth为8

        return proj_cloud, mask, aligned_depth, proj_cloud_color


    def depth_to_color_cloud3(self, depth, shape, calib):
        

        # import scipy.io
        # matfn='/data/lichunchi/pose/voxelpose-pytorch/depth.mat'
        # data = scipy.io.loadmat(matfn) # 假设文件名为trainset.mat
        # print(data.keys())
        # # 计算过程是没有问题的，问题出在p2d不同，是flatten的时候先行后列出了问题，还是depth不同造成的问题呢？
        # # st()
        # depth = data['depth']
        # 看来是depth不同造成的问题，不是是flatten的时候先行后列出了问题


        
        X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # 注意matlab展开成一维是先列后行，python是先行，mdzz，我估计应该影响不大，毕竟坐标对应都是一样的
        p2dd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        d_K, d_d = calib.depth_proj()
        R_depth, t_depth, T_M_depth = calib.lcc_M_depth()
        Kdist = d_K
        Mdist = T_M_depth[:3,:]
        distCoeffs = d_d
        lcc_p2d = np.concatenate([p2dd, depth.reshape(-1, 1)], axis=1)
        p3d = self.unproject(lcc_p2d, Kdist, Mdist, distCoeffs)
        cloud_local = p3d
        # st()


        # st()
        # import scipy.io
        # matfn='/data/lichunchi/pose/voxelpose-pytorch/p2d.mat'
        # data = scipy.io.loadmat(matfn) # 假设文件名为trainset.mat
        # print(data.keys())
        # # 计算过程是没有问题的，问题出在p2d不同，是flatten的时候先行后列出了问题，还是depth不同造成的问题呢？
        # p2d = data['p2d']
        # st()

        # p3d = self.unproject(p2d, Kdist, Mdist, distCoeffs)
        # cloud_local = p3d


        # dfx, dfy, dcx, dcy = d_K[0, 0], d_K[1, 1], d_K[0, 2], d_K[1, 2]

        # d_d = d_d[:5]

        # # undistort注释之后会右偏得不那么严重
        # depth = cv.undistort(depth, d_K, d_d) / 1e3
        # # depth = depth / 1e3

        # ###### depth image -> 3d point cloud_local
        # # depth取值范围为1.5~3之间 应该基本上都是人了
        # # 这里就是小孔成像，相似三角形
        # # cloud_local = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1]) if depth[v, u] < 3 and depth[v, u] > 1.5])
        # # depth取值无限制
        # cloud_local = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1])])
        # # (Pdb) cloud_local.shape
        # # (217088, 3)


        # # 为什么这里没有先用M_depth转化为世界坐标系，而是直接用M_color?
        # #我试一下先用M_depth转化为世界坐标系，的结果
        # # cloud_local = (cloud_local - t_depth.reshape(1, 3)) @ R_depth
        # # cloud_local = cloud_local @ np.linalg.inv(R_depth) + t_depth.reshape(1, 3)
        # # st()
        # import scipy.io
        # matfn='/data/lichunchi/pose/voxelpose-pytorch/p3d.mat'
        # data = scipy.io.loadmat(matfn) # 假设文件名为trainset.mat
        # print(data.keys())
        # cloud_local = data['p3d']

        # 感觉应该是实锤了，就是这个之前投影的不对劲
        # st()
        # print(cloud_local.mean(axis=0))
        # todo 投影到世界坐标系

        ######## org project_pts
        R, t = calib.k_depth_color()
        ## 居然是原本的kinoptic-dataset-tools这里R_color进行了改动造成的问题，md
        ## 看来更有必要还原原汁原味的matlab代码了
        R_color, t_color, T_M_color = calib.lcc_M_color_all()
        cloud_forproj = cloud_local @ np.linalg.inv(R_color) + t_color.reshape(1, 3)
        # cloud_world = self.lcc_camera_to_world(cloud_local, calib)
        # st()
        print(cloud_local.mean(axis=0))
        print(cloud_forproj.mean(axis=0))

        # ###### 3d point cloud_local -> rgb image
        img_pts = self.project_pts(cloud_forproj, calib)
        # 一直到这里都是一致的
        # print(img_pts.mean(axis=0))
        img_pts = np.array(img_pts + 0.5).astype(np.int32)


        ####### matlab project points
        # R_color, t_color, T_M_color = calib.lcc_M_color_all()
        # c_K, c_d = calib.color_proj()
        # img_pts = self.matlab_poseproject2d(cloud_local, R_color, t_color, c_K, c_d)
        # img_pts = np.array(img_pts + 0.5).astype(np.int32)


        panoptic_calibData_R, panoptic_calibData_t = calib.lcc_panoptic_calibData()
        M = np.concatenate([panoptic_calibData_R, panoptic_calibData_t], axis=1)
        T_panopticWorld2KinectColor = np.row_stack([M, [0,0,0,1]])
        T_kinectColor2PanopticWorld = np.linalg.inv(T_panopticWorld2KinectColor)
        
        scale_kinoptic2panoptic = np.eye(4)
        scaleFactor = 100
        scale_kinoptic2panoptic[:3,:3] = scaleFactor*scale_kinoptic2panoptic[:3,:3]

        R_color, t_color, T_M_color = calib.lcc_M_color()
        T_kinectColor2KinectLocal = T_M_color
        T_kinectLocal2KinectColor = np.linalg.inv(T_kinectColor2KinectLocal) 
          
        T_kinectLocal2PanopticWorld =  T_kinectColor2PanopticWorld.dot(scale_kinoptic2panoptic).dot(T_kinectLocal2KinectColor)
        

        point3d_panopticWorld = T_kinectLocal2PanopticWorld.dot(np.concatenate([cloud_local.T, np.ones((1, cloud_local.shape[0]))], axis=0))
        point3d_panopticWorld = point3d_panopticWorld[:3, :].T

        cloud_world = point3d_panopticWorld


        # delete points out of image space
        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud_world, shape[:2])
        # print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        # (Pdb) cloud_filter.shape
        # (30445, 3)
        # (Pdb) img_pts_filter.shape
        # (30445, 2)
        # st()
        # proj_cloud其实就是image u,v 到 3d point xyz的映射
        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter

        # st()
        proj_cloud = np.flip(proj_cloud, axis=0)

        mask = np.multiply((proj_cloud[:,:,0]==0), (proj_cloud[:,:,1]==0))
        mask = np.multiply(mask, (proj_cloud[:,:,2]==0))
        # st()

        # 相机坐标系到世界坐标系
        # proj_cloud = self.lcc_camera_to_world(proj_cloud.reshape(-1,3), calib).reshape(proj_cloud.shape)
        # proj_cloud_2 = self.joints_to_color(proj_cloud_1.reshape(-1,3), calib).reshape(proj_cloud_1.shape)

        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
        # show_np_image(proj_cloud_show_z_color1)
        # show_np_image(proj_cloud_show_z_color2)
        
        return proj_cloud, mask


    def depth_to_color_cloud2(self, depth, shape, calib):
        # depth相机的内参
        d_K, d_d = calib.depth_proj()
        
        # color的外参
        R, t = calib.k_depth_color()

        dfx, dfy, dcx, dcy = d_K[0, 0], d_K[1, 1], d_K[0, 2], d_K[1, 2]

        d_d = d_d[:5]

        # undistort注释之后会右偏得不那么严重
        depth = cv.undistort(depth, d_K, d_d) / 1e3
        # depth = depth / 1e3

        ###### depth image -> 3d point cloud
        # depth取值范围为1.5~3之间 应该基本上都是人了
        # 这里就是小孔成像，相似三角形
        # cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1]) if depth[v, u] < 3 and depth[v, u] > 1.5])
        # depth取值无限制
        cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1])])
        # (Pdb) cloud.shape
        # (217088, 3)

        print(cloud.mean(axis=0))


        # 为什么这里没有先用M_depth转化为世界坐标系，而是直接用M_color?
        #我试一下先用M_depth转化为世界坐标系，的结果
        # cloud = (cloud - t_depth.reshape(1, 3)) @ R_depth
        # cloud = cloud @ np.linalg.inv(R_depth) + t_depth.reshape(1, 3)
        # st()
        # import scipy.io
        # matfn='/data/lichunchi/pose/voxelpose-pytorch/p3d.mat'
        # data = scipy.io.loadmat(matfn) # 假设文件名为trainset.mat
        # print(data.keys())
        # cloud = data['p3d']

        # 感觉应该是实锤了，就是这个之前投影的不对劲
        # st()

        

        cloud = cloud @ np.linalg.inv(R) + t.reshape(1, 3)

        ###### 3d point cloud -> rgb image
        img_pts = self.project_pts(cloud, calib)
        img_pts = np.array(img_pts + 0.5).astype(np.int32)

        # delete points out of image space
        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud, shape[:2])
        # print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        # (Pdb) cloud_filter.shape
        # (30445, 3)
        # (Pdb) img_pts_filter.shape
        # (30445, 2)
        # st()
        # proj_cloud其实就是image u,v 到 3d point xyz的映射
        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter


        # mask应该在相机坐标系到世界坐标系之前计算
        mask = np.multiply((proj_cloud[:,:,0]==0), (proj_cloud[:,:,1]==0))
        mask = np.multiply(mask, (proj_cloud[:,:,2]==0))

        # 相机坐标系到世界坐标系
        # proj_cloud = self.lcc_camera_to_world(proj_cloud.reshape(-1,3), calib).reshape(proj_cloud.shape)
        # proj_cloud_2 = self.joints_to_color(proj_cloud_1.reshape(-1,3), calib).reshape(proj_cloud_1.shape)

        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
        # show_np_image(proj_cloud_show_z_color1)
        # show_np_image(proj_cloud_show_z_color2)
        
        return proj_cloud, mask

    def _filter_pts(self, img, cloud, shape):
        img_pts, cloud_pts = img, cloud
        h, w = shape

        for _idx in [
                lambda pts: 0 < pts[:, 0],
                lambda pts: pts[:, 0] < w,
                lambda pts: 0 < pts[:, 1],
                lambda pts: pts[:, 1] < h
                ]:
            idx = np.array(_idx(img_pts))
            img_pts = img_pts[idx]
            cloud_pts = cloud_pts[idx]

        return img_pts, cloud_pts
    
    # rgb内参
    def project_pts(self, cloud, calib):
        K, d = calib.color_proj()
        proj_pts, _ = cv.projectPoints(cloud, np.zeros(3), np.zeros(3), K, d)
        proj_pts = proj_pts.reshape(-1, 2)
        return proj_pts

    # 世界坐标系到相机坐标系
    def joints_to_color(self, joints, calib):
        R, t = calib.joints_k_color()

        joints = joints[:, :3]

        joints = joints @ np.linalg.inv(R) + t.reshape(1, 3)
        return joints

    # 相机坐标系到世界坐标系
    def lcc_camera_to_world(self, joints, calib):
        R, t = calib.joints_k_color()

        joints = joints[:, :3]

        joints = (joints - t.reshape(1, 3)) @ R 
        # st()
        # joints = joints @ np.linalg.inv(R) + t.reshape(1, 3)
        return joints

    def _check(self, seq):
        
        for i in range(len(self._sync)):
            bodies_idx = self._sync[i]
            # st()
            if bodies_idx is None:
                break
            bodies, d_id_list, c_id_list = bodies_idx
            

            continue_flag = False
            # self._sync其实是没有进行过滤的，d_id[1]，c_id[1]可能有很多-1，这里过滤
            for index in range(len(d_id_list)):
                d_id = d_id_list[index]
                c_id = c_id_list[index]

                # st()
                # print(len(d_id_list))
                # print(index)
                
                # st()
                depth_len = len(self._depth_reader_list[seq][index])
                color_len = len(self._color_reader_list[seq][index])

                # print(f'd_id:{d_id}, c_id:{c_id}')
                # print(f'depth_len:{depth_len}, color_len:{color_len}')

                if d_id >= depth_len or c_id >= color_len:
                    # st()
                    continue_flag = True
                    break
            
            if continue_flag:
                continue

            depth_list = d_id_list
            color_list = c_id_list

            # st()
            yield (bodies, depth_list, color_list)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        

        # st()
    
        for seq in self.sequence_list:

            # 选出需要的相机的参数
            cameras = self._get_cam(seq)

            # st()
            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')

            # Voxel代码需要给定一个标注，找到所有node对应的depth image和rgb image 需要改一下panoptic-dataset-tools同步的代码
            selected_cams = self.CAMS[:self.num_views]
            
            seq_root = osp.join(self.dataset_root, seq)
            
            self._sync = SyncReader(path.join(seq_root, f'ksynctables_{seq}.json'), \
                                    path.join(seq_root, 'hdPose3d_stage1_coco19'), selected_cams)
            
            
            self._data = list(self._check(seq))

            # st()

            print(f'seq : {seq}, self._sync : {len(self._sync)}, self._data : {len(self._data)}')

            # st()


            for idx, _data_item in enumerate(self._data):
                
                # lcc debugging
                # if idx < 50:continue
                # debugging color index 1174
                # set shuffle=False
                # set interval=1 

                # python序号从0开始，-1对齐
                # if idx < 1000+100*0-1:continue

                if idx % self._interval == 0:
                    # st()
                    bodies, depth_id_list, color_id_list = _data_item
                    # bodies, depth_id_list, color_id_list = self._data[100] # lcc debugging


                    # # panoptic-dataset-tools 中的投影
                    # # for vis
                    # # img_list_to_show = []

                    # # 在db中存数据需要保证每次存储的，要么是连续的num_view个要么是0个，不能出现其他数字否则会乱掉
                    
                    # _calib_list_seq = self._calib_list[seq]

                    # for k, v in cameras.items():
                        
                    #     camera_index = k[1] - 1 
                    #     _calib = _calib_list_seq[camera_index]

                    #     # st()

                    #     all_poses_3d = []
                    #     all_poses_vis_3d = []
                    #     all_poses = []
                    #     all_poses_vis = []

                    #     for body in bodies:
                            
                    #         pose3d = body
                    #         # st()
                    #         joints = self.joints_to_color(pose3d[:, 0:3], _calib)
                    #         # print(joints.shape)
                    #         # print(joints)
                    #         proj_joints = self.project_pts(joints, _calib)
                    #         # print(proj_joints.shape)
                    #         # print(proj_joints)


                    #         pose3d = pose3d[:self.num_joints]

                    #         # Coordinate transformation
                    #         M = np.array([[1.0, 0.0, 0.0],
                    #                     [0.0, 0.0, -1.0],
                    #                     [0.0, 1.0, 0.0]])
                    #         # pose3d[:, 0:3].dot(M).dot(np.linalg.inv(M))
                    #         pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                    #         proj_joints = proj_joints[:self.num_joints]

                    #         # st()
                    #         # 注意这里因为之前/100这里也/100
                    #         joints_vis = pose3d[:, -1] > (0.1 / 100)

                    #         # 这里再*100来保持原样
                    #         all_poses_3d.append(pose3d[:, 0:3] * 10.0 * 100) 
                    #         all_poses_vis_3d.append(
                    #             np.repeat(
                    #                 np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                    #         if not joints_vis[self.root_id]:
                    #             continue
                            
                    #         # 计算pose2d
                    #         # # Coordinate transformation
                    #         # M = np.array([[1.0, 0.0, 0.0],
                    #         #               [0.0, 0.0, -1.0],
                    #         #               [0.0, 1.0, 0.0]])
                    #         # pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                    #         # all_poses_3d.append(pose3d[:, 0:3] * 10.0) 
                    #         # all_poses_vis_3d.append(
                    #         #     np.repeat(
                    #         #         np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                    #         # pose2d = np.zeros((pose3d.shape[0], 2))

                    #         # # 2d 坐标是通过投影算出来的，用之前先可视化好
                    #         # pose2d[:, :2] = projectPoints(
                    #         #     pose3d[:, 0:3].transpose(), v['K'], v['R'],
                    #         #     v['t'], v['distCoef']).transpose()[:, :2]

                    #         pose2d = proj_joints

                    #         # 只要不超出image就是可见
                    #         x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                    #                                 pose2d[:, 0] <= width - 1)
                    #         y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                    #                                 pose2d[:, 1] <= height - 1)
                    #         check = np.bitwise_and(x_check, y_check)

                    #         joints_vis[np.logical_not(check)] = 0

                    #         all_poses.append(pose2d)
                    #         all_poses_vis.append(
                    #             np.repeat(
                    #                 np.reshape(joints_vis, (-1, 1)), 2, axis=1))
                        
                    #     # if len(all_poses_3d) > 0 and len(all_poses) == 0:
                    #     #     # 居然会这样
                    #     #     st()

                    #     if len(all_poses_3d) > 0 and len(all_poses) > 0:
                    #         our_cam = {}
                    #         our_cam['R'] = v['R']
                    #         our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                    #         our_cam['fx'] = np.array(v['K'][0, 0])
                    #         our_cam['fy'] = np.array(v['K'][1, 1])
                    #         our_cam['cx'] = np.array(v['K'][0, 2])
                    #         our_cam['cy'] = np.array(v['K'][1, 2])
                    #         our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                    #         our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                            
                    #         # 这里检查和panoptic出来的一致性
                    #         db.append({
                    #             'key': "",
                    #             'image': "",
                    #             'joints_3d': all_poses_3d,
                    #             'joints_3d_vis': all_poses_vis_3d,
                    #             'joints_2d': all_poses,
                    #             'joints_2d_vis': all_poses_vis,
                    #             'camera': our_cam, 
                    #             'camera_index': camera_index, # 这之后的是为了读取image和depth的
                    #             'depth_index': depth_id_list[camera_index], 
                    #             'color_index': color_id_list[camera_index], 
                    #             'seq': seq, 
                    #             'dataset_name': 'kinoptic'
                    #         })
                            
                    #         # st()
                    #         # print(db[-1]['joints_3d'])
                    #         # print(db[-1]['joints_2d'])
                    #         # st()

                    #         if idx % 1 == 0:
                    #             # if idx > 8000 and idx % 2 == 0:
                    #             # if idx > 3700 and idx % 2 == 0:
                    #             # new_shape = (1920 // 4, 1080 // 4)
                    #             # for vis
                    #             # img_list_to_show.append(self._color_reader_list[seq][camera_index][color_id_list[camera_index]])
                    #             print(all_poses)
                    #             img = self._color_reader_list[seq][camera_index][color_id_list[camera_index]]
                    #             for a_pose in all_poses:
                    #                 for point_i in range(15):
                    #                     # img = cv.putText(img, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
                    #                     img = cv.circle(img, (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 2, (255, 255, 0), 2)
                    #                     img = cv.putText(img, f'{point_i}', (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 0, 1, (255, 0, 255), 2)
                    #                     img = cv.putText(img, f'idx:{idx}', (30,30), 0, 1, (255, 0, 255), 2)

                    #             cv.namedWindow("color",0)
                    #             cv.resizeWindow("color", 960, 540)
                    #             cv.imshow('color', img)
                                
                    #             if 113 == cv.waitKey(100):
                    #                 st()


                    ###### voxelpose投影的代码中只有少量view正确
                    # for vis
                    # img_list_to_show = []
                    # 在db中存数据需要保证每次存储的，要么是连续的num_view个要么是0个，不能出现其他数字否则会乱掉

                    # st()
                    # print(cameras)
                    # print(bodies)

                    camera_index = 0
                    for k, v in cameras.items():

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        # 3d joint -> 2d joint
                        # array([[-109.202   , -146.703   ,    0.73081 ,    0.681366],
                        # [-109.531   , -166.822   ,  -16.5489  ,    0.664246],
                        # [-108.344   ,  -91.629   ,    1.34021 ,    0.413544],
                        # [ -97.9078  , -146.878   ,  -11.9947  ,    0.615387],
                        # [ -93.9524  , -118.337   ,  -15.5165  ,    0.651825],
                        # [-102.788   ,  -93.7093  ,  -19.8694  ,    0.635406],
                        # [-103.064   ,  -91.7136  ,   -7.53898 ,    0.420196],
                        # [-103.74    ,  -49.1241  ,   -8.26781 ,    0.471039],
                        # [-102.667   ,   -8.5043  ,   -9.90811 ,    0.6344  ],
                        # [-121.878   , -147.223   ,   13.6519  ,    0.580506],
                        # [-125.404   , -118.028   ,   16.9511  ,    0.399444],
                        # [-132.818   ,  -93.4912  ,    9.92613 ,    0.288025],
                        # [-113.625   ,  -91.5443  ,   10.2194  ,    0.391907],
                        # [-113.868   ,  -47.4008  ,    8.62231 ,    0.509888],
                        # [-114.357   ,   -6.72679 ,   10.5661  ,    0.651703],
                        # [-106.879   , -170.176   ,  -14.174   ,    0.629914],
                        # [-103.618   , -168.129   ,   -4.48493 ,    0.571411],
                        # [-112.644   , -170.398   ,  -14.9105  ,    0.415497],
                        # [-118.67    , -168.921   ,   -6.39634 ,    0.201447]])
                        for body in bodies:
                            
                            # 我觉得这里对于pose3d的改变也改变了body，所以在多视角的时候只有第一个视角是对的
                            # 果然是变了！天哪！
                            # 1
                            # [ 1.44041e+02 -1.28982e+02  4.44952e+01  6.25183e-01  1.38594e+02
                            # -1.43489e+02  2.91097e+01  6.17493e-01  1.46541e+02 -8.64528e+01
                            # 5.82469e+01  5.44250e-01  1.53823e+02 -1.27412e+02  3.80593e+01
                            # 5.73059e-01  1.47420e+02 -1.04368e+02  4.18614e+01  3.98804e-01
                            # 1.32429e+02 -1.14710e+02  3.30617e+01  2.49939e-01  1.53692e+02
                            # -8.53041e+01  5.40476e+01  5.07446e-01  1.50002e+02 -4.86249e+01
                            # 5.02555e+01  5.32166e-01  1.54491e+02 -1.00227e+01  5.07923e+01
                            # 5.06165e-01  1.32163e+02 -1.30464e+02  4.95971e+01  6.06079e-01
                            # 1.33064e+02 -1.03642e+02  4.97073e+01  5.00916e-01  1.26388e+02
                            # -9.74391e+01  3.16178e+01  4.18091e-01  1.39390e+02 -8.76015e+01
                            # 6.24462e+01  5.23682e-01  1.32146e+02 -5.24725e+01  5.35686e+01
                            # 5.47180e-01  1.33518e+02 -1.24042e+01  6.06574e+01  5.48035e-01
                            # 1.41653e+02 -1.46690e+02  2.89937e+01  3.20740e-01  1.49177e+02
                            # -1.43011e+02  3.43536e+01  1.05774e-01  1.37161e+02 -1.46736e+02
                            # 3.14794e+01  5.80505e-01  1.36980e+02 -1.44714e+02  3.98771e+01
                            # 5.84045e-01]
                            # 2
                            # [ 1.44041e+02  4.44952e+01  1.28982e+02  6.25183e-01  1.38594e+02
                            # 2.91097e+01  1.43489e+02  6.17493e-01  1.46541e+02  5.82469e+01
                            # 8.64528e+01  5.44250e-01  1.53823e+02  3.80593e+01  1.27412e+02
                            # 5.73059e-01  1.47420e+02  4.18614e+01  1.04368e+02  3.98804e-01
                            # 1.32429e+02  3.30617e+01  1.14710e+02  2.49939e-01  1.53692e+02
                            # 5.40476e+01  8.53041e+01  5.07446e-01  1.50002e+02  5.02555e+01
                            # 4.86249e+01  5.32166e-01  1.54491e+02  5.07923e+01  1.00227e+01
                            # 5.06165e-01  1.32163e+02  4.95971e+01  1.30464e+02  6.06079e-01
                            # 1.33064e+02  4.97073e+01  1.03642e+02  5.00916e-01  1.26388e+02
                            # 3.16178e+01  9.74391e+01  4.18091e-01  1.39390e+02  6.24462e+01
                            # 8.76015e+01  5.23682e-01  1.32146e+02  5.35686e+01  5.24725e+01
                            # 5.47180e-01  1.33518e+02  6.06574e+01  1.24042e+01  5.48035e-01
                            # 1.41653e+02 -1.46690e+02  2.89937e+01  3.20740e-01  1.49177e+02
                            # -1.43011e+02  3.43536e+01  1.05774e-01  1.37161e+02 -1.46736e+02
                            # 3.14794e+01  5.80505e-01  1.36980e+02 -1.44714e+02  3.98771e+01
                            # 5.84045e-01]

                            # print('1\n', body)

                            body_tmp = copy.deepcopy(body)

                            pose3d = body_tmp.reshape((-1, 4))

                            # print(pose3d)

                            pose3d = pose3d[:self.num_joints]

                            # st()
                            joints_vis = pose3d[:, -1] > 0.1
                            if not joints_vis[self.root_id]:
                                continue

                            
                            
                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                            # print('2\n', body)
                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)

                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))
                            pose2d = np.zeros((pose3d.shape[0], 2))

                            ### 注意这里投影的2d pose是投影到rgb cam上面去的，而不是kinoptic相机上去的
                            # 2d 坐标是通过投影算出来的，用之前先可视化好
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                                
                            # 只要不超出image就是可见
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0
                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))
                        
                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
                            # st()
                            # camera_index = k[1] - 1 
                            # st()
                            db.append({
                                'key': "",
                                'image': "",
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': our_cam, 
                                'camera_index': camera_index, # 这之后的是为了读取image和depth的
                                'depth_index': depth_id_list[camera_index], 
                                'color_index': color_id_list[camera_index], 
                                'seq': seq, 
                                'dataset_name': 'kinoptic'
                            })

                            if idx % 1 == 0:
                                # if idx > 8000 and idx % 2 == 0:
                                # if idx > 3700 and idx % 2 == 0:
                                # new_shape = (1920 // 4, 1080 // 4)
                                # for vis
                                # img_list_to_show.append(self._color_reader_list[seq][camera_index][color_id_list[camera_index]])
                                print(all_poses)
                                print('\n')
                                img = self._color_reader_list[seq][camera_index][color_id_list[camera_index]]
                                for a_pose in all_poses:
                                    for point_i in range(15):
                                        # img = cv.putText(img, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
                                        img = cv.circle(img, (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 2, (255, 255, 0), 2)
                                        img = cv.putText(img, f'{point_i}', (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 0, 1, (255, 0, 255), 2)
                                        img = cv.putText(img, f'idx:{idx}', (30,30), 0, 1, (255, 0, 255), 2)

                                cv.namedWindow("color",0)
                                cv.resizeWindow("color", 960, 540)
                                cv.imshow('color', img)
                                
                                if 113 == cv.waitKey(100):
                                    st()
                        
                        camera_index += 1
                    
                    # for vis
                    # 感觉没有太大问题
                    # st()
                    # show = np.concatenate(img_list_to_show, axis=1)
                    # cv.namedWindow("color",0)
                    # cv.resizeWindow("color", 1920, 540)
                    # cv.imshow('color', show)
                    # if 113 == cv.waitKey(10):
                    #     st()

        return db

    def _get_cam(self, seq):
        
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json_tricks.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            # 在这用cam_list选出需要的camera
            # st()
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras


    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))
        
        # 重载，一个batch中相当于是一帧的所有view
        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)

        # st()
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views
    
    ### 注意分清楚train和eval，train的时候不会跑这个
    def evaluate(self, preds):
        # 这里直接传进去所有的preds进行evaluate
        
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        ### total gt，其实就是一个全部的gt的index
        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue
            
            ### 对于其中一帧，匹配pred和gt
            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0] ### 选取之前grid_center没有被过滤的

            ### 对于每一个pred pose
            for pose in pred:
                mpjpes = []
                ### 遍历所有的gt pose，找mpjpe最小的gt pose
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    ### 这里我能理解用gt的vis去挑选pose中的，相当于只evaluate可见的pose
                    ### 但是问题是，怎么保证pose和gt是事先一一对应好的呢，难道是做了排序？
                    ### 难道说就是cpn中的proposal_layer中的那个topk就是做的排序？？？马萨卡
                    ### 都不是，而是这个eval循环里面已经写了，对于每一个预测pose，都找所有的gt与他计算mpjpe
                    ### 找最小的mpjpe作为这个预测pose match的gt
                    # st()
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]

                ### 到了这一步就不需要再考虑pred和gt的matching了，因为已经利用pred和gt的matching并且算出来mpjpe了
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt) ### 当前pose匹配的gt的全局index
                })

            total_gt += len(joints_3d)
        
        ### 上面选取之前grid_center没有被过滤的，可以理解
        ### 但是其实我不太明白为什么下面eval_ap的时候为什么要根据score从大到小排序，难道和顺序还有关系吗
        ### 是的，其实这就是匈牙利算法，先找每一个pred最近的gt，此时用的mpjpe作为标准，
        ### 做完后有的gt有匹配的pred，有的可能没有匹配的，有的可能有多个匹配的

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        # st()
        # exit()
        return  aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True) ### 相当于根据score从大到小排序

        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        
        ### 因为eval_list记录的是所有pred对应的gt pose，难免有多个pred对应了相同的gt pose
        ### 所以根据score排序就是为了让score比较大的pred优先占位
        for i, item in enumerate(eval_list):
            ### 注意这个规则就是说只要某一个gt被thres收到tp中去
            ### 就不会用这个gt去
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        # print(f"tp {tp} fp {fp}")
        # st()
        ### 不同，必须有置信度
        # array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  2.,
        #         2.,  2.,  2.,  2.,  3.,  4.,  4.,  4.,  5.,  5.,  5.,  6.,  6.,
        #         6.,  6.,  6.,  7.,  7.,  7.,  7.,  8.,  8.,  8.,  8.,  9.,  9.,
        #         9.,  9., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        #     10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 11., 12., 12.,
        #     12., 12., 12., 12., 12., 12., 12., 12., 12., 13., 13., 13., 13.,
        #     13., 14., 14., 14., 14., 14., 14., 14., 14., 14., 14., 14., 14.,
        #     14., 14., 14., 14., 14., 14., 15., 15., 15., 15., 16., 16., 16.,
        #     16., 17., 17., 17., 17., 17., 17., 17., 18., 18., 19., 19., 19.,
        #     19., 19., 19., 19., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
        #     20., 20., 21., 21., 21., 22., 22., 23., 23., 23., 23., 24., 24.,
        #     24., 24., 25., 25., 25., 25., 25., 25., 25., 25., 25., 25., 25.,
        #     25., 26., 26., 26., 26., 26., 26., 26., 26., 27., 27., 27., 27.,
        #     27., 27., 27., 27., 27., 28., 28., 28., 28., 28., 28., 28., 28.,
        #     28., 28., 28., 28., 29., 29., 29., 29., 29., 29., 30., 30., 30.,
        #     31., 31., 31., 31., 31., 31., 31., 31., 31., 31., 31., 31., 31.,
        #     31., 31., 31., 31., 31., 31., 31., 31., 32., 33., 33., 33., 33.,
        #     34., 34., 34., 34., 34., 34., 34., 34., 35., 35., 36., 37., 37.,
        #     37., 37., 38., 38., 38., 38., 38., 38., 39., 39., 39., 39., 39.,
        #     39., 39., 39., 39., 39., 39., 39., 39., 39., 39., 40., 40., 40.,
        #     40., 40., 40., 40., 41., 41., 41., 41., 41., 41., 42., 42., 42.,
        #     42., 42., 42., 43., 43., 43., 43., 44., 44., 44., 45., 45., 45.,
        #     45., 46., 47., 47., 47., 47., 47., 47., 48., 48., 48., 48., 48.,
        #     49., 49., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 51.,
        #     51., 51., 51., 51., 51., 51., 51., 51., 52., 52., 52., 52., 52.,
        #     52., 52., 52., 52., 53., 53., 53., 53., 53., 53., 53., 54., 54.,
        #     54., 54., 54., 54., 54., 54., 54., 54., 54., 54., 54., 54., 54.,
        #     54., 54., 54., 54., 54., 55., 55., 55., 55., 55., 55., 55., 55.,
        #     55., 55., 55., 55., 55., 55., 55., 55., 55., 55., 55., 55., 55.,
        #     55., 55., 55., 55., 55., 55., 55., 55., 55., 55., 55., 56., 56.,
        #     56., 56., 56., 56., 56., 57., 57., 57., 57., 57., 57., 57., 57.,
        #     57., 57., 57., 57., 57., 57., 57., 57., 58., 58., 58., 58., 58.,
        #     58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58.,
        #     58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58.,
        #     58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58.,
        #     58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58., 58.,
        #     58., 58.])
        
        # tp [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1.
        # 1. 1. 1. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 3. 3. 3. 3.
        # 3. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4.
        # 4. 4. 4. 4. 4. 4. 4. 5. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 6.
        # 6. 6. 6. 6. 6. 6. 6. 6. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7.
        # 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7.
        # 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7.] fp [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
        # 15.  16.  17.  18.  19.  19.  20.  21.  22.  23.  24.  25.  26.  26.
        # 27.  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.
        # 41.  42.  42.  43.  44.  45.  46.  46.  47.  48.  49.  50.  51.  52.
        # 53.  54.  55.  56.  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.
        # 67.  68.  69.  70.  71.  72.  73.  74.  75.  75.  75.  76.  77.  78.
        # 79.  80.  81.  82.  83.  84.  85.  86.  87.  88.  89.  90.  91.  92.
        # 93.  94.  95.  96.  97.  98.  98.  99. 100. 101. 102. 103. 104. 105.
        # 106. 107. 108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118. 119.
        # 120. 121. 122. 123. 124. 125. 126. 127. 128. 129. 130. 131. 132. 133.
        # 134. 135. 136. 137. 138. 139. 140. 141. 142. 143. 144. 145. 146. 147.
        # 148. 149. 150. 151. 152. 153. 154.]

        ### 我感觉无论排序与否，tp和fp的数组可能会变化，但是数组最后那个数字是不会变的
        ### 因为所有的gt只要是被匹配上的都不会漏掉，如果有多个pred对应了同一个gt也只会有一个gt被收到gt_det里面
        ### 但是下面的过程是用cumsum来做的，累加和，那就可能不同了

        # st()
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt


class SyncReader:
    def __init__(self, ksync_file, joint_dir, selected_cams) -> None:
        node_id_list = [cam[1] for cam in selected_cams]
        self._color_time_list = []
        self._depth_time_list = []
        
        for node_id in node_id_list:
            node_name = f'KINECTNODE{node_id}'

            with open(ksync_file) as kf:
                data = json.load(kf)
                # 这个color_time后面有巨量的-1.0的无效的univ_time，我也不知道为什么
                # 搞得一个17分钟的video只能用其中的4分钟

                color_time = data['kinect']['color'][node_name]['univ_time']
                depth_time = data['kinect']['depth'][node_name]['univ_time']
                
                self._color_time = color_time
                self._depth_time = depth_time

                # (Pdb) len(color_time)
                # 29520
                # (Pdb) len(depth_time)
                # 29521
                
                # color_time每两个帧之间的间隔为
                # 975184 ÷ 29520 = 33.03 无单位
                # color_time 总时长975184
                # 总时长为
                # 17.5 * 60 = 1050 秒
                # 所以1s大约无单位的color_time的975184 / 1050 = 928.74 ≈ 1000
                # 可以推测这个无单位的差不多是ms 因此只要color_time和depth_time的差别在1000以内
                # 两帧差距在1s之内

                self._color_time_list.append(self._color_time)
                self._depth_time_list.append(self._depth_time)

        
        assert path.isdir(joint_dir)

        # st()
        # 下面开始读取anno files
        timed_bodies = {}
        joints_file = os.listdir(joint_dir)
        for joint in sorted(joints_file):
            with open(path.join(joint_dir, joint)) as f:
                try:
                    data = json.load(f)
                    time = data['univTime']
                    bodies = data['bodies']
                    if len(bodies) == 0:
                        continue
                    # 这里bodies只取第0个人，改为取list
                    # bodies = np.array(bodies[0]['joints19'])
                    bodies = [np.array(bodies[index]['joints19']) for index in range(len(bodies))] 
                    timed_bodies[time] = bodies
                except:
                    print(f'error occurs while processing {path.join(joint_dir, joint)}, skipping...: ')
                    

        
        # (Pdb) len(timed_bodies)
        # 26603

        self._timed_bodies = timed_bodies
        self._bodies_idx = list(self._sync())

        # st()

    def __len__(self):
        return len(self._bodies_idx)

    def __getitem__(self, idx):
        if idx > len(self._bodies_idx):
            raise Exception('sync out of range')
        
        # voxelpose需要原始的joints
        bodies, did, cid = self._bodies_idx[idx]
        
        # panoptic-dataset-tools
        # bodies = list(map(lambda body: np.array(body).reshape(-1, 4) / 100, bodies))

        # 注意这里panoptic-dataset-tools处理先/100了

        return bodies, did, cid

    @staticmethod
    def _time_iter(time_list):
        idx_time_iter = iter(enumerate(time_list))
        none_val = (-1, 1e10)
        return lambda cur=None: (none_val if cur is None else cur[1], next(idx_time_iter, none_val))
    
    # 从pair的左边和右边选一个最接近的
    @staticmethod
    def _pick(aim_time, pair):
        last, cur = pair
        diff_l, diff_r = aim_time - last[1], cur[1] - aim_time
        target = last if diff_l <= diff_r else cur
        # print(f'pick  {aim_time} {target[1]} ,  {target[1] - aim_time}')
        return target

    @staticmethod
    def _is_in(aim_time, pairs, sync_offset=0):
        last, cur = pairs
        # print('aim', aim_time, 'in', f'{last[1]}, {cur[1]}')
        return (last[1] - sync_offset) < aim_time and (cur[1] - sync_offset) >= aim_time

    def _nearest(self, next_color_list, next_depth_list):
        color_pair_list = []
        depth_pair_list = []

        for index in range(len(next_color_list)):
            color_pair_list.append(next_color_list[index]())
            depth_pair_list.append(next_depth_list[index]())

        for aim_time in self._timed_bodies.keys():
            yield_color_pair_list = []
            yield_depth_pair_list = []
            # 为什么要加32，是不是往往前面的误差比较大，或者是为了过滤前几个-1？
            # 为什么那个代码用32能全部对齐我却不行？？？
            aim_time += 32 # 保证0对齐

            for index in range(len(next_color_list)):
                while not SyncReader._is_in(aim_time, color_pair_list[index], 6.25):
                    color_pair_list[index] = next_color_list[index](color_pair_list[index])
                yield_color_pair_list.append(SyncReader._pick(aim_time, color_pair_list[index]))
                
                while not SyncReader._is_in(aim_time, depth_pair_list[index]):
                    depth_pair_list[index] = next_depth_list[index](depth_pair_list[index])
                yield_depth_pair_list.append(SyncReader._pick(aim_time, depth_pair_list[index]))
            yield yield_color_pair_list, yield_depth_pair_list
    
    # 相当于两个简单的迭代器，每次会返回当前iter所在位置左边和右边的时间的pair
    # self._time_iter(self._color_time)
    # self._time_iter(self._depth_time)

    # 这是一个复合的迭代器，用来找和当前self._timed_bodies，也就是aim_time，最接近的color_pair和depth_pair
    # self._nearest

    def _sync(self):
        # 迭代器list
        color_time_iter_list = [self._time_iter(ct) for ct in self._color_time_list]
        depth_time_iter_list = [self._time_iter(dt) for dt in self._depth_time_list]


        for bodies, (color_list, depth_list) in zip(self._timed_bodies.values(), self._nearest(color_time_iter_list, depth_time_iter_list)):

            # print(f'iter {color_list} {depth_list}')

            end_flag = False
            continue_flag = False
            # 有任何一对出现问题就抛弃这个time？
            # color_list内部考虑同步的问题吗？暂时不考虑先
            for index in range(len(color_list)):
                color = color_list[index]
                depth = depth_list[index]
                if depth[0] == -1 or color[0] == -1:
                    end_flag = True
                    break
                if abs(depth[1] - color[1]) > 6.5:
                    continue_flag = True
                    break
            
            if end_flag:
                break
            if continue_flag:
                continue
            
            yield_color_list = []
            yield_depth_list = []

            # yield_color_list_show = []
            # yield_depth_list_show = []

            for index in range(len(color_list)):
                color = color_list[index]
                depth = depth_list[index]
                

                yield_color_list.append(color[0])
                yield_depth_list.append(depth[0])

                # yield_color_list_show.append(color)
                # yield_depth_list_show.append(depth)

            # st()

            # print(f'yield {yield_depth_list_show} {yield_color_list_show}')
            # 我居然弄反了，wtm找了半天，rnm退钱
            yield (bodies, yield_depth_list, yield_color_list)


# depth img就是424*512的

class DepthReader:
    shape = (424, 512)
    frame_len = 2 * 424 * 512

    def __init__(self, depth_file) -> None:

        self.depth_file = depth_file

        self._frames = open(depth_file, 'rb')
        self._frames.seek(0, 2)
        self._count = self._frames.tell() // DepthReader.frame_len

    def __len__(self):
        return self._count

    def __getitem__org(self, idx):
        # load 相同的depth_id和matlab会得到不同的结果
        # print(f'get depth id /: {idx}/{self._count}')
        # st()

        if idx > self._count:
            raise Exception('depth index out of range')

        self._frames.seek(idx * DepthReader.frame_len, 0)
        raw_data = self._frames.read(DepthReader.frame_len)
        frame = np.frombuffer(raw_data, dtype=np.uint16)
        return np.array(frame).reshape(DepthReader.shape[0], DepthReader.shape[1])

    def __getitem__(self, idx):
        # load 相同的depth_id和matlab会得到不同的结果
        # print(f'get depth id /: {idx}/{self._count}')
        

        if idx > self._count:
            raise Exception('depth index out of range')

        self._frames.seek(idx * DepthReader.frame_len, 0)
        raw_data = self._frames.read(DepthReader.frame_len)
        frame = np.frombuffer(raw_data, dtype=np.uint16)
        ret = np.array(frame).reshape(DepthReader.shape[0], DepthReader.shape[1])


        ret = np.flip(ret,axis=1)


        # 就差一个水平翻转
        # import scipy.io
        # matfn='/data/lichunchi/pose/voxelpose-pytorch/depth.mat'
        # data = scipy.io.loadmat(matfn) # 假设文件名为trainset.mat
        # print(data.keys())
        # # 计算过程是没有问题的，问题出在p2d不同，是flatten的时候先行后列出了问题，还是depth不同造成的问题呢？
        # # st()
        # depth = data['depth']
        # (Pdb) ((np.flip(ret,axis=1) == depth)==True).sum()
        # 217088

        return ret


# videocapture不能用多线程读同一个video，只能拆分成image
class ColorReader:
    def __init__(self, color_file) -> None:
        self._frames = cv.VideoCapture(color_file) 
        self._count = 0
        if self._frames.isOpened():
            self._count = int(self._frames.get(cv.CAP_PROP_FRAME_COUNT))
        
    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        print(f'get color id: {idx}/{self._count}')
        if idx > self._count:
            raise Exception('color index out of range')
        print(f'1, {idx}')
        self._frames.set(cv.CAP_PROP_POS_FRAMES, idx)
        print(f'2')
        ret, frame = self._frames.read()
        print(f'3')
        if not ret:
            raise Exception(f'error frame by idx:{idx}')
        return frame



class ColorImgReader:
    def __init__(self, color_img_file_dir) -> None:
        self.color_img_file_dir = color_img_file_dir
        self.color_img_file_list = glob.glob(osp.join(color_img_file_dir, '*.jpg'))
        self.color_img_file_list.sort()
        self._count = len(self.color_img_file_list)
        
    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        # print(f'get color id: {idx}/{self._count}')
        
        if idx > self._count:
            raise Exception('color index out of range')
        frame = cv.imread(self.color_img_file_list[idx], cv.IMREAD_COLOR)
        # print(f'idx {idx} | fname {self.color_img_file_list[idx]}')

        if frame is None:
            raise Exception(f'error frame by idx:{idx}')
        return frame


# 我知道了作者这里提供了两个相机参数
# kcalib是kinect的相机参数
# calib是rgb的相机参数
class CalibReader:
    def __init__(self, calib_file, kcalib_file, node_id) -> None:
        with open(calib_file) as f:
            calib = json.load(f)
            camera = calib['cameras'][-11 + node_id]
            self._pan_calib = camera

        with open(kcalib_file) as kf:
            calib = json.load(kf)
            camera = calib['sensors'][node_id - 1]
            self._k_calib = camera

    def joints_k_color(self):
        R = self._pan_calib['R']
        t = self._pan_calib['t']
        R, t = np.array(R), np.array(t) / 100
        return R, t

    def lcc_panoptic_calibData(self):
        R = self._pan_calib['R']
        t = self._pan_calib['t']
        R, t = np.array(R), np.array(t)
        return R, t

    def lcc_M_color(self):
        T = self._k_calib['M_color']
        T_M_color = np.array(T)
        T = copy.deepcopy(T_M_color)
        # st()
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        
        return T[:3, :3], T[:3, 3:], T_M_color

    ###
    def lcc_M_color_all(self):
        T = self._k_calib['M_color']
        T_M_color = np.array(T)
        T = copy.deepcopy(T_M_color)
        # st()
        # T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        
        return T[:3, :3], T[:3, 3:], T_M_color

    # rgb的外参
    def k_depth_color(self):
        T = self._k_calib['M_color']
        T = np.array(T)
        # st()
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        
        return T[:3, :3], T[:3, 3:]

        # rgb的外参
    def lcc_M_depth(self):
        T = self._k_calib['M_depth']
        T_M_depth = np.array(T)
        T = copy.deepcopy(T_M_depth)
        # st()
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        
        return T[:3, :3], T[:3, 3:], T_M_depth

    # depth内参
    def depth_proj(self):
        K = self._k_calib['K_depth']
        d = self._k_calib['distCoeffs_depth']
        K, d = np.array(K), np.array(d)
        return K, d
    
    # rgb内参
    def color_proj(self):
        K = self._k_calib['K_color']
        d = self._k_calib['distCoeffs_color']
        K, d = np.array(K), np.array(d)
        return K, d

