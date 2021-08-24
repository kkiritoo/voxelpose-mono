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

logger = logging.getLogger(__name__)



TRAIN_LIST = [
    '160906_band1',
]

VAL_LIST = ['160906_band1']


# TRAIN_LIST = [
#     '160422_ultimatum1',
#     '160224_haggling1',
#     '160226_haggling1',
#     '161202_haggling1',
#     '160906_ian1',
#     '160906_ian2',
#     '160906_ian3',
#     '160906_band1',
#     '160906_band2',
#     '160906_band3',
# ]
# VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5']


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

CAMS = [(50, 1),
         (50, 2),
         (50, 3),
         (50, 4),
         (50, 5),
         (50, 6),
         (50, 7),
         (50, 8),
         (50, 9),
         (50, 10)]

class Kinoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        
        self.pixel_std = 200.0 # 没有用过
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = CAMS[:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = CAMS[:self.num_views]
            self.num_views = len(self.cam_list)

        ### 创建reader_list
        # 这里最用img读取，并且把depth.dat拆分一下，否则内存肯定会爆炸，应该不会吧，open不是全部读到内存 read才是内存复用就行了
        self._color_reader_list = {}
        self._depth_reader_list = {}

        for seq in self.sequence_list:
            selected_cams = CAMS[:self.num_views]
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
            selected_cams = CAMS[:self.num_views]
            node_id_list = [cam[1] for cam in selected_cams]
            
            seq_root = osp.join(self.dataset_root, seq)
            
            _calib_seq_list = []

            for node_id in node_id_list:
                _calib_seq_list.append(CalibReader(path.join(seq_root, f'calibration_{seq}.json'), path.join(seq_root, f'kcalibration_{seq}.json'), node_id))

            self._calib_list[seq] = _calib_seq_list

        ### 读取db
        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        # for now
        # 这个如果用以前的pose可能会全乱掉
        if False:
            # if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
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

    def depth_to_color_cloud(self, depth, shape, calib):
        # depth相机的内参
        d_K, d_d = calib.depth_proj()
        
        # depth相机的外参，我怀疑这是depth到color的外参

        R, t = calib.k_depth_color()

        dfx, dfy, dcx, dcy = d_K[0, 0], d_K[1, 1], d_K[0, 2], d_K[1, 2]

        d_d = d_d[:5]

        depth = cv.undistort(depth, d_K, d_d) / 1e3

        ###### depth image -> 3d point cloud
        # depth取值范围为1.5~3之间 应该基本上都是人了
        # 这里就是小孔成像，相似三角形
        # cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1]) if depth[v, u] < 3 and depth[v, u] > 1.5])
        # depth取值无限制
        cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1])])
        # (Pdb) cloud.shape
        # (217088, 3)


        # 为什么这里没有先用M_depth转化为世界坐标系，而是直接用M_color?

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
        # 最后proj_cloud中没有被赋值的点就是缺失的

        # proj_cloud_show_z = proj_cloud[:, :, 2]
        # proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
        # proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
        # proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
        # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)

        # cv.imshow('proj_cloud_show_z_color1', proj_cloud_show_z_color1)
        # cv.imshow('proj_cloud_show_z_color2', proj_cloud_show_z_color2)
        # st()

        return proj_cloud

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
            selected_cams = CAMS[:self.num_views]
            
            seq_root = osp.join(self.dataset_root, seq)

            self._sync = SyncReader(path.join(seq_root, f'ksynctables_{seq}.json'), \
                                    path.join(seq_root, 'hdPose3d_stage1_coco19'), selected_cams)
            
            
            self._data = list(self._check(seq))

            print(f'self._sync : {len(self._sync)}, self._data : {len(self._data)}')

            # st()

            for idx, _data_item in enumerate(self._data):
                bodies, depth_id_list, color_id_list = _data_item

                # st()
                
                
                # panoptic-dataset-tools 中的投影
                # for vis
                # img_list_to_show = []

                # 在db中存数据需要保证每次存储的，要么是连续的num_view个要么是0个，不能出现其他数字否则会乱掉
                
                _calib_list_seq = self._calib_list[seq]

                for k, v in cameras.items():
                    
                    camera_index = k[1] - 1 
                    _calib = _calib_list_seq[camera_index]

                    # st()

                    all_poses_3d = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []

                    for body in bodies:
                        
                        pose3d = body
                        # st()
                        joints = self.joints_to_color(pose3d[:, 0:3], _calib)
                        # print(joints.shape)
                        # print(joints)
                        proj_joints = self.project_pts(joints, _calib)
                        # print(proj_joints.shape)
                        # print(proj_joints)


                        pose3d = pose3d[:self.num_joints]
                        proj_joints = proj_joints[:self.num_joints]

                        # st()
                        # 注意这里因为之前/100这里也/100
                        joints_vis = pose3d[:, -1] > (0.1 / 100)

                        all_poses_3d.append(pose3d[:, 0:3] * 10.0) 
                        all_poses_vis_3d.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                        if not joints_vis[self.root_id]:
                            continue
                        
                        # 计算pose2d
                        # # Coordinate transformation
                        # M = np.array([[1.0, 0.0, 0.0],
                        #               [0.0, 0.0, -1.0],
                        #               [0.0, 1.0, 0.0]])
                        # pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                        # all_poses_3d.append(pose3d[:, 0:3] * 10.0) 
                        # all_poses_vis_3d.append(
                        #     np.repeat(
                        #         np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                        # pose2d = np.zeros((pose3d.shape[0], 2))

                        # # 2d 坐标是通过投影算出来的，用之前先可视化好
                        # pose2d[:, :2] = projectPoints(
                        #     pose3d[:, 0:3].transpose(), v['K'], v['R'],
                        #     v['t'], v['distCoef']).transpose()[:, :2]

                        pose2d = proj_joints

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
                    
                    # if len(all_poses_3d) > 0 and len(all_poses) == 0:
                    #     # 居然会这样
                    #     st()

                    if len(all_poses_3d) > 0 and len(all_poses) > 0:
                        our_cam = {}
                        our_cam['R'] = v['R']
                        our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                        our_cam['fx'] = np.array(v['K'][0, 0])
                        our_cam['fy'] = np.array(v['K'][1, 1])
                        our_cam['cx'] = np.array(v['K'][0, 2])
                        our_cam['cy'] = np.array(v['K'][1, 2])
                        our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

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


                        # if idx > 11300 and idx % 2 == 0:
                        #     # if idx > 8000 and idx % 2 == 0:
                        #     # if idx > 3700 and idx % 2 == 0:
                        #     # new_shape = (1920 // 4, 1080 // 4)
                        #     # for vis
                        #     # img_list_to_show.append(self._color_reader_list[seq][camera_index][color_id_list[camera_index]])
                        #     print(all_poses)
                        #     img = self._color_reader_list[seq][camera_index][color_id_list[camera_index]]
                        #     for a_pose in all_poses:
                        #         for point_i in range(15):
                        #             # img = cv.putText(img, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
                        #             img = cv.circle(img, (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 2, (255, 255, 0), 2)
                        #             img = cv.putText(img, f'{point_i}', (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 0, 1, (255, 0, 255), 2)
                        #             img = cv.putText(img, f'idx:{idx}', (30,30), 0, 1, (255, 0, 255), 2)

                        #     cv.namedWindow("color",0)
                        #     cv.resizeWindow("color", 960, 540)
                        #     cv.imshow('color', img)
                            
                        #     if 113 == cv.waitKey(100):
                        #         st()


            # ###### voxelpose投影的代码中只有少量view正确
            # # for vis
            # # img_list_to_show = []
            # # 在db中存数据需要保证每次存储的，要么是连续的num_view个要么是0个，不能出现其他数字否则会乱掉
            # for k, v in cameras.items():
                

            #     all_poses_3d = []
            #     all_poses_vis_3d = []
            #     all_poses = []
            #     all_poses_vis = []
            #     # 3d joint -> 2d joint
            #     # array([[-109.202   , -146.703   ,    0.73081 ,    0.681366],
            #     # [-109.531   , -166.822   ,  -16.5489  ,    0.664246],
            #     # [-108.344   ,  -91.629   ,    1.34021 ,    0.413544],
            #     # [ -97.9078  , -146.878   ,  -11.9947  ,    0.615387],
            #     # [ -93.9524  , -118.337   ,  -15.5165  ,    0.651825],
            #     # [-102.788   ,  -93.7093  ,  -19.8694  ,    0.635406],
            #     # [-103.064   ,  -91.7136  ,   -7.53898 ,    0.420196],
            #     # [-103.74    ,  -49.1241  ,   -8.26781 ,    0.471039],
            #     # [-102.667   ,   -8.5043  ,   -9.90811 ,    0.6344  ],
            #     # [-121.878   , -147.223   ,   13.6519  ,    0.580506],
            #     # [-125.404   , -118.028   ,   16.9511  ,    0.399444],
            #     # [-132.818   ,  -93.4912  ,    9.92613 ,    0.288025],
            #     # [-113.625   ,  -91.5443  ,   10.2194  ,    0.391907],
            #     # [-113.868   ,  -47.4008  ,    8.62231 ,    0.509888],
            #     # [-114.357   ,   -6.72679 ,   10.5661  ,    0.651703],
            #     # [-106.879   , -170.176   ,  -14.174   ,    0.629914],
            #     # [-103.618   , -168.129   ,   -4.48493 ,    0.571411],
            #     # [-112.644   , -170.398   ,  -14.9105  ,    0.415497],
            #     # [-118.67    , -168.921   ,   -6.39634 ,    0.201447]])
            #     for body in bodies:
                    
                    
            #         pose3d = body.reshape((-1, 4))

            #         print(pose3d)

            #         pose3d = pose3d[:self.num_joints]
            #         joints_vis = pose3d[:, -1] > 0.1
            #         if not joints_vis[self.root_id]:
            #             continue
            #         # Coordinate transformation
            #         M = np.array([[1.0, 0.0, 0.0],
            #                       [0.0, 0.0, -1.0],
            #                       [0.0, 1.0, 0.0]])
            #         pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
            #         all_poses_3d.append(pose3d[:, 0:3] * 10.0)
            #         all_poses_vis_3d.append(
            #             np.repeat(
            #                 np.reshape(joints_vis, (-1, 1)), 3, axis=1))
            #         pose2d = np.zeros((pose3d.shape[0], 2))
            #         # 2d 坐标是通过投影算出来的，用之前先可视化好
            #         pose2d[:, :2] = projectPoints(
            #             pose3d[:, 0:3].transpose(), v['K'], v['R'],
            #             v['t'], v['distCoef']).transpose()[:, :2]
            #         # 只要不超出image就是可见
            #         x_check = np.bitwise_and(pose2d[:, 0] >= 0,
            #                                  pose2d[:, 0] <= width - 1)
            #         y_check = np.bitwise_and(pose2d[:, 1] >= 0,
            #                                  pose2d[:, 1] <= height - 1)
            #         check = np.bitwise_and(x_check, y_check)
            #         joints_vis[np.logical_not(check)] = 0
            #         all_poses.append(pose2d)
            #         all_poses_vis.append(
            #             np.repeat(
            #                 np.reshape(joints_vis, (-1, 1)), 2, axis=1))
            #     if len(all_poses_3d) > 0:
            #         our_cam = {}
            #         our_cam['R'] = v['R']
            #         our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
            #         our_cam['fx'] = np.array(v['K'][0, 0])
            #         our_cam['fy'] = np.array(v['K'][1, 1])
            #         our_cam['cx'] = np.array(v['K'][0, 2])
            #         our_cam['cy'] = np.array(v['K'][1, 2])
            #         our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
            #         our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
            #         # st()
            #         camera_index = k[1] - 1 
            #         # st()
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
                    
            #         # new_shape = (1920 // 4, 1080 // 4)
            #         # for vis
            #         # img_list_to_show.append(self._color_reader_list[seq][camera_index][color_id_list[camera_index]])
            #         print(all_poses)
            #         img = self._color_reader_list[seq][camera_index][color_id_list[camera_index]]
            #         for a_pose in all_poses:
            #             for point_i in range(15):
            #                 img = cv.putText(img, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
            #         cv.namedWindow("color",0)
            #         cv.resizeWindow("color", 960, 540)
            #         cv.imshow('color', img)
                    
            #         if 113 == cv.waitKey(100):
            #             st()

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

    def evaluate(self, preds):
        # 这里直接传进去所有的preds进行evaluate
        
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
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
                data = json.load(f)
                time = data['univTime']
                bodies = data['bodies']
                if len(bodies) == 0:
                    continue
                
                # 这里bodies只取第0个人，改为取list
                # bodies = np.array(bodies[0]['joints19'])
                bodies = [np.array(bodies[index]['joints19']) for index in range(len(bodies))] 
                timed_bodies[time] = bodies
        
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
        bodies = list(map(lambda body: np.array(body).reshape(-1, 4) / 100, bodies))
        # bodies = np.array(bodies).reshape(-1, 4)[:, :3] / 100
        # st()

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
        self._frames = open(depth_file, 'rb')
        self._frames.seek(0, 2)
        self._count = self._frames.tell() // DepthReader.frame_len

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        # print(f'get depth id /: {idx}/{self._count}')
        if idx > self._count:
            raise Exception('depth index out of range')

        self._frames.seek(idx * DepthReader.frame_len, 0)
        raw_data = self._frames.read(DepthReader.frame_len)
        frame = np.frombuffer(raw_data, dtype=np.uint16)
        return np.array(frame).reshape(DepthReader.shape[0], DepthReader.shape[1])


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

    # depth的外参
    def k_depth_color(self):
        T = self._k_calib['M_color']
        T = np.array(T)
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        return T[:3, :3], T[:3, 3:]

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

