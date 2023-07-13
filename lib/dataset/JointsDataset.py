# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging

import cv2
import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import Dataset
import os

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale

from pdb import set_trace as st
from os import path as osp
import gzip
import random
from random import gauss

logger = logging.getLogger(__name__)


# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# def show_np_image(np_image):
#     plt.figure()
#     # plt.imshow(cv.cvtColor(np_image, cv.COLOR_BGR2RGB))
#     plt.imshow(np_image)
#     plt.show()


vis_flag = False
if vis_flag:
    coordist_sum_1 = 0
    coordist_cnt_1 = 0
    coordist_avg_1 = 0
    coordist_sum_2 = 0
    coordist_cnt_2 = 0
    coordist_avg_2 = 0


show_w, show_h = 480, 270
def show_view(view_to_show, name=None):
    # view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name, show_w, show_h)
    cv2.imshow(name, view_to_show)
    if 113 == cv2.waitKey(100):
        st()

class JointsDataset(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        # note
        self.num_views = cfg.DATASET.CAMERA_NUM

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.rand_sigma = cfg.NETWORK.RAND_SIGMA
        self.rand_sigma_gau = cfg.NETWORK.RAND_SIGMA_GAU
        self.rand_sigma_sample = cfg.NETWORK.RAND_SIGMA_SAMPLE

        self.f_weight = cfg.NETWORK.F_WEIGHT

        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)

        self.use_precomputed_hm = cfg.NETWORK.USE_PRECOMPUTED_HM

        self.out_3d_pose_vid = cfg.OUT_3D_POSE_VID
        self.pseudep = cfg.NETWORK.PSEUDEP

        self.namedataset = cfg.DATASET.NAMEDATASET



    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def get_aligned_depth_db(self):
        aligned_depth_db = {}

        assert self.namedataset == 'large'
        save_dir = f'./aligned_depth_db_{self.image_set}_{self.num_views}cam'
        # save_dir = './aligned_depth_db'
        os.makedirs(save_dir, exist_ok=True)


        all_equals_num = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            # import time
            # time_start=time.time()
            data_numpy = self._color_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['color_index']]
            # time_end=time.time()
            # print('color cost\n',time_end-time_start)
                
            # import time
            # time_start=time.time()
            depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
            # time_end=time.time()
            # print('depth cost\n',time_end-time_start)

            calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]

            ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
            ### kinect相机坐标系 * rgb参数 = rgb相机空间
            ### kinect相机坐标系 * depth参数 = depth相机空间
            ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
            # time_start=time.time()
            
            # cloud, cloud_mask, aligned_depth, cloud_color = self.depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
            cloud, cloud_mask, aligned_depth, cloud_color = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
            
            # time_end=time.time()
            # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

            # st()
            ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
            # color cost
            # 0.03504300117492676
            # depth cost
            # 0.000743865966796875
            # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
            # 0.9353528022766113

            aligned_cloud_color = cloud_color
            aligned_depth = aligned_cloud_color[:, :, 2]
            

            ### 之后需要增加↓
            # aligned_depth = aligned_depth[np.newaxis, :]
            ### 之后需要transform↓
            # aligned_depth = (aligned_depth * 1000).astype(np.uint16) ### 压缩保存
            pass  ### 不压缩保存

            ### depth_to_color_cloud_w_aligned_depth_w_cloud_color出来的东西只有aligned_depth能用到
            # st()
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            aligned_depth_db[aligned_depth_key] = aligned_depth

            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            

            # 1save
            # np.save(save_path, aligned_depth) # 4m

            # 2save
            f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
            np.save(f, aligned_depth) 
            f.close()


            f = gzip.GzipFile(save_path, "r")
            aligned_depth_load = np.load(f)
            f.close()

            all_equals_num += ((aligned_depth_load == aligned_depth).sum() == 1080 * 1920)
            if idx % 100 == 0:
                print(f'idx:{idx+1}/all_equals_num:{all_equals_num}')
            
        print('all done!')
        st()
        pass

    def check_aligned_depth_db(self):
        # save_dir = './aligned_depth_db'
        save_dir = './aligned_depth_db_node1'
        os.makedirs(save_dir, exist_ok=True)

        cnt = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if '160906_band3_0' in save_path:
                # print(save_path)
                cnt += 1
        print(f'160906_band3_0 {cnt}')
            
        cnt = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if '160906_band3_1' in save_path:
                # print(save_path)
                cnt += 1
        print(f'160906_band3_1 {cnt}')

        cnt = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if '160906_band3_2' in save_path:
                # print(save_path)
                cnt += 1
        print(f'160906_band3_2 {cnt}')

        cnt = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if '160906_band3_3' in save_path:
                # print(save_path)
                cnt += 1
        print(f'160906_band3_3 {cnt}')

        cnt = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])
            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if '160906_band3_4' in save_path:
                # print(save_path)
                cnt += 1
        print(f'160906_band3_4 {cnt}')
        st()
        

        all_equals_num = 0
        st()
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])

            aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"

            save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
            if osp.exists(save_path):
                print(f'exists {save_path}')
            else:
                print(f'not exists {save_path}')

            # f = gzip.GzipFile(save_path, "r")
            # aligned_depth_load = np.load(f)
            # f.close()

            if idx % 100 == 0:
                print(f'idx:{idx+1}')
            
        print('all done!')
        st()
        pass
    
    def check_heatmap_db(self):
        save_dir = './heatmap_db'
        os.makedirs(save_dir, exist_ok=True)


        all_equals_num = 0
        for idx in range(len(self.db)):
            # for idx in range(10):
            db_rec = copy.deepcopy(self.db[idx])

            heatmap_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['color_index']}"

            save_path = osp.join(save_dir, f'{heatmap_key}.npy')

            f = gzip.GzipFile(save_path, "r")
            heatmap_load = np.load(f)
            f.close()

            if idx % 100 == 0:
                print(f'idx:{idx+1}')
            
        print('all done!')
        st()
        pass

    def smooth_depth_image(self, depth_image, max_hole_size=10, hole_value=0):
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
        erosion = cv2.erode(mask,kernel,iterations = 1)
        mask = mask - erosion
        smoothed_depth_image = cv2.inpaint(depth_image.astype(np.uint16),mask,max_hole_size,cv2.INPAINT_NS)
        # smoothed_depth_image = cv2.inpaint(depth_image.astype(np.uint16),mask,max_hole_size,cv.INPAINT_TELEA)
        return smoothed_depth_image



    def __getitem__(self, idx):
        
        ### lcc debugging
        # item = self.getitem_func(idx)

        try:
            item = self.getitem_func(idx)
        except:
            # print(f'idx {idx} getitem error, skippping...')
            # db_rec = copy.deepcopy(self.db[idx])
            # print(f"idx {idx} seq {db_rec['seq']} camera_index {db_rec['camera_index']} color_index {db_rec['color_index']}")

            ### 出错的话随机取
            ### 随机取万一还出错就会error退出
            ### 不是随机取的问题，是因为我把路径改了
            return self.getitem_func(np.random.randint(self.__len__()))

            ### 取第一个
            # return self.getitem_func(0)

        return item
        

    def getitem_func(self, idx):
        
        # print('before')

        # st()

        # self.get_aligned_depth_db()
        # self.check_aligned_depth_db()
        # self.check_heatmap_db()

        # st()

        db_rec = copy.deepcopy(self.db[idx])
        
        # cloud -> meta
        cloud = -1.0
        cloud_mask = -1.0
        aligned_depth = np.zeros((1, 1080, 1920))
        proj_cloud_color_mask = np.zeros((1080, 1920))
        depth_data_numpy = np.zeros((424, 512)).astype(np.int32)
        # aligned_depth_read_flag = False
        limb_depth = np.zeros((1, self.image_size[1], self.image_size[0])) 

        # st()
        if not 'seq' in db_rec.keys():
            db_rec['seq'] = -1.0
        if not 'camera_index' in db_rec.keys():
            db_rec['camera_index'] = -1.0
        if not 'color_index' in db_rec.keys():
            db_rec['color_index'] = -1.0
        if not 'depth_index' in db_rec.keys():
            db_rec['depth_index'] = -1.0

        # st()
        # 可以直接用子类定义的depth_reader等属性
        if 'dataset_name' in db_rec.keys():
            if db_rec['dataset_name'] == 'kinoptic':
                # st()
                image_file = ""
                

                ### 除非证明unet cat image进去确实有用，否则尽量不要读image，太费时间
                ### 不读的话要改的东西太多了
                # import time
                # time_start=time.time()
                data_numpy = self._color_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['color_index']]

                # 保证是同一张image即可
                # print(f"db_rec['color_index'] {db_rec['color_index']} ")
                # print(f"db_rec['depth_index'] {db_rec['depth_index']} ")
                # print(f"db_rec['camera_index'] {db_rec['camera_index']} ")

                # time_end=time.time()
                # print('color cost\n',time_end-time_start)

                # st()
                
                if self.f_weight:
                    ############### <节省时间> ###############
                    if self.namedataset == 'large' or \
                        self.namedataset == 'large_interp25j' or \
                        self.namedataset == 'large_wholebody' or \
                        self.namedataset == 'large_wholebody_mp':
                        save_dir = f'./aligned_depth_db_{self.image_set}_{self.num_views}cam_itv{self._interval}'
                    elif self.namedataset == 'OUT_3D_POSE_VID':
                        save_dir = './aligned_depth_db_OUT_3D_POSE_VID'
                    else:
                        save_dir = './aligned_depth_db'
                    
                    # save_dir_org = save_dir + '_org'

                    os.makedirs(save_dir, exist_ok=True)
                    # os.makedirs(save_dir_org, exist_ok=True)


                    aligned_depth_key = f"{db_rec['seq']}_{db_rec['camera_index']}_{db_rec['depth_index']}"
                    save_path = osp.join(save_dir, f'{aligned_depth_key}.npy')
                    save_path_mask = osp.join(save_dir, f'{aligned_depth_key}_mask.npy')

                    ### precompute org image after inpaint
                    # save_path_org = osp.join(save_dir_org, f'{aligned_depth_key}.png')



                    # ########################### <gpu precompute> ###########################
                    # if osp.exists(save_path):
                    #     # st()
                    #     # print('before load')

                    #     f = gzip.GzipFile(save_path, "r")
                    #     aligned_depth = np.load(f) ### 有时会报alllow_pickle=False
                    #     # aligned_depth = np.load(f, allow_pickle=True)
                    #     f.close()

                    #     # st()
                    #     # aligned_depth = (aligned_depth / 1000).astype(np.float64) ### 压缩保存1
                    #     pass ### 不压缩保存2，啥也不用干
                        
                    #     aligned_depth = aligned_depth[np.newaxis, :]

                    #     aligned_depth_read_flag = True

                    #     # st()
                    # else:
                    #     if not self.namedataset == 'OUT_3D_POSE_VID':
                    #         print('没有在预计算的范围之内，不可能！！！')
                    #         st()
                    #         assert False
                    #     else:
                    #         pass
                    #     # time_start=time.time()
                    #     depth_data_numpy_tmp = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                    #     depth_data_numpy = depth_data_numpy_tmp.copy().astype(np.int32)


                    #     # time_end=time.time()
                    #     # print('depth cost\n',time_end-time_start)



                    #     # ############## <下面的处理包含cuda的代码比较麻烦，放到外面处理> ##############
                    #     # # st()

                    #     # # st()
                    #     # # show_np_image(depth_data_numpy / depth_data_numpy.max())

                    #     # # color慢得多
                    #     # # color cost
                    #     # # 0.18369126319885254
                    #     # # depth cost
                    #     # # 0.011093378067016602

                    #     # # construct u,v -> 3d point cloud using depth image
                    #     # calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]


                    #     # # #### <old code>
                    #     # # ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
                    #     # # ### kinect相机坐标系 * rgb参数 = rgb相机空间
                    #     # # ### kinect相机坐标系 * depth参数 = depth相机空间
                    #     # # ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
                    #     # # # time_start=time.time()
                    #     # # # st()
                    #     # # cloud, cloud_mask, aligned_depth, cloud_color = self.depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    #     # # cloud, cloud_mask, aligned_depth, cloud_color = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                        
                    #     # # # time_end=time.time()
                    #     # # # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

                    #     # # # st()
                    #     # # ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
                    #     # # # color cost
                    #     # # # 0.03504300117492676
                    #     # # # depth cost
                    #     # # # 0.000743865966796875
                    #     # # # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
                    #     # # # 0.9353528022766113

                    #     # # aligned_cloud_color = cloud_color
                    #     # # aligned_depth = aligned_cloud_color[:, :, 2]
                    #     # ### depth_to_color_cloud_w_aligned_depth_w_cloud_color出来的东西只有aligned_depth能用到
                    #     # # st()

                    #     # # cloud, cloud_mask = self.depth_to_color_cloud(depth_data_numpy, data_numpy.shape, calib)

                    #     # # proj_cloud = copy.deepcopy(cloud)


                    #     # # M = np.array([[1.0, 0.0, 0.0],
                    #     # #             [0.0, 0.0, -1.0],
                    #     # #             [0.0, 1.0, 0.0]])
                    #     # # cloud = cloud.reshape(-1,3).dot(M).reshape(cloud.shape) * 10

                    #     # # # print(f'depth_data_numpy {depth_data_numpy.mean()}')
                    #     # # # print(f'cloud {cloud.mean()}')
                    #     # # # print('cloud {}'.format(cloud[50,:,:].mean()))
                    #     # # # print('cloud {}'.format(cloud[100,:,:].mean()))
                    #     # # # print('cloud {}'.format(cloud[500,:,:].mean()))
                    #     # # # st()
                    #     # # #### </old code>

                    #     # # (Pdb) depth_data_numpy.shape
                    #     # # (424, 512)
                    #     # # (Pdb) data_numpy.shape
                    #     # # (1080, 1920, 3)
                    #     # # (Pdb) calib
                    #     # # <dataset.kinoptic.CalibReader object at 0x7f17a6027588>
                    #     # st()
                    #     # aligned_depth = self.depth_to_aligned_depth(depth_data_numpy, data_numpy.shape, calib)

                    #     # # print('aligned_depth没找到，保存？')
                    #     # # st()

                    #     # # aligned_depth_save = (aligned_depth.copy() * 1000).astype(np.uint16) ### 压缩保存1
                    #     # aligned_depth_save = aligned_depth.copy() ### 不压缩保存2
                        
                    #     # # print('before save')

                    #     # f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                    #     # np.save(f, aligned_depth_save) 
                    #     # f.close()

                    #     # aligned_depth = aligned_depth[np.newaxis, :]

                    #     # ############## </下面的处理包含cuda的代码比较麻烦，放到外面处理> ##############
                    # ########################### </gpu precompute> ###########################

                    # ########################### <cpu precompute> ###########################
                    # if osp.exists(save_path):
                    #     # st()
                    #     f = gzip.GzipFile(save_path, "r")
                    #     aligned_depth = np.load(f)
                    #     f.close()

                    #     # st()
                    #     # aligned_depth = (aligned_depth / 1000).astype(np.float64) ### 压缩保存1
                    #     pass ### 不压缩保存2，啥也不用干
                        
                    #     aligned_depth = aligned_depth[np.newaxis, :]

                    #     # st()
                    # else:
                    #     # time_start=time.time()
                    #     depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                    #     # time_end=time.time()
                    #     # print('depth cost\n',time_end-time_start)

                    #     # st()

                    #     # st()
                    #     # show_np_image(depth_data_numpy / depth_data_numpy.max())

                    #     # color慢得多
                    #     # color cost
                    #     # 0.18369126319885254
                    #     # depth cost
                    #     # 0.011093378067016602

                    #     # construct u,v -> 3d point cloud using depth image
                    #     calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]


                    #     ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
                    #     ### kinect相机坐标系 * rgb参数 = rgb相机空间
                    #     ### kinect相机坐标系 * depth参数 = depth相机空间
                    #     ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
                    #     # time_start=time.time()
                    #     # st()
                    #     # cloud, cloud_mask, aligned_depth, cloud_color = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    #     cloud, cloud_mask, aligned_depth, cloud_color, proj_cloud_color_mask = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    #     # time_end=time.time()
                    #     # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

                    #     # st()
                    #     ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
                    #     # color cost
                    #     # 0.03504300117492676
                    #     # depth cost
                    #     # 0.000743865966796875
                    #     # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
                    #     # 0.9353528022766113

                    #     aligned_cloud_color = cloud_color
                    #     aligned_depth = aligned_cloud_color[:, :, 2]

                    #     # print('aligned_depth没找到，保存？')
                    #     # st()

                    #     # aligned_depth_save = (aligned_depth.copy() * 1000).astype(np.uint16) ### 压缩保存1
                    #     aligned_depth_save = aligned_depth.copy() ### 不压缩保存2
                        
                    #     f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                    #     np.save(f, aligned_depth_save) 
                    #     f.close()

                    #     aligned_depth = aligned_depth[np.newaxis, :]


                    #     # ### depth_to_color_cloud_w_aligned_depth_w_cloud_color出来的东西只有aligned_depth能用到
                    #     # # st()

                    #     # # cloud, cloud_mask = self.depth_to_color_cloud(depth_data_numpy, data_numpy.shape, calib)

                    #     # proj_cloud = copy.deepcopy(cloud)


                    #     # M = np.array([[1.0, 0.0, 0.0],
                    #     #             [0.0, 0.0, -1.0],
                    #     #             [0.0, 1.0, 0.0]])
                    #     # cloud = cloud.reshape(-1,3).dot(M).reshape(cloud.shape) * 10

                    #     # # print(f'depth_data_numpy {depth_data_numpy.mean()}')
                    #     # # print(f'cloud {cloud.mean()}')
                    #     # # print('cloud {}'.format(cloud[50,:,:].mean()))
                    #     # # print('cloud {}'.format(cloud[100,:,:].mean()))
                    #     # # print('cloud {}'.format(cloud[500,:,:].mean()))
                    #     # # st()
                    # ########################### </cpu precompute> ###########################

                    ########################### <cpu precompute-w mask> ###########################
                    # if False:
                    #     pass
                    # else:
                    #     # time_start=time.time()
                    #     depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                    #     # time_end=time.time()
                    #     # print('depth cost\n',time_end-time_start)

                    #     # st()
                    #     depth_data_numpy[depth_data_numpy==0] = 8000
                        
                    #     ### lcc precompute save org image
                    #     # show_view(depth_data_numpy / depth_data_numpy.max(), '1')
                    #     max_hole_size = 20
                    #     # max_hole_size = 30
                    #     depth_data_numpy = self.smooth_depth_image(depth_data_numpy, max_hole_size=max_hole_size, hole_value=8000)
                    #     # show_view(depth_data_numpy / depth_data_numpy.max(), '2')

                    #     # st()
                    #     # save_path_org
                    #     depth_data_numpy_255 = (depth_data_numpy / 8000 * 255).astype(np.uint8)
                    #     depth_data_numpy_255 = cv2.resize(depth_data_numpy_255, (512, 512))
                    #     depth_data_numpy_255 = depth_data_numpy_255[..., np.newaxis]    
                    #     depth_data_numpy_255 = np.repeat(depth_data_numpy_255, 3, axis=2)
                    #     # if True:
                    #     #     name = '3'
                    #     #     cv2.namedWindow(name,0)
                    #     #     # st()
                    #     #     show_w, show_h = 480, 270
                    #     #     cv2.resizeWindow(name, show_w, show_h)
                    #     #     cv2.imshow(name, depth_data_numpy_255)
                    #     #     if 113 == cv2.waitKey(100):
                    #     #         st()   
                    #     cv2.imwrite(save_path_org, depth_data_numpy_255)
                        


                    if osp.exists(save_path) and osp.exists(save_path_mask):
                        # st()
                        f = gzip.GzipFile(save_path, "r")
                        aligned_depth = np.load(f)
                        f.close()

                        # st()
                        # aligned_depth = (aligned_depth / 1000).astype(np.float64) ### 压缩保存1
                        pass ### 不压缩保存2，啥也不用干
                        
                        aligned_depth = aligned_depth[np.newaxis, :]


                        f = gzip.GzipFile(save_path_mask, "r")
                        proj_cloud_color_mask = np.load(f)
                        f.close()

                        # st()
                    else:
                        # time_start=time.time()
                        depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                        # time_end=time.time()
                        # print('depth cost\n',time_end-time_start)

                        # st()
                        # show_np_image(depth_data_numpy / depth_data_numpy.max())

                        # color慢得多
                        # color cost
                        # 0.18369126319885254
                        # depth cost
                        # 0.011093378067016602

                        # construct u,v -> 3d point cloud using depth image
                        calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]


                        ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
                        ### kinect相机坐标系 * rgb参数 = rgb相机空间
                        ### kinect相机坐标系 * depth参数 = depth相机空间
                        ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
                        # time_start=time.time()
                        # st()
                        # cloud, cloud_mask, aligned_depth, cloud_color = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                        cloud, cloud_mask, aligned_depth, cloud_color, proj_cloud_color_mask = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                        # time_end=time.time()
                        # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

                        # st()
                        ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
                        # color cost
                        # 0.03504300117492676
                        # depth cost
                        # 0.000743865966796875
                        # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
                        # 0.9353528022766113

                        aligned_cloud_color = cloud_color
                        aligned_depth = aligned_cloud_color[:, :, 2]

                        # print('aligned_depth没找到，保存？')
                        # st()

                        # aligned_depth_save = (aligned_depth.copy() * 1000).astype(np.uint16) ### 压缩保存1
                        aligned_depth_save = aligned_depth.copy() ### 不压缩保存2
                        
                        f = gzip.GzipFile(save_path, "w") # 0.329m 给力啊
                        np.save(f, aligned_depth_save) 
                        f.close()

                        proj_cloud_color_mask_save = proj_cloud_color_mask.copy() ### 不压缩保存2
                        
                        f = gzip.GzipFile(save_path_mask, "w") # 0.329m 给力啊
                        np.save(f, proj_cloud_color_mask_save) 
                        f.close()


                        aligned_depth = aligned_depth[np.newaxis, :]


                        # ### depth_to_color_cloud_w_aligned_depth_w_cloud_color出来的东西只有aligned_depth能用到
                        # # st()

                        # # cloud, cloud_mask = self.depth_to_color_cloud(depth_data_numpy, data_numpy.shape, calib)

                        # proj_cloud = copy.deepcopy(cloud)


                        # M = np.array([[1.0, 0.0, 0.0],
                        #             [0.0, 0.0, -1.0],
                        #             [0.0, 1.0, 0.0]])
                        # cloud = cloud.reshape(-1,3).dot(M).reshape(cloud.shape) * 10

                        # # print(f'depth_data_numpy {depth_data_numpy.mean()}')
                        # # print(f'cloud {cloud.mean()}')
                        # # print('cloud {}'.format(cloud[50,:,:].mean()))
                        # # print('cloud {}'.format(cloud[100,:,:].mean()))
                        # # print('cloud {}'.format(cloud[500,:,:].mean()))
                        # # st()
                    ########################### </cpu precompute-w mask> ###########################


                    # ########################### <cpu precompute-wo save> ###########################
                    # # time_start=time.time()
                    # depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                    # # time_end=time.time()
                    # # print('depth cost\n',time_end-time_start)

                    # # st()

                    # # st()
                    # # show_np_image(depth_data_numpy / depth_data_numpy.max())

                    # # color慢得多
                    # # color cost
                    # # 0.18369126319885254
                    # # depth cost
                    # # 0.011093378067016602

                    # # construct u,v -> 3d point cloud using depth image
                    # calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]


                    # ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
                    # ### kinect相机坐标系 * rgb参数 = rgb相机空间
                    # ### kinect相机坐标系 * depth参数 = depth相机空间
                    # ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
                    # # time_start=time.time()
                    # # st()
                    # # cloud, cloud_mask, aligned_depth, cloud_color = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    # cloud, cloud_mask, aligned_depth, cloud_color, proj_cloud_color_mask = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    # # time_end=time.time()
                    # # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

                    # # st()
                    # ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
                    # # color cost
                    # # 0.03504300117492676
                    # # depth cost
                    # # 0.000743865966796875
                    # # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
                    # # 0.9353528022766113

                    # aligned_cloud_color = cloud_color
                    # aligned_depth = aligned_cloud_color[:, :, 2]

                    # aligned_depth = aligned_depth[np.newaxis, :]

                    # ########################### </cpu precompute-wo save> ###########################


                    # ### 在这里inpaint没有什用，只能消除depthmap本身的空洞，不能消除视角差产生的空洞
                    # ########################### <cpu precompute+inpaint> ###########################
                    # save_dir_inp = save_dir + '_inp'
                    # os.makedirs(save_dir_inp, exist_ok=True)
                    # save_path_inp = osp.join(save_dir_inp, f'{aligned_depth_key}.npy')

                    # if osp.exists(save_path_inp):
                    #     # st()
                    #     f = gzip.GzipFile(save_path_inp, "r")
                    #     aligned_depth = np.load(f)
                    #     f.close()

                    #     # st()
                    #     # aligned_depth = (aligned_depth / 1000).astype(np.float64) ### 压缩保存1
                    #     pass ### 不压缩保存2，啥也不用干
                        
                    #     aligned_depth = aligned_depth[np.newaxis, :]

                    #     # st()
                    # else:
                    #     # import time
                    #     # time_start=time.time()
                    #     depth_data_numpy = self._depth_reader_list[db_rec['seq']][db_rec['camera_index']][db_rec['depth_index']]
                        
                    #     show_view(depth_data_numpy / depth_data_numpy.max(), '1')

                    #     # st()
                    #     depth_data_numpy = self.smooth_depth_image(depth_data_numpy)

                    #     show_view(depth_data_numpy / depth_data_numpy.max(), '2')

                    #     # time_end=time.time()
                    #     # print('depth cost\n',time_end-time_start)

                    #     # st()

                    #     # st()
                    #     # show_np_image(depth_data_numpy / depth_data_numpy.max())

                    #     # color慢得多
                    #     # color cost
                    #     # 0.18369126319885254
                    #     # depth cost
                    #     # 0.011093378067016602

                    #     # construct u,v -> 3d point cloud using depth image
                    #     calib = self._calib_list[db_rec['seq']][db_rec['camera_index']]


                    #     ### 3d坐标系有两个，kinect相机坐标系和世界坐标系
                    #     ### kinect相机坐标系 * rgb参数 = rgb相机空间
                    #     ### kinect相机坐标系 * depth参数 = depth相机空间
                    #     ### kinect相机坐标系 * kinectlocal2panopticworld参数 = panoptic相机空间（真实世界坐标系）
                    #     # time_start=time.time()
                    #     # st()
                    #     cloud, cloud_mask, aligned_depth, cloud_color, proj_cloud_color_mask = self.np_depth_to_color_cloud_w_aligned_depth_w_cloud_color(depth_data_numpy, data_numpy.shape, calib)
                    #     # time_end=time.time()
                    #     # print('depth_to_color_cloud_w_aligned_depth_w_cloud_color cost\n',time_end-time_start)

                    #     # st()
                    #     ### 最主要的时间还是depth_to_color_cloud_w_aligned_depth_w_cloud_color浪费了
                    #     # color cost
                    #     # 0.03504300117492676
                    #     # depth cost
                    #     # 0.000743865966796875
                    #     # depth_to_color_cloud_w_aligned_depth_w_cloud_color cost
                    #     # 0.9353528022766113

                    #     aligned_cloud_color = cloud_color
                    #     aligned_depth = aligned_cloud_color[:, :, 2]

                    #     # print('aligned_depth没找到，保存？')
                    #     # st()

                    #     # aligned_depth_save = (aligned_depth.copy() * 1000).astype(np.uint16) ### 压缩保存1
                    #     aligned_depth_save = aligned_depth.copy() ### 不压缩保存2
                        
                    #     f = gzip.GzipFile(save_path_inp, "w") # 0.329m 给力啊
                    #     np.save(f, aligned_depth_save) 
                    #     f.close()

                    #     aligned_depth = aligned_depth[np.newaxis, :]
                    # ########################### </cpu precompute+inpaint> ###########################


                    ############### <节省时间> ###############

        else:
            image_file = db_rec['image']

            if self.data_format == 'zip':
                from utils import zipreader
                data_numpy = zipreader.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            else:
                data_numpy = cv2.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if data_numpy is None:
                # logger.error('=> fail to read {}'.format(image_file))
                # raise ValueError('Fail to read {}'.format(image_file))
                return None, None, None, None, None, None
        
        # videocapture出来的也是bgr
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        # st()
        joints_3d_vis = db_rec['joints_3d_vis']
        

        if vis_flag:
            # st()

            #### lcc debugging
            ### showing 
            ### 被遮挡的点果然几乎完全不对，因为取的是最前面的点
            ### 没有被遮挡的点几乎没啥问题，虽然和gt有差距，这是不可避免的，毕竟人有厚度
            ### 

            import cv2 as cv
            if idx % 1 == 0:
                # if idx > 8000 and idx % 2 == 0:
                # if idx > 3700 and idx % 2 == 0:
                # new_shape = (1920 // 4, 1080 // 4)
                # for vis
                # img_list_to_show.append(self._color_reader_list[seq][camera_index][color_id_list[camera_index]])
                # print(joints[0])
                print('\n')
                img_color = copy.deepcopy(data_numpy)
                img_color = cv.cvtColor(img_color, cv.COLOR_RGB2BGR)

                # img_color = cv2.rotate(img_color, cv2.ROTATE_180)
                # img_color = cv.flip(img_color, 1, dst=None)

                # st()
                for a_pose in [joints[0]]:
                    for point_i in range(15):
                        # img_color = cv.putText(img_color, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
                        
                        img_color = cv.circle(img_color, (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 2, (255, 255, 0), 2)
                        img_color = cv.putText(img_color, f'{point_i}', (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 0, 1, (255, 0, 255), 2)
                        img_color = cv.putText(img_color, f'idx:{idx}', (30,30), 0, 1, (255, 0, 255), 2)
                        # st()
                        # cv.namedWindow("color",0)
                        # cv.resizeWindow("color", 960, 540)
                        # cv.imshow('color', img_color)
                        # if 113 == cv.waitKey(100):
                        #     st()
                # show_np_image(img_color)
                # st()
                cv.namedWindow("color",0)
                cv.resizeWindow("color", 960, 540)
                cv.imshow('color', img_color)
                
                if 113 == cv.waitKey(100):
                    st()

            # st()
            # 我知道问题所在了
            # 其实我那个公式写的没错，但是经过变换到世界坐标系之后不能用cloud[:,:,0]==0来计算mask了，因此mask需要提前算好


            coords_to_cloud = np.zeros((joints[0].shape[0], 3))
            raw_coords = joints[0]
            raw_coords = raw_coords.astype(np.uint32)
            new_coords = raw_coords
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
                for r in range(max(cloud_mask.shape[0], cloud_mask.shape[1])): # 一圈一圈找
                    
                    # st()
                    # 大部分r都能控制在5以内
                    # 5之外的joints拿不到深度就不加权了把，直接保留原状
                    if r > 10:
                        too_far = True
                        break

                    cur_x = raw_x - r
                    for delta_y in range(-r, r+1):
                        cur_y = raw_y + delta_y # 卡住
                        if cur_x < 0 or cur_x >= cloud_mask.shape[0] or cur_y < 0 or cur_y >= cloud_mask.shape[1]:continue
                        
                        if cloud_mask[cur_x, cur_y] == False:
                            find_non_zero = True
                            break

                    if find_non_zero:
                        break

                    cur_x = raw_x + r
                    for delta_y in range(-r, r+1):
                        cur_y = raw_y + delta_y # 卡住
                        if cur_x < 0 or cur_x >= cloud_mask.shape[0] or cur_y < 0 or cur_y >= cloud_mask.shape[1]:continue
                        if cloud_mask[cur_x, cur_y] == False:
                            break

                    if find_non_zero:
                        break

                    cur_y = raw_y - r
                    for delta_x in range(-r, r+1):
                        cur_x = raw_x + delta_x # 卡住
                        if cur_x < 0 or cur_x >= cloud_mask.shape[0] or cur_y < 0 or cur_y >= cloud_mask.shape[1]:continue
                        if cloud_mask[cur_x, cur_y] == False:
                            break
                    
                    if find_non_zero:
                        break

                    cur_y = raw_y + r
                    for delta_x in range(-r, r+1):
                        cur_x = raw_x + delta_x # 卡住
                        if cur_x < 0 or cur_x >= cloud_mask.shape[0] or cur_y < 0 or cur_y >= cloud_mask.shape[1]:continue
                        if cloud_mask[cur_x, cur_y] == False:
                            break
                    
                    if find_non_zero:
                        break
                
                if too_far:
                    coords_to_cloud[joint_i] = np.array([-1.0, -1.0, -1.0])
                else:
                    coords_to_cloud[joint_i] = cloud[cur_x][cur_y]
                # print(f"r {r} | dist {(cur_x - raw_x) ** 2 + (cur_y - raw_y) ** 2}| raw_xy: ({raw_x}, {raw_y}) | cur_xy: ({cur_x}, {raw_y}) | coords_to_cloud: {coords_to_cloud[joint_i]} | joints_3d: {joints_3d[0][joint_i]} | coord_dist {np.sqrt(((coords_to_cloud[joint_i] - joints_3d[0][joint_i])**2).sum())}")
                coord_dist = np.sqrt(((coords_to_cloud[joint_i] - joints_3d[0][joint_i])**2).sum())
                global coordist_sum_1
                global coordist_cnt_1
                global coordist_avg_1

                global coordist_sum_2
                global coordist_cnt_2
                global coordist_avg_2
                if coord_dist < 100:
                    coordist_sum_1 += coord_dist
                    coordist_cnt_1 += 1
                    coordist_avg_1 = coordist_sum_1 / coordist_cnt_1
                else:
                    coordist_sum_2 += coord_dist
                    coordist_cnt_2 += 1
                    coordist_avg_2 = coordist_sum_2 / coordist_cnt_2
                print(f'coordist_sum_1 {coordist_sum_1} coordist_cnt_1 {coordist_cnt_1} coordist_avg_1 {coordist_avg_1}')
                print(f'coordist_sum_2 {coordist_sum_2} coordist_cnt_2 {coordist_cnt_2} coordist_avg_2 {coordist_avg_2}')
                if coordist_cnt_1 + coordist_cnt_2 > 500:
                    st()
                print(f"r {r} | coord_dist {np.sqrt(((coords_to_cloud[joint_i] - joints_3d[0][joint_i])**2).sum())}")
                new_coords[joint_i] = [cur_y, cur_x]
                # Depth投影到3d之后，有一些点的差距相当的大，看一下是不是由于这些点被遮挡了
                # time_end=time.time()
                # print(f'time cost for joint_i {joint_i}\n',time_end-time_start)

            proj_cloud_show_z = proj_cloud[:, :, 2]
            proj_cloud_show_z = proj_cloud_show_z / proj_cloud_show_z.max()
            proj_cloud_show_z = (proj_cloud_show_z * 255.0).astype(np.uint8) 
            import cv2 as cv
            proj_cloud_show_z_color1 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_JET)
            # proj_cloud_show_z_color2 =  cv.applyColorMap(proj_cloud_show_z, cv.COLORMAP_PARULA)
            # proj_cloud_show_z_color1 = proj_cloud_show_z
            # print(joints[0])
            print('\n')
            img_depth = copy.deepcopy(proj_cloud_show_z_color1)
            # st()
            for a_pose in [joints[0]]:
                for point_i in range(15):
                    # img_depth = cv.putText(img_depth, '{}'.format(point_i), (int(a_pose[point_i][0]),int(a_pose[point_i][1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
                    
                    img_depth = cv.circle(img_depth, (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 2, (255, 255, 0), 2)
                    img_depth = cv.circle(img_depth, (int(new_coords[point_i][0]),int(a_pose[point_i][1])), 2, (0, 0, 0), 2)

                    img_depth = cv.putText(img_depth, f'{point_i}', (int(a_pose[point_i][0]),int(a_pose[point_i][1])), 0, 1, (255, 0, 255), 2)
                    img_depth = cv.putText(img_depth, f'idx:{idx}', (30,30), 0, 1, (255, 0, 255), 2)
                    # st()
                    # cv.namedWindow("color",0)
                    # cv.resizeWindow("color", 960, 540)
                    # cv.imshow('color', img_depth)
                    # if 113 == cv.waitKey(100):
                    #     st()
            # show_np_image(img_depth)
            cv.namedWindow("depth reproject to rgb",0)
            cv.resizeWindow("depth reproject to rgb", 960, 540)
            cv.imshow('depth reproject to rgb', img_depth)
            if 113 == cv.waitKey(100):
                st()

            cv.namedWindow("composed",0)
            cv.resizeWindow("composed", 960, 540)
            
            cv.imshow('composed', (img_color*0.5 + img_depth*0.5).astype(np.uint8))
            if 113 == cv.waitKey(100):
                st()
            # 看一下cloud最近的有值的点是不是和joint的坐标大差不差
            # st()


        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0


        # (Pdb) data_numpy.shape
        # (1080, 1920, 3)
        # show_view(data_numpy, name='input_1')
        # st()

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        
        # show_view(input, name='input_2')
        # st()

        input_before_transfrom = input.copy()

        if self.transform:
            input = self.transform(input)


        ### lcc debugging cloud transform
        # if True:
            # st()
            ### 这个transform莫名奇妙插值太邪门了，还是在之后pad把
            # height, width, _ = data_numpy.shape
            # c = np.array([width / 2.0, height / 2.0])
            # # s = get_scale((width, height), self.image_size)
            # s = get_scale((width, height), (2025, 1080))
            # r = 0
            # trans_cloud = get_affine_transform(c, s, r, self.image_size)

            # cloud = cv2.warpAffine(cloud, trans_cloud, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
            # ### 默认pad的地方cloud_mask=0，表示有值，但是应该问题不大
            # cloud_mask = cv2.warpAffine(cloud_mask.astype(np.uint8), trans_cloud, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
            # cloud_mask = cloud_mask<0.5
            # st()

        for n in range(nposes):
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        if 'pred_pose2d' in db_rec and db_rec['pred_pose2d'] != None:
            # For convenience, we use predicted poses and corresponding values at the original heatmaps
            # to generate 2d heatmaps for Campus and Shelf dataset.
            # You can also use other 2d backbone trained on COCO to generate 2d heatmaps directly.
            pred_pose2d = db_rec['pred_pose2d']
            for n in range(len(pred_pose2d)):
                for i in range(len(pred_pose2d[n])):
                    pred_pose2d[n][i, 0:2] = affine_transform(pred_pose2d[n][i, 0:2], trans)

            input_heatmap = self.generate_input_heatmap(pred_pose2d)
            input_heatmap = torch.from_numpy(input_heatmap)
        else:
            input_heatmap = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])

        # hm_sigma = self.sigma if not self.rand_sigma else random.random() * 3 + 1
        ### test fix sigma=3
        # hm_sigma = random.random() * 3 + 1 if (self.rand_sigma and self.is_train) else self.sigma

        if self.rand_sigma and self.is_train:
            if self.rand_sigma_gau:
                hm_sigma = max(min(gauss(2.5, 0.5), 4), 1) ### 2.5为中心的高斯
                # print(f'rand_sigma_gau hm_sigma:{hm_sigma}')
            elif self.rand_sigma_sample:
                hm_sigma = random.sample([1, 1.5, 2, 2.5, 3, 3.5, 4], 1)[0]
            else:
                hm_sigma = random.random() * 3 + 1
                # print(f'rand_sigma hm_sigma:{hm_sigma}')
        else:
            hm_sigma = self.sigma
            # print(f'fixed hm_sigma:{hm_sigma}')


        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis, hm_sigma)

        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        if self.rand_sigma:
            hm_sigma_map = torch.ones(1, input.shape[1], input.shape[2]) * hm_sigma
            input = torch.cat((input, hm_sigma_map), dim=0)
            
        ##### <generate depth pseudo label> #####
        if self.pseudep:
            # st()
            # joints
            # self.limbs
            ### 先不考虑vis，即是否超出image，直接warp
            ### 有一些limb不贴上也没关系，直接用org depth即可
            limb_depth_width = 16
            # limb_depth_width = 26
            limb_depth_tmp_all = np.zeros((self.image_size[1], self.image_size[0]))

            for one_body_3d, one_body_2d, one_body_vis in zip(joints_3d, joints, joints_vis): ### 注意这里用joints_vis，因为包含了是否超出image
                
                one_body_3d_cp = one_body_3d.copy()
                one_body_2d_cp = one_body_2d.copy()
                one_body_vis_cp = one_body_vis.copy()

                ### 为了尽量不动kinoptic db，在这里变换
                import pickle
                with open('calib_list.pkl','rb') as calib_list_f:
                    calib_list = pickle.load(calib_list_f)
                
                # print(f'one_body_3d1:{one_body_3d_cp}')

                # # st()
                ### 先变换到raw data坐标系下
                one_body_3d_cp /= 1000
                # Coordinate transformation
                M = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])
                one_body_3d_cp = one_body_3d_cp.dot(np.linalg.inv(M))

                # print(f'one_body_3d2:{one_body_3d_cp}')

                # ### 转化为rgb坐标系
                one_body_3d_cp = self.joints_to_color(one_body_3d_cp, calib_list[db_rec['seq']][db_rec['camera_index']])

                # print(f'one_body_3d3:{one_body_3d_cp}')

                joint_ab_2d_list = []
                depth_ab_list = []
                for limb in self.limbs:
                    limb_depth_tmp = np.zeros((self.image_size[1], self.image_size[0]))
                    limb_depth_tmp_center = [self.image_size[1] // 2, self.image_size[0] // 2]

                    joint_a_3d = one_body_3d_cp[limb[0]]
                    joint_b_3d = one_body_3d_cp[limb[1]]

                    joint_a_vis = one_body_vis_cp[limb[0]]
                    joint_b_vis = one_body_vis_cp[limb[1]]
                    if not(joint_a_vis[0] and joint_b_vis[0]):
                        continue


                    depth_a = joint_a_3d[2]
                    depth_b = joint_b_3d[2]
                    # print(f'joint_a_3d:{joint_a_3d}, joint_b_3d:{joint_b_3d}|depth_a:{depth_a}, depth_b:{depth_b}')

                    joint_a_2d = one_body_2d_cp[limb[0]]
                    joint_b_2d = one_body_2d_cp[limb[1]]
                    joint_ab_2d_list.append((joint_a_2d, joint_b_2d))
                    depth_ab_list.append((depth_a, depth_b))

                    dist_ab = int(np.sqrt(((joint_a_2d - joint_b_2d) ** 2).sum()))


                    # todo
                    ### contruct limb depth rect
                    ### 测试旋转之后的端点是否对应
                    # add_pix_b_check = 20 # 末尾代表b，加长的对应的关节点应该是b(黑色)，check才通过
                    add_pix_b_check = 0

                    ### 这个为了保证关节点处不留缝隙，延长limb depth
                    # add_pix_ab = 0
                    add_pix_ab = 10

                    if dist_ab + add_pix_b_check + 2*add_pix_ab > limb_depth_tmp.shape[1] - 1: ### limb_rect太长的不要
                        continue

                    limb_rect = np.tile(np.linspace(depth_a, depth_b, dist_ab + add_pix_b_check + 2*add_pix_ab), (limb_depth_width, 1))
                    if dist_ab % 2 == 0:
                        limb_depth_tmp[limb_depth_tmp_center[0] - limb_depth_width // 2:limb_depth_tmp_center[0] + limb_depth_width // 2, \
                            limb_depth_tmp_center[1] - dist_ab // 2 - add_pix_ab:limb_depth_tmp_center[1] + dist_ab // 2 + add_pix_b_check + add_pix_ab] = limb_rect
                    else:
                        limb_depth_tmp[limb_depth_tmp_center[0] - limb_depth_width // 2:limb_depth_tmp_center[0] + limb_depth_width // 2, \
                            limb_depth_tmp_center[1] - dist_ab // 2 - add_pix_ab:limb_depth_tmp_center[1] + dist_ab // 2 + 1 + add_pix_b_check + add_pix_ab] = limb_rect

                    # print(f'limb_rect.shape:{limb_rect.shape}')
                    # print(f'limb_depth_tmp select shape:{limb_depth_tmp[limb_depth_tmp_center[0] - limb_depth_width // 2:limb_depth_tmp_center[0] + limb_depth_width // 2, limb_depth_tmp_center[1] - dist_ab // 2:limb_depth_tmp_center[1] + dist_ab // 2].shape}')
                    
                    ### check rect
                    # (Pdb) depth_a
                    # 3.045184261560469
                    # (Pdb) depth_b
                    # 3.001152206921019
                    # (Pdb) limb_depth_tmp.max()
                    # 3.045184261560469
                    # (Pdb) limb_depth_tmp[limb_depth_tmp>0].min()
                    # 3.001152206921019


                    vector_b2a = joint_a_2d - joint_b_2d
                    vector_b2a_unit = vector_b2a / LA.norm(vector_b2a)
                    vector_b2a_rect_unit = [-1.0, 0]

                    inner = np.inner(vector_b2a_unit, vector_b2a_rect_unit)
                    norms = LA.norm(vector_b2a_unit) * LA.norm(vector_b2a_rect_unit)

                    cos = inner / norms
                    rad = np.arccos(np.clip(cos, -1.0, 1.0))
                    deg = np.rad2deg(rad)

                    if np.isnan(deg):
                        # print('deg nan')
                        # print(f'joint_a_2d:{joint_a_2d}, joint_b_2d:{joint_b_2d}')
                        continue

                    # print(f'deg:{deg}')

                    ### check
                    c_tmp, s_tmp = np.cos(rad), np.sin(rad)
                    R_tmp = np.array(((c_tmp,-s_tmp), (s_tmp, c_tmp)))
                    vector_b2a_unit_rot = np.dot(R_tmp, vector_b2a_unit)

                    # print(f'org_vector_b2a_unit_rot:{vector_b2a_unit_rot}')
                    # st()
                    if vector_b2a[1] < 0:
                        deg = 360 - deg

                    rad = np.deg2rad(deg)
                    c_tmp, s_tmp = np.cos(rad), np.sin(rad)
                    R_tmp = np.array(((c_tmp,-s_tmp), (s_tmp, c_tmp)))
                    vector_b2a_unit_rot = np.dot(R_tmp, vector_b2a_unit)

                    # print(f'now_vector_b2a_unit_rot:{vector_b2a_unit_rot}')


                    ### warp limb depth rect to image space
                    height, width = limb_depth_tmp.shape
                    c_limb = np.array([(joint_a_2d[0] + joint_b_2d[0]) / 2.0, (joint_a_2d[1] + joint_b_2d[1])  / 2.0])
                    s_limb = get_scale(self.image_size, self.image_size)

                    # 最主要的得到rotation rotation指的是旋转的θ，正值表示逆时针，负值表示逆时针
                    r_limb = 360 - deg

                    trans_limb_depth_tmp = get_affine_transform(c_limb, s_limb, r_limb, self.image_size, inv=1)
                    limb_depth_tmp_warp = cv2.warpAffine(
                        limb_depth_tmp,
                        trans_limb_depth_tmp, (int(self.image_size[0]), int(self.image_size[1])),
                        flags=cv2.INTER_LINEAR)

                    # st()
                    ### check
                    # (Pdb) limb_depth_tmp_warp.max()
                    # 3.045184261560469
                    # (Pdb) limb_depth_tmp_warp[limb_depth_tmp_warp>0].min()
                    # 3.001152206921019


                    ### vis
                    import cv2 as cv
                    show_h = 540;show_w = 960
                    def show_depth(depthmap, name=None):
                        # depthmap = depthmap / 8.0
                        depthmap = depthmap / depthmap.max()
                        depthmap = (depthmap * 255.0).astype(np.uint8) 
                        depthmap =  cv.applyColorMap(depthmap, cv.COLORMAP_PARULA)

                        cv.namedWindow(name,0)
                        cv.resizeWindow(name, show_w, show_h)
                        cv.imshow(name, depthmap)
                        if 113 == cv.waitKey(100):
                            st()
                    
                    def show_depth_wimg(depthmap, input_before_transfrom, name=None):
                        # depthmap = depthmap / 8.0
                        depthmap = depthmap / depthmap.max()
                        depthmap = (depthmap * 255.0).astype(np.uint8) 
                        depthmap =  cv.applyColorMap(depthmap, cv.COLORMAP_PARULA)

                        img_to_show = (input_before_transfrom*0.3 + depthmap*0.7).astype(np.uint8) 

                        cv.namedWindow(name,0)
                        cv.resizeWindow(name, show_w, show_h)
                        cv.imshow(name, img_to_show)
                        if 113 == cv.waitKey(100):
                            st()
                    
                    def show_depth_wimg_wj(depthmap, input_before_transfrom, joint_a_2d, joint_b_2d, name=None):
                        # depthmap = depthmap / 8.0
                        min_depth = min(depth_a, depth_b)
                        max_depth = max(depth_a, depth_b)
                        depthmap[depthmap<min_depth] = min_depth
                        depthmap -= min_depth

                        ### org show
                        depthmap = depthmap / depthmap.max()
                        # st()
                        depthmap = (depthmap * 255.0).astype(np.uint8) 
                        depthmap =  cv.applyColorMap(depthmap, cv.COLORMAP_PARULA)

                        img_to_show = (input_before_transfrom*0.3 + depthmap*0.7).astype(np.uint8) 

                        img_to_show = cv.circle(img_to_show, (int(joint_a_2d[0]),int(joint_a_2d[1])), 2, (255, 255, 0), 2)
                        
                        ### 黑色点是b
                        img_to_show = cv.circle(img_to_show, (int(joint_b_2d[0]),int(joint_b_2d[1])), 2, (0, 0, 0), 2)

                        cv.namedWindow(name,0)
                        cv.resizeWindow(name, show_w, show_h)
                        cv.imshow(name, img_to_show)
                        if 113 == cv.waitKey(100):
                            st()
                    
                    # show_depth(limb_depth_tmp, name='limb_depth_tmp')
                    # show_depth(limb_depth_tmp_warp, name='limb_depth_tmp_warp')
                    # show_depth_wimg(limb_depth_tmp_warp, input_before_transfrom, name='show_depth_wimg')
                    # show_depth_wimg_wj(limb_depth_tmp_warp, input_before_transfrom, joint_a_2d, joint_b_2d, name='show_depth_wimg_wj')
                    # from time import sleep;sleep(0.2)

                    limb_depth_tmp_all = np.maximum(limb_depth_tmp_all, limb_depth_tmp_warp) 

                ### 对每一个人用depth max min限制
                # from time import sleep
                # sleep(0.3)
                def show_depth_wimg_wj_all(depthmap, input_before_transfrom, joint_ab_2d_list, depth_ab_list, name=None):
                    min_depth = 1e4
                    max_depth = -1e4
                    for (depth_a, depth_b) in depth_ab_list:
                        min_depth = min(min_depth, depth_a)
                        min_depth = min(min_depth, depth_b)
                        max_depth = max(max_depth, depth_a)
                        max_depth = max(max_depth, depth_a)

                    depthmap[depthmap<min_depth] = min_depth
                    depthmap -= min_depth

                    ### org show
                    depthmap = depthmap / depthmap.max()
                    # st()
                    depthmap = (depthmap * 255.0).astype(np.uint8) 
                    depthmap =  cv.applyColorMap(depthmap, cv.COLORMAP_PARULA)

                    img_to_show = (input_before_transfrom*0.3 + depthmap*0.7).astype(np.uint8) 

                    ### to debug
                    for (joint_a_2d_tmp, joint_b_2d_tmp) in joint_ab_2d_list:
                        img_to_show = cv.circle(img_to_show, (int(joint_a_2d_tmp[0]),int(joint_a_2d_tmp[1])), 2, (255, 0, 255), 2)
                        ### 黑色点是b
                        img_to_show = cv.circle(img_to_show, (int(joint_b_2d_tmp[0]),int(joint_b_2d_tmp[1])), 2, (255, 0, 255), 2)

                    cv.namedWindow(name,0)
                    cv.resizeWindow(name, show_w, show_h)
                    cv.imshow(name, img_to_show)
                    if 113 == cv.waitKey(100):
                        st()
                # show_depth_wimg_wj_all(limb_depth_tmp_all, input_before_transfrom, joint_ab_2d_list, depth_ab_list, name='show_depth_wimg_wj_all')
        
            # print('over')
            limb_depth = limb_depth_tmp_all[np.newaxis, :]
        ##### </generate depth pseudo label> #####
        


        # 注意这里面出去的joints_3d和joints
        # (Pdb) meta[0]['joints'].shape
        # (10, 15, 2)
        # (Pdb) meta[0]['joints_3d'].shape
        # (10, 15, 3)
        # 这个10代表的是self.maximum_person

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]
        
        # 我需要知道3d的空间的范围一般是多少
        # st()
        # (Pdb) len(joints_3d)
        # 3
        # 3个instance
        # 看了这个函数之后，我感觉应该是joints_3d这个之前就已经映射到grid坐标了
        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
        
        ### org meta
        ### 加上cloud batch_size只能为1
        # meta = {
        #     'image': image_file,
        #     'num_person': nposes,
        #     'joints_3d': joints_3d_u,
        #     'joints_3d_vis': joints_3d_vis_u,
        #     'roots_3d': roots_3d,
        #     'joints': joints_u,
        #     'joints_vis': joints_vis_u,
        #     'center': c,
        #     'scale': s,
        #     'rotation': r,
        #     'camera': db_rec['camera'], 
        #     'seq': db_rec['seq'],
        #     'camera_index': db_rec['camera_index'],
        #     'color_index': db_rec['color_index'],
        #     'depth_index': db_rec['depth_index'],
        #     'cloud': cloud,
        #     'cloud_mask': cloud_mask,
        #     'aligned_depth' : aligned_depth,
        # }
        

        ### 在多gpu的时候，下面这里不能传seq进去，而应该传一个seq_code
        ### lcc meta1
        # meta = {
        #     'image': image_file,
        #     'num_person': nposes,
        #     'joints_3d': joints_3d_u,
        #     'joints_3d_vis': joints_3d_vis_u,
        #     'roots_3d': roots_3d,
        #     'joints': joints_u,
        #     'joints_vis': joints_vis_u,
        #     'center': c,
        #     'scale': s,
        #     'rotation': r,
        #     'camera': db_rec['camera'], 
        #     'seq': db_rec['seq'],
        #     'camera_index': db_rec['camera_index'],
        #     'color_index': db_rec['color_index'],
        #     'depth_index': db_rec['depth_index'],
        #     # 'cloud': cloud,
        #     # 'cloud_mask': cloud_mask,
        #     'aligned_depth' : aligned_depth,
        #     'limb_depth' : limb_depth,
        #     'depth_data_numpy' : depth_data_numpy,
        #     'aligned_depth_read_flag' : aligned_depth_read_flag,
        # }

        ### lcc meta2
        ### 注意seq2code
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera'], 
            'seq_code': self.seq_to_code[db_rec['seq']],
            'camera_index': db_rec['camera_index'],
            'color_index': db_rec['color_index'],
            'depth_index': db_rec['depth_index'],
            # 'cloud': cloud,
            # 'cloud_mask': cloud_mask,
            'aligned_depth' : aligned_depth,
            'limb_depth' : limb_depth,
            # 'depth_data_numpy' : depth_data_numpy,
            # 'aligned_depth_read_flag' : aligned_depth_read_flag,
            'proj_cloud_color_mask':proj_cloud_color_mask[np.newaxis,:],
            'tensor_inputs':input
        }


        # st()
        # input已经resize了，pose同样
        # (Pdb) input.shape
        # torch.Size([3, 512, 960])

        # st()
        # (Pdb) input.mean()
        # tensor(0.5164)
        # (Pdb) input.max()
        # tensor(2.6400)
        # (Pdb) input.min()
        # tensor(-2.1179)
        
        ### 
        ### 注意这里直接传出去data_numpy
        # return input, target_heatmap, target_weight, target_3d, meta, input_heatmap
        return data_numpy, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)


    def generate_target_heatmap(self, joints, joints_vis, hm_sigma):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = hm_sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if joints_vis[n][joint_id, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_target_heatmaporg(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if joints_vis[n][joint_id, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        # st()
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                # st()
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)

        # st()
        return target

    def generate_input_heatmap(self, joints):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: input_heatmap
        '''
        nposes = len(joints)
        num_joints = self.cfg.NETWORK.NUM_JOINTS

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n][:, 0:2] / feat_stride, np.ones((num_joints, 1)))
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    if 'campus' in self.dataset_name:
                        max_value = 1.0
                    else:
                        max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0
                        # max_value = max_value**0.5
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2)) * max_value

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        return target




