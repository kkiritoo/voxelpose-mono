# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pdb import set_trace as st
from os import path as osp
import warnings
import torch.nn as nn
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

from mmpose.core.post_processing import get_affine_transform as mmp_get_affine_transform

from utils.transforms import get_affine_transform, get_scale


import cv2 as cv
import cv2
import numpy as np
import torch
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False




show_hm_flag = False
# show_hm_flag = True



mmpose_root = '/data/lichunchi/pose/tmp/mmpose'

det_config = osp.join(mmpose_root, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_config = osp.join(mmpose_root, 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py')
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

det_cat_id = 1
bbox_thr = 0.3

class HRNetWholebody(nn.Module):
    def __init__(self):
        super(HRNetWholebody, self).__init__()
        self.det_model = init_detector(det_config, det_checkpoint)
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(pose_config, pose_checkpoint)
    
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

        

    def forward(self, input_image):

        # st()
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, input_image)
 
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)
        
        ### hrnet只能接受mmdet的结果，不能接受全图
        ### 没有det到就直接放弃这张图吧
        if len(person_results) == 0:
            # person_results = [{'bbox': np.array([0, 0, 1920, 1080, 1.0], dtype=np.float32)}]
            raise Exception('no person detected!')
        
        if len(person_results) > 1:
            # print(f'len(person_results) {len(person_results)}')
            raise Exception('too many person detected!')


        assert len(person_results) >= 1
 
        # test a single image, with a list of bboxes.
 
        # optional
        # return_heatmap = False
        return_heatmap = True
 
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            input_image,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        img = input_image

        if len(returned_outputs) == 0:
            raise Exception('no person pose found!')

        heatmaps = returned_outputs[0]['heatmap'][0]

        # [ 564,  115, 1343, 1019,    0]
        # person_results[0]['bbox']
        
        bbox_xywh = pose_results[0]['bbox_xywh']
        bbox = bbox_xywh ### xywh

        from mmpose.core.bbox import bbox_xywh2cs
        from mmpose.core.bbox import bbox_cs2xywh
        mmp_image_size = np.array([288, 384]) ### hrnet image_size
        aspect_ratio = mmp_image_size[0] / mmp_image_size[1]
        center, scale = bbox_xywh2cs(
            bbox,
            aspect_ratio=aspect_ratio,
            padding=1.25,
            pixel_std=200.0)
        
        ### 先构造出经过crop&&resize之后的image
        # (Pdb) img.shape
        # (384, 288, 3)
        

        mmp_trans = mmp_get_affine_transform(center, scale, 0, mmp_image_size, inv=True)

        resized_hm = cv2.resize(heatmaps.transpose(1, 2, 0), (int(mmp_image_size[0]), int(mmp_image_size[1]))) ### resize到[288, 384]
        resized_hm = cv2.warpAffine(resized_hm, mmp_trans, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR) ### 贴到原本的[1080, 1920]

        image_size = np.array([960, 512]) ### resnet image_size
        image_size = image_size // 4 ### to [240, 128]

        ### pad && resize resized_hm
        height, width, _ = input_image.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), image_size)
        r = 0

        # (Pdb) data_numpy.shape
        # (1080, 1920, 3)
        # show_view(data_numpy, name='input_1')
        # st()

        trans = get_affine_transform(c, s, r, image_size)
        resized_hm_np = cv2.warpAffine(resized_hm, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        
        resized_hm_np = resized_hm_np.transpose(2, 0, 1)
        resized_hm_np_vis = resized_hm_np.copy()
        resized_hm_np = resized_hm_np[np.newaxis, :]
        resized_hm_out = torch.from_numpy(resized_hm_np)
        

        
        if show_hm_flag:
            # heatmap_all = np.zeros_like(heatmaps[0, :, :])
            # for j in range(heatmaps.shape[0]):
            #     heatmap = heatmaps[j, :, :]
            #     heatmap_all = np.maximum(heatmap_all, heatmap)

            # heatmap_all = (heatmap_all * 255).clip(0, 255).astype(np.uint8)
            # colored_heatmap = cv2.applyColorMap(heatmap_all, cv2.COLORMAP_JET)
        
            # resized_hm_vis = cv2.resize(colored_heatmap, (int(image_size[0]), int(image_size[1])))
            # resized_hm_vis = cv2.warpAffine(
            #     resized_hm_vis,
            #     mmp_trans, (img.shape[1], img.shape[0]),
            #     flags=cv2.INTER_LINEAR)
            # masked_image = resized_hm_vis * 0.7 + cv.cvtColor(img, cv.COLOR_RGB2BGR) * 0.3
            # masked_image = masked_image.astype(np.uint8)
            
            
            # show_h = 540;show_w = 960
            # def show_view(view_to_show, name=None):
            #     # view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)
            #     cv.namedWindow(name,0)
            #     cv.resizeWindow(name, show_w, show_h)
            #     cv.imshow(name, view_to_show)
            #     if 113 == cv.waitKey(100):
            #         st()
            
            # show_view(masked_image, 'haha')

            show_img_w_hm(heatmaps, img, image_size=mmp_image_size, mmp_trans=mmp_trans, name='org')
            # st()
            img_vis = cv2.warpAffine(img, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
            show_img_w_hm(resized_hm_np_vis, img_vis, name='warped')
            # st()

        return resized_hm_out



def show_img_w_hm(heatmaps, img, image_size=None, mmp_trans=None, name='default'):
    heatmap_all = np.zeros_like(heatmaps[0, :, :])
    for j in range(heatmaps.shape[0]):
        heatmap = heatmaps[j, :, :]
        heatmap_all = np.maximum(heatmap_all, heatmap)

    heatmap_all = (heatmap_all * 255).clip(0, 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_all, cv2.COLORMAP_JET)
    # st()
    if image_size is None and mmp_trans is None:
        resized_hm_vis = colored_heatmap
    else:
        resized_hm_vis = cv2.resize(colored_heatmap, (int(image_size[0]), int(image_size[1])))
        resized_hm_vis = cv2.warpAffine(
            resized_hm_vis,
            mmp_trans, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR)

    masked_image = resized_hm_vis * 0.7 + cv.cvtColor(img, cv.COLOR_RGB2BGR) * 0.3
    masked_image = masked_image.astype(np.uint8)
    
    
    show_h = 540;show_w = 960
    def show_view(view_to_show, name=None):
        # view_to_show = cv.cvtColor(view_to_show, cv.COLOR_RGB2BGR)
        cv.namedWindow(name,0)
        cv.resizeWindow(name, show_w, show_h)
        cv.imshow(name, view_to_show)
        if 113 == cv.waitKey(100):
            st()
    
    show_view(masked_image, name=name)

def get_pose_net():
    
    return HRNetWholebody()
