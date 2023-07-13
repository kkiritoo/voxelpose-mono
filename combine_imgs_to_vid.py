import os
import glob
import cv2

# ######### small dataset #########
# dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/small_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# s_valid_imgs_list_wo_unet = glob.glob(dirname1 + '/*_3d.png')

# dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/small_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unetd_lrecon'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# s_valid_imgs_list_w_unet = glob.glob(dirname1 + '/*_3d.png')

# dirname = '/data/lichunchi/pose/baselines/mono_depth/part-affinity/exp/vgg19/model_3d_0416'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# valid_imgs_list_rp = glob.glob(dirname1 + '/*_3d.png')

# image_lists1 = [s_valid_imgs_list_wo_unet, s_valid_imgs_list_w_unet, valid_imgs_list_rp]


# ######### whole dataset #########
# dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# valid_imgs_list_wo_unet = glob.glob(dirname1 + '/*_3d.png')

# dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unet_lrecon'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# valid_imgs_list_w_unet = glob.glob(dirname1 + '/*_3d.png')

# dirname = '/data/lichunchi/pose/baselines/mono_depth/part-affinity/exp/vgg19/model_3d_0416'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# valid_imgs_list_rp = glob.glob(dirname1 + '/*_3d.png')

# image_lists2 = [valid_imgs_list_wo_unet, valid_imgs_list_w_unet, valid_imgs_list_rp]

# ######### rgb + hm #########
# dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960'
# dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
# valid_imgs_list_rgb = glob.glob(dirname1 + '/*_rgb_hmpred_hmgt.png')




# list_name1 = ['s_voxelpose_wo_unet', 's_voxelpose_w_unet', 'residual pose']
# list_name2 = ['voxelpose_wo_unet', 'voxelpose_w_unet', 'residual pose']



#### wholebody
dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic_wholebody/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_jrn_wholebody_facehandsepv'
dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid', 'large_wholebody')
valid_imgs_list_wholebody = glob.glob(dirname1 + '/*_3d.png')

######### rgb + hm #########
dirname = '/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic_wholebody/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_jrn_wholebody_facehandsepv'
dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid', 'large_wholebody')
valid_imgs_list_rgb = glob.glob(dirname1 + '/*_rgb_hmpred_hmgt.png')

list_name = 'voxelpose-mono'


def image2video(image_list, name, fps=25):
    image_path_list = []
    for image_path in image_list:
        image_path_list.append(image_path)
    image_path_list.sort()
    temp = cv2.imread(image_path_list[0])
    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + name + '.mp4', fourcc, fps, size)
    for image_path in image_path_list:
        if image_path.endswith(".png"):
            image_data_temp = cv2.imread(image_path)
            video.write(image_data_temp)
    print("Video done！")

import numpy as np

def imagepair2video(image_list1, image_list2, name, fps=25):
    image_path_list1 = []
    for image_path in image_list1:
        image_path_list1.append(image_path)
    image_path_list1.sort()
    image_path_list2 = []
    for image_path in image_list2:
        image_path_list2.append(image_path)
    image_path_list2.sort()


    temp1 = cv2.imread(image_path_list1[0])
    temp2 = cv2.imread(image_path_list2[0])

    temp = np.concatenate([temp1, temp2], axis=0)
    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + name + '.mp4', fourcc, fps, size)

    cnt = 0
    for image_path1, image_path2 in zip(image_path_list1, image_path_list2):
        # if cnt > 100:break
        image_data_temp1 = cv2.imread(image_path1)

        image_data_temp2 = cv2.imread(image_path2)
        image_data_temp = np.concatenate([image_data_temp1, image_data_temp2], axis=0)
        video.write(image_data_temp)
        cnt += 1

    print("Video done！")

from pdb import set_trace as st

def imagelists2video(image_lists, save_name, fps=25, list_name=None):
    for image_list in image_lists:
        image_list.sort()
    
    temp_list = []
    for list_index, image_list in enumerate(image_lists):
        img_tmp = cv2.imread(image_list[0])
        if list_name is not None:
            img_tmp = cv2.putText(img_tmp, list_name[list_index], (20,20), 0, 2, (0,255,0), 3)
        temp_list.append(img_tmp)
    temp = np.concatenate(temp_list, axis=1)

    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + save_name + '.mp4', fourcc, fps, size)

    ### 最小的list长度作为frame_len
    frame_len = 1e5
    for image_list in image_lists:
        frame_len = min(frame_len, len(image_list)) - 1


    for frame_i in range(frame_len):
        image_data_temp_list = []
        for list_index, image_list in enumerate(image_lists):
            img_tmp = cv2.imread(image_list[frame_i])
            if list_name is not None:
                img_tmp = cv2.putText(img_tmp, list_name[list_index], (20,20), 0, 1, (0,255,0), 1)
            image_data_temp_list.append(img_tmp)
        try:
            image_data_temp = np.concatenate(image_data_temp_list, axis=1)
        except:
            print('fuck, skipping...')
            continue

        video.write(image_data_temp)
    
    print("Video done！")


def doubleimagelists2video(image_lists1, image_lists2, save_name, fps=25, list_name1=None, list_name2=None):
    for image_list in image_lists1:
        image_list.sort()

    for image_list in image_lists2:
        image_list.sort()

    temp_list = []
    for list_index, image_list in enumerate(image_lists1):
        img_tmp = cv2.imread(image_list[0])
        if list_name1 is not None:
            img_tmp = cv2.putText(img_tmp, list_name1[list_index], (20,20), 0, 2, (0,255,0), 3)
        temp_list.append(img_tmp)
    temp1 = np.concatenate(temp_list, axis=1)

    temp_list = []
    for list_index, image_list in enumerate(image_lists2):
        img_tmp = cv2.imread(image_list[0])
        if list_name2 is not None:
            img_tmp = cv2.putText(img_tmp, list_name2[list_index], (20,20), 0, 2, (0,255,0), 3)
        temp_list.append(img_tmp)
    temp2 = np.concatenate(temp_list, axis=1)

    temp = np.concatenate([temp1, temp2], axis=0)


    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + save_name + '.mp4', fourcc, fps, size)

    ### 最小的list长度作为frame_len
    frame_len = 1e5
    for image_list in image_lists1:
        frame_len = min(frame_len, len(image_list))
    for image_list in image_lists2:
        frame_len = min(frame_len, len(image_list))
    frame_len -= 1


    for frame_i in range(frame_len):
        image_data_temp_list = []
        for list_index, image_list in enumerate(image_lists1):
            img_tmp = cv2.imread(image_list[frame_i])
            if list_name1 is not None:
                img_tmp = cv2.putText(img_tmp, list_name1[list_index], (20,20), 0, 1, (0,255,0), 1)
            image_data_temp_list.append(img_tmp)
        try:
            image_data_temp1 = np.concatenate(image_data_temp_list, axis=1)
        except:
            print('fuck, skipping...')
            continue

        image_data_temp_list = []
        for list_index, image_list in enumerate(image_lists2):
            img_tmp = cv2.imread(image_list[frame_i])
            if list_name2 is not None:
                img_tmp = cv2.putText(img_tmp, list_name2[list_index], (20,20), 0, 1, (0,255,0), 1)
            image_data_temp_list.append(img_tmp)
        try:
            image_data_temp2 = np.concatenate(image_data_temp_list, axis=1)
        except:
            print('fuck, skipping...')
            continue
        
        image_data_temp = np.concatenate([image_data_temp1, image_data_temp2], axis=0)

        video.write(image_data_temp)
    
    print("Video done！")


def doubleimagelists2video_wrgbhm(image_lists1, image_lists2, valid_imgs_list_rgb, save_name, fps=25, list_name1=None, list_name2=None):
    for image_list in image_lists1:
        image_list.sort()

    for image_list in image_lists2:
        image_list.sort()
    
    valid_imgs_list_rgb.sort()

    temp_list = []
    for list_index, image_list in enumerate(image_lists1):
        img_tmp = cv2.imread(image_list[0])
        if list_name1 is not None:
            img_tmp = cv2.putText(img_tmp, list_name1[list_index], (20,20), 0, 2, (0,255,0), 3)
        temp_list.append(img_tmp)
    temp1 = np.concatenate(temp_list, axis=1)

    temp_list = []
    for list_index, image_list in enumerate(image_lists2):
        img_tmp = cv2.imread(image_list[0])
        if list_name2 is not None:
            img_tmp = cv2.putText(img_tmp, list_name2[list_index], (20,20), 0, 2, (0,255,0), 3)
        temp_list.append(img_tmp)
    temp2 = np.concatenate(temp_list, axis=1)


    ###### 
    image_list = valid_imgs_list_rgb
    img_tmp = cv2.imread(image_list[0])
    img_tmp_h = temp1.shape[0] # 400
    img_tmp_w = temp1.shape[1] # 1200
    
    
    temp3_resize_shape_w = img_tmp_w
    temp3_resize_shape_h = img_tmp.shape[0] * img_tmp_w // img_tmp.shape[1]

    img_tmp = cv2.resize(img_tmp,
                               (int(temp3_resize_shape_w), int(temp3_resize_shape_h)))     
    temp3 = img_tmp
    temp = np.concatenate([temp3, temp1, temp2], axis=0)
    ######


    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + save_name + '.mp4', fourcc, fps, size)

    ### 最小的list长度作为frame_len
    frame_len = 1e5
    for image_list in image_lists1:
        frame_len = min(frame_len, len(image_list))
    for image_list in image_lists2:
        frame_len = min(frame_len, len(image_list))
    frame_len -= 1


    for frame_i in range(frame_len):
        image_data_temp_list = []
        for list_index, image_list in enumerate(image_lists1):
            img_tmp = cv2.imread(image_list[frame_i])
            if list_name1 is not None:
                img_tmp = cv2.putText(img_tmp, list_name1[list_index], (20,20), 0, 1, (0,255,0), 1)
            image_data_temp_list.append(img_tmp)
        try:
            image_data_temp1 = np.concatenate(image_data_temp_list, axis=1)
        except:
            print('fuck, skipping...')
            continue

        image_data_temp_list = []
        for list_index, image_list in enumerate(image_lists2):
            img_tmp = cv2.imread(image_list[frame_i])
            if list_name2 is not None:
                img_tmp = cv2.putText(img_tmp, list_name2[list_index], (20,20), 0, 1, (0,255,0), 1)
            image_data_temp_list.append(img_tmp)
        try:
            image_data_temp2 = np.concatenate(image_data_temp_list, axis=1)
        except:
            print('fuck, skipping...')
            continue
        
        
        image_list = valid_imgs_list_rgb
        img_tmp = cv2.imread(image_list[frame_i])

        img_tmp_h = temp1.shape[0] # 400
        img_tmp_w = temp1.shape[1] # 1200
        
        temp3_resize_shape_w = img_tmp_w
        temp3_resize_shape_h = img_tmp.shape[0] * img_tmp_w // img_tmp.shape[1]

        img_tmp = cv2.resize(img_tmp,
                                (int(temp3_resize_shape_w), int(temp3_resize_shape_h)))     
        temp3 = img_tmp



        image_data_temp = np.concatenate([temp3, image_data_temp1, image_data_temp2], axis=0)

        video.write(image_data_temp)
    
    print("Video done！")


### for whole body
def singleimagelist2video_wrgbhm(image_list, valid_imgs_list_rgb, save_name, fps=25, list_name=None):
    image_list.sort()
    
    valid_imgs_list_rgb.sort()


    img_tmp = cv2.imread(image_list[0])
    if list_name is not None:
        img_tmp = cv2.putText(img_tmp, list_name, (20,20), 0, 2, (0,255,0), 3)
    temp = img_tmp

    ###### 
    image_list_rgb = valid_imgs_list_rgb
    # st()
    img_rgb_tmp = cv2.imread(image_list_rgb[0])
    
    ### 把img_tmp pad到合适的大小
    pad_len = (img_rgb_tmp.shape[1] - temp.shape[1]) // 2 # 690
    temp = np.pad(temp, ((0,0), (pad_len,pad_len), (0,0)), 'constant', constant_values=255)

    # st()
    temp = np.concatenate([img_rgb_tmp, temp], axis=0)
    # (Pdb) temp.shape
    # (2012, 2880, 3)
    ######


    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + save_name + '.mp4', fourcc, fps, size)

    ### 最小的list长度作为frame_len
    # frame_len = len(image_list)
    # frame_range = range(frame_len)

    frame_range = range(55, 250)


    for frame_i in frame_range:

        img_tmp = cv2.imread(image_list[frame_i])
        if list_name is not None:
            img_tmp = cv2.putText(img_tmp, list_name, (20,20), 0, 2, (0,255,0), 3)
        temp = img_tmp

        ###### 
        image_list_rgb = valid_imgs_list_rgb
        # st()
        img_rgb_tmp = cv2.imread(image_list_rgb[frame_i])

        
        ### 把img_tmp pad到合适的大小
        pad_len = (img_rgb_tmp.shape[1] - temp.shape[1]) // 2 # 690
        temp = np.pad(temp, ((0,0), (pad_len,pad_len), (0,0)), 'constant', constant_values=255)

        ######
        
        # st()
        image_data_temp = np.concatenate([img_rgb_tmp, temp], axis=0)

        video.write(image_data_temp)
    
    print("Video done！")


### name todo
# imagepair2video(valid_imgs_list_wo_unet, valid_imgs_list_w_unet, 'test', fps=5)

# imagelists2video(image_lists1, 'test', fps=5, list_name=list_name1)
# imagelists2video(image_lists, 'test', fps=5)

### 包含small和完整数据集的结果
# doubleimagelists2video(image_lists1, image_lists2, 'test2', fps=5, list_name1=list_name1, list_name2=list_name2)

### 包含small和完整数据集的结果，加上rgb和hm
# doubleimagelists2video_wrgbhm(image_lists1, image_lists2, valid_imgs_list_rgb, 'test2', fps=5, list_name1=list_name1, list_name2=list_name2)


singleimagelist2video_wrgbhm(valid_imgs_list_wholebody, valid_imgs_list_rgb, 'test3', fps=5, list_name=None)