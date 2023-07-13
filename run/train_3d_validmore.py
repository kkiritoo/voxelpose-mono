# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json

import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, validate_3d, train_3d_validmore
from utils.utils import create_logger, create_logger_lcc
from utils.utils import save_checkpoint, load_checkpoint, load_model_state, lcc_load_checkpoint, lcc_save_checkpoint, lcc_load_checkpoint_cpnprn
from utils.utils import load_backbone_panoptic, lcc_load_backbone_kinoptic
import dataset
import models

from pdb import set_trace as st

print('shit')
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    parser.add_argument('--use_small_dataset', action='store_true', help='use small dataset')
    parser.add_argument('--use_large_dataset', action='store_true', help='use large dataset')
    parser.add_argument('--out_3d_pose_vid', action='store_true', help='out 3d pose vid')
    parser.add_argument('--out_3d_pose_vid_namedset', action='store_true', help='out 3d pose vid')
    parser.add_argument('--jrn_space', type=str, help='out 3d pose vid')
    parser.add_argument('--jrn_grid', type=str, help='out 3d pose vid')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    ################### <org param settings> ###################
    # fix backbone
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            ### debugging train 2d net
            params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
            # params.requires_grad = True
    for params in model.module.root_net.parameters():
        params.requires_grad = True
    for params in model.module.pose_net.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)
    ################### </org param settings> ###################

    # ################### <lcc param settings> ###################
    # # load 现成的model，只train unet

    # if model.module.backbone is not None:
    #     for params in model.module.backbone.parameters():
    #         ### debugging train 2d net
    #         params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    #         # params.requires_grad = True
    # for params in model.module.root_net.parameters():
    #     params.requires_grad = False
    # for params in model.module.pose_net.parameters():
    #     params.requires_grad = False
    # if hasattr(model.module, 'unet'):
    #     for params in model.module.unet.parameters():
    #         params.requires_grad = True    

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # # optimizer = optim.Adam(model.module.parameters(), lr=lr)
    # ################### </lcc param settings> ###################

    return model, optimizer


def lcc_get_optimizer(model):
    lr = config.TRAIN.LR
    use_unet = config.NETWORK.USE_UNET
    unet_train_by_turn = config.NETWORK.UNET_TRAIN_BY_TURN

    use_jrn = config.NETWORK.USE_JRN
    

    # ################### <org param settings> ###################
    # # fix backbone
    # if model.module.backbone is not None:
    #     for params in model.module.backbone.parameters():
    #         ### debugging train 2d net
    #         params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    #         # params.requires_grad = True
    
    
    # for params in model.module.root_net.parameters():
    #     params.requires_grad = True
    # for params in model.module.pose_net.parameters():
    #     params.requires_grad = True

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # # optimizer = optim.Adam(model.module.parameters(), lr=lr)
    # ################### </org param settings> ###################

    # ################### <lcc param settings> ###################
    # # load 现成的model，只train unet

    # if model.module.backbone is not None:
    #     for params in model.module.backbone.parameters():
    #         ### debugging train 2d net
    #         params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    #         # params.requires_grad = True
    # for params in model.module.root_net.parameters():
    #     params.requires_grad = False
    # for params in model.module.pose_net.parameters():
    #     params.requires_grad = False
    # if hasattr(model.module, 'unet'):
    #     for params in model.module.unet.parameters():
    #         params.requires_grad = True    

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # # optimizer = optim.Adam(model.module.parameters(), lr=lr)
    # ################### </lcc param settings> ###################

    
    # load 现成的model，只train unet
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            ### debugging train 2d net
            params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
            # params.requires_grad = True
    for params in model.module.root_net.parameters():
        # st()
        if unet_train_by_turn:
            params.requires_grad = True
        else:
            # params.requires_grad = False if use_unet or use_jrn else True
            params.requires_grad = False if use_unet else True
    
    if hasattr(model.module, 'pose_net'):
        for params in model.module.pose_net.parameters():
            # st()
            if unet_train_by_turn:
                params.requires_grad = True
            else:
                # params.requires_grad = False if use_unet or use_jrn else True
                params.requires_grad = False if use_unet else True
    
    # if hasattr(model.module, 'unet'):
    if use_unet:
        for params in model.module.unet.parameters():
            params.requires_grad = True    
            
    if hasattr(model.module, 'joint_refine_net'):
        for params in model.module.joint_refine_net.parameters():
            params.requires_grad = True

    ### 注意这里optimizer已经记录了所有的requires_grad的参数，也就是说，你之后再设置requires_grad是没用的！！！
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    return model, optimizer


def valid_process(config, model, test_loader, logger, final_output_dir, epoch, optimizer, best_precision):
    precision = validate_3d(config, model, test_loader, final_output_dir)

    print(f"now best_precision {best_precision}, precision {precision}")
    if precision > best_precision:
        best_precision = precision
        best_model = True
        print(f"best_model = True")
    else:
        best_model = False
        print(f"best_model = False")
        

    logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'state_dict': model.module.state_dict(),
    #     'precision': best_precision,
    #     'optimizer': optimizer.state_dict(),
    # }, best_model, final_output_dir)
    lcc_save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.module.state_dict(),
        'precision': best_precision,
        'optimizer': optimizer.state_dict(),
    }, best_model, final_output_dir)

    return best_precision

def valid_process_randsigma(config, model, test_loader_list, logger, final_output_dir, epoch, optimizer, best_precision):
    for test_loader, test_dataset_sigma in test_loader_list:
        # st()
        precision = validate_3d(config, model, test_loader, final_output_dir, test_dataset_sigma=test_dataset_sigma)

        print(f"test_dataset_sigma {test_dataset_sigma} now best_precision {best_precision}, precision {precision}")
        if precision > best_precision:
            best_precision = precision
            best_model = True
            print(f"best_model = True")
        else:
            best_model = False
            print(f"best_model = False")
            

        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))

        # st()
        # print('debugging no save')
        ## lcc debugging
        lcc_save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    return best_precision

def main():
    # torch.multiprocessing.set_start_method('spawn')
    
    args = parse_args()

    config['USE_SMALL_DATASET'] = args.use_small_dataset
    config['USE_LARGE_DATASET'] = args.use_large_dataset
    config['OUT_3D_POSE_VID'] = args.out_3d_pose_vid
    config['OUT_3D_POSE_VID_NAMEDSET'] = args.out_3d_pose_vid_namedset
    
    config.JRN_SPACE = 300
    config.JRN_GRID = 64

    if args.use_large_dataset:
        print('using large dataset......')
        config.DATASET.NAMEDATASET = "large"
        # config.DATASET.NODE = 1
        ### 第一个epoch要全部BATCH_SIZE = 1否则pickle会报错？不行还是报错
        ### 貌似是worker必须是0？
        # config.TRAIN.BATCH_SIZE = 1
        # config.TEST.BATCH_SIZE = 1
        # config.WORKERS = 0

        if config.DATASET.INTERP_25J:
            config.DATASET.NAMEDATASET = "large_interp25j"
        
        if config.DATASET.KIPROJ:
            config.DATASET.NAMEDATASET = "large_kiproj"
            
        if config.DATASET.WHOLEBODY:
            config.DATASET.NAMEDATASET = "large_wholebody"
            if args.jrn_space is not None:
                config.JRN_SPACE = int(args.jrn_space)
            # st()
            if args.jrn_grid is not None:
                config.JRN_GRID = int(args.jrn_grid)

            if config.DATASET.WHOLEBODY_MULTIPERSON:
                config.DATASET.NAMEDATASET = "large_wholebody_mp"

    if config.NETWORK.USE_UNET:
        assert config.NETWORK.F_WEIGHT == True
    
    from os import path as osp
    if args.use_small_dataset or args.use_large_dataset:
        logger, final_output_dir, tb_log_dir = create_logger_lcc(
            config, args.cfg, 'train', args.use_small_dataset, args.use_large_dataset)
    else:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'train')

    # st()

    if config['OUT_3D_POSE_VID']:
        print('OUT_3D_POSE_VID mode, org TEST BATCH_SIZE:{config.TEST.BATCH_SIZE}')
        config.TEST.BATCH_SIZE = 1 # out 3d pose vid
        print('now TEST BATCH_SIZE:{config.TEST.BATCH_SIZE}')
        if config['OUT_3D_POSE_VID_NAMEDSET']:
            config.DATASET.NAMEDATASET = "OUT_3D_POSE_VID"

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # st()
    if not config['OUT_3D_POSE_VID']:
        train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
            config, config.DATASET.TRAIN_SUBSET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True)
            
        print(f'train : {len(train_dataset)}')


    if config.NETWORK.RAND_SIGMA:
        test_loader_list = []
        # sigma_list = [1, 1.5, 2, 2.5, 3, 3.5, 4] ### rand_sigma
        # sigma_list = [1.5] ### rand_sigma test
        # sigma_list = [1, 1.5, 2, 2.5, 3, 3.5, 4] ### rand_sigma test

        sigma_list = [2.5] ### rand_sigma_gau 
        for sigma in sigma_list:
            test_dataset_sigma = eval('dataset.' + config.DATASET.TEST_DATASET)(
                config, config.DATASET.TEST_SUBSET, False,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            test_dataset_sigma.sigma = sigma
            
            test_loader_sigma = torch.utils.data.DataLoader(
                test_dataset_sigma,
                batch_size=config.TEST.BATCH_SIZE * len(gpus),
                shuffle=False,
                num_workers=config.WORKERS,
                pin_memory=True)
            # ### debugging test shuffle
            # test_loader_sigma = torch.utils.data.DataLoader(
            #     test_dataset_sigma,
            #     batch_size=config.TEST.BATCH_SIZE * len(gpus),
            #     shuffle=True,
            #     num_workers=config.WORKERS,
            #     pin_memory=True)

            test_loader_list.append((test_loader_sigma, test_dataset_sigma.sigma))

        print(f'test : {len(test_dataset_sigma)}')
    else:
        test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
            config, config.DATASET.TEST_SUBSET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.TEST.BATCH_SIZE * len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)

        print(f'test : {len(test_dataset)}')

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # model, optimizer = get_optimizer(model)
    model, optimizer = lcc_get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0

    if not config['OUT_3D_POSE_VID'] and not config.NETWORK.USE_WHOLEBODY_BACKBONE:
        if (not config.NETWORK.USE_PRECOMPUTED_HM) and config.NETWORK.PRETRAINED_BACKBONE:
            if config.NETWORK.PRETRAINED_BACKBONE == "models/pose_resnet50_panoptic.pth.tar":
                model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
            else:
                model = lcc_load_backbone_kinoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
        pass
    
    #### 注意这里在测试的时候一定要load全部的backbone，否则jrn没有输出
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = lcc_load_checkpoint(model, optimizer, final_output_dir, config=config)
        # st()
        # if config.NETWORK.RAND_SIGMA:
        #     ### 只load cpn prn
        #     start_epoch, model, optimizer, best_precision = lcc_load_checkpoint_cpnprn(model, optimizer, final_output_dir, config=config)
        # else:
        #     # start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)
        #     start_epoch, model, optimizer, best_precision = lcc_load_checkpoint(model, optimizer, final_output_dir, config=config)

    ### lcc debugging: show model params 为了验证是不是真的改变了pose_net或者root_net的参数
    # test_net_state_dict = model.module.unet.state_dict()
    # for key in test_net_state_dict:
    #     # st()
    #     print(test_net_state_dict[key])

        # (Pdb) test_net_state_dict[key].mean()                              
        # tensor(-0.0037, device='cuda:0')                                   
        # (Pdb) test_net_state_dict[key].min()                               
        # tensor(-0.4895, device='cuda:0')                                   
        # (Pdb) test_net_state_dict[key].max()                               
        # tensor(0.4297, device='cuda:0')
        ### 结果验证pose_net参数并没有改变，是一样的，那为啥validate的结果不同呢


        # (Pdb) test_net_state_dict[key].mean()                              
        # tensor(0.0176, device='cuda:0')                                    
        # (Pdb) test_net_state_dict[key].min()                               
        # tensor(-0.1780, device='cuda:0')                                   
        # (Pdb) test_net_state_dict[key].max()                               
        # tensor(0.1923, device='cuda:0') 
        ### 结果验证unet参数改变了

        ### 在pose_net root_net参数都没有改变的情况下



    ### lcc debugging no unet:测试不适用unet的结果，load checkpoint之后再设置self.use_unet=False即可，计划通
    ### 注意这里必须用model.module
    # st()
    # model.module.use_unet = False
    # config.NETWORK.USE_UNET = False

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }


    ### 1jrn turn
    best_precision = 0


    ### debugging 只train一个epoch
    end_epoch = start_epoch+1


    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        # # lcc debugging validate_3d
        # precision = validate_3d(config, model, test_loader, final_output_dir);st()
        # if config.NETWORK.RAND_SIGMA:
        #     best_precision = valid_process_randsigma(config, model, test_loader_list, logger, final_output_dir, epoch, optimizer, best_precision);st()
        # else:
        #     best_precision = valid_process(config, model, test_loader, logger, final_output_dir, epoch, optimizer, best_precision);st()
        
        

        if config['OUT_3D_POSE_VID']:
            # st()
            precision = validate_3d(config, model, test_loader_list[0][0], final_output_dir)
            # st()
        
        if config.NETWORK.RAND_SIGMA:
                
            ### cpn prn turn
            # best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=False)



            ### debugging for wholebody 单人 全身
            ### 2jrn turn
            best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=True)

            # if config.TRAIN.CPNPRNF > 0:
            #     if epoch <= config.TRAIN.CPNPRNF:
            #         best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=False)
            #     else:
            #         best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=True)
            # else:
            #     if epoch % 2 == 0:
            #         best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=False)
            #     else:
            #         best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process_randsigma, test_loader_list, logger, best_precision, JRN_TRAIN_BY_TURN=True, jrn_turn=True)
        else:
            best_precision = train_3d_validmore(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, valid_process, test_loader, logger, best_precision)

        if config.NETWORK.RAND_SIGMA:
            best_precision = valid_process_randsigma(config, model, test_loader_list, logger, final_output_dir, epoch, optimizer, best_precision)
        else:
            best_precision = valid_process(config, model, test_loader, logger, final_output_dir, epoch, optimizer, best_precision)


    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()