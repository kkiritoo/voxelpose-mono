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
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state, lcc_load_checkpoint, lcc_save_checkpoint
from utils.utils import load_backbone_panoptic
import dataset
import models

from pdb import set_trace as st

print('shit')
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

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

    # st()
    # load 现成的model，只train unet
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            ### debugging train 2d net
            params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
            # params.requires_grad = True
    for params in model.module.root_net.parameters():
        params.requires_grad = False if use_unet else True
    
    if hasattr(model.module, 'pose_net'):
        for params in model.module.pose_net.parameters():
            params.requires_grad = False if use_unet else True
    
    # if hasattr(model.module, 'unet'):
    if use_unet:
        for params in model.module.unet.parameters():
            params.requires_grad = True    

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    return model, optimizer

def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # st()

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

    print(f'train : {len(train_dataset)} | test : {len(test_dataset)}')

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
    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
    if config.TRAIN.RESUME:
        # start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)
        start_epoch, model, optimizer, best_precision = lcc_load_checkpoint(model, optimizer, final_output_dir)


    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        # # lcc debugging validate_3d
        # precision = validate_3d(config, model, test_loader, final_output_dir)

        # ap@25: 0.0067   ap@50: 0.4769   ap@75: 0.7304   ap@100: 0.7901  ap@125: 0.8372  ap@150: 0.8681  recall@500mm: 0.9148    mpjpe@500mm: 62.542
        # ap@25: 0.0022   ap@50: 0.3367   ap@75: 0.5989   ap@100: 0.6700  ap@125: 0.7482  ap@150: 0.7984  recall@500mm: 0.9148    mpjpe@500mm: 62.542
        # st()
        # exit()

        # lr_scheduler.step()
        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        
        precision = validate_3d(config, model, test_loader, final_output_dir)

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

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



    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
