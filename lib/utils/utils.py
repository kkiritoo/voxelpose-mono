# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))  ##
    root_output_dir = (this_dir / '..' / '..' / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (this_dir / '..' / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = tensorboard_log_dir / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def create_logger_lcc(cfg, cfg_name, phase='train', use_small_dataset=False, use_large_dataset=False):
    this_dir = Path(os.path.dirname(__file__))  ##
    root_output_dir = (this_dir / '..' / '..' / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (this_dir / '..' / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    ### 即使是同时use_small_dataset和use_large_dataset也是打到small_dataset里面
    if use_small_dataset:
        final_output_dir = root_output_dir / dataset / model / 'small_dataset' / cfg_name
    elif use_large_dataset:
        final_output_dir = root_output_dir / dataset / model / 'large_dataset' / cfg_name
    else:
        final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = tensorboard_log_dir / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def load_model_state(model, output_dir, epoch):
    file = os.path.join(output_dir, 'checkpoint_3d_epoch'+str(epoch)+'.pth.tar')
    if os.path.isfile(file):
        model.module.load_state_dict(torch.load(file))
        print('=> load models state {} (epoch {})'
              .format(file, epoch))
        return model
    else:
        print('=> no checkpoint found at {}'.format(file))
        return model


def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        precision = checkpoint['precision'] if 'precision' in checkpoint else 0
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, precision

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, 0

from pdb import set_trace as st

def orglcc_load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        precision = checkpoint['precision'] if 'precision' in checkpoint else 0

        # ### org load
        # model.module.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        # st()

        ### lcc load
        model_dict = model.module.state_dict()
        # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        new_pretrained_state_dict_filter = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(new_pretrained_state_dict_filter)
        model.module.load_state_dict(model_dict)

        ### 因为fix之前pretrain的全部，所以不需要load optim参数
        # st()
        # optim_dict = optimizer.state_dict()
        # # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        # new_pretrained_state_dict_filter = {k:v for k,v in checkpoint['optimizer'].items() if k in optim_dict.keys()}
        # optim_dict.update(new_pretrained_state_dict_filter)
        # optimizer.load_state_dict(optim_dict)
        
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, precision

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, 0


def lcc_load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar', config=None):

    ### default
    # filename = "checkpoint.pth.tar"
    filename = "model_best.pth.tar"

    ### lcc debugging: load designated checkpoint
    # filename = "checkpoint.pth.tar.bak"
    # filename = "model_best_to_show.pth.tar"
    ### 在out s_wo_unet的时候需要注释下面
    if config['OUT_3D_POSE_VID']:
        filename = "model_best.pth.tar"
    
    print('todo unet+large dataset!!!!!!!!')
    ### todo unet+large dataset!!!!!!!!

    if os.path.exists(os.path.join(output_dir, filename)): # 只要可以继续训练一定继续训练
        file = os.path.join(output_dir, filename)
    else:
        # ### 之前的node2 checkpoint
        # if config.NETWORK.USE_UNET and config.USE_SMALL_DATASET: # unet+小数据集
        #     # ap@25: 0.0025	ap@50: 0.3885	ap@75: 0.6631	ap@100: 0.7590	ap@125: 0.8010	ap@150: 0.8240	recall@500mm: 0.8767	mpjpe@500mm: 59.578
        #     file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/small_dset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960', filename)
        # elif config.NETWORK.USE_UNET and (not config.USE_SMALL_DATASET): # unet+完整数据集
        #     # ap@25: 0.0306	ap@50: 0.5165	ap@75: 0.7106	ap@100: 0.7738	ap@125: 0.8135	ap@150: 0.8344	recall@500mm: 0.8771	mpjpe@500mm: 51.086
        #     file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960', filename)
        # else: # 不用unet
        #     file = os.path.join(output_dir, filename)

        if config.NETWORK.USE_UNET and config.USE_SMALL_DATASET: # unet+小数据集
            assert False
        elif config.NETWORK.USE_UNET and (not config.USE_SMALL_DATASET): # unet+大数据集 itv6
            file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5', filename)
        else: # 不用unet
            file = os.path.join(output_dir, filename)

    # st()

    if os.path.isfile(file):
        checkpoint = torch.load(file)
        if 'state_dict' in checkpoint:
            print('load full checkpoint with epoch and precision...')
            start_epoch = checkpoint['epoch']
            precision = checkpoint['precision'] if 'precision' in checkpoint else 0
            checkpoint_state_dict = checkpoint['state_dict']
        else:
            print('load checkpoint state dict without epoch and precision...')
            print('注意这里默认load的是model_best.pth.tar并且改名为checkpoint.pth.tar')
            start_epoch = 20
            precision = 0
            checkpoint_state_dict = checkpoint

        # ### org load
        # model.module.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        # st()

        ### lcc load
        model_dict = model.module.state_dict()
        # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        new_pretrained_state_dict_filter = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(new_pretrained_state_dict_filter)
        model.module.load_state_dict(model_dict)

        ### 因为fix之前pretrain的全部，所以不需要load optim参数
        # st()
        # optim_dict = optimizer.state_dict()
        # # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        # new_pretrained_state_dict_filter = {k:v for k,v in checkpoint['optimizer'].items() if k in optim_dict.keys()}
        # optim_dict.update(new_pretrained_state_dict_filter)
        # optimizer.load_state_dict(optim_dict)
        
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, precision

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, 0

def lcc_load_checkpoint_cpnprn(model, optimizer, output_dir, filename='checkpoint.pth.tar', config=None):
    print('load only cpn prn')

    ### default
    # filename = "checkpoint.pth.tar"
    filename = "model_best.pth.tar"

    ### lcc debugging: load designated checkpoint
    # filename = "checkpoint.pth.tar.bak"
    # filename = "model_best_to_show.pth.tar"
    ### 在out s_wo_unet的时候需要注释下面
    if config['OUT_3D_POSE_VID']:
        filename = "model_best.pth.tar"
    
    print('todo unet+large dataset!!!!!!!!')
    ### todo unet+large dataset!!!!!!!!
    
    # st()
    if os.path.exists(os.path.join(output_dir, filename)): # 只要可以继续训练一定继续训练
        file = os.path.join(output_dir, filename)
    else:
        if not config.USE_SMALL_DATASET: # unet+大数据集 itv6
            if not config.NETWORK.RAND_SIGMA_FSCRATCH:
                file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5', filename)
            else:
                file = os.path.join(output_dir, filename)

            if config.NETWORK.USE_JRN:
                # file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_2', filename)

                

                if config.DATASET.WHOLEBODY_MULTIPERSON:
                    # file = os.path.join('', filename) ### todo
                    
                    ### 由于train的效果不是很好，试一下直接用现成的效果 不能直接用joint数量不同，只能finetune
                    # file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2/output/kinoptic/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_2', filename)
                    file = os.path.join(output_dir, filename)
                else:
                    print('single person load cpn prn')
                    ### 用已经train好的body
                    file = os.path.join('/data/lichunchi/pose/voxelpose-pytorch-unet2-wholebody/output/kinoptic_wholebody/multi_person_posenet_50/large_dataset/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_jrn_wholebody_face', filename)

                    # file = os.path.join(output_dir, filename)
                
        else: # 不用unet
            pass
    
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        if 'state_dict' in checkpoint:
            print('load full checkpoint with epoch and precision...')
            start_epoch = checkpoint['epoch']
            precision = checkpoint['precision'] if 'precision' in checkpoint else 0
            checkpoint_state_dict = checkpoint['state_dict']
        else:
            print('load checkpoint state dict without epoch and precision...')
            print('注意这里默认load的是model_best.pth.tar并且改名为checkpoint.pth.tar')
            start_epoch = 20
            precision = 0
            checkpoint_state_dict = checkpoint

        # ### org load
        # model.module.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('load optimizer state_dict success')
        except:
            print('load optimizer state_dict failed')

        # st()

        ### lcc load
        # model_dict = model.module.state_dict()
        # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        pretrained_state_dict_pose_net = {k.replace('pose_net.', ''):v for k,v in checkpoint['state_dict'].items() if 'pose_net.' in k}
        model.module.pose_net.load_state_dict(pretrained_state_dict_pose_net)
        pretrained_state_dict_pose_net = {k.replace('root_net.', ''):v for k,v in checkpoint['state_dict'].items() if 'root_net.' in k}
        model.module.root_net.load_state_dict(pretrained_state_dict_pose_net)

        ### 因为fix之前pretrain的全部，所以不需要load optim参数
        # st()
        # optim_dict = optimizer.state_dict()
        # # 去掉new_pretrained_state_dict中，backbone_dict不存在的weight（其实这一步不需要也行，毕竟所有都在）
        # new_pretrained_state_dict_filter = {k:v for k,v in checkpoint['optimizer'].items() if k in optim_dict.keys()}
        # optim_dict.update(new_pretrained_state_dict_filter)
        # optimizer.load_state_dict(optim_dict)
        
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, precision

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, 0


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

### 上面的checkpoint没有办法得到完整的checkpoint信息
def lcc_save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states, os.path.join(output_dir, f'model_best.pth.tar'))


def load_backbone_panoptic(model, pretrained_file):
    this_dir = os.path.dirname(__file__)
    pretrained_file = os.path.abspath(os.path.join(this_dir, '../..', pretrained_file))
    pretrained_state_dict = torch.load(pretrained_file)
    model_state_dict = model.module.backbone.state_dict()

    prefix = "module."
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
    logging.info("load backbone statedict from {}".format(pretrained_file))
    model.module.backbone.load_state_dict(new_pretrained_state_dict)

    return model

def lcc_load_backbone_kinoptic(model, pretrained_file):
    this_dir = os.path.dirname(__file__)
    pretrained_file = os.path.abspath(os.path.join(this_dir, '../..', pretrained_file))
    pretrained_state_dict = torch.load(pretrained_file)


    pretrained_state_dict = pretrained_state_dict['state_dict']
    # st()
    model_state_dict = model.module.backbone.state_dict()

    # prefix = "module."
    prefix = "backbone."

    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
    logging.info("load backbone statedict from {}".format(pretrained_file))
    model.module.backbone.load_state_dict(new_pretrained_state_dict)

    return model
