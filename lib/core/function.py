from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

from utils.vis import save_debug_3d_images_wholebody_for_vid, save_debug_images_multi, save_batch_heatmaps_multi_gt_pred_lcc, save_batch_heatmaps_multi_gt_pred_lcc2
from utils.vis import save_debug_3d_images, save_debug_3d_images_for_vid, save_debug_3d_images_wholebody, save_debug_3d_images_for_vid_meta
from utils.vis import save_debug_3d_cubes


from pdb import set_trace as st
logger = logging.getLogger(__name__)


### lcc train 3d with depth recon loss
def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()
    losses_depth_recon = AverageMeter()
    

    model.train()
    # model.eval() # 这里设置eval输出才会和eval时候完全一样


    ### 注意
    if model.module.backbone is not None:
        model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly

    accumulation_steps = 4
    # accumulation_steps = 1
    accu_loss_3d = 0

    end = time.time()

    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):

        data_time.update(time.time() - end)

        if 'panoptic' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
                                                                              targets_2d=targets_2d,
                                                                              weights_2d=weights_2d,
                                                                              targets_3d=targets_3d[0])
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                              input_heatmaps=input_heatmap)
        elif 'kinoptic' in config.DATASET.TEST_DATASET:
            if config.NETWORK.USE_UNET:
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
                                                                              targets_2d=targets_2d,
                                                                              weights_2d=weights_2d,
                                                                              targets_3d=targets_3d[0])
            else:                                                           
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
                                                                                targets_2d=targets_2d,
                                                                                weights_2d=weights_2d,
                                                                                targets_3d=targets_3d[0])
                                                                            

        loss_2d = loss_2d.mean()
        loss_3d = loss_3d.mean()
        loss_cord = loss_cord.mean()
        loss_depth_recon = loss_depth_recon.mean()

        losses_2d.update(loss_2d.item())
        losses_3d.update(loss_3d.item())
        losses_cord.update(loss_cord.item())
        losses_depth_recon.update(loss_depth_recon.item())

        loss = loss_2d + loss_3d + loss_cord
        losses.update(loss.item())

        ### 由于参数只增加了unet中的参数，因此下面两个loss应该是对unet都有梯度

        # ### org step
        # # # lcc debugging不更新参数，纳尼，居然变得和val一样了，都是完全
        # if loss_cord > 0:
        #     optimizer.zero_grad()
        #     # (loss_2d + loss_cord).backward()
        #     # 为了传2d loss
        #     ### 增加了unet之后的
        #     (loss_2d + loss_cord).backward()
        #     # loss_cord.backward(retain_graph=True)
        #     optimizer.step()
        
        # if vis_flag:print("5:{}".format(torch.cuda.memory_allocated(0)))

        # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
        #     optimizer.zero_grad()
        #     accu_loss_3d.backward()
        #     optimizer.step()
        #     accu_loss_3d = 0.0
        # else:
        #     accu_loss_3d += loss_3d / accumulation_steps

        ### lcc step
        if config.NETWORK.USE_UNET and loss_depth_recon > 0:
            optimizer.zero_grad()
            loss_depth_recon.backward(retain_graph=True)
            optimizer.step()
        
        # lcc debugging不更新参数，纳尼，居然变得和val一样了，都是完全
        if loss_cord > 0:
            optimizer.zero_grad()
            loss_cord.backward()
            # 为了传2d loss
            # (loss_2d + loss_cord).backward(retain_graph=True)
            optimizer.step()

        if not config.NETWORK.USE_UNET:
            if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
                optimizer.zero_grad()
                accu_loss_3d.backward()
                optimizer.step()
                accu_loss_3d = 0.0
            else:
                accu_loss_3d += loss_3d / accumulation_steps



        batch_time.update(time.time() - end)
        end = time.time()

        ### lcc debugging: testing running
        # config.PRINT_FREQ = 1

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            if config.NETWORK.USE_UNET:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'loss_depth_recon: {loss_depth_recon.val:.6f} ({loss_depth_recon.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                    loss_cord=losses_cord, loss_depth_recon=losses_depth_recon, memory=gpu_memory_usage)
            else:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed: {speed:.1f} samples/s\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                    'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                    'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                    'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                    'Memory {memory:.1f}'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                        loss_cord=losses_cord, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_loss_cord', losses_cord.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            for k in range(len(inputs)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{:08}_{}'.format(
                    os.path.join(output_dir, 'train'), i, view_name)
                save_debug_images_multi(config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)
            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'train'), i)
            
            
            save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
            save_debug_3d_images(config, meta[0], pred, prefix2)
        


### lcc train 3d with depth recon loss
def train_3d_validmore(config, model, optimizer, loader, epoch, output_dir, writer_dict, valid_process, test_loader, logger, best_precision, JRN_TRAIN_BY_TURN=False, jrn_turn=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()
    losses_depth_recon = AverageMeter()
    losses_cord_refine = AverageMeter()
    
    model.train()
    # model.eval() # 这里设置eval输出才会和eval时候完全一样


    ### 注意
    if model.module.backbone is not None:
        model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly
    

    ### 开eval会让loss_cord突然上升
    ### 只要有参数是training=True，整个模型的training=True
    # if config.NETWORK.USE_JRN:
    #     model.module.root_net.eval()
    #     model.module.pose_net.eval()
    #     model.module.joint_refine_net.train()

    accumulation_steps = 4
    # accumulation_steps = 1
    accu_loss_3d = 0

    if config.NETWORK.UNET_TRAIN_BY_TURN:
        ### 每100个step
        unet_yici_steps_max = 100
        cpn_prn_yici_steps_max = 100
        unet_yici_step = 0
        cpn_prn_yici_step = 0
        unet_turn = True
    
    ### debugging 0729
    ### end-to-end显存不够，而且刚开始还是太慢了，还是依次训练吧
    #·jrn_turn = True

    if config.NETWORK.USE_JRN and JRN_TRAIN_BY_TURN:
        if jrn_turn:
            print('training jrn...')
        else:
            print('training cpn & prn...')

        ### lcc debugging yici:
        if not jrn_turn:
            model.module.joint_refine_net.eval() 
            model.module.root_net.train()
            model.module.pose_net.train()

            for params in model.module.root_net.parameters():
                params.requires_grad = True

            for params in model.module.pose_net.parameters():
                params.requires_grad = True

            for params in model.module.joint_refine_net.parameters():
                params.requires_grad = False
        else:
            model.module.joint_refine_net.train() 
            model.module.root_net.eval()
            model.module.pose_net.eval()

            ### 感觉好像莫名其妙的.eval()之后也变化
            ### 看一下是不是真的
            ### .eval并不会不让网络的参数被fix，只是改变了网络的self.training，还有bn?
            ### 只有像下面这样设置，才可以真正fix

            # for params in model.module.pose_net.parameters():print('before', params[0].requires_grad);break

            for params in model.module.root_net.parameters():
                params.requires_grad = False

            for params in model.module.pose_net.parameters():
                params.requires_grad = False

            for params in model.module.joint_refine_net.parameters():
                params.requires_grad = True

            # for params in model.module.pose_net.parameters():print('after', params[0].requires_grad);break


    end = time.time()
    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
        data_time.update(time.time() - end)


        # if config.NETWORK.UNET_TRAIN_BY_TURN:
        #     if unet_turn:
        #         unet_yici_step += 1
        #         if unet_yici_step >= unet_yici_steps_max:
        #             unet_yici_step = 0
        #             unet_turn = False
        #             print(f'change turn unet_turn {unet_turn}...')

        #     if not unet_turn:
        #         cpn_prn_yici_step += 1
        #         if cpn_prn_yici_step >= cpn_prn_yici_steps_max:
        #             cpn_prn_yici_step = 0
        #             unet_turn = True
        #             print(f'change turn unet_turn {unet_turn}...')

        #     ### lcc debugging yici:
        #     if not unet_turn:
        #         model.module.unet.eval() 
        #     else:
        #         model.module.unet.train() 

        #     # print(f'MASSAGE:unet_turn {unet_turn} | unet_yici_step {unet_yici_step} | cpn_prn_yici_step {cpn_prn_yici_step}')


                    
        # if 'panoptic' in config.DATASET.TEST_DATASET:
        #     pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
        #                                                                     targets_2d=targets_2d,
        #                                                                     weights_2d=weights_2d,
        #                                                                     targets_3d=targets_3d[0])
        # elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
        #     pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
        #                                                                     input_heatmaps=input_heatmap)
        # elif 'kinoptic' in config.DATASET.TEST_DATASET:
        #     if config.NETWORK.USE_UNET:
        #         if config.NETWORK.UNET_TRAIN_BY_TURN:
        #             pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
        #                                                                         targets_2d=targets_2d,
        #                                                                         weights_2d=weights_2d,
        #                                                                         targets_3d=targets_3d[0], unet_turn=unet_turn)
        #         else:
        #             pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
        #                                                                         targets_2d=targets_2d,
        #                                                                         weights_2d=weights_2d,
        #                                                                         targets_3d=targets_3d[0])
                
        #     else:
        #         # ret = model(views=inputs, meta=meta,
        #         #                                                                 targets_2d=targets_2d,
        #         #                                                                 weights_2d=weights_2d,
        #         #                                                                 targets_3d=targets_3d[0], jrn_turn=jrn_turn)
        #         # st()

        #         pred, pred_wholebody, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_cord_refine = model(views=inputs, meta=meta,
        #                                                                         targets_2d=targets_2d,
        #                                                                         weights_2d=weights_2d,
        #                                                                         targets_3d=targets_3d[0], jrn_turn=jrn_turn)
        #         # st()
        #         if pred.shape == torch.Size([1]): ### backbone no output
        #             i -= 1
        #             continue



        # 1KeyError: '/data/lichunchi/pose/voxelpose-pytorch-unet2/run/../lib/models/multi_person_posenet.py'
        # 2allow_pickle=False
        ### forward莫名其妙KeyError，valid应该不会有问题，我试过，valid如果有问题不能直接continue，否则pred gt数量不一致
        try:
            if 'panoptic' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(views=inputs, meta=meta,
                                                                                targets_2d=targets_2d,
                                                                                weights_2d=weights_2d,
                                                                                targets_3d=targets_3d[0])
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                                input_heatmaps=input_heatmap)
            elif 'kinoptic' in config.DATASET.TEST_DATASET:
                if config.NETWORK.USE_UNET:
                    if config.NETWORK.UNET_TRAIN_BY_TURN:
                        pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
                                                                                    targets_2d=targets_2d,
                                                                                    weights_2d=weights_2d,
                                                                                    targets_3d=targets_3d[0], unet_turn=unet_turn)
                    else:
                        pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
                                                                                    targets_2d=targets_2d,
                                                                                    weights_2d=weights_2d,
                                                                                    targets_3d=targets_3d[0])
                    
                else:                                                           
                    pred, pred_wholebody, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_cord_refine = model(views=inputs, meta=meta,
                                                                                    targets_2d=targets_2d,
                                                                                    weights_2d=weights_2d,
                                                                                    targets_3d=targets_3d[0], jrn_turn=jrn_turn)
        # except:
        except Exception as e:
            # print("except:", e)
            i -= 1
            continue
        
        loss_2d = loss_2d.mean()
        loss_3d = loss_3d.mean()
        loss_cord = loss_cord.mean()

        if config.NETWORK.USE_UNET:
            loss_depth_recon = loss_depth_recon.mean()
        if config.NETWORK.USE_JRN:
            loss_cord_refine = loss_cord_refine.mean()

        losses_2d.update(loss_2d.item())
        losses_3d.update(loss_3d.item())
        losses_cord.update(loss_cord.item())

        if config.NETWORK.USE_UNET:
            losses_depth_recon.update(loss_depth_recon.item())
        if config.NETWORK.USE_JRN:
            losses_cord_refine.update(loss_cord_refine.item())     

        loss = loss_2d + loss_3d + loss_cord
        losses.update(loss.item())

        ### 由于参数只增加了unet中的参数，因此下面两个loss应该是对unet都有梯度

        # ### org step
        # # # lcc debugging不更新参数，纳尼，居然变得和val一样了，都是完全
        # if loss_cord > 0:
        #     optimizer.zero_grad()
        #     # (loss_2d + loss_cord).backward()
        #     # 为了传2d loss
        #     ### 增加了unet之后的
        #     (loss_2d + loss_cord).backward()
        #     # loss_cord.backward(retain_graph=True)
        #     optimizer.step()
        
        # if vis_flag:print("5:{}".format(torch.cuda.memory_allocated(0)))

        # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
        #     optimizer.zero_grad()
        #     accu_loss_3d.backward()
        #     optimizer.step()
        #     accu_loss_3d = 0.0
        # else:
        #     accu_loss_3d += loss_3d / accumulation_steps




        ########yici
        if not config.NETWORK.USE_JRN or (not jrn_turn):
            # print('haha')
            ### lcc step
            if config.NETWORK.UNET_TRAIN_BY_TURN:
                if unet_turn and config.LOSS.USE_DEPTH_RECON_LOSS and loss_depth_recon > 0:
                    optimizer.zero_grad()
                    (loss_depth_recon*config.LOSS.DEPTH_RECON_LOSS_WEIGHT).backward(retain_graph=True)
                    optimizer.step()
            else:
                if config.LOSS.USE_DEPTH_RECON_LOSS and loss_depth_recon > 0:
                    optimizer.zero_grad()
                    (loss_depth_recon*config.LOSS.DEPTH_RECON_LOSS_WEIGHT).backward(retain_graph=True)
                    optimizer.step()

            # lcc debugging不更新参数，纳尼，居然变得和val一样了，都是完全
            if loss_cord > 0:
                optimizer.zero_grad()
                loss_cord.backward()
                # 为了传2d loss
                # (loss_2d + loss_cord).backward(retain_graph=True)

                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)
                # print('loss_cord')
                # st()

                optimizer.step()
            
            if config.NETWORK.UNET_TRAIN_BY_TURN:
                if not unet_turn:
                    # if not config.NETWORK.USE_UNET:
                    if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
                        optimizer.zero_grad()
                        accu_loss_3d.backward()

                        # for name, param in model.named_parameters():
                        #     if param.grad is None:
                        #         print(name)
                        # print('accu_loss_3d')
                        # st()

                        optimizer.step()
                        accu_loss_3d = 0.0
                    else:
                        accu_loss_3d += loss_3d / accumulation_steps
            else:
                if not config.NETWORK.USE_UNET:
                    if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
                        optimizer.zero_grad()
                        accu_loss_3d.backward()

                        # for name, param in model.named_parameters():
                        #     if param.grad is None:
                        #         print(name)
                        # print('accu_loss_3d')
                        # st()

                        optimizer.step()
                        accu_loss_3d = 0.0
                    else:
                        accu_loss_3d += loss_3d / accumulation_steps
        else:
            if loss_cord_refine > 0:
                optimizer.zero_grad()
                loss_cord_refine.backward()
                optimizer.step()    


        ### tensor inputs
        tensor_input = [meta[0]['tensor_inputs']]

        batch_time.update(time.time() - end)
        end = time.time()

        ### lcc debugging: testing running
        # config.PRINT_FREQ = 1

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            if config.NETWORK.USE_UNET:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'loss_depth_recon: {loss_depth_recon.val:.6f} ({loss_depth_recon.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=len(tensor_input) * tensor_input[0].size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                    loss_cord=losses_cord, loss_depth_recon=losses_depth_recon, memory=gpu_memory_usage)
            else:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed: {speed:.1f} samples/s\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                    'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                    'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                    'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                    'Loss_cord_refine: {loss_cord_refine.val:.6f} ({loss_cord_refine.avg:.6f})\t' \
                    'Memory {memory:.1f}'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        speed=len(tensor_input) * tensor_input[0].size(0) / batch_time.val,
                        data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                        loss_cord=losses_cord, loss_cord_refine=losses_cord_refine, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_loss_cord', losses_cord.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


            ### lcc debugging:j30
            for k in range(len(tensor_input)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{:08}_{}'.format(
                    os.path.join(output_dir, 'train'), i, view_name)
                    
                # save_debug_images_multi(config, tensor_input[k], meta[k], targets_2d[k], heatmaps[k], prefix)
                save_debug_images_multi(config, tensor_input[k][:,:3,:,:], meta[k], targets_2d[k], heatmaps[k], prefix)
            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'train'), i)
            
            
            save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)

            save_debug_3d_images(config, meta[0], pred, prefix2)

            # st()
            # save_debug_3d_images_wholebody(config, meta[0], pred_wholebody, prefix2)
       
        # valid_freq = 1e10
        if config['USE_LARGE_DATASET']:
            if config.DATASET.CAMERA_NUM == 1:
                # valid_freq = 1e10 ### 不valid

                ### lcc debugging inp5
                valid_freq = 2000
                # valid_freq = 3000
                # valid_freq = 300
                # valid_freq = 5000

            elif config.DATASET.CAMERA_NUM == 4:
                # valid_freq = 6000
                valid_freq = 5000
            else:
                assert False
        else:
            ### 每2000 step valid一次
            # valid_freq = 1
            valid_freq = 2000

        # st()
            
        if (not config.USE_SMALL_DATASET) and i > 0 and i % valid_freq == 0:
            ### 注意这里会设置model.eval()
            best_precision = valid_process(config, model, test_loader, logger, output_dir, epoch, optimizer, best_precision)
            model.train()
            
            ### 下面这句话不加的话会让2d backbone打开训练，造成2d loss急剧上升
            if model.module.backbone is not None:
                model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly

    return best_precision


def validate_3d(config, model, loader, output_dir, test_dataset_sigma=None):
    
    # test_dataset_sigma = None
    if test_dataset_sigma == None:
        test_dataset_sigma = 'fixed'

    # precompute_pred = True ### 现在只用这个控制是否预计算
    precompute_pred = False
    

    preds_dir = './eval_tmp'
    import os.path as osp
    import pickle
    preds_path = osp.join(preds_dir, f'{test_dataset_sigma}_preds.npy')

    if osp.exists(preds_path) and precompute_pred:
        with open(preds_path,'rb') as preds_f:
            preds = pickle.load(preds_f)
    else:

        batch_time = AverageMeter()
        data_time = AverageMeter()
        # # lcc debugging 
        # losses = AverageMeter()
        losses_2d = AverageMeter()
        # losses_3d = AverageMeter()
        # losses_cord = AverageMeter()
        losses_cord_refine = AverageMeter()

        ### 原本的eval操作 
        model.eval()

        # ### training时候的操作
        # model.train()
        # if model.module.backbone is not None:
        #     model.module.backbone.eval()  # Comment out this line if you want to train 2D backbone jointly


        preds = []
        preds_wholebody = []
        preds_valid = []
        with torch.no_grad():
            end = time.time()

            # ### debugging for vis 
            # for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
            #     prefix2 = '{}_{:08}'.format(
            #             os.path.join(output_dir, 'validation'), i)
            #     # st()
            #     save_debug_3d_images_for_vid_meta(config, meta[0], prefix2, config.DATASET.NAMEDATASET)
            
            # st()

            for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
                
                ### lcc debugging
                # if i > 10:break

                data_time.update(time.time() - end)


                # if 'panoptic' in config.DATASET.TEST_DATASET:
                #     pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                #                                                 weights_2d=weights_2d, targets_3d=targets_3d[0])
                # elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                #     pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                #                                                 input_heatmaps=input_heatmap)
                # elif 'kinoptic' in config.DATASET.TEST_DATASET:
                #     if config.NETWORK.USE_UNET:
                #         pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
                #                                                                     targets_2d=targets_2d,
                #                                                                     weights_2d=weights_2d,
                #                                                                     targets_3d=targets_3d[0])
                #     else:                                                           
                #         pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_cord_refine = model(views=inputs, meta=meta,
                #                                                                         targets_2d=targets_2d,
                #                                                                         weights_2d=weights_2d,
                #                                                                         targets_3d=targets_3d[0])

                try:
                    if 'panoptic' in config.DATASET.TEST_DATASET:
                        pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                                    weights_2d=weights_2d, targets_3d=targets_3d[0])
                    elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                        pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                                    input_heatmaps=input_heatmap)
                    elif 'kinoptic' in config.DATASET.TEST_DATASET:
                        if config.NETWORK.USE_UNET:
                            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_depth_recon = model(views=inputs, meta=meta,
                                                                                        targets_2d=targets_2d,
                                                                                        weights_2d=weights_2d,
                                                                                        targets_3d=targets_3d[0])
                        else:                                                           
                            pred, pred_wholebody, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_cord_refine = model(views=inputs, meta=meta,
                                                                                            targets_2d=targets_2d,
                                                                                            weights_2d=weights_2d,
                                                                                            targets_3d=targets_3d[0])
                    # st()
                    gpu_num = inputs[0].size(0)
                    preds_valid.extend([idx for idx in range(i*gpu_num, i*gpu_num+gpu_num)])
                # except:
                except Exception as e:
                    # print("except:", e)
                    continue  



                # st()
                # print(f'step {i} heatmaps[0].mean() {heatmaps[0].mean()}')
                # print(f'step {i} loss_2d.mean() {loss_2d.mean()}')

                # # lcc debugging 
                loss_2d = loss_2d.mean()
                # loss_3d = loss_3d.mean()
                # loss_cord = loss_cord.mean()

                # print(f'loss_2d {loss_2d} loss_3d {loss_3d} loss_cord {loss_3d}')

                losses_2d.update(loss_2d.item())
                # losses_3d.update(loss_3d.item())
                # losses_cord.update(loss_cord.item())
                # loss = loss_2d + loss_3d + loss_cord
                # losses.update(loss.item())

                pred = pred.detach().cpu().numpy()
                pred_wholebody_tensor = pred_wholebody.clone()
                pred_wholebody = pred_wholebody.detach().cpu().numpy()

                # st()
                # (Pdb) pred.shape
                # (4, 10, 15, 5)
                # (Pdb) pred[0][:,0,3] ### visibility
                # array([ 0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1.], dtype=float32)
                # (Pdb) pred[0][:,0,4] ### confidence
                # array([0.9337361 , 0.9093438 , 0.8922758 , 0.7408172 , 0.02298843,
                #     0.01545834, 0.0092029 , 0.00901857, 0.00717135, 0.00681221],
                #     dtype=float32)
                # (Pdb) pred[0][0,:,:] 
                # array([[-8.0240417e+02, -8.1142358e+02,  1.4024762e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-7.5469409e+02, -6.4437024e+02,  1.5737432e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-8.0869714e+02, -7.6547876e+02,  8.5172546e+02,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-9.6816187e+02, -7.7864191e+02,  1.4104233e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-1.0366121e+03, -7.5460187e+02,  1.1512948e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-8.6469183e+02, -6.0803320e+02,  1.2231427e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-9.1160376e+02, -7.3587927e+02,  8.6032922e+02,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-9.2683398e+02, -7.3280225e+02,  4.3996906e+02,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-9.3051837e+02, -7.7443005e+02,  9.3117111e+01,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-6.6814716e+02, -8.2536682e+02,  1.4019128e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-6.1707672e+02, -8.1962378e+02,  1.1563583e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-6.9500220e+02, -6.2659283e+02,  1.2597683e+03,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-7.1707886e+02, -7.9836310e+02,  8.5272137e+02,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-6.7629968e+02, -8.5999493e+02,  4.5305896e+02,  0.0000000e+00,
                #         9.3373609e-01],
                #     [-7.1506262e+02, -9.8309509e+02,  9.6438950e+01,  0.0000000e+00,
                #         9.3373609e-01]], dtype=float32)
                
                for b in range(pred.shape[0]):
                    preds.append(pred[b])
                    preds_wholebody.append(pred_wholebody[b])

                # (Pdb) len(preds)
                # 4
                # (Pdb) preds[0].shape
                # (10, 15, 5)


                ### tensor inputs
                # st()
                tensor_input = [meta[0]['tensor_inputs']]


                batch_time.update(time.time() - end)
                end = time.time()


                if not os.path.exists('.'):
                    ### 如果自动断开则暂停
                    st()


                # # # lcc debugging 
                # if i % 100 == 0:
                #     print(inputs[0])
                #     print(meta[0])
                #     print(inputs[0].max(), inputs[0].min(), inputs[0].mean())
                #     print(pred[0].max(), pred[0].min(), pred[0].mean())


                ### lcc debugging: testing running
                # config.PRINT_FREQ = 1

                # config.PRINT_FREQ = 3
                ### lcc debugging
                # config.PRINT_FREQ = 10

                print_freq = config.PRINT_FREQ
                # print_freq = 3
                
                if i % print_freq == 0 or i == len(loader) - 1:
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = 'Test: [{0}/{1}]\t' \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed: {speed:.1f} samples/s\t' \
                        'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                        'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                        'Memory {memory:.1f}'.format(
                            i, len(loader), batch_time=batch_time,
                            speed=len(tensor_input) * tensor_input[0].size(0) / batch_time.val,
                            data_time=data_time, loss_2d=losses_2d, memory=gpu_memory_usage)
                    logger.info(msg)

                    # # lcc debugging 
                    # msg = '\n' \
                    #   'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    #   'Speed: {speed:.1f} samples/s\t' \
                    #   'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    #   'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                    #   'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                    #   'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                    #   'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                    #   'Memory {memory:.1f}'.format(batch_time=batch_time,
                    #     speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    #     data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                    #     loss_cord=losses_cord, memory=gpu_memory_usage)
                    # logger.info(msg)

                    for k in range(len(tensor_input)):
                        view_name = 'view_{}'.format(k + 1)
                        prefix = '{}_{:08}_{}'.format(
                            os.path.join(output_dir, 'validation'), i, view_name)
                        # save_debug_images_multi(config, tensor_input[k], meta[k], targets_2d[k], heatmaps[k], prefix)
                        save_debug_images_multi(config, tensor_input[k][:,:3,:,:], meta[k], targets_2d[k], heatmaps[k], prefix)
                        
                        # save_debug_images_multi_lcc(config, tensor_input[k], meta[k], targets_2d[k], heatmaps[k], target_depthoffset[k], depthoffset[k], refined_depth[k], prefix)
                    prefix2 = '{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), i)

                    save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                    save_debug_3d_images(config, meta[0], pred, prefix2)

                    # st()
                    save_debug_3d_images_wholebody(config, meta[0], pred_wholebody_tensor, prefix2)
                
                if config.OUT_3D_POSE_VID:
                    # st()
                    prefix2 = '{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), i)
                    
                    # st()
                    save_debug_3d_images_wholebody_for_vid(config, meta[0], pred_wholebody_tensor, prefix2, config.DATASET.NAMEDATASET)
                    
                    basename = os.path.basename(prefix2)
                    dirname = os.path.dirname(prefix2)
                    dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid', config.DATASET.NAMEDATASET)
                    prefix3 = os.path.join(dirname1, basename)

                    save_batch_heatmaps_multi_gt_pred_lcc(tensor_input[0][:,:3,:,:], targets_2d[0], heatmaps[0], prefix3)
                    # save_batch_heatmaps_multi_gt_pred_lcc2(tensor_input[0][:,:3,:,:], targets_2d[0], heatmaps[0], prefix3)
    
    if config['OUT_3D_POSE_VID']:
        print('all images done!')
        st()
    # if out_3d_pose_vid:
    #     valid_imgs_list = []
    #     dirname = os.path.dirname(prefix2)
    #     dirname1 = os.path.join(dirname, '3d_joints', 'valid_imgs_for_vid')
    #     import glob
    #     import cv2
    #     valid_imgs_list = glob.glob(dirname1 + '/*.png')
        
    #     def image2video(image_dir, name, fps=25):
    #         image_path_list = []
    #         for image_path in image_dir:
    #             image_path_list.append(image_path)
    #         image_path_list.sort()
    #         temp = cv2.imread(image_path_list[0])
    #         size = (temp.shape[1], temp.shape[0])
    #         fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    #         video = cv2.VideoWriter('./output/' + name + '.mp4', fourcc, fps, size)
    #         for image_path in image_path_list:
    #             if image_path.endswith(".png"):
    #                 image_data_temp = cv2.imread(image_path)
    #                 video.write(image_data_temp)
    #         print("Video done！")
    #     ### name todo
    #     image2video(valid_imgs_list, '1')
    # st()

    metric = None
    if 'panoptic' in config.DATASET.TEST_DATASET or 'kinoptic' in config.DATASET.TEST_DATASET:
        # st()
        if precompute_pred:
            # st()
            ### save preds for eval
            preds_dir = './eval_tmp'
            os.makedirs(preds_dir, exist_ok=True)
            import os.path as osp
            preds_path = osp.join(preds_dir, f'{test_dataset_sigma}_preds.npy')
            with open(preds_path,'wb') as preds_f:
                pickle.dump(preds, preds_f)


        # aps, _, mpjpe, recall = loader.dataset.evaluate(preds)
        # st()
        aps, _, mpjpe, recall = loader.dataset.evaluate_w_pv(preds, preds_valid)
        aps_body = aps.copy()
        msg = 'ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        logger.info(msg)
        

        # st()
        aps, _, mpjpe, recall = loader.dataset.evaluate_w_pv_face(preds_wholebody, preds_valid)
        aps_face = aps.copy()
        msg = 'face ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        logger.info(msg)

        aps, _, mpjpe, recall = loader.dataset.evaluate_w_pv_hand(preds_wholebody, preds_valid)
        aps_hand = aps.copy()
        msg = 'hand ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        
        metric = np.mean(aps_body) ### use preds metric
        # metric = np.mean(aps_face + aps_hand) ### use preds metric
        # st()
        logger.info(msg)


    elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
        actor_pcp, avg_pcp, _, recall = loader.dataset.evaluate(preds)
        msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
              ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp*100, recall=recall)
        logger.info(msg)
        metric = np.mean(avg_pcp)

    return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
