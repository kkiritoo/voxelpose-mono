# CUDA_VISIBLE_DEVICES= python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unet.yaml

# out_3d_pose_vid
CUDA_VISIBLE_DEVICES=8 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unet_lrecon.yaml \
                        --out_3d_pose_vid \
                        --out_3d_pose_vid_namedset