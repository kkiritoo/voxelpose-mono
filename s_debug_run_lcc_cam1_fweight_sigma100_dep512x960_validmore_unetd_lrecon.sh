CUDA_VISIBLE_DEVICES=3 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/unet/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unetd_lrecon.yaml \
                        --use_small_dataset 

# 换个gpu
# CUDA_VISIBLE_DEVICES=6 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/unet/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unetd_lrecon.yaml \
#                         --use_small_dataset 

# out_3d_pose_vid
# CUDA_VISIBLE_DEVICES=7 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/unet/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_unetd_lrecon.yaml \
#                         --use_small_dataset \
#                         --out_3d_pose_vid 