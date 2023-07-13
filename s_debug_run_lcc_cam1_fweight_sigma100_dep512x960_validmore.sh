CUDA_VISIBLE_DEVICES=3 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960.yaml \
                        --use_small_dataset 

# CUDA_VISIBLE_DEVICES=6 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960.yaml \
#                         --use_small_dataset \
#                         --out_3d_pose_vid 