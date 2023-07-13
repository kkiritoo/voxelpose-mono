CUDA_VISIBLE_DEVICES=7 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960.yaml

# CUDA_VISIBLE_DEVICES=6 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_c1.yaml

# CUDA_VISIBLE_DEVICES=7 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960_c2.yaml


# CUDA_VISIBLE_DEVICES=0 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep512x960.yaml \
#                         --out_3d_pose_vid \
#                         --out_3d_pose_vid_namedset