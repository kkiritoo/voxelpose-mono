CUDA_VISIBLE_DEVICES=7 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
                        --use_large_dataset 

### multi-gpu
# CUDA_VISIBLE_DEVICES=8,9 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset 

# CUDA_VISIBLE_DEVICES=1,2 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset 

# CUDA_VISIBLE_DEVICES=5,6,7,8 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset 

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset 


# CUDA_VISIBLE_DEVICES=6 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam.yaml \
#                         --use_large_dataset \
#                         --out_3d_pose_vid 