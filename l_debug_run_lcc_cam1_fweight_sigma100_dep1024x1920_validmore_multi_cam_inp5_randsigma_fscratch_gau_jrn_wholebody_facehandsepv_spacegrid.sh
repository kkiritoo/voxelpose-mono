CUDA_VISIBLE_DEVICES=5 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_jrn_wholebody_facehandsepv.yaml \
                        --use_large_dataset  \
                        --jrn_space 1200

# CUDA_VISIBLE_DEVICES=8 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5_randsigma_fscratch_gau_jrn_wholebody_facehandsepv.yaml \
#                         --use_large_dataset \
#                         --out_3d_pose_vid