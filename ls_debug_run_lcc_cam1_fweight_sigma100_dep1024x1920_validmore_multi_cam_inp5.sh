CUDA_VISIBLE_DEVICES=5 python run/train_3d_validmore.py --cfg configs/kinoptic/resnet50/debug_prn64_cpn80x80x20_960x512_cam1_fweight_sigma100_dep1024x1920_multi_cam_inp5.yaml \
                        --use_large_dataset \
                        --use_small_dataset 