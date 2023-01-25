#!/bin/bash

# JSRT using images as input
python scripts/test_model_images.py --train_images_dir=data/Exp_Extendido_Oclusion/Extended/Train/Images \
                                    --train_points_dir=data/Exp_Extendido_Oclusion/Extended/Train/landmarks_txt \
                                    --test_images_dir=data/Exp_Extendido_Oclusion/Padchest_Occlusion/Images \
                                    --results_dir=results/test/Exp_Extendido_Oclusion/Padchest_Occlusion/images \
                                    --num_atlas=5 \
                                    --image_measure='mutual_info' \
                                    --label_fusion='voting' \
                                    --remove_tmp_dir