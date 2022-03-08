#!/bin/bash

# JSRT using images as input
python scripts/test_model_images.py --train_images_dir=data/JSRT/Train/Images \
                                    --train_points_dir=data/JSRT/Train/landmarks_txt \
                                    --test_images_dir=data/JSRT/Test/Images \
                                    --results_dir=results/test/JSRT/images \
                                    --num_atlas=5 \
                                    --image_measure='mutual_info' \
                                    --label_fusion='voting' \
                                    --remove_tmp_dir

# JSRT using labels as input
python scripts/test_model_labels.py --train_images_dir=data/JSRT/Train/masks \
                                   --train_points_dir=data/JSRT/Train/landmarks_txt \
                                   --test_images_dir=data/JSRT/Test/masks \
                                   --results_dir=results/test/JSRT/labels \
                                   --num_atlas=5 \
                                   --image_measure='mutual_info' \
                                   --label_fusion='voting' \
                                   --remove_tmp_dir