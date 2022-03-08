#!/bin/bash

python scripts/test_model_images.py --train_images_dir=data/JSRT/Train/Images \
                                    --train_points_dir=data/JSRT/Train/landmarks_txt \
                                    --test_images_dir=data/Montgomery/Images \
                                    --results_dir=results/test/Montgomery/images \
                                    --num_atlas=5 \
                                    --image_measure='mutual_info' \
                                    --label_fusion='voting' \
                                    --remove_tmp_dir