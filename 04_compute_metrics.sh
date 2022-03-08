#!/bin/bash

# JSRT using images as input
python scripts/compute_metrics.py --pred_points_dir=results/test/JSRT/images/output_points \
                                  --true_points_dir=data/JSRT/Test/landmarks \
                                  --output_dir=results/metrics/JSRT/images

# JSRT using labels as input
python scripts/compute_metrics.py --pred_points_dir=results/test/JSRT/labels/output_points \
                                  --true_points_dir=data/JSRT/Test/landmarks \
                                  --output_dir=results/metrics/JSRT/labels