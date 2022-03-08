#!/bin/bash

python scripts/format_points.py --points_dir=data/JSRT/Train/landmarks \
                                --output_dir=data/JSRT/Train/landmarks_txt

python scripts/format_points.py --points_dir=data/JSRT/Val/landmarks \
                                --output_dir=data/JSRT/Val/landmarks_txt

python scripts/format_points.py --points_dir=data/JSRT/Test/landmarks \
                                --output_dir=data/JSRT/Test/landmarks_txt