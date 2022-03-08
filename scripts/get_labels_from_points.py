import os
import argparse
import numpy as np
from PIL import Image

from src.utils.io import read_dir
from src.preprocess import get_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--points_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    points_files = read_dir(args.points_dir)

    num_points = len(points_files)
    print('\nGetting segmentations from points:\n')
    for i, points_path in enumerate(points_files):
        points_name = os.path.basename(points_path)

        print(f'\t{i+1:>3}/{num_points} File {points_name}', end=' ', flush=True)

        points = np.load(points_path)

        # Get label image from points
        label = get_seg(points)

        # Save label image
        output_path = os.path.join(args.output_dir, points_name.split('.')[0] + '.png')
        Image.fromarray(np.uint8(label)).save(output_path)

        print('Ok')

    print('\nDone.\n')
