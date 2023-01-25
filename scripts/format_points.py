import os
import argparse
import numpy as np

from src.utils.io import read_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--points_dir', type=str)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    points_files = read_dir(args.points_dir)

    num_points = len(points_files)
    print('\nFormatting points in numpy files to txt files:\n')
    for i, points_path in enumerate(points_files):
        points_name = os.path.basename(points_path)

        print(f'\t{i+1:>3}/{num_points} File {points_name}', end=' ', flush=True)

        points = np.load(points_path).reshape(-1, 2)[:120]
        points *= args.scale  # rescale points if required

        n_points = points.shape[0]
        output_path = os.path.join(args.output_dir, points_name.split('.')[0]) + '.txt'

        # Save points according to SimpleElastix format
        with open(output_path, 'w') as f:
            f.write('point\n')
            f.write(f'{n_points}\n')
            np.savetxt(f, points, fmt='%s')

        print('Ok')

    print('\nDone.\n')
