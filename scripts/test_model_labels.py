import os
import argparse
import shutil
import time
import numpy as np
import SimpleITK as sitk

from src.utils.io import read_dir
from src.models import MAS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images_dir', type=str)
    parser.add_argument('--train_points_dir', type=str)
    parser.add_argument('--test_images_dir', type=str)
    parser.add_argument('--num_atlas', type=int)
    parser.add_argument('--image_measure', type=str)
    parser.add_argument('--label_fusion', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--remove_tmp_dir', action='store_true')
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    output_dir = os.path.join(args.results_dir, 'output_points')
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(args.results_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Read list of atlases (i.e., label and its corresponding point set)
    atlas_files = {
        'images': read_dir(args.train_images_dir),
        'points': read_dir(args.train_points_dir)
    }

    # Read list of input labels
    test_images_list = read_dir(args.test_images_dir)

    # Configure registration procedure
    elx_params = sitk.VectorOfParameterMap()

    affine_params = sitk.GetDefaultParameterMap('affine', 5)
    elx_params.append(affine_params)

    bspline_params = sitk.GetDefaultParameterMap('bspline', 5, 64)
    bspline_params['MaximumNumberOfIterations'] = ['1500']
    bspline_params['FinalBSplineInterpolationOrder'] = ['0']
    elx_params.append(bspline_params)

    # Configure multi-atlas
    multiatlas = MAS(atlas_files, args.num_atlas, args.image_measure, args.label_fusion)

    num_images = len(test_images_list)
    print('\nPredicting points for input labels:\n')
    for i, image_path in enumerate(test_images_list):
        image_name = os.path.basename(image_path)

        print(f'\t{i+1:>3}/{num_images} File {image_name}', end=' ', flush=True)

        cur_tmp_dir = os.path.join(tmp_dir, f'file_{i+1:0>3}')
        os.makedirs(cur_tmp_dir, exist_ok=True)

        # Run multi-atlas
        t_start = time.time()
        pred_points = multiatlas.predict_points(image_path, elx_params, cur_tmp_dir)[0]
        t_elapsed = time.time() - t_start

        # Save results
        output_path = os.path.join(output_dir, image_name.split('.')[0])
        np.save(output_path, pred_points)

        print(f'({t_elapsed:.2f} sec)')

    if args.remove_tmp_dir:
        shutil.rmtree(tmp_dir)

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
