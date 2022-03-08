import os
import argparse
import numpy as np
import csv

from src.utils.io import read_dir
from src.metric.image import mean_absolute_error as mae, mean_squared_error as mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_points_dir', type=str)
    parser.add_argument('--pred_points_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    os.makedirs(args.output_dir, exist_ok=True)

    true_points_files = read_dir(args.true_points_dir)
    pred_points_files = read_dir(args.pred_points_dir)

    result_metrics = {'mae': [], 'mse': []}

    print('\nComputing metrics of predicted points:\n')
    num_points = len(true_points_files)
    for i, (true_path, pred_path) in enumerate(zip(true_points_files, pred_points_files)):
        file_name = os.path.basename(true_path)

        print(f'\t{i+1:>3}/{num_points} File {file_name}', end=' ', flush=True)

        true_points = np.load(true_path)
        pred_points = np.load(pred_path)

        result_metrics['mae'].append(mae(true_points, pred_points))
        result_metrics['mse'].append(mse(true_points, pred_points))

        print('Ok')

    # Save results
    with open(os.path.join(args.output_dir, 'metrics.csv'), 'w') as f:
        writer = csv.DictWriter(f, result_metrics.keys())
        writer.writeheader()
        writer.writerow(result_metrics)

    print('\nResults:\n')
    for k, v in result_metrics.items():
        print('\t{}: mean: {:.4f} - std: {:.4f}'.format(k.upper(), np.mean(v), np.std(v)))

    print('\nDone.\n') 
