import os
import numpy as np
import SimpleITK as sitk

from src.utils.io import read_image_to_itk


def register_label(fixed_image_path, moving_image_path, moving_label_path, elx_params, output_dir):
    """ Register a pair of images and transform a label map. """

    # Set parameters for registration
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(read_image_to_itk(fixed_image_path))
    elx.SetMovingImage(read_image_to_itk(moving_image_path))
    elx.SetParameterMap(elx_params)

    # Run registration
    elx.SetLogToConsole(False)
    elx.SetOutputDirectory(output_dir)
    elx.Execute()

    # Get transformed image
    result_image = elx.GetResultImage()

    # Set transform parameters for labels
    tfx_params = elx.GetTransformParameterMap()
    tfx_params[-1]['FinalBSplineInterpolationOrder'] = ['0']
    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(read_image_to_itk(moving_label_path))
    tfx.SetTransformParameterMap(tfx_params)

    # Transform moving label
    tfx.SetLogToConsole(False)
    tfx.SetOutputDirectory(output_dir)
    tfx.Execute()

    # Get transformed label
    result_label = tfx.GetResultImage()

    return result_image, result_label


def register_points(fixed_image_path, moving_image_path, fixed_points_path, elx_params, output_dir):
    """ Register a pair of images and transform a set of points. """

    # Set parameters for registration
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(read_image_to_itk(fixed_image_path))
    elx.SetMovingImage(read_image_to_itk(moving_image_path))
    elx.SetParameterMap(elx_params)

    # Run registration
    elx.SetLogToConsole(False)
    elx.SetOutputDirectory(output_dir)
    elx.Execute()

    # Get transformed image
    result_image = elx.GetResultImage()

    # Set transform parameters for points
    tfx_params = elx.GetTransformParameterMap()
    tfx = sitk.TransformixImageFilter()
    tfx.SetFixedPointSetFileName(fixed_points_path)
    tfx.SetTransformParameterMap(tfx_params)

    # Transform fixed points
    tfx.SetLogToConsole(False)
    tfx.SetOutputDirectory(output_dir)
    tfx.Execute()

    # Get transformed points
    output_points = np.genfromtxt(os.path.join(output_dir, 'outputpoints.txt'))
    result_points = output_points[:, (27, 28)].flatten()

    return result_image, result_points
