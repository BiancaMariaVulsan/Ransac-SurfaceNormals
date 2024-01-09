from matplotlib import pyplot as plt
import torch
from torch.nn.functional import normalize

import numpy as np

def eval_pred_sn(pred_norm, gt_norm, mask):
    pixels = []

    pred = pred_norm.astype(float)
    gt = gt_norm.astype(float)
    nv = mask[:, :, 0]

    # pred = np.clip(pred, 1e-5, pred.max())
    # gt = np.clip(gt, 1e-5, gt.max())

    # Normalize both vectors
    gt = gt / np.linalg.norm(gt, axis=2, keepdims=True)
    pred = pred / np.linalg.norm(pred, axis=2, keepdims=True)

    # Compute dot product and keep only valid pixels
    dp = np.sum(gt * pred, axis=2)
    t = np.clip(dp, -1, 1)
    pixels.append(t[nv > 0])

    e = np.degrees(np.arccos(np.concatenate(pixels)))
    e = e[~np.isnan(e)]  # Exclude NaN values

    # Check if there are non-NaN values in the array before calculating metrics
    if len(e) > 0:
        nums_e = [
            np.mean(e),
            np.median(e),
            np.sqrt(np.mean(e ** 2)),
            np.mean(e < 11.25) * 100,
            np.mean(e < 22.5) * 100,
            np.mean(e < 30) * 100
        ]

        print('---------------------------------------')
        print('Mean:', nums_e[0])
        print('Median:', nums_e[1])
        print('RMSE:', nums_e[2])
        print('11.25:', nums_e[3])
        print('22.5:', nums_e[4])
        print('30:', nums_e[5])
        print('---------------------------------------')

def angle_error_under_threshold(values: torch.tensor, threshold: float):
    return torch.sum(values < threshold) / values.count_nonzero()

def normals_angle_difference(prediction: torch.tensor, target: torch.tensor,
                              mask: torch.tensor=None) -> torch.tensor:
    """Compute the angles between predicted and target normal vectors, in degrees.

    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
    """
    if prediction.shape != target.shape:
        raise ValueError('Input tensors have different shapes.')
    
    if mask is not None and mask.shape != target.shape:
        raise ValueError("Mask shape doesn't match that of the input tensors.")

    if mask is not None:
        prediction = torch.where(mask == 1, prediction, target)
    
    prediction = normalize(prediction, dim=-1)
    target = normalize(target, dim=-1)

    cosines = torch.sum(prediction * target, dim=-1)
    radian_angles = torch.acos(cosines)
    degree_angles = torch.rad2deg(radian_angles)
    return degree_angles

def get_normals_error_visualization(prediction: torch.Tensor, target: torch.Tensor, validity_mask: torch.Tensor=None) -> plt.figure:
    """Get a visualization for surface normals prediction error.
    
    The input tensors must be of shape [Height x Width X Channel].

    Args:
        prediction: Predicted depth
        target: Ground truth depth
        validity_mask: Marks valid values with `1.0` and invalid ones with `0.0`
    """
    error = torch.abs(prediction - target)

    if validity_mask is not None:
        error *= validity_mask

    figure, axis_handles = plt.subplots(3, 1, figsize=[12.0, 12.0])
    axes = ['X', 'Y', 'Z']
    for index, axis in enumerate(axes):
        plt.sca(axis_handles[index])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title(f'Normal {axis}')
        image = plt.imshow(error[..., index], cmap='Greys', vmin=0)
        plt.colorbar(image)
    plt.show()