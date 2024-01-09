import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import math

import torch

from evaluation import angle_error_under_threshold, get_normals_error_visualization, normals_angle_difference, eval_pred_sn

def load_data(folder_path):
    """Load the data stored as a matlab file. 
        
        The folder containing the data should not contain data of other type!

        Args:
            folder_path: Path to the local folder where .mat files are located.
    
        Returns:
            image_array: The original image extracted from the matlab file.
            depth_array: The depth map extracted from the matlab file.
            norm_array: The surface normals extracted from the matlab file.
            mask_array: The mask for the image extracted from the matlab file.
    """
    # List all files in the folder
    file_list = os.listdir(folder_path)
    image_array = []
    depth_array = []
    norm_array = []
    mask_array = []
    # Iterate through each file in the folder
    try:
        for filename in file_list:
            # Get the full path of the file by joining the folder path with the filename
            file_path = os.path.join(folder_path, filename)
            # Load the matlab image as a dictionary
            matlab_data = scipy.io.loadmat(file_path)
            # Extract the data from the MATLAB file
            image_data = matlab_data['img']
            image_array.append(image_data)
            # The depth is given in meters
            depth_data = matlab_data['depth']
            depth_array.append(depth_data)
            surface_data = matlab_data['norm']
            norm_array.append(surface_data)
            mask_data = matlab_data['mask']
            mask_array.append(mask_data)
    except:
        print("Something went wrong when reading the files!")
        exit(1)

    return image_array, depth_array, norm_array, mask_array

def depth_map_to_3d_points(depth_map, intrinsics):
    """Convert each point from the depth map to it's corresponding 3D coordinates.

        Args:
            depth_map: The depth map of the image.
            intrinsics: The intrinsic parameters of the camera

        Returns:
            np.dstack((x_3d, y_3d, z)): A 2D array containing at each position the corresponding 3D point
    """
    # Unpack camera intrinsics
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    height, width = depth_map.shape
    # Create a mesh grid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    z = depth_map
    # Convert depth map to 3D points in camera coordinate system
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Return also the image coordinates to be able to perform labeling
    return np.dstack((x_3d, y_3d, z)), np.dstack((x, y))

def fit_plane(x_noisy, y_noisy, z_noisy):
    """Perform plane fitting to find a, b, c parameters from the equation a*x + b*y + cz + d = 0.

        We consider d=-1 => a*x + b*y + c*z = 1

        Args:
            x_noisy: The noisy x coordinates of the points.
            y_noisy: The noisy y coordinates of the points.
            z_noisy: The noisy z coordinates of the points.

        Returns:
            a: The coefficient of x in the equation a*x + b*y + c*z = 1.
            b: The coefficient of y in the equation a*x + b*y + c*z = 1.
            c: The coefficient of z in the equation a*x + b*y + c*z = 1.
    """
    # Convert lists to NumPy arrays for efficient operations
    x_noisy = np.array(x_noisy)
    y_noisy = np.array(y_noisy)
    z_noisy = np.array(z_noisy)

    # Find the indices of valid points (where z_noisy is not equal to 0)
    valid_indices = np.where(z_noisy != 0)

    # Filter out the invalid points using the valid_indices
    x_noisy_valid = x_noisy[valid_indices]
    y_noisy_valid = y_noisy[valid_indices]
    z_noisy_valid = z_noisy[valid_indices]

    if len(x_noisy_valid) == 0 or len(y_noisy_valid) == 0 or len(z_noisy_valid) == 0:
        return 0, 0, 0
    
    # Estimate the a, b, c coefficients
    A = np.column_stack((x_noisy_valid, y_noisy_valid, z_noisy_valid))
    AP = np.linalg.pinv(A)
    X = np.matmul(AP,np.ones(len(x_noisy_valid)))
    a, b, c = X

    norm = math.sqrt(a**2 + b**2 + c**2)
    if norm == 0:
        return 0, 0, 0
    return a/norm, b/norm, c/norm

def findConnectedComponents(inliers):
    height, width = inliers.shape[:2]
    labels = np.zeros_like(inliers, dtype=np.uint16)
    current_label = 1
    stats = []

    def dfs(i, j):
        if labels[i, j] == 0 and inliers[i, j]:
            labels[i, j] = current_label
            stats[current_label - 1] += 1
            for ni, nj in neighbors(i, j, height, width):
                dfs(ni, nj)

    for i in range(height):
        for j in range(width):
            if labels[i, j] == 0 and inliers[i, j]:
                stats.append(0)
                dfs(i, j)
                current_label += 1

    return labels

def neighbors(i, j, height, width):
    for ni in range(i-1, i+2):
        for nj in range(j-1, j+2):
            if 0 <= ni < height and 0 <= nj < width and (ni != i or nj != j):
                yield ni, nj

def cc_ransac(image, patch, indices_patch, depth, points_3d, s, t, Tc, T, N):

    for _ in range(N):
        # Step 1: Randomly select a sample
        flat_points_3d = points_3d.reshape((-1, 3))
        sample_indices = np.random.choice(flat_points_3d.shape[0], size=s, replace=False)
        sample_points = flat_points_3d[sample_indices, :]

        # Fit a plane to the sample points
        x_values = sample_points[:3, :1].flatten()
        y_values = sample_points[:3, 1:2].flatten()
        z_values = sample_points[:3, 2:3].flatten()
        a, b, c = fit_plane(x_values, y_values, z_values)

        if a == 0:
            continue
        
        # Step 2: Determine consensus set
        inliers = np.zeros_like(image, dtype=np.uint16)
        for i in range (points_3d.shape[0]):
            for j in range (points_3d.shape[1]):
                x, y, z = points_3d[i, j]
                img_x, img_y = indices_patch[:,i,j]
                dist = abs(a * x + b * y + c * z + 1) / math.sqrt(a**2 + b**2 + c**2)
                if dist < t:
                    inliers[img_x, img_y] = 255

            if len(inliers) >= T:
                norm = math.sqrt(a**2 + b**2 + c**2)
                if norm == 0:
                    return 0, 0, 0
                return a/norm, b/norm, c/norm

    norm = math.sqrt(a**2 + b**2 + c**2)
    if norm == 0:
        return 0, 0, 0
    return a/norm, b/norm, c/norm


def cc_ransac_for_patch(image, depth, mask, patch_size, s, t, Tc, T, N):
    height, width, _ = image.shape
    normals = np.zeros((height, width, 3))

    # RGB Intrinsic Parameters (specific to the dataset)
    intrinsics = {
        # focal distance
        'fx': 5.1885790117450188e+02,
        'fy': 5.1946961112127485e+02,
        # principal point
        'cx': 3.2558244941119034e+02,
        'cy': 2.5373616633400465e+02
    }

    # Apply the mask on the depth
    mask = np.array(mask)
    depth = np.multiply(depth, mask)

    # Calculated the corresponding 3D coordinates
    points_3d, image_coord = depth_map_to_3d_points(depth, intrinsics)
    for x in range(patch_size // 2, height - patch_size // 2 - 1):
        for y in range(patch_size // 2, width - patch_size // 2 - 1):
            patch_image = image[x - patch_size // 2 : x + patch_size // 2, y - patch_size // 2 : y + patch_size // 2, :]
            patch_depth = depth[x - patch_size // 2 : x + patch_size // 2, y - patch_size // 2 : y + patch_size // 2]
            patch_points = points_3d[x - patch_size // 2 : x + patch_size // 2, y - patch_size // 2 : y + patch_size // 2]

            indices_patch = np.empty((2, patch_size, patch_size), dtype=int)

             # Get indices of each value in patch_image
            indices_x, indices_y = np.indices(patch_image.shape[:-1])
            
            # Add the starting indices of the patch to get absolute indices
            indices_x = indices_x + x - patch_size // 2
            indices_y = indices_y + y - patch_size // 2

            # Stack the indices along the first axis to form a (2, patch_size, patch_size) array
            indices_patch[0, :, :] = indices_x
            indices_patch[1, :, :] = indices_y

            if np.where(patch_points != 0)[0].size >= 3:
                patch_normals = cc_ransac(image, patch_image, indices_patch, patch_depth, patch_points, s, t, Tc, T, N)
                normals[x, y] = patch_normals
            else:
                normals[x, y] = [0, 0, 0]

    return normals 

def visualize_normals(normals_matrix, image, depth, mask, ground_truth):
    """Visualize the normals_matrix, image, depth, mask and ground_truth on the same image.

        Args:
            normals_matrix: A 2D array containing the normals to each point in the original image.
            image: The original image
            mask: The mask that tells what values from the depth map are valid.
            ground_truth: The desired result for the surface normals.
    """
    # Convert the values of the image according to the ReadMe for a better visualization.
    image[:,:,0] = image[:,:,0] + 2* 122.175
    image[:,:,1] = image[:,:,1] + 2* 116.169  
    image[:,:,2] = image[:,:,2] + 2* 103.508
    image = np.uint8(image)

    # Create a new figure with five subplots (1 row, 5 columns)
    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    ax1.imshow(image)
    ax1.set_title('Image')
    ax1.axis('off')

    ax2.imshow(depth)
    ax2.set_title('Depth')
    ax2.axis('off')

    ax3.imshow(normals_matrix)
    ax3.set_title('Surface Normals')
    ax3.axis('off')

    ax4.imshow(mask)
    ax4.set_title('Mask')
    ax4.axis('off')

    ax5.imshow(ground_truth)
    ax5.set_title('Ground Truth')
    ax5.axis('off')

    # Adjust the layout to avoid overlapping titles and axis labels
    plt.tight_layout()
    plt.show()

def main():
    folder_path = "C:\\DFM\\tasks\\sprint1\\surface_normal_initial_impl\\surface_normals_nyu\\"
    images, depths, ground_truth, masks = load_data(folder_path)

    # Set parameters
    s = 40     # Number of samples for RANSAC
    t = 20     # Distance threshold for inliers
    Tc = 4     # Size threshold for connected components
    T = 80     # Size threshold for termination
    N = 10     # Number of RANSAC trials
    patch_size = 10  # Adjust patch size as needed

    normals = cc_ransac_for_patch(images[0], depths[0], masks[0], patch_size, s, t, Tc, T, N)

    visualize_normals(normals, images[0], depths[0], masks[0], ground_truth[0])

    depth_mask = torch.from_numpy(masks[0]).unsqueeze(2)
    # Expand the single-channel depth_mask_tensor to have three channels
    normals_mask = depth_mask.expand(-1, -1, 3)
 
    normals_mask_np = normals_mask.cpu().detach().numpy()
    eval_pred_sn(normals, ground_truth[0], normals_mask_np)

    gt = torch.from_numpy(ground_truth[0])
    pred = torch.from_numpy(normals)

    angle_difference = normals_angle_difference(pred, gt, normals_mask)
    print('SN angle error < 11.25 deg', angle_error_under_threshold(angle_difference, 11.25))
    print('SN angle error < 22.5 deg', angle_error_under_threshold(angle_difference, 22.5))
    print('SN angle error < 30 deg', angle_error_under_threshold(angle_difference, 30))

    get_normals_error_visualization(pred, gt, normals_mask)

if __name__ == "__main__":
    main()