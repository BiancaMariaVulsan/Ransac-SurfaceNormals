import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import math
import matplotlib
import torch
from evaluation import normals_angle_difference, angle_error_under_threshold, get_normals_error_visualization, eval_pred_sn

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

    return np.dstack((x_3d, y_3d, z))


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


def determine_normals(depth, mask, K):
    """Determine the normals corresponding to each pixel.

        Args:
            depth: The depth map of the image.
            K: The size of the neighborhood of a pixel, used for plane fitting.

        Returns:
            normals: The 2D array containing at each location the (nx, nt, nz) coordinates of the normal to that specific point.
    """
    width, height = depth.shape

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
    points_3d = depth_map_to_3d_points(depth, intrinsics)

    depth_region = []
    normals = np.zeros_like(points_3d)
    # Perform plane fitting on neighboring pixels
    for x in range(K, width-K):
        for y in range(K, height-K):
            # Extract depth values from the region around the pixel
            depth_region = points_3d[x-K:x+K+1, y-K:y+K+1]

            # Reshape the depth_region
            X = depth_region[:,:,0]
            Y = depth_region[:,:,1]
            Z = depth_region[:,:,2]

            # Fit a plane to the points in the region
            a,b,c = fit_plane(X, Y, Z)
            # Save the normal
            normals[x, y] = (a, b, c)

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


def visualize_normals_channels(normals_matrix, mask):
    """Visualize separately the channels of the normals

        Args:
            normals_matrix: A 2D array containing the normals to each point in the original image.
    """
    # Separate the channels and apply validity mask
    channel_r = np.multiply(normals_matrix[:, :, 0], mask)
    channel_g = np.multiply(normals_matrix[:, :, 1], mask)
    channel_b = np.multiply(normals_matrix[:, :, 2], mask)
    # Apply mask to each channel and set invalid pixels to NaN
    channel_r = np.where(mask == 0, np.nan, channel_r)
    channel_g = np.where(mask == 0, np.nan, channel_g)
    channel_b = np.where(mask == 0, np.nan, channel_b)

    # Display each channel using jet colormap
    plt.figure(figsize=(15, 5))

    # Red channel
    plt.subplot(131) # 131 = one grid with one row and three columns and select the first subplot
    plt.imshow(channel_r, cmap='jet')
    plt.title('Red Channel - x')
    plt.axis('off')
    plt.colorbar()
    plt.gca().add_patch(plt.Rectangle((0, 0), channel_r.shape[1], channel_r.shape[0], color='black', fill=False, linewidth=2))

    # Green channel
    plt.subplot(132) # 132 = one grid with one row and three columns and select the second subplot
    plt.imshow(channel_g, cmap='jet')
    plt.title('Green Channel - y')
    plt.axis('off')
    plt.colorbar()
    plt.gca().add_patch(plt.Rectangle((0, 0), channel_r.shape[1], channel_r.shape[0], color='black', fill=False, linewidth=2))

    # Blue channel
    plt.subplot(133) # 133 = one grid with one row and three columns and select the third subplot
    plt.imshow(channel_b, cmap='jet')
    plt.title('Blue Channel - z')
    plt.axis('off')
    plt.colorbar()
    plt.gca().add_patch(plt.Rectangle((0, 0), channel_r.shape[1], channel_r.shape[0], color='black', fill=False, linewidth=2))

    plt.tight_layout()
    plt.show()


def plot_3d_coordinate_system():
    """
        Print the 3D coordinate system.
    """
    fig_legend = plt.figure()
    ax_legend = fig_legend.add_subplot(111, projection='3d')

    # Define the axis limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    z_min, z_max = 0, 1

    # Set axis limits
    ax_legend.set_xlim(x_min, x_max)
    ax_legend.set_ylim(y_min, y_max)
    ax_legend.set_zlim(z_min, z_max)

    # Draw the x, y, and z-axis lines
    ax_legend.quiver(0, 0, 0, 1, 0, 0, color='red', label='X')
    ax_legend.quiver(0, 0, 0, 0, 1, 0, color='blue', label='Y')
    ax_legend.quiver(0, 0, 0, 0, 0, 1, color='green', label='Z')

    plt.tight_layout()
    plt.figtext(0.5, 0.91, "X-axis: Red, Y-axis: Green, Z-axis: Blue", ha="center", fontsize=10, color='black')
    plt.show()


def compute_evaluation_metrics(result_img, ground_truth, mask, median=True):
    """Calculate the Mean/Median Absolute Error between two images, considering the validity mask.

    Args:
        result_img: The image resulting from our algorithm.
        ground_truth: The image given in the dataset.
        mask: A 2D array that contains 0 at the positions that should not be considered.
        median: If true, the median absolute error is calculated, otherwise, the mean absolute error is calculated.

    Returns:
        float: The Mean/Median Absolute Error between the two images, considering the mask.
    """
    # Ensure that the image and the ground truth have the same shape
    if result_img.shape != ground_truth.shape:
        print("The image has to have the same shape as the ground truth.")

    # Calculate the absolute difference between the two images
    absolute_diff = np.abs(result_img - ground_truth)

    # Make the mean between the channels to reshape the array
    absolute_diff = np.mean(absolute_diff, axis=2)

    # Apply the mask to the absolute difference
    masked_diff = np.multiply(absolute_diff, mask)

    # Flatten the masked absolute difference array to 1D, excluding masked positions (where mask is 0)
    flat_diff = masked_diff.ravel()
    non_zero_diff = flat_diff[flat_diff != 0]

    # Calculate the absolute error considering the mask
    if median == True:
        return np.median(non_zero_diff)
    else:
        return np.mean(non_zero_diff)
    

def main():
    folder_path = "C:\\DFM\\tasks\\sprint1\\surface_normal_initial_impl\\surface_normals_nyu\\"
    images, depths, ground_truth, masks = load_data(folder_path)
    normals = determine_normals(depths[0], masks[0], K=10)
    visualize_normals(normals, images[0], depths[0], masks[0], ground_truth[0])
    # visualize_normals_channels(normals, masks[0])
    # plot_3d_coordinate_system()

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

    # print("Median Absolute Error: ", compute_evaluation_metrics(normals, ground_truth[0], masks[0], median=True))
    # print("Mean Absolute Error: ", compute_evaluation_metrics(normals, ground_truth[0], masks[0], median=False))
    
    # for depth in depths:
    #     determine_normals(depth)

if __name__ == "__main__":
    main()