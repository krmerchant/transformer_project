"""
nuScenes Data Processing for SalsaNext Model
This script demonstrates how to load nuScenes LiDAR data and pass it through
the Qualcomm SalsaNext model for semantic segmentation.
"""

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import qai_hub_models
from qai_hub_models.models.salsanext import Model


# ============================================================================
# STEP 1: Initialize nuScenes Dataset
# ============================================================================

def init_nuscenes(dataroot='/data/nuscenes', version='v1.0-mini'):
    """
    Initialize the nuScenes dataset.

    Args:
        dataroot: Path to nuScenes dataset root directory
        version: Dataset version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')

    Returns:
        nusc: NuScenes object
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    return nusc


# ============================================================================
# STEP 2: Load LiDAR Point Cloud and Camera Image from nuScenes
# ============================================================================

def load_lidar_pointcloud(nusc, sample_token):
    """
    Load LiDAR point cloud data for a given sample.

    Args:
        nusc: NuScenes object
        sample_token: Token identifying the sample

    Returns:
        points: Nx4 array [x, y, z, intensity]
    """
    # Get sample data
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']

    # Get LiDAR data path
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = nusc.get_sample_data_path(lidar_token)

    # Load point cloud
    pc = LidarPointCloud.from_file(lidar_filepath)

    # Points are in format [x, y, z, intensity] (4 x N)
    points = pc.points.T  # Convert to N x 4

    return points


def load_camera_image(nusc, sample_token, camera_channel='CAM_FRONT'):
    """
    Load camera image for a given sample.

    Args:
        nusc: NuScenes object
        sample_token: Token identifying the sample
        camera_channel: Camera to load ('CAM_FRONT', 'CAM_FRONT_LEFT', etc.)

    Returns:
        image: RGB image as numpy array
    """
    from PIL import Image

    # Get sample data
    sample = nusc.get('sample', sample_token)
    camera_token = sample['data'][camera_channel]

    # Get camera image path
    camera_filepath = nusc.get_sample_data_path(camera_token)

    # Load image
    image = Image.open(camera_filepath)
    image = np.array(image)

    return image


# ============================================================================
# STEP 3: Convert Point Cloud to Range View (Spherical Projection)
# ============================================================================

def pointcloud_to_rangeview(points, H=64, W=2048, fov_up=10.0, fov_down=-30.0):
    """
    Convert 3D point cloud to range view image (spherical projection).
    This is the input format expected by SalsaNext.

    Args:
        points: Nx4 array [x, y, z, intensity]
        H: Height of range image (vertical resolution)
        W: Width of range image (horizontal resolution)
        fov_up: Upper field of view in degrees
        fov_down: Lower field of view in degrees

    Returns:
        range_image: HxWx5 tensor with channels [x, y, z, range, intensity]
    """
    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    # Calculate range (depth)
    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Calculate angles
    yaw = -np.arctan2(y, x)  # Horizontal angle
    pitch = np.arcsin(z / (depth + 1e-8))  # Vertical angle

    # Convert FOV to radians
    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov = abs(fov_up_rad) + abs(fov_down_rad)

    # Project to image coordinates
    u = 0.5 * (yaw / np.pi + 1.0)  # Normalized to [0, 1]
    v = 1.0 - (pitch + abs(fov_down_rad)) / fov  # Normalized to [0, 1]

    # Convert to pixel coordinates
    u = np.floor(u * W).astype(np.int32)
    v = np.floor(v * H).astype(np.int32)

    # Clip to valid range
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # Initialize range image with zeros
    # 5 channels: x, y, z, depth, intensity
    range_image = np.zeros((H, W, 5), dtype=np.float32)

    # Fill range image (keep closest point for each pixel)
    # Sort by depth to keep closest points
    order = np.argsort(depth)[::-1]  # Reverse to put farthest first

    for idx in order:
        range_image[v[idx], u[idx], 0] = x[idx]
        range_image[v[idx], u[idx], 1] = y[idx]
        range_image[v[idx], u[idx], 2] = z[idx]
        range_image[v[idx], u[idx], 3] = depth[idx]
        range_image[v[idx], u[idx], 4] = intensity[idx]

    return range_image


# ============================================================================
# STEP 4: Preprocess for SalsaNext
# ============================================================================

def preprocess_for_salsanext(range_image):
    """
    Preprocess range image for SalsaNext model.
    Model expects input shape: [1, 5, 64, 2048]

    Args:
        range_image: HxWx5 numpy array

    Returns:
        input_tensor: Torch tensor ready for model [1, 5, H, W]
    """
    # Convert to torch tensor and rearrange dimensions
    # From [H, W, C] to [C, H, W]
    input_tensor = torch.from_numpy(range_image).permute(2, 0, 1)

    # Add batch dimension [1, C, H, W]
    input_tensor = input_tensor.unsqueeze(0)

    # Normalize (optional - depends on model training)
    # You may need to adjust normalization based on the specific model

    return input_tensor


# ============================================================================
# STEP 5: Run SalsaNext Inference
# ============================================================================

def run_salsanext_inference(input_tensor):
    """
    Run inference using SalsaNext model from Qualcomm AI Hub.

    Args:
        input_tensor: [1, 5, 64, 2048] torch tensor

    Returns:
        predictions: Semantic segmentation predictions
    """
    # Load pretrained SalsaNext model
    model = Model.from_pretrained().to('cuda')
    model.eval()

    with torch.no_grad():
        # Run inference
        output = model(input_tensor).to('cuda')

    return output


# ============================================================================
# STEP 6: Post-process Results
# ============================================================================

def postprocess_predictions(predictions, range_image):
    """
    Post-process model predictions and map back to 3D points.

    Args:
        predictions: Model output [1, num_classes, H, W]
        range_image: Original range image [H, W, 5]

    Returns:
        point_labels: Per-point semantic labels
    """
    # Get predicted class for each pixel
    pred_labels = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()

    # Get valid points (non-zero depth)
    valid_mask = range_image[:, :, 3] > 0

    # Extract labels for valid points only
    point_labels = pred_labels[valid_mask]

    return point_labels, pred_labels


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def process_nuscenes_sample(nusc, sample_token):
    """
    Complete pipeline: nuScenes sample -> SalsaNext -> predictions

    Args:
        nusc: NuScenes object
        sample_token: Token identifying the sample

    Returns:
        point_labels: Per-point semantic labels
        pred_image: 2D semantic segmentation map
    """
    print(f"Processing sample: {sample_token}")

    # Step 1: Load point cloud
    print("Loading point cloud...")
    points = load_lidar_pointcloud(nusc, sample_token)
    print(f"Loaded {len(points)} points")

    # Step 2: Convert to range view
    print("Converting to range view...")
    range_image = pointcloud_to_rangeview(points, H=64, W=2048)

    # Step 3: Preprocess
    print("Preprocessing...")
    input_tensor = preprocess_for_salsanext(range_image)
    print(f"Input shape: {input_tensor.shape}")

    # Step 4: Run inference
    print("Running inference...")
    predictions = run_salsanext_inference(input_tensor)
    print(f"Output shape: {predictions.shape}")

    # Step 5: Post-process
    print("Post-processing...")
    point_labels, pred_image = postprocess_predictions(predictions, range_image)

    print("Done!")
    return point_labels, pred_image, range_image


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize nuScenes
    nusc = init_nuscenes(
        dataroot='/home/komelmerchant/data/sets/nuscenes',
        version='v1.0-mini'
    )



    # Get first sample
    sample_token = nusc.sample[15]['token']

    image =  load_camera_image(nusc, sample_token)

    # Process sample
    point_labels, pred_image, range_image = process_nuscenes_sample(
        nusc,
        sample_token
    )

    # Visualize results
    import matplotlib.pyplot as
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Show range (depth) view
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Range View (Depth)')
    axes[0, 0].axis('off')

    # Show predictions
    axes[0, 1].imshow(pred_image, cmap='tab20')
    axes[0, 1].set_title('Semantic Segmentation')
    axes[0, 1].axis('off')

    # Top-down view of point cloud (BEV - Bird's Eye View)
    valid_mask = range_image[:, :, 3] > 0
    x_coords = range_image[valid_mask, 0]
    y_coords = range_image[valid_mask, 1]
    depths = range_image[valid_mask, 3]

    scatter = axes[1, 0].scatter(x_coords, y_coords, c=depths,
                                 cmap='viridis', s=0.1, alpha=0.6)
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Top-Down View (Colored by Depth)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Depth (m)')

    # Top-down view with semantic labels
    pred_labels_flat = pred_image[valid_mask]
    scatter2 = axes[1, 1].scatter(x_coords, y_coords, c=pred_labels_flat,
                                  cmap='tab20', s=0.1, alpha=0.6)
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].set_title('Top-Down View (Semantic Labels)')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 1], label='Class ID')

    plt.tight_layout()
    plt.savefig('salsanext_results.png', dpi=150, bbox_inches='tight')

    plt.show()
    print("Results saved to salsanext_results.png")