import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np

class NuScenesRangeView(Dataset):
    def __init__(self, dataroot, version='v1.0-mini',
                 H=32, W=1024, fov_up=10.0, fov_down=-30.0):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = [s['data']['LIDAR_TOP'] for s in self.nusc.sample]

        self.H = H
        self.W = W
        self.fov_up = fov_up * np.pi / 180.0
        self.fov_down = fov_down * np.pi / 180.0
        self.fov = abs(self.fov_up - self.fov_down)

    def __len__(self):
        return len(self.samples)

    def lidar_to_range_view(self, points):
        """Convert point cloud to range view"""
        x = points[0, :]
        y = points[1, :]
        z = points[2, :]
        intensity = points[3, :] if points.shape[0] > 3 else np.ones_like(x)

        # Calculate range
        depth = np.sqrt(x**2 + y**2 + z**2)

        # Calculate angles
        yaw = -np.arctan2(y, x)  # Horizontal angle
        pitch = np.arcsin(z / np.clip(depth, 1e-8, None))  # Vertical angle

        # Project to image coordinates
        u = 0.5 * (yaw / np.pi + 1.0)  # [0, 1]
        v = 1.0 - (pitch + abs(self.fov_down)) / self.fov  # [0, 1]

        # Scale to pixel coordinates
        u = np.floor(u * self.W).astype(np.int32)
        v = np.floor(v * self.H).astype(np.int32)

        # Clip to valid range
        u = np.clip(u, 0, self.W - 1)
        v = np.clip(v, 0, self.H - 1)

        # Initialize range image (5 channels: range, x, y, z, intensity)
        range_image = np.zeros((self.H, self.W, 5), dtype=np.float32)

        # Sort points by depth (keep closest point per pixel)
        order = np.argsort(depth)[::-1]  # Far to near (overwrite far with near)

        for idx in order:
            range_image[v[idx], u[idx], 0] = x[idx]
            range_image[v[idx], u[idx], 1] = y[idx]
            range_image[v[idx], u[idx], 2] = z[idx]
            range_image[v[idx], u[idx], 3] = depth[idx]
            range_image[v[idx], u[idx], 4] = intensity[idx]

        return range_image

    def __getitem__(self, idx):
        lidar_token = self.samples[idx]

        # Load point cloud
        pc_path = self.nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(pc_path)

        # Convert to range view
        range_image = self.lidar_to_range_view(pc.points)

        # Get bounding boxes
        b, boxes, a = self.nusc.get_sample_data(lidar_token)

        # Extract information from boxes
        box_list = []
        class_list = []
        
        for box in boxes:
            # Get class
            category = box.name.lower()
           
            class_idx  = 1 if 'vehicle' in category else 0 # I only care about vehicls
            if class_idx == 0:
                continue  # Skip non-vehicle classes 
            
            class_list.append(class_idx)
            
            # Get center position
            cx, cy, cz = box.center
            
            # Get dimensions (width, length, height)
            w, l, h = box.wlh
            
            # Get rotation (yaw)
            yaw = box.orientation.yaw_pitch_roll[0]
            
            # Combine into 7-dimensional vector [cx, cy, cz, w, l, h, theta]
            box_params = torch.tensor([cx, cy, cz, w, l, h, yaw], dtype=torch.float32)
            box_list.append(box_params)
            
            
        
        # Convert lists to tensors
        if box_list:
            boxes_tensor = torch.stack(box_list)
            classes_tensor = torch.tensor(class_list, dtype=torch.long)
        else:
            # Handle empty case
            boxes_tensor = torch.zeros((0, 7), dtype=torch.float32)
            classes_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            'range_image': torch.from_numpy(range_image).permute(2, 0, 1),  # (5, H, W)
            'boxes': boxes_tensor,  # (N, 7) where N is number of boxes
            'classes': classes_tensor  # (N,) class indices
        }

