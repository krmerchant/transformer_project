import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from PIL import Image
from pyquaternion import Quaternion


class NuScenesCameraView(Dataset):
    """
    NuScenes dataset for camera images with 2D bounding boxes.
    Returns front camera images and projected 2D boxes.
    """
    def __init__(self, dataroot, version='v1.0-mini', 
                 camera='CAM_FRONT', image_size=(900, 1600)):
        """
        Args:
            dataroot: Path to NuScenes dataset
            version: Dataset version ('v1.0-mini' or 'v1.0-trainval')
            camera: Camera sensor to use (default: 'CAM_FRONT')
            image_size: Target image size (H, W)
        """
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.camera = camera
        self.image_size = image_size
        
        # Get all camera samples
        self.samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                if camera in sample['data']:
                    self.samples.append(sample['data'][camera])
                sample_token = sample['next']
    
    def __len__(self):
        return len(self.samples)
    
    def get_2d_boxes(self, camera_token, target_width, target_height):
        """
        Get 2D bounding boxes for a camera image.
        Projects 3D boxes onto the image plane.
        """
        # Get camera calibration and pose
        cam_data = self.nusc.get('sample_data', camera_token)
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
        
        # Get image dimensions
        img_path = self.nusc.get_sample_data_path(camera_token)
        img = Image.open(img_path)
        orig_width, orig_height = img.size
        
        # Calculate scale factors for resizing
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        # Scale the intrinsic matrix to match the resized image
        cam_intrinsic_scaled = cam_intrinsic.copy()
        cam_intrinsic_scaled[0, :] *= scale_x  # Scale focal length and principal point x
        cam_intrinsic_scaled[1, :] *= scale_y  # Scale focal length and principal point y
        
        # Get ego pose
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Get all 3D boxes in the scene
        _, boxes_3d, _ = self.nusc.get_sample_data(camera_token)
        
        box_list = []
        class_list = []
        
        for box in boxes_3d:
            # Filter by category
            category = box.name.lower()
            class_idx = 0 if 'vehicle' in category else 1 if 'pedestrian' in category else -1
            
            if class_idx < 0:
                continue
            
            # Get 3D box corners in global frame
            corners_3d = box.corners()  # (3, 8) array of 8 corners in global coordinates
            
            # Transform from global to ego vehicle frame
            # First apply inverse rotation, then subtract translation
            rotation_ego_inv = Quaternion(ego_pose['rotation']).inverse.rotation_matrix
            corners_3d_ego = rotation_ego_inv @ (corners_3d - np.array(ego_pose['translation']).reshape(3, 1))
            
            # Transform from ego to camera frame
            # First apply inverse rotation, then subtract translation
            rotation_cam_inv = Quaternion(cam_calib['rotation']).inverse.rotation_matrix
            corners_3d_cam = rotation_cam_inv @ (corners_3d_ego - np.array(cam_calib['translation']).reshape(3, 1))
            
            # Remove points behind camera
            if np.any(corners_3d_cam[2, :] <= 0):
                continue
            
            # Project to 2D image plane using SCALED intrinsic matrix
            corners_2d = view_points(corners_3d_cam, cam_intrinsic_scaled, normalize=True)[:2, :]
            
            # Get bounding box from projected corners
            x_min = np.min(corners_2d[0, :])
            x_max = np.max(corners_2d[0, :])
            y_min = np.min(corners_2d[1, :])
            y_max = np.max(corners_2d[1, :])
            
            # Check if box is within image bounds (target size)
            #if x_max < 0 or x_min > target_width or y_max < 0 or y_min > target_height:
            #    continue
            
            # Clip to image boundaries (target size)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(target_width, x_max)
            y_max = min(target_height, y_max)
            
            # Skip tiny boxes
            box_width = x_max - x_min
            box_height = y_max - y_min
           # if box_width < 5 or box_height < 5:
           #     continue
            
            # Convert to center format: [cx, cy, w, h]
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = box_width
            h = box_height
            
            # Normalize to [0, 1] using target dimensions
            cx_norm = cx / target_width
            cy_norm = cy / target_height
            w_norm = w / target_width
            h_norm = h / target_height
            
            box_params = torch.tensor([cx_norm, cy_norm, w_norm, h_norm], dtype=torch.float32)
            box_list.append(box_params)
            class_list.append(class_idx)
        
        return box_list, class_list
    
    def __getitem__(self, idx):
        camera_token = self.samples[idx]
        
        # Load image
        img_path = self.nusc.get_sample_data_path(camera_token)
        image = Image.open(img_path).convert('RGB')
        
        # Resize image if needed
        if image.size != (self.image_size[1], self.image_size[0]):
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Get 2D boxes (pass target dimensions)
        box_list, class_list = self.get_2d_boxes(camera_token, self.image_size[1], self.image_size[0])
        
        # Convert lists to tensors
        if box_list:
            boxes_tensor = torch.stack(box_list)
            classes_tensor = torch.tensor(class_list, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            classes_tensor = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image_tensor,      # (3, H, W) normalized [0, 1]
            'boxes': boxes_tensor,      # (N, 4) - [cx, cy, w, h] normalized [0, 1]
            'labels': classes_tensor    # (N,) class indices
        }

