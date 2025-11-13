"""
3D Point Cloud and Bounding Box Visualization Tool

Visualizes LiDAR point clouds and 3D bounding boxes from NuScenes dataset.
Supports interactive 3D visualization using Open3D.

Uses the NuScenesRangeView dataset class for consistent data handling.
"""

import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from dataset import NuScenesRangeView
from nuscenes.utils.data_classes import LidarPointCloud
from typing import Tuple, List, Optional, Dict


class PointCloudVisualizer:
    """Interactive 3D point cloud and bounding box visualizer."""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', 
                 H: int = 32, W: int = 1024, fov_up: float = 10.0, fov_down: float = -30.0):
        """
        Initialize the visualizer using NuScenesRangeView dataset.
        
        Args:
            dataroot: Path to NuScenes dataset root directory
            version: NuScenes version (e.g., 'v1.0-mini', 'v1.0-trainval')
            H: Height of range image
            W: Width of range image
            fov_up: Upper field of view in degrees
            fov_down: Lower field of view in degrees
        """
        self.dataset = NuScenesRangeView(
            dataroot=dataroot, 
            version=version,
            H=H, W=W,
            fov_up=fov_up,
            fov_down=fov_down
        )
        self.nusc = self.dataset.nusc
        self.dataroot = dataroot
        self.current_idx = 0
        
    def load_point_cloud(self, sample_idx: int) -> np.ndarray:
        """
        Load point cloud for a specific sample.
        
        Args:
            sample_idx: Index of the sample
            
        Returns:
            Point cloud as (N, 3) numpy array of (x, y, z) coordinates
        """
        # Use the dataset's range view to reconstruct xyz points.
        data = self.dataset[sample_idx]
        range_img = data['range_image']

        # range_img stored as (C, H, W) where channels are [x, y, z, depth, intensity]
        if isinstance(range_img, torch.Tensor):
            arr = range_img.numpy()
        else:
            arr = range_img

        # Support both (C, H, W) and (H, W, C)
        if arr.ndim == 3 and arr.shape[0] == 5:
            x = arr[0]
            y = arr[1]
            z = arr[2]
            depth = arr[3]
        elif arr.ndim == 3 and arr.shape[2] == 5:
            x = arr[:, :, 0]
            y = arr[:, :, 1]
            z = arr[:, :, 2]
            depth = arr[:, :, 3]
        else:
            raise ValueError(f"Unexpected range_image shape: {arr.shape}")

        mask = depth > 0
        if not np.any(mask):
            return np.zeros((0, 3), dtype=np.float32)

        points = np.stack([x[mask], y[mask], z[mask]], axis=1).astype(np.float32)
        return points
    
    def get_bounding_boxes(self, sample_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Get bounding boxes for a specific sample using the dataset's box format.
        
        Args:
            sample_idx: Index of the sample
            
        Returns:
            Tuple of:
            - List of box centers (3,) each
            - List of box dimensions (3,) each (width, length, height)
            - List of rotation matrices (3, 3) each
            - List of class names
        """
        sample = self.nusc.sample[sample_idx]
        lidar_token = sample['data']['LIDAR_TOP']
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        
        centers = []
        dimensions = []
        rotations = []
        classes = []
        
        for box in boxes:
            # Center position
            centers.append(np.array(box.center, dtype=np.float32))
            
            # Dimensions (width, length, height)
            dimensions.append(np.array(box.wlh, dtype=np.float32))
            
            # Rotation matrix
            rotations.append(box.rotation_matrix)
            
            # Class name
            classes.append(box.name)
        
        return centers, dimensions, rotations, classes
    
    def get_frame_data(self, sample_idx: int) -> Dict:
        """
        Get complete frame data (point cloud + boxes) in a single call.
        
        Args:
            sample_idx: Index of the sample
            
        Returns:
            Dictionary with:
            - 'points': (N, 3) point cloud
            - 'boxes': (M, 7) box parameters [cx, cy, cz, w, l, h, yaw]
            - 'classes': (M,) class indices (1=vehicle, 0=other)
            - 'box_names': List of class names
            - 'range_image': (5, H, W) range image tensor
        """
        # Use the dataset indexing to get range image, boxes and classes
        data = self.dataset[sample_idx]

        # Reconstruct points from range image
        points = self.load_point_cloud(sample_idx)

        # Boxes and classes come directly from the custom dataset
        boxes_tensor = data.get('boxes', None)
        classes_tensor = data.get('classes', None)

        if isinstance(boxes_tensor, torch.Tensor):
            boxes_np = boxes_tensor.numpy()
        elif boxes_tensor is None:
            boxes_np = np.zeros((0, 7), dtype=np.float32)
        else:
            boxes_np = np.array(boxes_tensor, dtype=np.float32)

        if isinstance(classes_tensor, torch.Tensor):
            classes_np = classes_tensor.numpy()
        elif classes_tensor is None:
            classes_np = np.zeros((0,), dtype=np.int64)
        else:
            classes_np = np.array(classes_tensor, dtype=np.int64)

        # Generate human-readable names from class encoding (dataset uses 1=vehicle)
        box_names = [
            ('vehicle' if int(c) == 1 else 'other') for c in classes_np
        ]

        range_image = data['range_image'].numpy() if isinstance(data['range_image'], torch.Tensor) else data['range_image']

        return {
            'points': points,
            'boxes': boxes_np,
            'classes': classes_np,
            'box_names': box_names,
            'range_image': range_image
        }
    
    def create_bounding_box_mesh(
        self,
        box_params: np.ndarray,
        color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> o3d.geometry.LineSet:
        """
        Create a wireframe bounding box mesh from box parameters.
        
        Args:
            box_params: Box parameters [cx, cy, cz, w, l, h, yaw] (7,)
            color: RGB color for the box
            
        Returns:
            Open3D LineSet object
        """
        cx, cy, cz, w, l, h, yaw = box_params
        center = np.array([cx, cy, cz])
        dimensions = np.array([w, l, h])
        
        # Create rotation matrix from yaw
        cos_yaw = np.sin(yaw)
        sin_yaw = np.cos(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Define box corners in local coordinates
        corners_local = np.array([
            [-w/2, -l/2, -h/2],  # 0
            [w/2, -l/2, -h/2],   # 1
            [w/2, l/2, -h/2],    # 2
            [-w/2, l/2, -h/2],   # 3
            [-w/2, -l/2, h/2],   # 4
            [w/2, -l/2, h/2],    # 5
            [w/2, l/2, h/2],     # 6
            [-w/2, l/2, h/2],    # 7
        ], dtype=np.float32)
        
        # Apply rotation and translation
        corners = (rotation @ corners_local.T).T + center
        
        # Define edges connecting the corners
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))
        
        return line_set
    
    def visualize_sample(
        self,
        sample_idx: int,
        point_size: float = 2.0,
        show_classes: bool = True
    ):
        """
        Visualize a single sample with point cloud and bounding boxes.
        
        Args:
            sample_idx: Index of the sample to visualize
            point_size: Size of points in visualization
            show_classes: Whether to print class information
        """
        print(f"\n{'='*60}")
        print(f"Visualizing Sample {sample_idx} / {len(self.dataset) - 1}")
        print(f"{'='*60}")
        
        # Get frame data
        frame_data = self.get_frame_data(sample_idx)
        points = frame_data['points']
        boxes = frame_data['boxes']
        box_names = frame_data['box_names']
        
        print(f"Point cloud shape: {points.shape}")
        
        # Create point cloud geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color points by height for better visualization
        colors = points.copy()
        colors[:, 0] = (colors[:, 0] - colors[:, 0].min()) / (colors[:, 0].max() - colors[:, 0].min() + 1e-6)
        colors[:, 1] = (colors[:, 1] - colors[:, 1].min()) / (colors[:, 1].max() - colors[:, 1].min() + 1e-6)
        colors[:, 2] = (colors[:, 2] - colors[:, 2].min()) / (colors[:, 2].max() - colors[:, 2].min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"Number of objects: {len(boxes)}")
        
        if show_classes:
            print("\nObjects found:")
            for i, cls in enumerate(box_names):
                print(f"  {i+1}. {cls}")
        
        # Color map for different classes
        color_map = {
            'vehicle': (0.0, 1.0, 0.0),      # Red
            'pedestrian': (1.0, 0.0, 0.0),   # Green
            'bicycle': (0.0, 0.0, 1.0),      # Blue
            'motorcycle': (1.0, 1.0, 0.0),   # Yellow
            'bus': (1.0, 0.0, 1.0),          # Magenta
            'truck': (0.0, 1.0, 1.0),        # Cyan
        }
        
        # Create bounding box meshes
        bbox_meshes = []
        for box_params, cls_name in zip(boxes, box_names):
            # Determine color based on class
            color = (0.0, 1.0, 0.0)  # Default green
            for key, col in color_map.items():
                if key.lower() in cls_name.lower():
                    color = col
                    break
            
            bbox = self.create_bounding_box_mesh(box_params, color)
            bbox_meshes.append(bbox)
        
        # Create visualization
        geometries = [pcd] + bbox_meshes
        
        # Print controls
        print("\n" + "="*60)
        print("Visualization Controls:")
        print("  - Left mouse: Rotate")
        print("  - Right mouse: Pan")
        print("  - Scroll: Zoom")
        print("  - P: Pick point")
        print("  - H: Show help")
        print("  - Q: Close window")
        print("="*60 + "\n")
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Sample {sample_idx} - {len(boxes)} objects",
            width=1200,
            height=800
        )
    
    def visualize_range(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """
        Sequentially visualize a range of samples.
        
        Args:
            start_idx: Starting sample index
            end_idx: Ending sample index (inclusive). If None, visualizes all remaining samples.
        """
        if end_idx is None:
            end_idx = len(self.dataset) - 1
        
        for idx in range(start_idx, min(end_idx + 1, len(self.dataset))):
            self.visualize_sample(idx)
    
    def browse_mode(self, start_idx: int = 0):
        """
        Browse mode with arrow key navigation (using keyboard input).
        
        Args:
            start_idx: Starting frame index
        """
        current_idx = start_idx
        max_idx = len(self.dataset) - 1
        
        print("\n" + "="*60)
        print("3D Point Cloud Visualizer - Browse Mode")
        print("="*60)
        print(f"Total samples: {len(self.dataset)}")
        print("\nCommands:")
        print("  'n' or 'next'   - Show next frame")
        print("  'p' or 'prev'   - Show previous frame")
        print("  '<number>'      - Jump to frame")
        print("  'q' or 'quit'   - Exit")
        print("="*60 + "\n")
        
        while True:
            try:
                # Show current frame
                self.visualize_sample(current_idx)
                
                # Get user input
                user_input = input(
                    f"\nFrame {current_idx}/{max_idx} - Enter command (n/p/number/q): "
                ).strip().lower()
                
                if user_input in ['q', 'quit']:
                    print("Exiting visualizer.")
                    break
                elif user_input in ['n', 'next']:
                    if current_idx < max_idx:
                        current_idx += 1
                    else:
                        print("Already at last frame.")
                elif user_input in ['p', 'prev']:
                    if current_idx > 0:
                        current_idx -= 1
                    else:
                        print("Already at first frame.")
                else:
                    try:
                        idx = int(user_input)
                        if 0 <= idx < len(self.dataset):
                            current_idx = idx
                        else:
                            print(f"Invalid index. Available: 0-{max_idx}")
                    except ValueError:
                        print("Invalid command. Use n/p/<number>/q")
            
            except KeyboardInterrupt:
                print("\n\nExiting visualizer.")
                break
            except RuntimeError as e:
                print(f"Error: {e}")
    
    def interactive_visualizer(self):
        """
        Interactive mode for exploring samples.
        """
        print("\n" + "="*60)
        print("3D Point Cloud and Bounding Box Visualizer")
        print("="*60)
        print(f"Total samples available: {len(self.dataset)}\n")
        
        while True:
            try:
                user_input = input(
                    "Enter sample index (or 'quit' to exit, 'help' for options): "
                ).strip().lower()
                
                if user_input == 'quit':
                    print("Exiting visualizer.")
                    break
                elif user_input == 'help':
                    print("\nAvailable commands:")
                    print("  <number>  - Visualize sample at index")
                    print("  range <start> <end> - Visualize range of samples")
                    print("  browse    - Browse mode with next/prev navigation")
                    print("  quit      - Exit the program")
                    print("  help      - Show this message\n")
                elif user_input.startswith('range'):
                    parts = user_input.split()
                    if len(parts) == 3:
                        try:
                            start = int(parts[1])
                            end = int(parts[2])
                            self.visualize_range(start, end)
                        except ValueError:
                            print("Invalid range format. Use: range <start> <end>")
                    else:
                        print("Invalid range format. Use: range <start> <end>")
                elif user_input == 'browse':
                    try:
                        start = int(input("Start at frame index: "))
                        if 0 <= start < len(self.dataset):
                            self.browse_mode(start)
                        else:
                            print(f"Invalid index. Available: 0-{len(self.dataset)-1}")
                    except ValueError:
                        print("Invalid index.")
                else:
                    try:
                        idx = int(user_input)
                        if 0 <= idx < len(self.dataset):
                            self.visualize_sample(idx)
                        else:
                            print(f"Index out of range. Available: 0-{len(self.dataset)-1}")
                    except ValueError:
                        print("Invalid input. Enter a number or 'help'.")
            
            except KeyboardInterrupt:
                print("\n\nExiting visualizer.")
                break
            except RuntimeError as e:
                print(f"Error: {e}")


def main():
    """
    Main function to run the visualizer.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D point clouds and bounding boxes')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuScenes',
        help='Path to NuScenes dataset root directory'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        help='NuScenes version (v1.0-mini, v1.0-trainval, etc.)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Visualize a specific sample index'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PointCloudVisualizer(
        dataroot=args.dataroot,
        version=args.version
    )
    
    if args.sample is not None:
        visualizer.visualize_sample(args.sample)
    elif args.interactive:
        visualizer.interactive_visualizer()
    else:
        visualizer.interactive_visualizer()


if __name__ == '__main__':
    main()
