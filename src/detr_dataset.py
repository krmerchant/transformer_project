import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image


class CPPE5Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None, device="cuda"):
        """
        Args:
            hf_dataset: Hugging Face dataset split (e.g., ds['train'])
            transform: Optional transform to apply to images
        """
        self.device = device
        self.dataset = hf_dataset
        self.transform = transform
        self.label_names = ['Coverall', 'Face_Shield',
                            'Gloves', 'Goggles', 'Mask']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image and convert to RGB
        image = item['image']
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Store original dimensions for bbox scaling
            orig_width, orig_height = image.size

            # Apply transform
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

        # Get bboxes and labels
        # CPPE-5 format: [x_min, y_min, width, height] in absolute coordinates
        bboxes = torch.tensor(item['objects']['bbox'],
                              dtype=torch.float32, device=self.device)
        labels = torch.tensor(
            item['objects']['category'], dtype=torch.int64, device=self.device)

        # Convert from [x_min, y_min, w, h] to [center_x, center_y, w, h]
        # and normalize by original image dimensions
        if len(bboxes) > 0:
            # Extract components
            x_min = bboxes[:, 0]
            y_min = bboxes[:, 1]
            width = bboxes[:, 2]
            height = bboxes[:, 3]

            # Convert to center coordinates
            center_x = x_min + width / 2
            center_y = y_min + height / 2

            # Normalize by original image dimensions to [0, 1]
            center_x = center_x / orig_width
            center_y = center_y / orig_height
            width = width / orig_width
            height = height / orig_height

            # Stack back together: [center_x, center_y, w, h]
            bboxes = torch.stack([center_x, center_y, width, height], dim=1)
        else:
            # Empty bbox tensor if no objects
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        # Create target dictionary
        target = {
            'boxes': bboxes,
            'labels': labels
        }

        # IMPORTANT: Return as tuple (image, target) for DETR collate_fn
        return image, target
