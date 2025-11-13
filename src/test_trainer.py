import os
import sys
import torch
from torch.utils.data import DataLoader
from model import DeTR
from trainer import DeTRTrainer
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)



# Add notebooks directory to Python path
notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
sys.path.append(notebooks_dir)
from dataset import NuScenesRangeView

def collate_fn(batch):
    """Custom collate function to handle NuScenes range view data"""
    images = torch.stack([item['range_image'] for item in batch])
    
    # Find maximum number of objects in this batch
    max_objects = max(item['boxes'].shape[0] for item in batch)
    
    batch_size = len(batch)
    no_object_idx = 0  # Index for no object
    labels = torch.full((batch_size, max_objects), no_object_idx)  # -1 padding for labels
    boxes = torch.zeros(batch_size, max_objects, 7)
    
    # Fill in the actual values
    for i, item in enumerate(batch):
        num_objects = item['boxes'].shape[0]
        # Assuming classes are already encoded as integers
        labels[i, :num_objects] = item['classes']
        boxes[i, :num_objects] = item['boxes']
    
    targets = {
        'labels': labels,
        'boxes': boxes
    }
    
    return images, targets

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create NuScenes dataset
    nuscenes_root = '/home/komelmerchant/data/sets/nuscenes'
    dataset = NuScenesRangeView(nuscenes_root, H=32, W=1024)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    num_classes = 1
    model = DeTR(num_classes=num_classes)
    
    # Initialize trainer
    trainer = DeTRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_classes=num_classes,
        learning_rate=1e-3
        

    )
    
    # Train for a few epochs
    try:
        logger.debug("Starting training...")
        trainer.train(num_epochs=3)
        logger.debug("Training completed successfully!")
        
        # Save final checkpoint
        trainer.save_checkpoint('test_checkpoint.pth')
        print("Checkpoint saved successfully!")
        
    except Exception as e:
        logger.debug(f"An error occurred during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()