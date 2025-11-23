import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from trainer import DeTRTrainer
import numpy as np
import logging
from detr_dataset import CPPE5Dataset
from datasets import load_dataset
from torchvision import transforms
from misc import collate_fn
from dataclasses import dataclass
from models.detr import build as build_model

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


@dataclass
class Args:
    num_classes: int = 4
    device: str = "cuda"
    hidden_dim = 512
    position_embedding: str = "sine"
    lr_backbone: float = 1e-3
    dilation: bool = False
    set_cost_class: float = 1
    set_cost_bbox: float = 5
    set_cost_giou: float = 2
    dice_loss_coef: float = 1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    eos_coef = 0.1
    masks = False
    backbone = 'resnet50'
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = True
    num_queries = 100
    aux_loss = False


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    ds = load_dataset("rishitdagli/cppe-5")
    dataset = CPPE5Dataset(ds['train'], transform=transform)
    dataset = Subset(dataset, indices=range(20))
    # only extract 20 samples for train val split
    logger.warning(f"Dataset size: {len(dataset)} samples")

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Initialize model
    num_classes = 4

    args = Args()
    model, criterion, _ = build_model(args)

    # Initialize trainer
    trainer = DeTRTrainer(
        model=model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_classes=num_classes,
        learning_rate=1e-4
        #   normalization_factors=dataset.dataset.normalization_factors
    )

    # Train for a few epochs
    logger.info("Starting training...")
    trainer.train(num_epochs=40)
    logger.info("Training completed successfully!")

    # Save final checkpoint
    trainer.save_checkpoint('test_checkpoint.pth')
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
