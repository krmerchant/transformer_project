import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import numpy as np

import logging
logging.basicConfig( level=logging.INFO)
logger = logging.getLogger(__name__)


class HungarianMatcher(nn.Module):
    """
    Compute the assignment between ground truth boxes and predictions using Hungarian algorithm.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Compute the classification cost
        out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes+1]
        tgt_ids = targets["labels"]  # [batch_size, num_target_boxes]
        
        # Compute the L1 cost between boxes
        out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 7]
        tgt_bbox = targets["boxes"]  # [batch_size, num_target_boxes, 7]
        
        indices = []
        for i in range(bs):
            # Classification cost: -log probability of the correct class
            cost_class = -out_prob[i, :, tgt_ids[i]]
            
            # Box L1 cost
            cost_bbox = torch.cdist(out_bbox[i], tgt_bbox[i], p=1)

            #@todo: maybe add GIoU cost here
             
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
            
            # Hungarian algorithm
            indices.append(
                linear_sum_assignment(C.cpu().numpy())
            )
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class DeTRLoss(nn.Module):
    """
    Loss function for DETR
    """
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        
        # Loss weights
        self.weight_dict = {
            'loss_ce': 1,
            'loss_bbox': 1,
        }
        
        # Create loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, outputs, targets):
        # Debug prints and sanity checks
        logger.debug(f"Pred logits shape: {outputs['pred_logits'].shape}")
        logger.debug(f"Pred boxes shape: {outputs['pred_boxes'].shape}")
        logger.debug(f"Target boxes shape: {targets['boxes'].shape}")
        logger.debug(f"Target labels shape: {targets['labels'].shape}")
        
        # Verify no NaNs or infinities
        assert torch.isfinite(outputs['pred_boxes']).all(), "Predicted boxes contain NaN or inf"
        assert torch.isfinite(targets['boxes']).all(), "Target boxes contain NaN or inf"
        
        # Retrieve the matching between predictions and targets
        indices = self.matcher(outputs, targets)
        
        # Compute all the requested losses
        losses = {}
        
        # Classification loss
        pred_logits = outputs['pred_logits']
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=pred_logits.device)
                                  
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            target_classes[batch_idx, pred_idx] = targets['labels'][batch_idx, tgt_idx]
            
        # Compute classification loss with diagnostics on failure
        losses['loss_ce'] = self.ce_loss(pred_logits.flatten(0, 1), target_classes.flatten())
       
        # Bounding box loss
        pred_boxes = outputs['pred_boxes']
        target_boxes = torch.zeros(pred_boxes.shape, device=pred_boxes.device)
        
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            logger.debug(f"Batch {batch_idx}, Pred indices: {pred_idx}, Target indices: {tgt_idx}")
            logger.debug(f"Target boxes shape: {targets['boxes'].shape}")
            logger.debug(f"Batch targets shape: {targets['boxes'][batch_idx].shape}")
            
            # Convert indices to CPU for safe indexing
            pred_idx_cpu = pred_idx.cpu()
            tgt_idx_cpu = tgt_idx.cpu()
            
            # Get boxes for this batch
            
            # More careful assignment
            for i, (p, t) in enumerate(zip(pred_idx_cpu, tgt_idx_cpu)):
                target_boxes[batch_idx, p.item()] = targets['boxes'][batch_idx][t.item()]
            
            logger.debug(f"Assignment completed for batch {batch_idx}")
            
        losses['loss_bbox'] = self.l1_loss(pred_boxes, target_boxes)
        
        # Compute total weighted loss
        total_loss = sum(self.weight_dict[k] * losses[k] for k in losses.keys())
        losses['total_loss'] = total_loss
        
        return losses

class DeTRTrainer:
    def __init__(self, 
                 model,
                 train_dataloader,
                 val_dataloader=None,
                 num_classes=10,
                 learning_rate=1e-4,
                 weight_decay=1e-4):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize matcher and loss function
        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=1)
        self.criterion = DeTRLoss(num_classes=num_classes, matcher=self.matcher)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            # Move data to device
            images = images.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                class_loss = losses['loss_ce'].item()
                bbox_loss = losses['loss_bbox'].item()
                logger.info(f'Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}')
                logger.info(f'Batch [{batch_idx}/{num_batches}], Class Loss: {class_loss:.4f}')
                logger.info(f'Batch [{batch_idx}/{num_batches}], Bbox Loss: {bbox_loss:.4f}')
                 
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        for images, targets in self.val_dataloader:
            # Move data to device
            images = images.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            loss = losses['total_loss']
            
            total_loss += loss.item()
            
        return total_loss / num_batches
    
    def train(self, num_epochs):
        """Train the model for specified number of epochs"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            
            # Train for one epoch
            train_loss = self.train_epoch()
            print(f'Training Loss: {train_loss:.4f}')
            
            # Validate
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f'Validation Loss: {val_loss:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    }, 'best_model.pth')
            
            # Update learning rate
            self.lr_scheduler.step()
            
    def save_checkpoint(self, path):
        """Save a checkpoint of the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, path)
        
    def load_checkpoint(self, path):
        """Load a checkpoint of the model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

# Example usage:
"""
# Initialize model and dataloaders
model = DeTR(num_classes=10)
train_dataloader = ...  # Your training dataloader
val_dataloader = ...    # Your validation dataloader

# Initialize trainer
trainer = DeTRTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_classes=10,
    learning_rate=1e-4
)

# Train the model
trainer.train(num_epochs=100)
"""