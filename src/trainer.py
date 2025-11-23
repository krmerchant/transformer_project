from models.detr import SetCriterion
from models.matcher import HungarianMatcher
import torch
import torch.optim as optim
# from detr_loss import SetCriterion
# from detr_matcher import HungarianMatcher


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeTRTrainer:
    def __init__(self,
                 model,
                 criterion,
                 train_dataloader,
                 val_dataloader=None,
                 normalization_factors=None,
                 num_classes=10,
                 learning_rate=1e-2,
                 weight_decay=1e-4,
                 test_query_collapse=True):
        self.test_query_collapse = test_query_collapse
        self.box_norm_factors = normalization_factors
        # print number of trianable parameters

        # @todo probably should get this from data
        self.W = 224
        self.H = 224
        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params}")
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.box_norm_factors = normalization_factors
        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize matcher and loss function
    #    self.matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    #    weight_dict = {'loss_ce': 1, 'loss_bbox': 1}
    #    weight_dict['loss_giou'] = 1
    #    self.criterion = SetCriterion(
    #        num_classes=num_classes,
    #        matcher=self.matcher,
    #        losses=['labels', 'boxes', 'cardinality'],
    #        eos_coef=0.1,
    #        weight_dict=weight_dict)       # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        # self.lr_scheduler = optim.lr_scheduler.StepLR(
        #    self.optimizer,
        #    step_size=30,
        #    gamma=0.1
        # )

    def check_query_collapse(self, pred_logits, pred_boxes, threshold=0.5):
        """Detect if queries are collapsing to same predictions"""

        scales = torch.tensor(
            [self.W, self.H, self.W, self.H], device=pred_boxes.device)
        norm_pred_boxes = pred_boxes * scales
        # Get predicted classes (argmax)
        pred_classes = pred_logits.argmax(dim=-1)  # [B, num_queries]
        pred_probs = pred_logits.softmax(
            dim=-1).max(dim=-1)[0]  # [B, num_queries]

        # Filter by confidence threshold
        confident_preds = (pred_probs > threshold).float()

        # Check 1: Are all queries predicting the same class?
        for b in range(pred_classes.shape[0]):

            unique_classes = pred_classes[b, confident_preds[b] > 0].unique()
            print(f"Batch {b}: {len(unique_classes)} unique classes predicted")
            if len(unique_classes) <= 2:  # Only background + 1 object class
                print(f"  ⚠️  WARNING: Query collapse detected!")

        # Check 2: Are box predictions clustered?
        box_std = norm_pred_boxes.std(dim=1)  # std across queries
        box_mean = norm_pred_boxes.mean(dim=1)  # std across queries

        mean_logits = pred_logits.mean(dim=1)  # mean across queries
        std_logits = pred_logits.std(dim=1)  # mean across queries
        logger.info(f"Mean predicted logits: {mean_logits}")
        logger.info(f"STD of predicted logits: {std_logits}")
        # Print as table
        logger.info(
            f"{'Metric':<10} {'X':<10} {'Y':<10} {'Z':<10} {'W':<10} {'L':<10} {'H':<10}")
        logger.info("-" * 70)
        logger.info(
            f"{'STD':<10} {box_std[0,0]:<10.4f} {box_std[0,1]:<10.4f} {box_std[0,2]:<10.4f} {box_std[0,3]:<10.4f}")
        logger.info(
            f"{'MEAN':<10} {box_mean[0,0]:<10.4f} {box_mean[0,1]:<10.4f} {box_mean[0,2]:<10.4f} {box_mean[0,3]:<10.4f}  ")

        # Check 3: Are query embeddings diverse?
        return {
            'unique_classes': len(unique_classes),
            'box_diversity': box_std.mean().item(),
            'confident_predictions': confident_preds.sum().item()
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.criterion.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        num_params = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params}")

        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            # Move data to device

            logger.info(f"Number of trainable parameters: {num_params}")
            images = images.to(self.device)
            images = images.tensors
#            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
#                       for k, v in targets.items()}
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=0.1)

            self.optimizer.step()

            total_loss += loss.item()

            logger.info(f"Number of trainable parameters: {num_params}")

            if batch_idx % 10 == 0 or True:
                class_loss = loss_dict['loss_ce'].item()
                bbox_loss = loss_dict['loss_bbox'].item()
                giou_loss = loss_dict['loss_giou'].item()
                logger.info(
                    f'Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f},Class Loss: {class_loss:.4f}  Bbox Loss: {bbox_loss:.4f}, giou_loss: {giou_loss:4f}')

        return total_loss / num_batches

    @ torch.no_grad()
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
     #       targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
     #                  for k, v in targets.items()}

            images = images.tensors
            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)

            total_loss += loss.item()
            if (self.test_query_collapse):
                self.model.eval()  # Set model to eval mode for analysis
                re_classify = self.model(images)
                collapse_stats = self.check_query_collapse(
                    re_classify['pred_logits'],
                    re_classify['pred_boxes']
                )

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

            # self.lr_scheduler.step()

    def save_checkpoint(self, path):
        """Save a checkpoint of the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """Load a checkpoint of the model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
