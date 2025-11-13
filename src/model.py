import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


from qai_hub_models.models.salsanext import Model
from einops import rearrange


class ResNetLiDARExtractor(nn.Module):
    """
    ResNet50 feature extractor for 4-channel LiDAR range images.

    Features:
    - Handles 4x32x1024 input natively (no padding needed)
    - All weights trainable (not frozen) for fine-tuning
    - Pretrained on ImageNet for first 3 channels
    - 4th channel initialized intelligently
    """

    def __init__(self, pretrained=True, num_classes=None):
        """
        Args:
            pretrained: Load ImageNet pretrained weights
            num_classes: If provided, adds classification head
                        If None, returns features only
        """
        super().__init__()

        # Load pretrained ResNet50 (all weights trainable by default)
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)

        # ===== Modify first conv layer: 3 channels -> 4 channels =====
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=5,  # 4 LiDAR channels
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        if pretrained:
            with torch.no_grad():
                # Copy pretrained weights for first 3 channels
                self.conv1.weight[:, :3, :, :] = original_conv.weight
                # Initialize 4th channel as average of RGB channels
                self.conv1.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

        # ===== Keep all other ResNet layers (all trainable) =====
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 -> 256 channels
        self.layer2 = resnet.layer2  # 256 -> 512 channels
        self.layer3 = resnet.layer3  # 512 -> 1024 channels
        self.layer4 = resnet.layer4  # 1024 -> 2048 channels

        self.avgpool = resnet.avgpool

        # Feature dimension
        self.feature_dim = 2048

        # Optional classification head
        self.fc = None
        if num_classes is not None:
            self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_features=False):
        """
        Forward pass through ResNet.

        Args:
            x: (B, 4, 32, 1024) LiDAR range image
            return_features: If True, return intermediate features

        Returns:
            If num_classes is None: (B, 2048) feature vector
            If num_classes is set: (B, num_classes) logits
            If return_features: dict with features from each layer
        """
        # Initial layers
        x = self.conv1(x)  # (B, 64, 16, 512)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 8, 256)

        # Residual blocks
        x1 = self.layer1(x)  # (B, 256, 8, 256)
        x2 = self.layer2(x1)  # (B, 512, 4, 128)
        x3 = self.layer3(x2)  # (B, 1024, 2, 64)
        x4 = self.layer4(x3)  # (B, 2048, 1, 32)

        # Global average pooling
        features = self.avgpool(x4)  # (B, 2048, 1, 1)
        features = torch.flatten(features, 1)  # (B, 2048)

        # Return intermediate features if requested (useful for FPN, etc.)
        if return_features:
            return {
                'layer1': x1,  # 1/4 resolution
                'layer2': x2,  # 1/8 resolution
                'layer3': x3,  # 1/16 resolution
                'layer4': x4,  # 1/32 resolution
                'features': features
            }

        # Classification head if present
        if self.fc is not None:
            return self.fc(features)

        return features

    def freeze_backbone(self):
        """Freeze all layers except the classification head."""
        for name, param in self.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SalsaNextDownsamplingBackbone(nn.Module):
    """Extract only the downsampling blocks from SalsaNext"""

    def __init__(self, pretrained=True, freeze_backbone=False):
        super(SalsaNextDownsamplingBackbone, self).__init__()

        # Load the full SalsaNext model
        if pretrained:
            full_model = Model.from_pretrained()
        else:
            full_model = Model()

        # Handle DataParallel wrapper
        if hasattr(full_model, 'module'):
            salsanext = full_model.module
        else:
            salsanext = full_model

        # Extract only the downsampling blocks
        self.downCntx = salsanext.downCntx  # ResContextBlock: 5 -> 32 channels
        self.downCntx2 = salsanext.downCntx2  # ResContextBlock: 32 -> 32 channels
        self.downCntx3 = salsanext.downCntx3  # ResContextBlock: 32 -> 32 channels
        self.resBlock1 = salsanext.resBlock1  # ResBlock: 32 -> 64 channels + downsample
        self.resBlock2 = salsanext.resBlock2  # ResBlock: 64 -> 128 channels + downsample
        self.resBlock3 = salsanext.resBlock3  # ResBlock: 128 -> 256 channels + downsample
        self.resBlock4 = salsanext.resBlock4  # ResBlock: 256 -> 256 channels + downsample
        self.resBlock5 = salsanext.resBlock5  # ResBlock: 256 -> 256 channels (no downsample)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Pass through downsampling blocks sequentially
        features = []

        # Context blocks (no downsampling)
        x = self.downCntx(x)  # (B, 5, H, W) -> (B, 32, H, W)
        features.append(x)

        x = self.downCntx2(x)  # (B, 32, H, W) -> (B, 32, H, W)
        features.append(x)

        x = self.downCntx3(x)  # (B, 32, H, W) -> (B, 32, H, W)
        features.append(x)

        # ResBlocks with downsampling
        x = self.resBlock1(x)  # (B, 32, H, W) -> (B, 64, H/2, W/2)
        features.append(x)

        x = self.resBlock2(x)  # (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
        features.append(x)

        x = self.resBlock3(x)  # (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
        features.append(x)

        x = self.resBlock4(x)  # (B, 256, H/8, W/8) -> (B, 256, H/16, W/16)
        features.append(x)

        x = self.resBlock5(x)  # (B, 256, H/16, W/16) -> (B, 256, H/16, W/16)
        features.append(x)

        return {
            'features': features,  # All intermediate features
            'final_features': x,  # Final downsampled features (256 channels)
            'multi_scale': {
                'full_res': features[2],  # 32 channels, full resolution
                'half_res': features[3],  # 64 channels, 1/2 resolution
                'quarter_res': features[4],  # 128 channels, 1/4 resolution
                'eighth_res': features[5],  # 256 channels, 1/8 resolution
                'sixteenth_res': features[7]  # 256 channels, 1/16 resolution
            }
        }







class DeTR(nn.Module):
    """
    Demo DETR implementation.
    Minimal DeTR 
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, feature_map_size=(8,256), 
                 max_num_objects=20):
        super().__init__()

        self.backbone = ResNetLiDARExtractor()

        # create conversion layer
        self.conv = nn.Conv2d(256, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 7) # 3d boxes (cx,cy,cz,w,l,h, theta) 

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(max_num_objects, hidden_dim))

        # spatial positional encodings for 8x256 feature map
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(feature_map_size[0], hidden_dim // 2))  # H=8
        self.col_embed = nn.Parameter(torch.rand(feature_map_size[1], hidden_dim // 2))  # W=256

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone(inputs,return_features=True)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x['layer1'])

        # construct positional encodings
        H, W = h.shape[-2:]
        B = h.shape[0]  # batch size
        
        # Create and expand positional encodings for each item in the batch
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # Expand pos to match batch size
        pos = pos.repeat(1, B, 1)  # [HW, B, hidden_dim]
        
        # Prepare queries
        query_pos = self.query_pos.unsqueeze(1).repeat(1, B, 1)  # [100, B, hidden_dim]
        
        # Reshape features: [B, C, H, W] -> [HW, B, C]
        h = h.flatten(2).permute(2, 0, 1)
        
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h, query_pos).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h)}
    


model_backbone = ResNetLiDARExtractor()
model = DeTR(num_classes=10)
inputs = torch.rand(2, 5, 32, 1024)  # batch of 2 samples, 5 input channels, 200x176 image size


outputs = model(inputs)

