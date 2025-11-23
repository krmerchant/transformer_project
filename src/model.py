import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load(
            'facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(
            in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


class ResnetBackbone(nn.Module):
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
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        del self.resnet.fc

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.resnet.conv1(inputs)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

    def get_num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = models.resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


class DeTR(nn.Module):
    """
    Demo DETR implementation.
    Minimal DeTR
    """

    def __init__(self, num_classes, hidden_dim=128, nheads=8,
                 num_encoder_layers=8, num_decoder_layers=8, feature_map_size=(7, 7),
                 max_num_objects=10):
        super().__init__()

        # spatial positional encodings for 8x256 feature map
        self.backbone = ResnetBackbone()

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)  # adapter conv

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dropout=0.1, activation='relu')

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # self.linear_class = MLP(hidden_dim, hidden_dim,  num_classes + 1, 3)
        # 3d boxes (cx,cy,w,l cos(theta))
        self.linear_bbox = MLP(hidden_dim, hidden_dim,  4, 3)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(max_num_objects, hidden_dim))

        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(
            feature_map_size[0], hidden_dim // 2))  # H=8
        self.col_embed = nn.Parameter(torch.rand(
            feature_map_size[1], hidden_dim // 2))  # W=256

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone(inputs)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

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
        query_pos = self.query_pos.unsqueeze(
            1).repeat(1, B, 1)  # [100, B, hidden_dim]

        # Reshape features: [B, C, H, W] -> [HW, B, C]
        h = h.flatten(2).permute(2, 0, 1)

        # propagate through the transformer
        h = self.transformer(pos + h, query_pos).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


model = DeTR(num_classes=10)
# batch of 2 samples, 5 input channels, 200x176 image size
inputs = torch.rand(2, 3, 224, 224)


outputs = model(inputs)
print(outputs)
