import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .components import MomentumFinalBlock, SpatialAttention, MemoryBank
from config import ModelConfig

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', config=None):
        super(ChestXrayModel, self).__init__()
        
        # Use provided config or the default ModelConfig
        self.config = config if config is not None else ModelConfig
        
        # Backbone and Final Block
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(
                self.base_model.conv1, self.base_model.bn1, self.base_model.relu,
                self.base_model.maxpool, self.base_model.layer1, self.base_model.layer2,
                self.base_model.layer3
            )
            self.final_block = self.base_model.layer4
            self.feature_dim = 2048
        elif model_name == 'densenet121':
            self.base_model = models.densenet121(pretrained=True)
            features = list(self.base_model.features.children())
            self.backbone = nn.Sequential(*features[:-1])
            self.final_block = nn.Sequential(features[-1])
            self.feature_dim = 1024
        elif model_name in ['efficientnet_b0', 'efficientnet_b1']:
            self.base_model = models.efficientnet_v2_s(pretrained=True) if model_name == 'efficientnet_b0' else models.efficientnet_b1(pretrained=True)
            features = list(self.base_model.features)
            self.backbone = nn.Sequential(*features[:-1])
            self.final_block = nn.Sequential(features[-1])
            self.feature_dim = self.base_model.features[-1][0].out_channels
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.base_model.fc = nn.Identity() if hasattr(self.base_model, 'fc') else None
        self.base_model.classifier = nn.Identity() if hasattr(self.base_model, 'classifier') else None

        # Momentum Encoder
        self.momentum_final_block = MomentumFinalBlock(self.final_block, momentum=self.config.MOMENTUM)

        # Spatial Attention
        self.spatial_attention = SpatialAttention(self.feature_dim, reduction=self.config.ATTENTION_REDUCTION)

        # Memory Bank
        self.memory_bank = MemoryBank(
            self.feature_dim, 
            bank_size=self.config.BANK_SIZE, 
            rarity_threshold=self.config.RARITY_THRESHOLD
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.HIDDEN_DIM, num_classes)
        )
        self.model_name = model_name

    def forward(self, x):
        # Extract features
        backbone_features = self.backbone(x)
        main_features = self.final_block(backbone_features)
        with torch.no_grad():
            momentum_features = self.momentum_final_block(backbone_features)

        # Spatial attention and ROI extraction
        attention_map = self.spatial_attention(main_features)
        roi_features = main_features * attention_map
        roi_pooled = F.adaptive_avg_pool2d(roi_features, (1, 1)).flatten(1)

        # Momentum features
        momentum_pooled = F.adaptive_avg_pool2d(momentum_features, (1, 1)).flatten(1)

        # Combine ROI and momentum features (simple addition)
        fused_features = roi_pooled + momentum_pooled

        # Update and retrieve from memory bank
        if self.training:
            mean_norm = torch.mean(torch.norm(fused_features, dim=1))
            rarity_scores = torch.abs(torch.norm(fused_features, dim=1) - mean_norm) / mean_norm
            self.memory_bank.update(fused_features.detach(), rarity_scores)
        
        memory_features = self.memory_bank.retrieve(fused_features, k=self.config.RETRIEVAL_K)
        enhanced_features = fused_features + memory_features

        # Classification
        out = self.classifier(enhanced_features)

        # Update momentum encoder during training
        if self.training:
             self.momentum_final_block.update(self.final_block)

        return out 