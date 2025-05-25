import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from config import ModelConfig

class MomentumFinalBlock(nn.Module):
    def __init__(self, final_block, momentum=None):
        super(MomentumFinalBlock, self).__init__()
        self.momentum = momentum if momentum is not None else ModelConfig.MOMENTUM
        self.final_block = deepcopy(final_block)
        for param in self.final_block.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.final_block(x)

    def update(self, main_final_block):
        for param_q, param_k in zip(main_final_block.parameters(), self.final_block.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=None):
        super(SpatialAttention, self).__init__()
        reduction = reduction if reduction is not None else ModelConfig.ATTENTION_REDUCTION
        reduced_channels = max(in_channels // reduction, 8)
        
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, reduced_channels, kernel_size=5, padding=2)
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(reduced_channels * 3, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        
        features = torch.cat([f1, f3, f5], dim=1)
        attention = self.spatial_att(features)  # [batch_size, 1, H, W]
        return attention

class MemoryBank(nn.Module):
    def __init__(self, feature_dim, bank_size=None, rarity_threshold=None):
        super(MemoryBank, self).__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size if bank_size is not None else ModelConfig.BANK_SIZE
        self.rarity_threshold = rarity_threshold if rarity_threshold is not None else ModelConfig.RARITY_THRESHOLD
        
        self.register_buffer('memory', torch.zeros(self.bank_size, feature_dim))
        self.register_buffer('index', torch.tensor(0))

    def update(self, features, rarity_scores):
        batch_size = features.size(0)
        mask = rarity_scores < self.rarity_threshold
        rare_features = features[mask]
        
        if rare_features.size(0) > 0:
            num_to_add = min(rare_features.size(0), self.bank_size - self.index.item())
            if num_to_add > 0:
                self.memory[self.index:self.index + num_to_add] = rare_features[:num_to_add]
                self.index = (self.index + num_to_add) % self.bank_size

    def retrieve(self, query, k=None):
        k = k if k is not None else ModelConfig.RETRIEVAL_K
        valid_memory = self.memory
        if valid_memory.size(0) == 0:
            return torch.zeros_like(query)
        
        norm_query = F.normalize(query, dim=1)
        norm_memory = F.normalize(valid_memory, dim=1)
        similarity = torch.matmul(norm_query, norm_memory.T)
        
        # Create a mask for entries where similarity != 1
        mask = similarity != 1.0
        
        k = min(k, valid_memory.size(0))
        
        # Initialize containers for results
        batch_size = query.size(0)
        result = torch.zeros_like(query)
        
        for i in range(batch_size):
            # Get indices where similarity is not 1 for this query
            valid_indices = torch.where(mask[i])[0]
            
            if len(valid_indices) == 0:
                # If all memories have similarity=1, just return zeros
                continue
            
            # Get similarities only for valid indices
            valid_similarities = similarity[i, valid_indices]
            
            # Get top-k among valid similarities
            k_valid = min(k, valid_similarities.size(0))
            weights, rel_indices = valid_similarities.topk(k_valid)
            
            # Convert relative indices to absolute indices
            abs_indices = valid_indices[rel_indices]
            
            # Get features for these indices
            retrieved = valid_memory[abs_indices]
            
            # Apply weights
            weights = weights.unsqueeze(1).expand_as(retrieved)
            weighted_features = (retrieved * weights).sum(dim=0)
            
            result[i] = weighted_features
            
        return result 