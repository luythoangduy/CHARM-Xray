import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss to address class imbalance and hard sample mining
        
        Args:
            alpha (float): Weighting factor for positive samples
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
        
        Returns:
            torch.Tensor: Computed loss
        """
        # Apply sigmoid to convert logits to probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal Loss modification
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Asymmetric Loss to handle class imbalance and hard negative mining
        
        Args:
            gamma_neg (float): Focusing parameter for negative samples
            gamma_pos (float): Focusing parameter for positive samples
            clip (float): Clip the predictions to prevent extreme values
            eps (float): Small epsilon to prevent log(0)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        Compute asymmetric loss
        
        Args:
            x (torch.Tensor): Model predictions (logits)
            y (torch.Tensor): Ground truth labels
        
        Returns:
            torch.Tensor: Computed loss
        """
        # Convert to probabilities
        x_sigmoid = torch.sigmoid(x)
        
        # Clip predictions to prevent extreme values
        xs_min = x_sigmoid.clamp(min=self.eps)
        xs_max = x_sigmoid.clamp(max=1-self.eps)
        
        # Asymmetric term for positive and negative samples
        loss_pos = -y * torch.log(xs_min) * torch.pow(1 - xs_min, self.gamma_pos)
        loss_neg = -(1 - y) * torch.log(1 - xs_max) * torch.pow(xs_max, self.gamma_neg)
        
        loss = loss_pos + loss_neg
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class ChestXrayLoss(nn.Module):
    def __init__(self, loss_type='focal', **kwargs):
        super().__init__()
        if loss_type == 'focal':
            self.loss = FocalLoss(
                alpha=kwargs.get('focal_alpha', 1),
                gamma=kwargs.get('focal_gamma', 2)
            )
        elif loss_type == 'asymmetric':
            self.loss = AsymmetricLoss(
                gamma_neg=kwargs.get('asymmetric_gamma_neg', 4),
                gamma_pos=kwargs.get('asymmetric_gamma_pos', 1)
            )
        elif loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, preds, targets):
        return self.loss(preds, targets) 