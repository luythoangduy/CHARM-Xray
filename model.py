import torch
import torch.nn as nn
from config import MODEL_CONFIG

class XRayClassifier(nn.Module):
    def __init__(self):
        super(XRayClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(MODEL_CONFIG['input_channels'], MODEL_CONFIG['hidden_channels'][0], kernel_size=3, padding=1),
            nn.BatchNorm2d(MODEL_CONFIG['hidden_channels'][0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(MODEL_CONFIG['hidden_channels'][0], MODEL_CONFIG['hidden_channels'][1], kernel_size=3, padding=1),
            nn.BatchNorm2d(MODEL_CONFIG['hidden_channels'][1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(MODEL_CONFIG['hidden_channels'][1], MODEL_CONFIG['hidden_channels'][2], kernel_size=3, padding=1),
            nn.BatchNorm2d(MODEL_CONFIG['hidden_channels'][2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of flattened features
        self.flatten_size = MODEL_CONFIG['hidden_channels'][2] * (224 // 8) * (224 // 8)
        
        self.classifier = nn.Sequential(
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(512, MODEL_CONFIG['num_classes'])
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model():
    """Factory function to create a new model instance."""
    model = XRayClassifier()
    return model 