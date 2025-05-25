import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config import DATA_CONFIG, TRAINING_CONFIG

class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else self._get_default_transforms()
        self.samples = self._load_samples()
        
    def _get_default_transforms(self):
        return transforms.Compose([
            transforms.Resize(DATA_CONFIG['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def _load_samples(self):
        samples = []
        classes = sorted(os.listdir(self.data_dir))
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, class_idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir):
    """Create train, validation, and test data loaders."""
    # Create dataset
    dataset = XRayDataset(data_dir)
    
    # Calculate lengths for splits
    total_size = len(dataset)
    train_size = int(DATA_CONFIG['train_split'] * total_size)
    val_size = int(DATA_CONFIG['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader 