import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 85
IMAGE_SIZE = [224, 224]
BATCH_SIZE = {
    'phase1': 64,  # Phase 1 batch size
    'phase2': 128, # Phase 2 batch size
    'test': 32     # Test phase batch size
}
EPOCHS = 10

# Model Configuration
class ModelConfig:
    # Momentum Encoder parameters
    MOMENTUM = 0.99
    
    # Spatial Attention parameters
    ATTENTION_REDUCTION = 8
    
    # Memory Bank parameters
    BANK_SIZE = 512
    RARITY_THRESHOLD = 0.2
    RETRIEVAL_K = 3
    
    # Model architecture parameters
    DROPOUT_RATE = 0.3
    HIDDEN_DIM = 512
    NUM_CLASSES = 14
    
    @staticmethod
    def get_feature_dim(model_name):
        if model_name == 'resnet50':
            return 2048
        elif model_name == 'densenet121':
            return 1024
        elif model_name in ['efficientnet_b0', 'efficientnet_b1']:
            return None
        else:
            raise ValueError(f"Model {model_name} not supported")

# Disease Labels
DISEASE_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# Training Configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

# Data Paths
DATA_PATHS = {
    'train_val_list': '/kaggle/input/data/train_val_list.txt',
    'test_list': '/kaggle/input/data/test_list.txt',
    'data_entry': '/kaggle/input/data/Data_Entry_2017.csv',
    'image_dir': '/kaggle/input/data/*/images/*.png'
}

# Model Save Paths
MODEL_PATHS = {
    'model_save_dir': '/kaggle/working/models/',
    'results_dir': '/kaggle/working/parameter_search_results'
}

# FastAI specific configurations
FASTAI_CONFIG = {
    'mixed_precision': True,
    'early_stopping_patience': 5,
    'min_delta': 0.001,
    'phase1_epochs': 3,
    'phase2_epochs': 20,
    'phase2_lr_range': (2e-5, 8e-5)
} 