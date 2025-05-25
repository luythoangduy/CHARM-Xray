import os
import pandas as pd
import numpy as np
from glob import glob
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
from config import DEVICE, SEED, IMAGE_SIZE, BATCH_SIZE, DISEASE_LABELS, DATA_PATHS

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_disease_labels():
    return [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]

def load_data():
    """Load and prepare the dataset"""
    # Load labels
    labels_df = pd.read_csv('data/Data_Entry_2017.csv')
    labels_df.columns = [
        'Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
        'Patient_Age', 'Patient_Gender', 'View_Position',
        'Original_Image_Width', 'Original_Image_Height',
        'Original_Image_Pixel_Spacing_X',
        'Original_Image_Pixel_Spacing_Y', 'dfd'
    ]
    
    # One-hot encoding for disease labels
    disease_labels = get_disease_labels()
    for disease in disease_labels:
        labels_df[disease] = labels_df['Finding_Labels'].map(
            lambda x: 1 if disease in x else 0
        )
    
    # Convert Finding_Labels to list format
    labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(
        lambda s: [l for l in str(s).split('|')]
    )
    
    # Map image paths
    num_glob = glob('data/*/images/*.png')
    img_path = {os.path.basename(x): x for x in num_glob}
    labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)
    
    return labels_df

def get_transforms():
    """Get data transforms for testing"""
    item_tfms = [
        Resize((224, 224)),
    ]
    
    batch_tfms = [
        Normalize.from_stats(*imagenet_stats),
    ]
    
    return item_tfms, batch_tfms

def get_test_dls(batch_size=32, num_samples=-1):
    """Create test DataLoaders"""
    # Load and prepare data
    df = load_data()
    
    # Limit samples if specified
    if num_samples > 0:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    # Get transforms
    item_tfms, batch_tfms = get_transforms()
    
    # Create DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=get_disease_labels())),
        get_x=lambda row: row['Paths'],
        get_y=lambda row: row[get_disease_labels()].tolist(),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    
    # Create DataLoaders
    return dblock.dataloaders(df, bs=batch_size)

def get_dls(labels_df, phase='train', bs=None):
    """Create FastAI DataLoaders with proper transforms and splits."""
    # Use appropriate batch size based on phase if not specified
    if bs is None:
        bs = BATCH_SIZE[phase]

    # Define transforms
    item_transforms = [
        Resize(IMAGE_SIZE),
    ]

    batch_transforms = [
        Flip(),
        Rotate(),
        Normalize.from_stats(*imagenet_stats),
    ]

    def get_x(row):
        """Get image path."""
        return row['Paths']

    def get_y(row):
        """Get multi-label targets."""
        return row[DISEASE_LABELS].tolist()

    # Create DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=DISEASE_LABELS)),
        splitter=RandomSplitter(valid_pct=0.125, seed=SEED),
        get_x=get_x,
        get_y=get_y,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms
    )

    # Create dataloaders based on phase
    if phase == 'train':
        dls = dblock.dataloaders(labels_df, bs=bs)
    elif phase == 'test':
        dls = dblock.dataloaders(labels_df, bs=bs, shuffle=False)
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    return dls

def prepare_data():
    """Main function to prepare all data splits."""
    # Set random seed
    seed_everything(SEED)
    
    # Load data
    labels_df = load_data()
    
    # Split data by patient ID
    unique_patients = np.unique(labels_df['Patient_ID'])
    train_val_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )
    
    # Create dataframes for each split
    train_val_df = labels_df[labels_df['Patient_ID'].isin(train_val_patients)]
    test_df = labels_df[labels_df['Patient_ID'].isin(test_patients)]
    
    return None, None, train_val_df, test_df 