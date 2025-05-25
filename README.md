# Chest X-Ray Classification with Advanced Deep Learning

This project implements a state-of-the-art deep learning model for multi-label chest X-ray classification using PyTorch and FastAI. The model incorporates several advanced techniques for improved performance:

## Key Features

- **Advanced Model Architecture**:
  - Momentum Encoder for robust feature learning (coefficient: 0.99)
  - Spatial Attention mechanism for ROI detection
  - Memory Bank for rare feature storage and retrieval
  - Support for multiple backbone networks (EfficientNet, ResNet50, DenseNet121)

- **Training Optimizations**:
  - Two-phase training strategy with different batch sizes
  - Mixed precision training for faster computation
  - Multiple loss functions (BCE, Focal Loss, Asymmetric Loss)
  - Early stopping and model checkpointing

## Project Structure

```
.
├── config.py                # Configuration parameters and constants
├── data.py                 # Data loading and preprocessing
├── losses.py              # Custom loss function implementations
├── models/
│   ├── __init__.py
│   ├── components.py      # Model components (Momentum, Attention, Memory Bank)
│   └── chest_xray_model.py # Main model architecture
├── train.py               # Training script
├── test_checkpoint.py     # Quick checkpoint testing utility
└── README.md
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
fastai>=2.7.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
pillow>=8.3.1
```

## Dataset Structure

The code expects the following data structure:
```
data/
├── train_val_list.txt
├── test_list.txt
├── Data_Entry_2017.csv
└── images/
    └── *.png
```

## Training Process

The model training consists of two phases:

### Phase 1: Initial Training
- Batch size: 64
- Loss function: BCE Loss
- Frozen backbone layers
- Learning rate finder for optimal LR

### Phase 2: Fine-tuning
- Batch size: 128
- Loss function: Asymmetric Loss
- Unfrozen layers
- One-cycle policy with LR range (2e-5, 8e-5)

### Testing
- Batch size: 32
- ROC-AUC evaluation for each disease class

## Model Components

1. **Momentum Encoder**
   - Updates target network with EMA of online network
   - Momentum coefficient: 0.99
   - Warm-up period during initial training

2. **Spatial Attention**
   - Multi-scale convolution (1x1, 3x3, 5x5)
   - Channel reduction ratio: 8
   - Adaptive ROI detection

3. **Memory Bank**
   - Bank size: 512 features
   - Rarity threshold: 0.2
   - Top-k retrieval: k=3
   - Dynamic feature storage and retrieval

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure parameters in `config.py`:
```python
# Example configuration
BATCH_SIZE = {
    'train': 64,
    'phase2': 128,
    'test': 32
}
```

3. Run training:
```bash
python train.py
```

4. Test checkpoints:
```bash
# Quick test on 100 samples
python test_checkpoint.py --checkpoint path/to/checkpoint

# Full test set evaluation
python test_checkpoint.py --checkpoint path/to/checkpoint --num_samples -1

# Custom batch size
python test_checkpoint.py --checkpoint path/to/checkpoint --batch_size 64
```

## Testing Utility

The `test_checkpoint.py` script provides a quick way to evaluate model checkpoints:

- **Features**:
  - Configurable sample size for quick testing
  - Per-class ROC-AUC evaluation
  - Overall model performance metrics
  - Detailed error reporting

- **Output Metrics**:
  - Overall ROC-AUC score
  - Mean class ROC-AUC
  - Individual disease class scores
  - Training time metrics

## Disease Classes

The model classifies 14 different conditions:
- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural Thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia

## Results

Model evaluation results are saved in:
- Model checkpoints: `/kaggle/working/models/`
- Predictions: `/kaggle/working/parameter_search_results/`
- Training metrics: `training_results.csv`

## Performance Monitoring

The training process includes:
- Learning rate scheduling
- Model checkpointing
- Early stopping
- Performance visualization
- ROC-AUC tracking

## Citation

If you use this code in your research, please cite:
```
@misc{charm_xray_2024,
  title={Advanced Deep Learning for Chest X-Ray Classification},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/charm_xray}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 