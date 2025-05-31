# CHARM-Xray: A Consistent Hybrid Attention Network with Rare Memory Bank for Multi-label Chest X-ray Classification

This repository contains the official implementation of the paper:
> **CHARM-Xray: A Consistent Hybrid Attention Network with Rare Memory Bank for Multi-label Chest X-ray Classification**

We propose CHARM-Xray, a novel architecture for multi-label CXR classification that addresses this imbalance through three key components: 
(1) a momentum-stabilized pathway that promotes feature consistency 
(2) a multi-scale feature fusion module that captures diverse spatial patterns
(3) an adaptive memory bank that retains discriminative features from rare pathologies.

---

## 🖼️ Model Overview
![CHARM-Xray Overview](images/Overview_CHARMXray.png)

---

## 📊 Hyperparameter Comparison Results

### Comparison of hyperparameter settings with EfficientNetV2-S backbone

| **Momentum** | ***k*** | **Threshold** | **AUC** |
|--------------|---------|---------------|---------|
| 0.99         | 2       | 0.05          | 0.8610  |
|              | 2       | 0.1           | 0.8677  |
|              | 2       | 0.2           | 0.8386  |
|              | 3       | 0.05          | 0.8384  |
|              | 3       | 0.1           | 0.8627  |
|              | 3       | 0.2           | 0.8495  |
|              | 4       | 0.05          | 0.8632  |
|              | 4       | 0.1           | 0.8617  |
|              | 4       | 0.2           | 0.8723  |
|              | 5       | 0.2           | **0.8752** |
| 0.999        | 2       | 0.05          | 0.8497  |
|              | 2       | 0.1           | 0.8471  |
|              | 2       | 0.2           | 0.8665  |
|              | 3       | 0.05          | 0.8445  |
|              | 3       | 0.1           | 0.8564  |
|              | 3       | 0.2           | 0.8678  |
|              | 4       | 0.05          | 0.8570  |
|              | 4       | 0.1           | 0.8506  |
|              | 4       | 0.2           | 0.8445  |
|              | 5       | 0.1           | 0.8555  |
|              | 5       | 0.2           | 0.8568  |
| 0.9999       | 2       | 0.1           | 0.8509  |
|              | 3       | 0.1           | 0.8568  |
|              | 4       | 0.1           | 0.8611  |
|              | 4       | 0.2           | 0.8708  |
|              | 5       | 0.1           | 0.8601  |

**Best performance:** Momentum = 0.99, k = 5, Threshold = 0.2, AUC = **0.8752**

---

## 📁 Project Structure
```
├── train.ipynb     # Training notebook with pre-training and fine-tuning
├── test.ipynb      # Evaluation and testing notebook
├── image.png       # Model architecture or training pipeline illustration
└── README.md       # This file
```

---

## ⚙️ Setup

- **Platform:** Kaggle Notebooks
- **GPU:** P100 (enable in Notebook settings)
- **Dependencies:** Available in Kaggle by default (fastai, PyTorch, scikit-learn)

---

## 🚀 Training and Evaluation Instructions

### 🎯 Training Configuration
The training process in `train.ipynb` includes both phases (pre-training and fine-tuning) in a single notebook. You can adjust key hyperparameters as follows:

```python
# Define the parameter grid for training configuration
momentum_values = [0.99]
k_values = [2, 3, 4, 5]
threshold_values = [0.2]

# Example: Set config for optimal training
momentum_values = [0.99]
k_values = [5]
threshold_values = [0.2]

# Pass config to your training function as needed
```

---

## ➕ Adding Custom Data to Kaggle

1. **Go to your Kaggle Notebook**
2. On the right sidebar, click on **"Add data"**
3. Search for a dataset or click **"Upload"** to add your own files
4. After adding, access your data in the notebook under `/kaggle/input/data/`

---

## 🔧 Quick Start Guide

### Training
- Simply open `train.ipynb` and **Run All** cells
- The notebook handles both pre-training and fine-tuning phases automatically

### Testing
1. Update the configuration parameters in `test.ipynb` as needed
2. Add your trained model to `/kaggle/input/best-model/` directory
3. Run the evaluation cells

### Pre-trained Models
Pre-trained models will be available on Google Drive: https://drive.google.com/drive/folders/1QXrxM_Ip0jchvykSWNiV5HSdVl09UUpa?usp=drive_link.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{charm-xray2024,
  title={CHARM-Xray: A Consistent Hybrid Attention Network with Rare Memory Bank for Multi-label Chest X-ray Classification},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue in this repository.
