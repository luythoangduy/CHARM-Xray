import torch
import pandas as pd
import numpy as np
from fastai.vision.all import *
from sklearn.metrics import roc_auc_score
from models.chest_xray_model import ChestXrayModel, ModelConfig
from data import get_test_dls
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test a checkpoint of the chest X-ray model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing (default: 32)')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples to test (default: 100, use -1 for all)')
    return parser.parse_args()

def get_roc_auc(learner, disease_labels):
    """Calculate ROC-AUC scores for the model"""
    learner.freeze()
    preds, y_test = learner.get_preds(ds_idx=1)
    
    # Overall ROC-AUC
    roc_auc = roc_auc_score(y_test, preds)
    
    # Per-class ROC-AUC
    scores = []
    for i in range(len(disease_labels)):
        label_roc_auc = roc_auc_score(y_test[:,i], preds[:,i])
        scores.append(label_roc_auc)
    
    return {
        'overall_roc_auc': roc_auc,
        'class_roc_auc': dict(zip(disease_labels, scores)),
        'mean_class_roc_auc': np.mean(scores)
    }

def main():
    args = parse_args()
    
    # Disease labels
    disease_labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    try:
        # Get test dataloaders
        dls = get_test_dls(batch_size=args.batch_size, num_samples=args.num_samples)
        
        # Create model and learner
        model = ChestXrayModel(num_classes=len(disease_labels))
        learn = Learner(
            dls,
            model,
            loss_func=nn.BCEWithLogitsLoss(),
            metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()]
        )
        
        # Load checkpoint
        learn.load(args.checkpoint)
        
        # Run evaluation
        print("\nRunning evaluation...")
        results = get_roc_auc(learn, disease_labels)
        
        # Print results
        print("\nResults:")
        print(f"Overall ROC-AUC: {results['overall_roc_auc']:.4f}")
        print(f"Mean Class ROC-AUC: {results['mean_class_roc_auc']:.4f}")
        print("\nPer-class ROC-AUC scores:")
        for disease, score in results['class_roc_auc'].items():
            print(f"{disease}: {score:.4f}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == '__main__':
    main() 