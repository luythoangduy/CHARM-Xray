import os
import time
import torch
import pandas as pd
import numpy as np
from fastai.vision.all import *
from sklearn.metrics import roc_auc_score
from config import (
    DEVICE, SEED, MODEL_PATHS, DISEASE_LABELS, 
    FASTAI_CONFIG, ModelConfig, BATCH_SIZE
)
from models.chest_xray_model import ChestXrayModel
from losses import ChestXrayLoss
from data import seed_everything, prepare_data, get_dls

def create_learner(
    dls,
    loss_type='focal',
    cbs=None,
    **loss_kwargs
):
    """Create a FastAI learner with the chest x-ray model."""
    # Create model
    model = ChestXrayModel(
        num_classes=ModelConfig.NUM_CLASSES,
        model_name='efficientnet_b0'
    )
    
    # Default callbacks
    default_cbs = [
        SaveModelCallback(
            monitor='valid_loss',
            min_delta=FASTAI_CONFIG['min_delta'],
            with_opt=True
        ),
        EarlyStoppingCallback(
            monitor='valid_loss',
            min_delta=FASTAI_CONFIG['min_delta'],
            patience=FASTAI_CONFIG['early_stopping_patience']
        ),
        ShowGraphCallback()
    ]
    
    # Add custom callbacks
    if cbs:
        if isinstance(cbs, list):
            default_cbs.extend(cbs)
        else:
            default_cbs.append(cbs)
    
    # Create learner
    learn = Learner(
        dls,
        model,
        loss_func=ChestXrayLoss(loss_type=loss_type, **loss_kwargs),
        metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()],
        cbs=default_cbs
    )
    
    # Enable mixed precision if configured
    if FASTAI_CONFIG['mixed_precision']:
        learn.to_fp16()
    
    return learn

def get_roc_auc(learner):
    """Calculate ROC AUC scores for the model."""
    learner.freeze()
    preds, y_test = learner.get_preds(ds_idx=1)
    roc_auc = roc_auc_score(y_test, preds)
    
    scores = []
    for i in range(len(DISEASE_LABELS)):
        label_roc_auc_score = roc_auc_score(y_test[:,i], preds[:,i])
        scores.append(label_roc_auc_score)
    
    print('ROC_AUC_Labels:', list(zip(DISEASE_LABELS, scores)))   
    print(f'Overall ROC AUC: {roc_auc:.4f}')
    
    return {
        'roc_auc': roc_auc,
        'class_auc': scores,
        'preds': preds,
        'y_test': y_test
    }

def run_training(
    train_df,
    test_df,
    exp_name='experiment'
):
    """Run the complete training process including both phases."""
    print(f"\n=== Starting training experiment: {exp_name} ===")
    
    # Create save directories
    os.makedirs(MODEL_PATHS['model_save_dir'], exist_ok=True)
    os.makedirs(MODEL_PATHS['results_dir'], exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Phase 1: Initial training with BCE loss
        print('--------------Begin Phase 1---------------')
        # Create dataloaders with phase 1 batch size
        train_dls_phase1 = get_dls(train_df, phase='train', bs=BATCH_SIZE['phase1'])
        learn = create_learner(train_dls_phase1, loss_type='bce')
        
        # Find learning rate
        lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        lr = lrs.valley if lrs.valley is not None else 1e-4
        print('Initial learning rate:', lr)
        
        # Train with frozen layers
        learn.fine_tune(
            freeze_epochs=FASTAI_CONFIG['phase1_epochs'],
            epochs=FASTAI_CONFIG['phase2_epochs'],
            base_lr=lr
        )
        
        # Save phase 1 model
        learn.save(f"{MODEL_PATHS['model_save_dir']}/phase1_{exp_name}")
        
        # Phase 2: Fine-tuning with asymmetric loss
        print('--------------Begin Phase 2---------------')
        # Create new dataloaders with phase 2 batch size
        train_dls_phase2 = get_dls(train_df, phase='train', bs=BATCH_SIZE['phase2'])
        learn = create_learner(train_dls_phase2, loss_type='asymmetric')
        learn = learn.load(f"{MODEL_PATHS['model_save_dir']}/phase1_{exp_name}")
        
        # Unfreeze and train
        learn.unfreeze()
        learn.fit_one_cycle(
            5,
            slice(*FASTAI_CONFIG['phase2_lr_range'])
        )
        
        # Save final model
        learn.save(f"{MODEL_PATHS['model_save_dir']}/final_{exp_name}")
        
        # Testing phase
        print('--------------Begin Testing---------------')
        # Create test dataloaders with test batch size
        test_dls = get_dls(test_df, phase='test', bs=BATCH_SIZE['test'])
        test_learn = create_learner(test_dls, loss_type='asymmetric')
        test_learn = test_learn.load(f"{MODEL_PATHS['model_save_dir']}/final_{exp_name}")
        
        # Get results
        results = get_roc_auc(test_learn)
        torch.save(
            results['preds'],
            f"{MODEL_PATHS['results_dir']}/preds_{exp_name}.pt"
        )
        
        training_time = time.time() - start_time
        
        return {
            'mean_auc': results['roc_auc'],
            'class_auc': results['class_auc'],
            'training_time': training_time,
            'exp_name': exp_name
        }
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return {
            'mean_auc': float('nan'),
            'class_auc': [],
            'training_time': float('nan'),
            'exp_name': exp_name,
            'error': str(e)
        }

if __name__ == "__main__":
    # Set random seed
    seed_everything(SEED)
    
    # Prepare data
    print("Preparing data...")
    _, _, train_df, test_df = prepare_data()
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Run training
    result = run_training(train_df, test_df)
    
    # Save results
    pd.DataFrame([result]).to_csv(
        f"{MODEL_PATHS['results_dir']}/training_results.csv",
        index=False
    ) 