"""
Multi-Task Cross-Validation (WEIGHTED)
This version is modified to INCLUDE the 'unknown' activity in training.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import librosa
from tqdm import tqdm
import json
import warnings
import time
warnings.filterwarnings('ignore')

# cahange paths accordingly
class Config:
    # Paths - Correctly placed inside the class
    VOCAL_ANNOTATION_CSV = 'vocal_annotation_all.csv'
    DATA_BASE_FOLDER = 'Data'
    OUTPUT_DIR = '/root/weighted_cross_outputs_3'
    
    # Model
    MODEL_NAME = 'facebook/hubert-base-ls960'
    NUM_FOLDS = 5           
    BATCH_SIZE = 8          
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 40        
    
    # Audio settings
    SEGMENT_LENGTH = 10.0
    SAMPLE_RATE = 16000
    PADDING_BEFORE = 2.0
    PADDING_AFTER = 2.0
    
    # Multi-task settings
    ACTIVITY_LABEL_COLUMN = 'label_activity'
    VOCALIZATION_LABEL_COLUMN = 'vocalization_type'
    EXCLUDE_UNKNOWN = False 
    # Task weights (for combining loss)
    ACTIVITY_WEIGHT = 0.5
    VOCALIZATION_WEIGHT = 0.5
    
    # Device - Tell PyTorch to use the GPU
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42


# MULTI-TASK MODEL

class MultiTaskHuBERT(nn.Module):
    """HuBERT with two classification heads"""
    def __init__(self, model_name, num_activity_labels, num_vocalization_labels):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        hidden_size = self.hubert.config.hidden_size
        self.activity_classifier = nn.Linear(hidden_size, num_activity_labels)
        self.vocalization_classifier = nn.Linear(hidden_size, num_vocalization_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        pooled = self.dropout(pooled)
        activity_logits = self.activity_classifier(pooled)
        vocalization_logits = self.vocalization_classifier(pooled)
        return activity_logits, vocalization_logits



# DATASET CLASS
class MultiTaskAudioDataset(Dataset):
    """Dataset for multi-task learning"""
    def __init__(self, dataframe, data_base_folder, feature_extractor,
                 segment_length=10.0, sample_rate=16000,
                 padding_before=2.0, padding_after=2.0):
        self.df = dataframe.reset_index(drop=True)
        self.data_base_folder = data_base_folder
        self.feature_extractor = feature_extractor
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.padding_before = padding_before
        self.padding_after = padding_after
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_base_folder, row['folder'], row['original_audio'])
        
        try:
            # Calculate segment boundaries with padding
            start_time = max(0, row['start_time'] - self.padding_before)
            end_time = row['end_time'] + self.padding_after
            duration = min(end_time - start_time, self.segment_length)
            
            # Load audio segment
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, duration=duration)
            
            # Pad or truncate to fixed length
            target_length = int(self.segment_length * self.sample_rate)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Extract features
            inputs = self.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'activity_label': torch.tensor(row['activity_label_id'], dtype=torch.long),
                'vocalization_label': torch.tensor(row['vocalization_label_id'], dtype=torch.long),
                'filename': row['original_audio']
            }
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            target_length = int(self.segment_length * self.sample_rate)
            dummy_audio = np.zeros(target_length)
            inputs = self.feature_extractor(dummy_audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            return {
                'input_values': inputs.input_values.squeeze(0),
                'activity_label': torch.tensor(0, dtype=torch.long), # Will use a default label
                'vocalization_label': torch.tensor(0, dtype=torch.long),
                'filename': 'error'
            }


# TRAINING FUNCTIONS
def train_epoch(model, dataloader, optimizer, activity_criterion, vocalization_criterion, 
                device, activity_weight, vocalization_weight):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    activity_preds_all = []
    activity_labels_all = []
    vocalization_preds_all = []
    vocalization_labels_all = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Filter out error batches
        if 'error' in batch['filename']:
            continue
            
        input_values = batch['input_values'].to(device)
        activity_labels = batch['activity_label'].to(device)
        vocalization_labels = batch['vocalization_label'].to(device)
        
        optimizer.zero_grad()
        activity_logits, vocalization_logits = model(input_values)
        
        activity_loss = activity_criterion(activity_logits, activity_labels)
        vocalization_loss = vocalization_criterion(vocalization_logits, vocalization_labels)
        loss = activity_weight * activity_loss + vocalization_weight * vocalization_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        activity_preds = torch.argmax(activity_logits, dim=1)
        activity_preds_all.extend(activity_preds.cpu().numpy())
        activity_labels_all.extend(activity_labels.cpu().numpy())
        
        vocalization_preds = torch.argmax(vocalization_logits, dim=1)
        vocalization_preds_all.extend(vocalization_preds.cpu().numpy())
        vocalization_labels_all.extend(vocalization_labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'act_acc': f'{accuracy_score(activity_labels_all, activity_preds_all):.3f}',
            'voc_acc': f'{accuracy_score(vocalization_labels_all, vocalization_preds_all):.3f}'
        })
    
    # Handle cases where all batches were errors
    if len(dataloader) == 0 or len(activity_labels_all) == 0:
        return 0, 0, 0
        
    avg_loss = total_loss / len(dataloader)
    activity_acc = accuracy_score(activity_labels_all, activity_preds_all)
    vocalization_acc = accuracy_score(vocalization_labels_all, vocalization_preds_all)
    
    return avg_loss, activity_acc, vocalization_acc


def evaluate(model, dataloader, activity_criterion, vocalization_criterion, 
             device, activity_weight, vocalization_weight):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    activity_preds_all = []
    activity_labels_all = []
    vocalization_preds_all = []
    vocalization_labels_all = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Filter out error batches
            if 'error' in batch['filename']:
                continue
                
            input_values = batch['input_values'].to(device)
            activity_labels = batch['activity_label'].to(device)
            vocalization_labels = batch['vocalization_label'].to(device)
            
            activity_logits, vocalization_logits = model(input_values)
            
            activity_loss = activity_criterion(activity_logits, activity_labels)
            vocalization_loss = vocalization_criterion(vocalization_logits, vocalization_labels)
            loss = activity_weight * activity_loss + vocalization_weight * vocalization_loss
            
            total_loss += loss.item()
            
            activity_preds = torch.argmax(activity_logits, dim=1)
            activity_preds_all.extend(activity_preds.cpu().numpy())
            activity_labels_all.extend(activity_labels.cpu().numpy())
            
            vocalization_preds = torch.argmax(vocalization_logits, dim=1)
            vocalization_preds_all.extend(vocalization_preds.cpu().numpy())
            vocalization_labels_all.extend(vocalization_labels.cpu().numpy())
    
    # Handle cases where all batches were errors
    if len(dataloader) == 0 or len(activity_labels_all) == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [], [])
        
    avg_loss = total_loss / len(dataloader)
    
    activity_acc = accuracy_score(activity_labels_all, activity_preds_all)
    activity_precision, activity_recall, activity_f1, _ = precision_recall_fscore_support(
        activity_labels_all, activity_preds_all, average='weighted', zero_division=0
    )
    
    vocalization_acc = accuracy_score(vocalization_labels_all, vocalization_preds_all)
    vocalization_precision, vocalization_recall, vocalization_f1, _ = precision_recall_fscore_support(
        vocalization_labels_all, vocalization_preds_all, average='weighted', zero_division=0
    )
    
    return (avg_loss, 
            activity_acc, activity_precision, activity_recall, activity_f1,
            vocalization_acc, vocalization_precision, vocalization_recall, vocalization_f1,
            activity_preds_all, activity_labels_all,
            vocalization_preds_all, vocalization_labels_all)

def cross_validate(config):
    """Perform K-Fold Cross Validation"""
    print(f"Device: {config.DEVICE}")
    print(f"Folds: {config.NUM_FOLDS} ")
    print(f"Epochs: {config.NUM_EPOCHS} ")
    print(f"Batch Size: {config.BATCH_SIZE}")
    
    # Load data
    print(f"\n Loading: {config.VOCAL_ANNOTATION_CSV}")
    try:
        df = pd.read_csv(config.VOCAL_ANNOTATION_CSV)
    except FileNotFoundError:
        print(f"ERROR: File not found at {config.VOCAL_ANNOTATION_CSV}")
        print("Please update the 'VOCAL_ANNOTATION_CSV' in the Config class.")
        return
    print(f"   Total segments: {len(df)}")
    
    # Filter unknown
    if config.EXCLUDE_UNKNOWN:
        df = df[
            (df[config.ACTIVITY_LABEL_COLUMN] != 'unknown') & 
            (df[config.VOCALIZATION_LABEL_COLUMN] != 'unknown') &
            (df['status'] == 'matched') # Explicitly use 'matched'
        ].reset_index(drop=True)
        print(f"   After filtering for 'matched' and 'unknown': {len(df)} segments")
    else:
        print("   Training with 'unknown' data included.")
    
    if len(df) == 0:
        print("\n No valid data! Check your CSV or filters.")
        return None
    
    # Clean Vocalization Labels
    # We must do this *before* creating the label mappings
    # HANDLE 'unknown' VOCALIZATION 
    label_map = {
        'W': 'W', 'w': 'W', "W'": 'W', 'WW': 'W',
        'NOISE': 'NOISE', 'NOIS': 'NOISE', 'W+NOISE': 'NOISE', 
        'RUMORE CON FISCHIO': 'NOISE',
        # Group rare labels into 'OTHER'
        'ct': 'OTHER', 'PSB': 'OTHER', 'CR': 'OTHER', 'b': 'OTHER',
        'PbS': 'OTHER', 'M': 'OTHER', 'unknown': 'OTHER' # <-- Added unknown
    }
    
    df[config.VOCALIZATION_LABEL_COLUMN] = df[config.VOCALIZATION_LABEL_COLUMN].apply(
        lambda x: label_map.get(x, x)
    )
    # Drop 'OTHER'
    df = df[df[config.VOCALIZATION_LABEL_COLUMN] != 'OTHER'].reset_index(drop=True)
    print(f"   After cleaning vocalization labels: {len(df)} segments")

    
    # Create label mappings
    activity_labels = sorted(df[config.ACTIVITY_LABEL_COLUMN].unique())
    activity_label2id = {label: idx for idx, label in enumerate(activity_labels)}
    activity_id2label = {idx: label for label, idx in activity_label2id.items()}
    
    vocalization_labels = sorted(df[config.VOCALIZATION_LABEL_COLUMN].unique())
    vocalization_label2id = {label: idx for idx, label in enumerate(vocalization_labels)}
    vocalization_id2label = {idx: label for label, idx in vocalization_label2id.items()}
    
    print(f"\n Activity Classes ({len(activity_labels)}):")
    for label, idx in activity_label2id.items():
        count = (df[config.ACTIVITY_LABEL_COLUMN] == label).sum()
        print(f"   {idx}: {label} ({count} segments)")
    
    print(f"\n Vocalization Classes ({len(vocalization_labels)}):")
    for label, idx in vocalization_label2id.items():
        count = (df[config.VOCALIZATION_LABEL_COLUMN] == label).sum()
        print(f"   {idx}: {label} ({count} segments)")
    
    df['activity_label_id'] = df[config.ACTIVITY_LABEL_COLUMN].map(activity_label2id)
    df['vocalization_label_id'] = df[config.VOCALIZATION_LABEL_COLUMN].map(vocalization_label2id)
    
    # CALCULATE CLASS WEIGHTS
    
    activity_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df['activity_label_id']),
        y=df['activity_label_id']
    )
    
    vocalization_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df['vocalization_label_id']),
        y=df['vocalization_label_id']
    )
    
    print("\nCalculated Activity Weights:")
    for i, w in enumerate(activity_weights_array):
        print(f"  {activity_id2label[i]}: {w:.4f}")
        
    print("\nCalculated Vocalization Weights:")
    for i, w in enumerate(vocalization_weights_array):
        print(f"  {vocalization_id2label[i]}: {w:.4f}")

    # Feature extractor
    print(f"\n Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)
    
    # Cross-validation
    # We stratify on the *most* imbalanced task, which is activity
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    fold_results = []
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save label mappings
    with open(os.path.join(config.OUTPUT_DIR, 'activity_id2label.json'), 'w') as f:
        json.dump(activity_id2label, f)
    with open(os.path.join(config.OUTPUT_DIR, 'vocalization_id2label.json'), 'w') as f:
        json.dump(vocalization_id2label, f)
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['activity_label_id']), 1):
        fold_start_time = time.time()
        print(f"FOLD {fold}/{config.NUM_FOLDS}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Datasets
        train_dataset = MultiTaskAudioDataset(
            train_df, config.DATA_BASE_FOLDER, feature_extractor,
            config.SEGMENT_LENGTH, config.SAMPLE_RATE,
            config.PADDING_BEFORE, config.PADDING_AFTER
        )
        val_dataset = MultiTaskAudioDataset(
            val_df, config.DATA_BASE_FOLDER, feature_extractor,
            config.SEGMENT_LENGTH, config.SAMPLE_RATE,
            config.PADDING_BEFORE, config.PADDING_AFTER
        )
        
        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Model
        print(f"\n Loading Multi-Task HuBERT model...")
        model = MultiTaskHuBERT(config.MODEL_NAME, len(activity_labels), len(vocalization_labels)).to(config.DEVICE)
        
        
        # APPLY WEIGHTS TO LOSS
        # Convert weights to tensors and send to device
        act_weights_tensor = torch.tensor(activity_weights_array, dtype=torch.float).to(config.DEVICE)
        voc_weights_tensor = torch.tensor(vocalization_weights_array, dtype=torch.float).to(config.DEVICE)
        
        # Pass the 'weight' argument
        activity_criterion = nn.CrossEntropyLoss(weight=act_weights_tensor)
        vocalization_criterion = nn.CrossEntropyLoss(weight=voc_weights_tensor)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        
        # Training loop
        best_val_loss = float('inf') # Save based on combined loss
        best_epoch = 0
        
        for epoch in range(config.NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
            
            # Train
            train_loss, train_activity_acc, train_voc_acc = train_epoch(
                model, train_loader, optimizer,
                activity_criterion, vocalization_criterion,
                config.DEVICE, config.ACTIVITY_WEIGHT, config.VOCALIZATION_WEIGHT
            )
            
            epoch_time = time.time() - epoch_start_time
            print(f"Train - Loss: {train_loss:.4f}, Activity: {train_activity_acc:.4f}, Vocalization: {train_voc_acc:.4f}")
            print(f"Epoch time: {epoch_time/60:.1f} minutes")
            
            # Validate
            (val_loss, val_activity_acc, val_activity_prec, val_activity_rec, val_activity_f1,
             val_voc_acc, val_voc_prec, val_voc_rec, val_voc_f1,
             _, _, _, _) = evaluate(
                model, val_loader,
                activity_criterion, vocalization_criterion,
                config.DEVICE, config.ACTIVITY_WEIGHT, config.VOCALIZATION_WEIGHT
            )
            
            print(f"Val - Loss: {val_loss:.4f}")
            print(f"  Activity: Acc={val_activity_acc:.4f}, F1={val_activity_f1:.4f}")
            print(f"  Vocalization: Acc={val_voc_acc:.4f}, F1={val_voc_f1:.4f}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f'multitask_fold{fold}_best.pt'))
                print(f" Saved best model (by val loss)")
            
            # Time estimate
            if epoch == 0:
                estimated_fold_time = epoch_time * config.NUM_EPOCHS / 60
                estimated_total_time = estimated_fold_time * config.NUM_FOLDS / 60
                print(f" Estimated fold time: {estimated_fold_time:.1f} minutes")
                print(f" Estimated total time: {estimated_total_time:.1f} hours")
        
        # Final evaluation
        fold_time = (time.time() - fold_start_time) / 3600
        print(f"\n   Fold {fold} completed in {fold_time:.2f} hours")
        
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, f'multitask_fold{fold}_best.pt')))
        
        (val_loss, val_activity_acc, val_activity_prec, val_activity_rec, val_activity_f1,
         val_voc_acc, val_voc_prec, val_voc_rec, val_voc_f1,
         activity_preds, activity_labels_true,
         voc_preds, voc_labels_true) = evaluate(
            model, val_loader,
            activity_criterion, vocalization_criterion,
            config.DEVICE, config.ACTIVITY_WEIGHT, config.VOCALIZATION_WEIGHT
        )
        
        print(f"\n Fold {fold} Final Results (from best model):")
        print(f"  Activity - Acc: {val_activity_acc:.4f}, F1: {val_activity_f1:.4f}")
        print(f"  Vocalization - Acc: {val_voc_acc:.4f}, F1: {val_voc_f1:.4f}")
        
        # Save confusion matrices
        activity_cm = confusion_matrix(
            activity_labels_true, activity_preds,
            labels=list(range(len(activity_labels)))
        )
        pd.DataFrame(activity_cm, index=activity_labels, columns=activity_labels).to_csv(
            os.path.join(config.OUTPUT_DIR, f'activity_confusion_matrix_fold{fold}.csv')
        )
        
        voc_cm = confusion_matrix(
            voc_labels_true, voc_preds,
            labels=list(range(len(vocalization_labels)))
        )
        pd.DataFrame(voc_cm, index=vocalization_labels, columns=vocalization_labels).to_csv(
            os.path.join(config.OUTPUT_DIR, f'vocalization_confusion_matrix_fold{fold}.csv')
        )
        
        fold_results.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'activity_accuracy': val_activity_acc,
            'activity_f1': val_activity_f1,
            'vocalization_accuracy': val_voc_acc,
            'vocalization_f1': val_voc_f1,
            'fold_time_hours': fold_time
        })
    
    # Summary
    total_time = (time.time() - start_time) / 3600
    print("CROSS VALIDATION SUMMARY")
    print(f"  Total training time: {total_time:.2f} hours")
    
    results_df = pd.DataFrame(fold_results)
    print(f"\n{results_df.to_string(index=False)}")
    
    print(f"\nAverage Results:")
    print(f"  Activity - Acc: {results_df['activity_accuracy'].mean():.4f} ± {results_df['activity_accuracy'].std():.4f}")
    print(f"  Activity - F1: {results_df['activity_f1'].mean():.4f} ± {results_df['activity_f1'].std():.4f}")
    print(f"  Vocalization - Acc: {results_df['vocalization_accuracy'].mean():.4f} ± {results_df['vocalization_accuracy'].std():.4f}")
    print(f"  Vocalization - F1: {results_df['vocalization_f1'].mean():.4f} ± {results_df['vocalization_f1'].std():.4f}")
    
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'multitask_cross_validation_summary.csv'), index=False)
    print(f"\n Results saved to {config.OUTPUT_DIR}/")
    
    return results_df

# RUN THE SCRIPT
if __name__ == '__main__':
    try:
        # Set seeds for reproducibility
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
    
        print("Starting cross-validation...")
        results = cross_validate(Config()) # Instantiate the config class
        print("Cross-validation finished.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
