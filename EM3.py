import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa
from tqdm import tqdm
import json
import warnings
import time
import traceback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

    
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



class InferenceConfig:
    ROOT_AUDIO_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Data'
    # --- CHANGE 2: Define the path to your Ground Truth file ---
    GROUND_TRUTH_FILE = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\vocal_annotation_all.csv'
    # Path to your FOLDER containing the .pt and .json files
    MODEL_LOAD_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\weighted_cross_outputs_3'
    # Path to the FOLDER where you want to save prediction_log.csv AND the evaluation CSV
    PREDICTION_SAVE_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\EM_4_results'

    # --- Model and Audio settings ---
    MODEL_NAME = 'facebook/hubert-base-ls960'
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 10.0 # The window size
    WINDOW_STEP = 5.0    # Sliding window step
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_FOLDS = 5 # Number of models in your ensemble

def predict_audio_file(file_path, config, feature_extractor, models, 
                       activity_id2label, vocalization_id2label):
    
    sr = config.SAMPLE_RATE
    segment_len_sec = config.SEGMENT_LENGTH
    step_sec = config.WINDOW_STEP
    device = config.DEVICE
    
    try:
        audio, sr = librosa.load(file_path, sr=sr)
    except Exception:
        return pd.DataFrame()

    target_length_samples = int(segment_len_sec * sr)
    step_samples = int(step_sec * sr)
    audio_len = len(audio)
    
    if audio_len < target_length_samples:
        return pd.DataFrame()
        
    num_segments = (audio_len - target_length_samples) // step_samples + 1
    
    results = []
    
    for i in range(num_segments):
        start = i * step_samples
        end = start + target_length_samples
        segment = audio[start:end]

        inputs = feature_extractor(
            segment, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)

        all_activity_probs = []
        all_vocalization_probs = []

        with torch.no_grad():
            for model in models:
                activity_logits, vocalization_logits = model(input_values)
                all_activity_probs.append(torch.nn.functional.softmax(activity_logits, dim=1))
                all_vocal_probs = torch.nn.functional.softmax(vocalization_logits, dim=1)
                all_vocalization_probs.append(all_vocal_probs)

        avg_activity_probs = torch.mean(torch.stack(all_activity_probs), dim=0)
        avg_vocalization_probs = torch.mean(torch.stack(all_vocalization_probs), dim=0)

        activity_pred_id = torch.argmax(avg_activity_probs, dim=1).item()
        vocalization_pred_id = torch.argmax(avg_vocalization_probs, dim=1).item()

        activity_confidence = avg_activity_probs[0, activity_pred_id].item()
        vocalization_confidence = avg_vocalization_probs[0, vocalization_pred_id].item()

        start_sec = start / sr
        end_sec = end / sr
        results.append({
            'file_path': file_path, 
            'start_time_s': f"{start_sec:.1f}",
            'end_time_s': f"{end_sec:.1f}",
            'activity': activity_id2label[activity_pred_id],
            'activity_confidence': f"{activity_confidence:.2f}",
            'vocalization': vocalization_id2label[vocalization_pred_id],
            'vocalization_confidence': f"{vocalization_confidence:.2f}"
        })

    return pd.DataFrame(results)

# 4. AGGREGATION AND DUAL-TASK EVALUATION FUNCTION (NEW)
def final_aggregate_evaluation(df_all_predictions, df_gt_all, activity_id2label, vocalization_id2label, config):
    """
    Performs dual-task evaluation for both Vocalization and Activity.
    """
    
    step_sec = config.WINDOW_STEP
    all_merged_df = []
    
    # --- Determine Negative/Background Labels ---
    negative_vocal_labels_list = ['silence', 'no_vocalisation', 'none', 'background', 'noise', 'no_call'] 
    negative_vocal_label = next((v for k, v in vocalization_id2label.items() if v.lower() in negative_vocal_labels_list), None)
    
    # ASSUME: Activity labels that are NOT the trained labels are 'None' or 'Background Activity'
    # We will set the default (non-event) activity label as the most common one in the predictions 
    # if it's not a common activity label, or use a placeholder.
    activity_labels_list = list(activity_id2label.values())
    
    # Common Dolphin Project Activities (replace with your non-event/default activity if needed)
    common_activity_labels = ['ORD', 'PLAY', 'FFR', 'UNKNOWN', 'NIGHT']
    
    # Use the most frequent activity label in the predictions as the default/non-event label
    # if no explicit activity is annotated in the GT for a given time window.
    default_activity_label = df_all_predictions['activity'].mode().iloc[0]

    if negative_vocal_label is None:
        print("\nERROR: Cannot proceed. Vocalization negative label not found.")
        return

    # --- 1. Alignment Loop for BOTH Tasks ---
    for full_file_name in tqdm(df_all_predictions['file_path'].apply(os.path.basename).unique(), desc="Aligning Data"):
        
        df_pred_file = df_all_predictions[df_all_predictions['file_path'].apply(os.path.basename) == full_file_name].copy()
        df_pred_file['start_time_s'] = df_pred_file['start_time_s'].astype(float)
        df_pred_file['time_bin'] = (df_pred_file['start_time_s'] / step_sec).round().astype(int)
        
        gt_file = df_gt_all[df_gt_all['file_name'] == full_file_name]

        if gt_file.empty:
            continue
        
        max_time_bin = df_pred_file['time_bin'].max()
        
        # Initialize GT arrays: Vocalization defaults to Noise, Activity defaults to the most frequent predicted activity
        gt_vocal_per_bin = [negative_vocal_label] * (max_time_bin + 1)
        gt_activity_per_bin = [default_activity_label] * (max_time_bin + 1)
        
        for _, row in gt_file.iterrows():
            gt_start_sec = row['start_sec']
            gt_end_sec = row['end_sec']
            
            true_vocal_label = row['vocal_label'] 
            true_activity_label = row['activity_label'] # NEW: Get the true activity label

            start_bin = int(np.floor(gt_start_sec / step_sec))
            end_bin = int(np.ceil(gt_end_sec / step_sec))
            
            for b in range(start_bin, min(end_bin, max_time_bin + 1)):
                # Assign the exact labels to the time bin
                gt_vocal_per_bin[b] = true_vocal_label
                gt_activity_per_bin[b] = true_activity_label # NEW: Assign Activity GT

        gt_df = pd.DataFrame({
            'time_bin': np.arange(len(gt_vocal_per_bin)), 
            'ground_truth_vocal': gt_vocal_per_bin,
            'ground_truth_activity': gt_activity_per_bin # NEW: Activity GT column
        })
        merged_df = pd.merge(df_pred_file, gt_df, on='time_bin', how='inner')
        all_merged_df.append(merged_df)

    if not all_merged_df:
        print("No predictions aligned with ground truth. Evaluation cannot be performed.")
        return

    df_combined = pd.concat(all_merged_df, ignore_index=True)
    

    # 2. VOCALIZATION METRICS (Repeated from previous step)
    
    y_true_vocal = df_combined['ground_truth_vocal']
    y_pred_vocal = df_combined['vocalization']
    vocalization_labels_list = sorted(list(vocalization_id2label.values()))
    
    p_vocal, r_vocal, f1_vocal, support_vocal = precision_recall_fscore_support(
        y_true_vocal, y_pred_vocal, labels=vocalization_labels_list, average=None, zero_division=0
    )
    accuracy_vocal = accuracy_score(y_true_vocal, y_pred_vocal)
    cm_vocal = confusion_matrix(y_true_vocal, y_pred_vocal, labels=vocalization_labels_list)

    print("FINAL AGGREGATED VOCALIZATION EVALUATION")
    print("Vocalization Metrics (Aggregated Across ALL Files)")
    print(f"Overall Accuracy: {accuracy_vocal:.4f}")
    
    metrics_df_vocal = pd.DataFrame({
        'Precision': p_vocal, 'Recall': r_vocal, 'F1-Score': f1_vocal, 'Support (GT)': support_vocal
    }, index=vocalization_labels_list)
    print(metrics_df_vocal.to_string(float_format="%.4f"))
    
    print("VOCALIZATION CONFUSION MATRIX (True Labels vs. Predicted)")
    print(pd.DataFrame(cm_vocal, index=vocalization_labels_list, columns=vocalization_labels_list).to_string())

    # ========================================================================
    # --- 3. ACTIVITY METRICS (NEW) ---
    # ========================================================================
    
    y_true_activity = df_combined['ground_truth_activity']
    y_pred_activity = df_combined['activity']
    activity_labels_list = sorted(list(activity_id2label.values()))
    
    p_activity, r_activity, f1_activity, support_activity = precision_recall_fscore_support(
        y_true_activity, y_pred_activity, labels=activity_labels_list, average=None, zero_division=0
    )
    accuracy_activity = accuracy_score(y_true_activity, y_pred_activity)
    cm_activity = confusion_matrix(y_true_activity, y_pred_activity, labels=activity_labels_list)

    print("FINAL AGGREGATED ACTIVITY EVALUATION")
    print("Activity Metrics (Aggregated Across ALL Files)")
    print(f"Overall Accuracy: {accuracy_activity:.4f}")
    
    metrics_df_activity = pd.DataFrame({
        'Precision': p_activity, 'Recall': r_activity, 'F1-Score': f1_activity, 'Support (GT)': support_activity
    }, index=activity_labels_list)
    print(metrics_df_activity.to_string(float_format="%.4f"))
    
    print(" ACTIVITY CONFUSION MATRIX (True Labels vs. Predicted)")
    print(pd.DataFrame(cm_activity, index=activity_labels_list, columns=activity_labels_list).to_string())


# ============================================================================
# 5. MAIN INFERENCE AND EVALUATION LOOP
# ============================================================================

def run_full_inference_and_evaluation(config):
    """
    Main loop to find all audio files, load models once, predict, and evaluate.
    """
    model_dir = config.MODEL_LOAD_DIR
    
    print(f"Loading models and settings from: {model_dir}")

    # --- Step 1: Load Label Mappings ---
    try:
        with open(os.path.join(model_dir, 'activity_id2label.json'), 'r') as f:
            activity_id2label = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(model_dir, 'vocalization_id2label.json'), 'r') as f:
            vocalization_id2label = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print(f"ERROR: Could not find label mapping files in {model_dir}")
        print("Please check MODEL_LOAD_DIR.")
        return

    num_activity_labels = len(activity_id2label)
    num_vocalization_labels = len(vocalization_id2label)
    
    # Step 2: Load Ground Truth File (FIXED COLUMN MAPPING for Activity)
    try:
        df_gt_all = pd.read_csv(config.GROUND_TRUTH_FILE)
        # RENAMED COLUMNS: Added 'label_activity' to 'activity_label'
        df_gt_all.rename(columns={
            'original_audio': 'file_name', 
            'start_time': 'start_sec',     
            'end_time': 'end_sec',
            'vocalization_type': 'vocal_label', 
            'label_activity': 'activity_label' # NEW: Added for Activity evaluation
        }, inplace=True)
        
        required_cols = ['file_name', 'start_sec', 'end_sec', 'vocal_label', 'activity_label']
        if not all(col in df_gt_all.columns for col in required_cols):
            print(f"ERROR: Ground truth renaming failed. Missing columns. Found: {df_gt_all.columns.tolist()}")
            return

        print(f"Loaded ground truth with {len(df_gt_all)} annotations.")
    except FileNotFoundError:
        print(f"ERROR: Ground truth file not found at {config.GROUND_TRUTH_FILE}. Cannot evaluate.")
        return
        
    # Step 3: Load Feature Extractor and Models (once)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)
    models = []
    for fold in range(1, config.NUM_FOLDS + 1):
        model_path = os.path.join(model_dir, f'multitask_fold{fold}_best.pt')
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found, skipping: {model_path}")
            continue

        model = MultiTaskHuBERT(config.MODEL_NAME, num_activity_labels, num_vocalization_labels)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        model.to(config.DEVICE)
        models.append(model)

    if not models:
        print(f"ERROR: No models were loaded from {model_dir}. Check your MODEL_LOAD_DIR path.")
        return

    print(f"Successfully loaded {len(models)} models on {config.DEVICE}.")
    
    # --- Step 4: Find all audio files ---
    all_audio_files = []
    for root, _, files in os.walk(config.ROOT_AUDIO_DIR):
        for file in files:
            if file.endswith('.wav'): 
                all_audio_files.append(os.path.join(root, file))
    
    print(f"Found a total of {len(all_audio_files)} WAV files in {config.ROOT_AUDIO_DIR}.")

    if not all_audio_files:
        print("No audio files found. Check ROOT_AUDIO_DIR.")
        return
    
    # --- Step 5: Run Inference and store ALL predictions ---
    all_predictions = []
    
    for file_path in tqdm(all_audio_files, desc="Predicting Audio Files"):
        df_pred = predict_audio_file(
            file_path, config, feature_extractor, models, 
            activity_id2label, vocalization_id2label
        )
        if not df_pred.empty:
            all_predictions.append(df_pred)
            
    if not all_predictions:
        print("No predictions were generated. Stopping.")
        return
        
    df_all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # --- Step 6: Final Aggregation and Metric Calculation ---
    final_aggregate_evaluation(df_all_predictions, df_gt_all, activity_id2label, vocalization_id2label, config)


# ============================================================================
# 6. RUN THE SCRIPT
# ============================================================================

if __name__ == '__main__':
    try:
        config = InferenceConfig()
        run_full_inference_and_evaluation(config)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()