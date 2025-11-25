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
warnings.filterwarnings('ignore')


# 1. DEFINE THE MODEL (Must be identical to training)
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


# 2. CONFIGURE THE PREDICTION
# Change this to the 5-minute file you want to test
TEST_AUDIO_FILE = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Data\Raw_recordings_Day1_pt5\20211120_152643_192.wav'

class InferenceConfig:
    
    # Path to your FOLDER containing the .pt and .json files
    MODEL_LOAD_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\weighted_cross_outputs'
    
    # Path to the FOLDER where you want to save prediction_log.csv
    PREDICTION_SAVE_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Weighted_predictions'
    
    # --- Model and Audio settings (Must match training) ---
    MODEL_NAME = 'facebook/hubert-base-ls960'
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 10.0 # The window size
    
    # --- Sliding window step (e.g., predict every 5 seconds) ---
    WINDOW_STEP = 5.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_FOLDS = 5 # Number of models in your ensemble

# 3. THE PREDICTION FUNCTION
def predict_audio_file(file_path, config):
    """
    Loads all 5 models and runs a sliding window prediction
    on the full audio file.
    """
    
    model_dir = config.MODEL_LOAD_DIR
    save_dir = config.PREDICTION_SAVE_DIR
    
    print(f"Loading models and settings from: {model_dir}")
    
    # --- Step 1: Load Label Mappings ---
    try:
        with open(os.path.join(model_dir, 'activity_id2label.json'), 'r') as f:
            activity_id2label = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(model_dir, 'vocalization_id2label.json'), 'r') as f:
            vocalization_id2label = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print(f"ERROR: Could not find label mapping files in {model_dir}")
        print("Please make sure MODEL_LOAD_DIR points to your training output folder.")
        return

    num_activity_labels = len(activity_id2label)
    num_vocalization_labels = len(vocalization_id2label)
    
    print(f"Loaded {num_activity_labels} activity labels and {num_vocalization_labels} vocalization labels.")

    # --- Step 2: Load Feature Extractor ---
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)

    # --- Step 3: Load Models (Ensemble) ---
    models = []
    print(f"Loading {config.NUM_FOLDS} models for ensembling...")
    for fold in range(1, config.NUM_FOLDS + 1):
        model_path = os.path.join(model_dir, f'multitask_fold{fold}_best.pt')
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found, skipping: {model_path}")
            continue
            
        model = MultiTaskHuBERT(config.MODEL_NAME, num_activity_labels, num_vocalization_labels)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval() # Set to evaluation mode
        model.to(config.DEVICE)
        models.append(model)
        
    if not models:
        print(f"ERROR: No models were loaded from {model_dir}. Check your MODEL_LOAD_DIR path.")
        return
        
    print(f"Successfully loaded {len(models)} models.")
    print(f"Using device: {config.DEVICE}")

    # --- Step 4: Load and Process Audio ---
    print(f"\nLoading audio file: {file_path}")
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    except FileNotFoundError:
        print(f"ERROR: Audio file not found at {file_path}")
        print("Please update the 'TEST_AUDIO_FILE' variable.")
        return
        
    target_length_samples = int(config.SEGMENT_LENGTH * config.SAMPLE_RATE)
    step_samples = int(config.WINDOW_STEP * config.SAMPLE_RATE)
    num_segments = (len(audio) - target_length_samples) // step_samples + 1
    
    print(f"Audio loaded. Length: {len(audio)/sr:.2f} seconds.")
    print(f"Scanning file with a {config.SEGMENT_LENGTH}s window, moving {config.WINDOW_STEP}s at a time.")
    print(f"Total segments to predict: {num_segments}")

    # --- Step 5: Run Inference Loop (Sliding Window) ---
    results = []
    for i in tqdm(range(num_segments), desc="Predicting"):
        start = i * step_samples
        end = start + target_length_samples
        segment = audio[start:end]

        # Pre-process the audio segment
        inputs = feature_extractor(
            segment, 
            sampling_rate=config.SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(config.DEVICE)

        # Get predictions from all models
        all_activity_probs = []
        all_vocalization_probs = []
        
        with torch.no_grad():
            for model in models:
                activity_logits, vocalization_logits = model(input_values)
                # Convert to probabilities using softmax
                all_activity_probs.append(torch.nn.functional.softmax(activity_logits, dim=1))
                all_vocalization_probs.append(torch.nn.functional.softmax(vocalization_logits, dim=1))

        # Average the probabilities from all models
        avg_activity_probs = torch.mean(torch.stack(all_activity_probs), dim=0)
        avg_vocalization_probs = torch.mean(torch.stack(all_vocalization_probs), dim=0)

        # Get the final prediction
        activity_pred_id = torch.argmax(avg_activity_probs, dim=1).item()
        vocalization_pred_id = torch.argmax(avg_vocalization_probs, dim=1).item()
        
        # Get the confidence (probability)
        activity_confidence = avg_activity_probs[0, activity_pred_id].item()
        vocalization_confidence = avg_vocalization_probs[0, vocalization_pred_id].item()

        # Store the result
        start_sec = start / config.SAMPLE_RATE
        end_sec = end / config.SAMPLE_RATE
        results.append({
            'start_time_s': f"{start_sec:.1f}",
            'end_time_s': f"{end_sec:.1f}",
            'activity': activity_id2label[activity_pred_id],
            'activity_confidence': f"{activity_confidence:.2f}",
            'vocalization': vocalization_id2label[vocalization_pred_id],
            'vocalization_confidence': f"{vocalization_confidence:.2f}"
        })

    # Step 6: Format and Print Output
    if not results:
        print("No predictions were made. The audio file might be too short.")
        return

    df = pd.DataFrame(results)
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Use new save variable
    # Get the name of the test file (e.g., "my_test_audio.wav")
    test_file_name = os.path.basename(file_path)
    # Create a unique log file name (e.g., "prediction_log_my_test_audio.csv")
    log_file_name = f"prediction_log_{os.path.splitext(test_file_name)[0]}.csv"
    log_file_path = os.path.join(save_dir, log_file_name)
    
    df.to_csv(log_file_path, index=False)
    print(f"\n Full prediction log saved to: {log_file_path}")
    
    # Print summaries   
    print("\n Activity Found (% of windows) ")
    activity_summary = df['activity'].value_counts(normalize=True) * 100
    print(activity_summary.to_string(float_format="%.1f%%"))
    
    print("\n Vocalization Found (% of windows) ")
    vocalization_summary = df['vocalization'].value_counts(normalize=True) * 100
    print(vocalization_summary.to_string(float_format="%.1f%%"))


# 4. RUN THE SCRIPT
# This "if" statement is REQUIRED to prevent errors on Windows
if __name__ == '__main__':
    try:
        config = InferenceConfig()
        predict_audio_file(TEST_AUDIO_FILE, config)
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()