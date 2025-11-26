import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.manifold import TSNE
from tqdm import tqdm
import json


#  CONFIGURATION
class Config:
    DATA_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Data'
    MODEL_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\weighted_cross_outputs_3'
    SAVE_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Analysis_Results'

    MODEL_NAME = 'facebook/hubert-base-ls960'
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 10.0
    WINDOW_STEP = 5.0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # t-SNE Settings
    MAX_POINTS_TO_PLOT = 15000 

#  MODEL DEFINITION
class MultiTaskHuBERT(nn.Module):
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
        
        return activity_logits, vocalization_logits, pooled

def find_all_raw_wavs(base_dir):
    search_pattern = os.path.join(base_dir, "Raw*", "**", "*.wav")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        print(f"No files found in {search_pattern}")
        files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    return sorted(files)

def run_global_tsne():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Checkpoint file path
    checkpoint_path = os.path.join(Config.SAVE_DIR, 'extracted_features_checkpoint.npz')

    # Load Labels
    print("Loading labels...")
    try:
        with open(os.path.join(Config.MODEL_DIR, 'activity_id2label.json'), 'r') as f:
            act_id2label = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(Config.MODEL_DIR, 'vocalization_id2label.json'), 'r') as f:
            voc_id2label = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print("ERROR: Label files not found.")
        return

    # CHECKPOINT
    if os.path.exists(checkpoint_path):
        print(f"\n[FAST LOAD] Found checkpoint at: {checkpoint_path}")
        print("Skipping audio processing and loading saved features")
        data = np.load(checkpoint_path, allow_pickle=True)
        all_embeddings = data['embeddings']
        all_act_preds = data['activities']
        all_voc_preds = data['vocalizations']
        print(f"Loaded {len(all_embeddings)} segments instantly.")
    
    else:
        # AUDIO PROCESSING 
        print("No checkpoint found. Processing audio files")
        print("Loading model")
        model = MultiTaskHuBERT(Config.MODEL_NAME, len(act_id2label), len(voc_id2label))
        model_path = os.path.join(Config.MODEL_DIR, 'multitask_fold1_best.pt')
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            return
            
        model.to(Config.DEVICE)
        model.eval()
        
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.MODEL_NAME)
        files = find_all_raw_wavs(Config.DATA_DIR)
        print(f"Found {len(files)} audio files to process.")
        
        all_embeddings = []
        all_act_preds = []
        all_voc_preds = []
        
        step_samples = int(Config.WINDOW_STEP * Config.SAMPLE_RATE)
        target_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLE_RATE)

        print("\n Starting Extraction Loop")
        for file_path in tqdm(files):
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                num_segments = (len(audio) - target_samples) // step_samples + 1
                if num_segments < 1: continue

                for i in range(num_segments):
                    start = i * step_samples
                    end = start + target_samples
                    segment = audio[start:end]

                    inputs = feature_extractor(segment, sampling_rate=Config.SAMPLE_RATE, return_tensors="pt", padding=True)
                    input_values = inputs.input_values.to(Config.DEVICE)

                    with torch.no_grad():
                        act_logits, voc_logits, features = model(input_values)
                        act_pred = torch.argmax(act_logits, dim=1).item()
                        voc_pred = torch.argmax(voc_logits, dim=1).item()
                        
                        # Store as numpy (RAM efficient)
                        all_embeddings.append(features.cpu().numpy().flatten())
                        all_act_preds.append(act_id2label[act_pred])
                        all_voc_preds.append(voc_id2label[voc_pred])
                        
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {e}")
                continue
        print(f"Saving checkpoint to {checkpoint_path}...")
        np.savez(checkpoint_path, embeddings=all_embeddings, activities=all_act_preds, vocalizations=all_voc_preds)

    # Downsampling
    total_points = len(all_embeddings)
    if total_points < 10:
        print("Not enough data points.")
        return

    if total_points > Config.MAX_POINTS_TO_PLOT:
        print(f"Sampling {Config.MAX_POINTS_TO_PLOT} points from {total_points} total...")
        indices = np.random.choice(total_points, Config.MAX_POINTS_TO_PLOT, replace=False)
        X = np.array(all_embeddings)[indices]
        y_act = np.array(all_act_preds)[indices]
        y_voc = np.array(all_voc_preds)[indices]
    else:
        X = np.array(all_embeddings)
        y_act = np.array(all_act_preds)
        y_voc = np.array(all_voc_preds)

    print(f"Running t-SNE on {len(X)} points")
    tsne = TSNE(n_components=2, perplexity=30, verbose=1, random_state=42, max_iter=1000)
    
    X_embedded = tsne.fit_transform(X)

    #  Plotting
    print("Generating Global Cluster Map")
    df_plot = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Predicted Activity': y_act,
        'Predicted Vocalization': y_voc
    })

    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        data=df_plot, 
        x='x', y='y', 
        hue='Predicted Activity',
        style='Predicted Vocalization',
        s=80, 
        alpha=0.7
    )
    
    plt.title(f"Global Audio Feature Map ({len(X)} segments)\nColored by Model Prediction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(Config.SAVE_DIR, 'global_tsne_map.png')
    plt.savefig(save_path)
    print(f"Global Map saved to: {save_path}")

if __name__ == '__main__':
    run_global_tsne()