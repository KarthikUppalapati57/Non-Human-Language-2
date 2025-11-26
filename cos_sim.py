import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import HubertModel


# CONFIGURATION
class Config:
    MODEL_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\weighted_cross_outputs_3'
    SAVE_DIR = r'C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Cos_Analysis_Results'
    MODEL_NAME = 'facebook/hubert-base-ls960'

# MODEL DEFINITION

class MultiTaskHuBERT(nn.Module):
    def __init__(self, model_name, num_activity_labels, num_vocalization_labels):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        hidden_size = self.hubert.config.hidden_size
        self.activity_classifier = nn.Linear(hidden_size, num_activity_labels)
        self.vocalization_classifier = nn.Linear(hidden_size, num_vocalization_labels)

    def forward(self, input_values):
        outputs = self.hubert(input_values)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return self.activity_classifier(pooled), self.vocalization_classifier(pooled)

# ANALYSIS SCRIPT
def run_weight_analysis():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print(f"Loading labels from {Config.MODEL_DIR}...")
    try:
        with open(os.path.join(Config.MODEL_DIR, 'activity_id2label.json'), 'r') as f:
            act_id2label = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(Config.MODEL_DIR, 'vocalization_id2label.json'), 'r') as f:
            voc_id2label = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        print("Error: Could not find JSON label files.")
        return

    # Load the best model (using Fold 1 as representative)
    model_path = os.path.join(Config.MODEL_DIR, 'multitask_fold1_best.pt')
    print(f"Loading model weights from {model_path}...")
    
    model = MultiTaskHuBERT(Config.MODEL_NAME, len(act_id2label), len(voc_id2label))
    
    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Extract Weights
    w_act = model.activity_classifier.weight.detach()
    w_voc = model.vocalization_classifier.weight.detach()

    # Normalize vectors
    w_act_norm = F.normalize(w_act, p=2, dim=1)
    w_voc_norm = F.normalize(w_voc, p=2, dim=1)

    similarity_matrix = torch.mm(w_act_norm, w_voc_norm.t()).numpy()

    # Plotting
    print("Generating Heatmap...")
    plt.figure(figsize=(12, 8))
    
    act_labels = [act_id2label[i] for i in range(len(act_id2label))]
    voc_labels = [voc_id2label[i] for i in range(len(voc_id2label))]

    sns.heatmap(similarity_matrix, 
                xticklabels=voc_labels, 
                yticklabels=act_labels,
                cmap="RdBu_r",
                center=0,
                annot=True,
                fmt=".2f")
    
    plt.title('Internal Model Representations\n(Do these classes share underlying features?)')
    plt.ylabel('Activity Weights')
    plt.xlabel('Vocalization Weights')
    plt.tight_layout()
    
    save_path = os.path.join(Config.SAVE_DIR, 'internal_weight_similarity.png')
    plt.savefig(save_path)
    print(f"Analysis complete. Heatmap saved to: {save_path}")

if __name__ == '__main__':
    run_weight_analysis()