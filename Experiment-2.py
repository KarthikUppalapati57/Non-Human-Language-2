import os
import pandas as pd
from gemini_setup import get_gemini_model

# --- CONFIGURATION ---
PREDICTIONS_DIR = r"C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\local_Weighted_3_predictions"

# --- DATA LOADER (Same as Exp 1) ---
def get_session_profile(audio_filename):
    """
    Reads the log and calculates the percentage of time spent in each state.
    """
    file_stem = os.path.splitext(audio_filename)[0]
    log_name = f"prediction_log_{file_stem}.csv"
    log_path = os.path.join(PREDICTIONS_DIR, log_name)
    
    print(f" Looking for log file: {log_name}...")
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f" Could not find log file at: {log_path}")

    print(f" Found log! Calculating distribution...")
    df = pd.read_csv(log_path)
    
    # Calculate normalized counts (percentages)
    if 'activity' in df.columns:
        act_probs = df['activity'].value_counts(normalize=True).round(3).to_dict()
    else:
        act_probs = {"Error": "Column 'activity' not found"}

    if 'vocalization' in df.columns:
        voc_probs = df['vocalization'].value_counts(normalize=True).round(3).to_dict()
    else:
        voc_probs = {"Error": "Column 'vocalization' not found"}
    
    return act_probs, voc_probs

# --- MAIN EXECUTION ---
def run():
    print("\n Experiment 2: Few-Shot Expert Analysis (Session Context)")
    
    audio_file = input("Enter audio filename (e.g., 20211120_114118_192.wav): ").strip()
    
    try:
        # 1. Get Profile
        act_probs, voc_probs = get_session_profile(audio_file)
        
        print(f"\n Session Profile:")
        print(f"   Act: {act_probs}")
        print(f"   Voc: {voc_probs}")
        
        # 2. Connect to Gemini
        model = get_gemini_model()
        
        # 3. Prompt with "Session-Based" Examples
        prompt = f"""
        Role: Expert Marine Biologist.
        Task: Interpret the behavioral profile of a dolphin recording session.
        
        [TEACHING EXAMPLES]
        
        Example 1 (Clear Activity):
        Input: 
          - Activity Profile: {{'PLAY': 0.85, 'unknown': 0.15}}
          - Vocalization Profile: {{'W': 0.90, 'NOISE': 0.10}}
        Output: 
          This session is clearly dominated by Play behavior (85%). The strong presence of Whistles ('W' at 90%) confirms this is a social, non-aggressive interaction. The small 'unknown' percentage suggests high model confidence throughout the clip.
        
        Example 2 (Sparse/Intermittent Activity):
        Input: 
          - Activity Profile: {{'unknown': 0.70, 'ORD': 0.25, 'FFR': 0.05}}
          - Vocalization Profile: {{'MW': 0.30, 'silence': 0.70}}
        Output: 
          This recording captures mostly silence or low-confidence data (70% unknown). However, the 25% presence of 'Training' (ORD) combined with Multi-Loop Whistles (MW) indicates the dolphin was highly engaged and actively communicating (likely with the trainer or another dolphin) during the moments of training activity

        [CURRENT SESSION DATA]
        - Activity Profile: {act_probs}
        - Vocalization Profile: {voc_probs}
        
        [YOUR ANALYSIS]
        Synthesize this data. What is the dolphin doing?
        """
        
        print("\n... Generating Expert Analysis ...")
        response = model.generate_content(prompt)
        print(f"\n Expert Report:\n{response.text}")
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    run()