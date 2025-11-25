import os
import pandas as pd
import google.generativeai as genai
from gemini_setup import get_gemini_model

PREDICTIONS_DIR = r"C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\local_Weighted_3_predictions"

AUDIO_DIR = r"C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\Data\Raw_recordings_Day1_pt2"

def get_hubert_opinion(audio_filename):
    """
    Finds the specific prediction log for this audio file and calculates
    the 'Dominant' behavior (Mode) from the time-series data.
    """
    # 1. Construct the expected log filename
    #    Rule: If audio is '2021...wav', log is 'prediction_log_2021...csv'
    file_stem = os.path.splitext(audio_filename)[0]
    log_name = f"prediction_log_{file_stem}.csv"
    log_path = os.path.join(PREDICTIONS_DIR, log_name)
    
    print(f" Looking for HuBERT log: {log_name}...")
    
    if not os.path.exists(log_path):
        print(f" Log file not found at: {log_path}")
        return "Log Not Found", "Log Not Found"

    try:
        print(f" Found log! Analyzing dominant behavior...")
        df = pd.read_csv(log_path)
        
        # 2. Calculate Dominant Activity (The most frequent class)
        if 'activity' in df.columns:
            # Try to ignore 'unknown' to find the specific signal
            known_acts = df[df['activity'] != 'unknown']['activity']
            if not known_acts.empty:
                # If we have specific labels (like ORD), pick the most common one
                hubert_act = known_acts.mode()[0]
            else:
                # If it's 100% unknown, then the prediction is unknown
                hubert_act = df['activity'].mode()[0]
        else:
            hubert_act = "Error (Column Missing)"

        # 3. Calculate Dominant Vocalization
        if 'vocalization' in df.columns:
            hubert_voc = df['vocalization'].mode()[0]
        else:
            hubert_voc = "Error (Column Missing)"
            
        return hubert_act, hubert_voc

    except Exception as e:
        print(f" Error reading CSV: {e}")
        return "Error", "Error"

def run():
    print("\n Experiment 3: Multimodal Head-to-Head Comparison")
    
    # 1. Get Audio Filename
    filename_input = input("Enter audio filename (e.g., 20211120_114118_192.wav): ").strip()
    
    # 2. Check Audio File
    full_audio_path = os.path.join(AUDIO_DIR, filename_input)
    if not os.path.exists(full_audio_path):
        print(f" Error: Audio file not found at {full_audio_path}")
        return

    try:
        model = get_gemini_model()
        
        # 3. Get HuBERT's Opinion (From the CSV Log)
        hubert_act, hubert_voc = get_hubert_opinion(filename_input)
        
        # 4. Get Gemini's Opinion (From the Audio)
        print(f" Uploading {filename_input} to Gemini...")
        uploaded_file = genai.upload_file(full_audio_path)
        
        prompt = f"""
        Role: Expert Marine Biologist and Data Scientist.
        
        Task: Compare an acoustic model's prediction against your own hearing.
        
        [CONTEXT: STATE DEFINITIONS]
        **Activity States:** 'FFR' (Feeding), 'ORD' (Training), 'PLAY' (Playing), 'NIGHT' (Probable Sleeping).
        **Vocalization States:** 'BPS' (Burst-pulse sounds), 'ECT' (Echolocations), 'FB' (Feeding Buzzes), 'MW' (Multi-Loop Whistles), 'W' (Whistles).

        [MODEL PREDICTION (HuBERT)]
        The acoustic model analyzed this file frame-by-frame.
        - Dominant Activity: {hubert_act}
        - Dominant Vocalization: {hubert_voc}
        
        [YOUR MISSION]
        1. **Acoustic Analysis**: Listen to the raw audio. What specific sounds do you hear? (Echolocation Clicks, whistles, burst pulses?)
        2. **Verification**: 
           - Does the model's prediction ({hubert_act}) match what you hear?
           - If the model said 'ORD' (Training) but you hear 'FFR' (Feeding), explain the discrepancy.
        3. **Final Verdict**: Classify the behavior [FFR, NIGHT, ORD, PLAY].
        """
        
        print("... Comparing ...")
        response = model.generate_content([prompt, uploaded_file])
        
        print(f"  HEAD-TO-HEAD RESULTS")
        print(f"1️  HuBERT Says:  [{hubert_act}] with [{hubert_voc}]")
        print(f"2  Gemini Says:\n{response.text}")

    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":

    run()
