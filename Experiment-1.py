import os
import pandas as pd
from gemini_setup import get_gemini_model

PREDICTIONS_DIR = r"C:\Users\ukart\OneDrive - University of Tennessee\M\3rd Sem\NLP\Dolphins\Project\local_Weighted_3_predictions"


def get_session_profile(audio_filename):
    """
    Reads the time-series log and calculates the prevalence of each behavior 
    across the entire recording duration.
    """
    # 1. Construct the log filename
    file_stem = os.path.splitext(audio_filename)[0]
    log_name = f"prediction_log_{file_stem}.csv"
    log_path = os.path.join(PREDICTIONS_DIR, log_name)
    
    print(f" Looking for log file: {log_name}...")
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f" Could not find log file at: {log_path}")

    # 2. Read the CSV
    print(f" Found log Analyzing session profile...")
    df = pd.read_csv(log_path)
    
    # 3. Calculate "Session Probabilities" (Frequency of each class)
    # This counts how often 'ORD' appears vs 'unknown' across all rows
    if 'activity' in df.columns:
        act_probs = df['activity'].value_counts(normalize=True).round(3).to_dict()
    else:
        act_probs = {"Error": "Column 'activity' not found"}

    if 'vocalization' in df.columns:
        voc_probs = df['vocalization'].value_counts(normalize=True).round(3).to_dict()
    else:
        voc_probs = {"Error": "Column 'vocalization' not found"}
    
    return act_probs, voc_probs


def run():
    print("\n Experiment 1: The Interpreter (Session Analysis)")
    
    audio_file = input("Enter audio filename (e.g., 20211120_114118_192.wav): ").strip()
    
    try:
        # 1. Get Session Profile
        act_probs, voc_probs = get_session_profile(audio_file)
        
        print(f"\n Session Behavioral Profile (Distribution across file):")
        print(f"   Activity Distribution:     {act_probs}")
        print(f"   Vocalization Distribution: {voc_probs}")
        
        # 2. Connect to Gemini
        model = get_gemini_model()
        
        # 3. Prompt (Updated for Session Context)
        prompt = f"""
        Role: You are an expert Marine Biologist.
        Task: Analyze the behavioral profile of a dolphin session based on model predictions.
        
        [DATA: BEHAVIORAL DISTRIBUTION]
        This represents the percentage of time the dolphin spent in each state during the recording:
        - Activity Profile: {act_probs}
        - Vocalization Profile: {voc_probs}
        
        [CONTEXT: STATE DEFINITIONS]
        **Activity States:**
        - 'FFR': Feeding
        - 'ORD': Training
        - 'PLAY': Playing
        - 'NIGHT': During probable sleeping
        
        **Vocalization States:**
        - 'BPS': Burst-pulse sounds (often associated with high arousal/social interaction)
        - 'ECT': Echolocations (used for navigation and foraging)
        - 'FB': Feeding Buzzes (specific, high-repetition echolocation associated with prey capture)
        - 'MW': Multi-Loop Whistles
        - 'W': Whistles (including signature whistles, used for communication)
        - 'unknown': Low confidence or silence.
        
        [INSTRUCTION]
        Synthesize this data into a session report. 
        1. What was the dominant behavior?
        2. **Interpretation of 'unknown':** If 'unknown' is the highest class in either profile (i.e., **>50%** of the time), what does the *presence* of the minority, defined classes (like ORD, FFR, or FB, MW) tell us about the moments when the dolphin *was* actively engaged or vocalizing? Specifically, does the presence of **Feeding** or **Training** states suggest a high-value operational period, despite the overall noise/silence?
        """
        
        print("\n... Generating Session Report ...")
        response = model.generate_content(prompt)
        print(f"\n Marine Biologist Report:\n{response.text}")
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":

    run()
