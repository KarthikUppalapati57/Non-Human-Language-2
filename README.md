# Dolphin Behavior Interpretation with MultiTask HuBERT & LLMs

**Translating raw acoustic signals into interpretable behavioral contexts using Deep Learning.**

## Overview
This project introduces a **MultiTask Deep Learning Framework** to interpret the semantic meaning of Bottlenose Dolphin (*Tursiops truncatus*) vocalizations. Unlike traditional approaches that only classify sounds, this system utilizes **HuBERT** and **Large Language Models (LLMs)** to map acoustic patterns to specific behaviors like playing, feeding, or social ordering

## Architecture
The model utilizes a **MultiTaskHuBERT** architecture based on the pre-trained `facebook/hubert-base-ls960` checkpoint.

* **Shared Body:** A pre-trained HuBERT model acts as the "ears," processing raw audio to create context-aware representations.
* **Dual Heads:**
    * **Activity Head:** Classifies behavioral context (5 neurons: FFR, NIGHT, ORD, PLAY, unknown).
    * **Vocalization Head:** Classifies sound type (6 neurons: BPS, ECT, FB, MW, NOISE, W).
* **Semantic Layer:** An LLM (Gemini) generates human-interpretable explanations of the model's predictions, resolving ambiguities where rigid classification fails.

##  Key Results
* **Activity Classification Accuracy:** **79.23%** (achieved after 40 epochs).
* **Vocalization Classification Accuracy:** **73.75%**.
* **Key Improvement:** Extended training and weighted loss strategies significantly improved the detection of "play-fighting" (FFR) and reduced confusion between acoustically similar pulses

##  Dataset
* **Source:** Bottlenose dolphin recordings collected during controlled training sessions (Oltremare Marine Park).
*  **Link**: https://www.seanoe.org/data/00979/109081/
* **Classes:**
    * *Activities:* FFR (Free Feeding/Play-fighting), NIGHT, ORD (Ordering), PLAY, unknown.
    * *Vocalizations:* BPS (Burst Pulse Sounds), ECT (Echolocation Click Trains), FB (Feeding Buzzes), MW (Multi-loop Whistles), W (Whistles), NOISE.

##  Contributors
* **Siva Sai Pavan Karthik Uppalapati** 
* **Jashikar Chowdary Malineni** 
* **Advait Joshi**
* **Gomathi**

* ## Testing INstructions:
* code names and description
* 1) Check.ipnb - Exploring dataset
  2) Vocalization_annotation.py - gathering all the vocalisation and actitiy information to on file (output: Vocalisation_annotation_all.csv).
  3) checking_percentage.ipynb - use =d to check weight so that we can train models accordingly.
  4) weighted_cv_40epoches.py - Code used to Fine-tune our Hubert model.(output: models access in this link: https://drive.google.com/drive/folders/1EDe7jbVom1nd5ljTO0Zexx3xR2Mj9uuB?usp=sharing
  5) HUBERT_to_Prob.py: code to loads the model and take audio files and give you probalities as output
  6) Experiment-1: zero-shot prompting - take the prediction output from the HUBERT_to_Prob.py file and give you human understandble context
  7) Experiment-2: few-shot prompting - take the prediction output from the HUBERT_to_Prob.py file and give you human understandble context
  8) Experiment-3: Head-to-Head - take the prediction output from the HUBERT_to_Prob.py & respective audio file LLM it self checks the audio file becomes judge and gives you final verdict in human understandble context
  9) EM3.py - code for evalution metrics
*Steps for testing:
* 1) Download the dataset
  2) Download All the files inside the Files folder in this repository
  3) Download the models from the link provided
  4) First run the HUBERT_to_prob.py code to get the probability prediction files
  5) Load thoes prediction the LLM Case studies code (Experiment-1,Experiment-2,Experiment-3)
  6) Experiment_Output - sample output file for reference inside the Files folder.
