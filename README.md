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
* 
Download the dataset
