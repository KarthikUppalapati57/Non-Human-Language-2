"""
Step 1: Create vocal_annotation.csv (UPDATED - Process ALL files)
This version processes BOTH matched and unknown files from the CSV
Marks files without annotations as 'unknown'
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_vocal_annotation_csv(matched_csv_file, data_base_folder, annotation_folder, output_csv='vocal_annotation.csv'):
    """
    Create vocal annotation CSV from annotation .txt files
    Processes ALL files (matched and unknown status)
    """
    print(" CREATE VOCAL ANNOTATION CSV (Process ALL Files)")
    
    # Load matched files
    print(f"\n Loading: {matched_csv_file}")
    df_all = pd.read_csv(matched_csv_file)
    
    print(f"   Total files in CSV: {len(df_all)}")
    
    # Show status distribution
    status_counts = df_all['status'].value_counts()
    print(f"\n Status Distribution:")
    for status, count in status_counts.items():
        print(f"   {status}: {count} files")
    
    # Process ALL files (not just matched!)
    print(f"\n Processing ALL files (matched + unknown)...")
    
    print(f"\n Folder Configuration:")
    print(f"   Base data folder: {data_base_folder}")
    print(f"   Annotation folder: {annotation_folder}")
    
    # Verify annotation folder exists
    if not os.path.exists(annotation_folder):
        print(f"\n Error: Annotation folder not found: {annotation_folder}")
        print(f"   Please verify the path to Vocalization_Labels folder")
        return None
    
    # Store all vocalization segments
    all_vocalizations = []
    
    # Statistics
    files_with_annotations = 0
    files_without_annotations = 0
    total_vocalizations = 0
    vocalization_type_counts = {}
    missing_files = []
    processed_by_status = {'matched': 0, 'unknown': 0}
    
    print(f"\n🔍 Processing annotation files...")
    
    for idx, row in df_all.iterrows():
        # Get file info from CSV (process ALL rows, not just matched)
        audio_filename = row['filename']
        audio_name = os.path.splitext(audio_filename)[0]  # Remove .wav extension
        folder = row['folder']
        label_activity = row['label_activity']  # May be 'unknown' for unmatched files
        file_status = row['status']  # 'matched' or 'unknown'
        
        # Track processing by status
        processed_by_status[file_status] = processed_by_status.get(file_status, 0) + 1
        
        # Construct annotation file path
        annotation_path = os.path.join(annotation_folder, f"{audio_name}.txt")
        
        # Check if annotation file exists
        if os.path.exists(annotation_path):
            try:
                # Read annotation file
                # Format: start_time \t end_time \t vocalization_type
                ann_df = pd.read_csv(
                    annotation_path,
                    sep='\t',
                    header=None,
                    names=['start_time', 'end_time', 'vocalization_type']
                )
                
                # Calculate duration for each vocalization
                ann_df['duration'] = ann_df['end_time'] - ann_df['start_time']
                
                # Add metadata
                ann_df['original_audio'] = audio_filename
                ann_df['folder'] = folder
                ann_df['label_activity'] = label_activity  # Keep original (may be 'unknown')
                ann_df['status'] = file_status  # Add status column
                
                # Reorder columns
                ann_df = ann_df[[
                    'original_audio', 'folder', 'start_time', 'end_time', 
                    'duration', 'vocalization_type', 'label_activity', 'status'
                ]]
                
                # Add to list
                all_vocalizations.append(ann_df)
                
                # Update statistics
                files_with_annotations += 1
                num_vocs = len(ann_df)
                total_vocalizations += num_vocs
                
                # Count vocalization types (only for valid ones)
                if label_activity != 'unknown':
                    for voc_type in ann_df['vocalization_type'].unique():
                        vocalization_type_counts[voc_type] = vocalization_type_counts.get(voc_type, 0) + len(ann_df[ann_df['vocalization_type'] == voc_type])
                
                if (idx + 1) % 20 == 0:
                    print(f"   Processed {idx + 1}/{len(df_all)} files... ({files_with_annotations} with annotations)")
                    
            except Exception as e:
                print(f"     Error reading annotation for {audio_filename}: {e}")
                files_without_annotations += 1
                missing_files.append(audio_filename)
                
                # Add unknown entry
                all_vocalizations.append(pd.DataFrame([{
                    'original_audio': audio_filename,
                    'folder': folder,
                    'start_time': 0,
                    'end_time': 0,
                    'duration': 0,
                    'vocalization_type': 'unknown',
                    'label_activity': 'unknown',
                    'status': file_status
                }]))
        else:
            # No annotation file found - mark as unknown
            files_without_annotations += 1
            missing_files.append(audio_filename)
            
            all_vocalizations.append(pd.DataFrame([{
                'original_audio': audio_filename,
                'folder': folder,
                'start_time': 0,
                'end_time': 0,
                'duration': 0,
                'vocalization_type': 'unknown',
                'label_activity': 'unknown',
                'status': file_status
            }]))
            
            if files_without_annotations <= 10:  # Show first 10 missing
                print(f"     No annotation file: {os.path.basename(annotation_path)}")
    
    # Combine all dataframes
    if all_vocalizations:
        df_vocal = pd.concat(all_vocalizations, ignore_index=True)
    else:
        print("\n No vocalizations found!")
        return None
    
    # Save to CSV
    df_vocal.to_csv(output_csv, index=False)
    
    print(f"\n Input Files:")
    print(f"   Total audio files in CSV: {len(df_all)}")
    print(f"   Files with 'matched' status: {(df_all['status'] == 'matched').sum()}")
    print(f"   Files with 'unknown' status: {(df_all['status'] == 'unknown').sum()}")
    
    print(f"\n Processing Results:")
    print(f"   Files processed: {len(df_all)}")
    print(f"    Files with annotation .txt: {files_with_annotations}")
    print(f"    Files without annotation .txt: {files_without_annotations}")

    print(f"\n Vocalizations:")
    print(f"   Total vocalization segments: {len(df_vocal)}")
    print(f"   Valid vocalizations (with timestamps): {total_vocalizations}")
    print(f"   Unknown entries (no annotation): {len(df_vocal[df_vocal['vocalization_type'] == 'unknown'])}")
    
    if vocalization_type_counts:
        print(f"\n Vocalization Types (from matched files):")
        for voc_type, count in sorted(vocalization_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_vocalizations) * 100 if total_vocalizations > 0 else 0
            print(f"   {voc_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n Duration Statistics (valid vocalizations only):")
    valid_durations = df_vocal[df_vocal['vocalization_type'] != 'unknown']['duration']
    if len(valid_durations) > 0:
        print(f"   Total duration: {valid_durations.sum():.1f} seconds ({valid_durations.sum()/60:.1f} minutes)")
        print(f"   Average duration: {valid_durations.mean():.2f} seconds")
        print(f"   Min duration: {valid_durations.min():.2f} seconds")
        print(f"   Max duration: {valid_durations.max():.2f} seconds")
        print(f"   Median duration: {valid_durations.median():.2f} seconds")
    
    print(f"\n Label Activity Distribution:")
    activity_counts = df_vocal['label_activity'].value_counts()
    for activity, count in activity_counts.items():
        percentage = (count / len(df_vocal)) * 100
        print(f"   {activity}: {count} ({percentage:.1f}%)")
    
    print(f"\n Folders Processed:")
    folder_counts = df_vocal['folder'].value_counts()
    print(f"   Total unique folders: {len(folder_counts)}")
    for folder, count in sorted(folder_counts.items()):
        print(f"   {folder}: {count} vocalizations")
    
    print(f"\n Output:")
    print(f"    Saved to: {output_csv}")
    print(f"   Total rows: {len(df_vocal)}")
    print(f"   Columns: {list(df_vocal.columns)}")
    
    # Show sample
    print(f"\n Sample rows (first 5 matched, first 5 unknown):")
    matched_sample = df_vocal[df_vocal['status'] == 'matched'].head(5)
    if len(matched_sample) > 0:
        print("\n  MATCHED:")
        print(matched_sample[['original_audio', 'folder', 'duration', 'vocalization_type', 'label_activity']].to_string(index=False))
    
    unknown_sample = df_vocal[df_vocal['label_activity'] == 'unknown'].head(5)
    if len(unknown_sample) > 0:
        print("\n  UNKNOWN:")
        print(unknown_sample[['original_audio', 'folder', 'vocalization_type', 'label_activity', 'status']].to_string(index=False))
    
    # Warning for unknown entries
    unknown_count = len(df_vocal[df_vocal['vocalization_type'] == 'unknown'])
    if unknown_count > 0:
        print(f"\n  NOTE:")
        print(f"   {unknown_count} entries marked as 'unknown' (no annotation file found)")
        print(f"   These will be excluded during training in Step 2")
        
        if len(missing_files) > 0 and len(missing_files) <= 20:
            print(f"\n   Files without annotation .txt:")
            for filename in missing_files:
                print(f"      - {filename}")
        elif len(missing_files) > 20:
            print(f"\n   Files without annotation .txt (first 20):")
            for filename in missing_files[:20]:
                print(f"      - {filename}")
            print(f"      ... and {len(missing_files) - 20} more")

    
    return df_vocal


if __name__ == "__main__":
    # ============ CONFIGURATION ============
    
    # Path to your matched CSV file
    MATCHED_CSV = 'matched_files_fixed.csv'
    
    # Base data folder (where Raw_recordings folders are)
    DATA_BASE_FOLDER = './Data'
    
    # Path to Vocalization_Labels folder (where .txt annotation files are)
    ANNOTATION_FOLDER = './Data/122902/Vocalization_Labels'
    
    # Output CSV filename
    OUTPUT_CSV = 'vocal_annotation_all.csv'
    print("\n Starting Step 1: Create Vocal Annotation CSV (Process ALL Files)\n")
    
    # Check if files exist
    if not os.path.exists(MATCHED_CSV):
        print(f" Error: {MATCHED_CSV} not found!")
        print(f"   Please make sure the file exists in the current directory.")
        exit(1)
    
    if not os.path.exists(ANNOTATION_FOLDER):
        print(f" Error: {ANNOTATION_FOLDER} not found!")
        print(f"   Please update ANNOTATION_FOLDER to point to your Vocalization_Labels directory.")
        exit(1)
    
    # Run the extraction
    df_vocal = create_vocal_annotation_csv(MATCHED_CSV, DATA_BASE_FOLDER, ANNOTATION_FOLDER, OUTPUT_CSV)
    
    if df_vocal is not None:
        print(f"\n Success! Ready for Step 2!")
        print(f"\nCreated: '{OUTPUT_CSV}'")
        print(f"Total segments: {len(df_vocal)}")
        print(f"  - With annotations: {len(df_vocal[df_vocal['vocalization_type'] != 'unknown'])}")
        print(f"  - Without annotations (unknown): {len(df_vocal[df_vocal['vocalization_type'] == 'unknown'])}")
    else:
        print(f"\n Failed to create annotation CSV")