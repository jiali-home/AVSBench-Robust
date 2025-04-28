"""
Audio Processing for Robust Audio-Visual Segmentation

This script processes audio datasets for audio-visual segmentation tasks, creating misaligned and noise versions
of audio files for training/testing robust AVS models. 

Required files before running:
1. Original datasets (s4 or ms3) from AVSBench
2. metadata.csv from AVSS dataset

Usage:
    python audio_processing.py --dataset [s4|ms3] --seed [random_seed]
"""

import os
import argparse
from pydub import AudioSegment
import numpy as np
import random
import shutil
import pickle
import librosa
import torch
import sys
import pandas as pd
from pydub import AudioSegment

def generate_silent_audio(duration_ms=5000, sample_rate=44100):
    """
    Generate a silent audio file.
    
    :param duration_ms: Duration in milliseconds (5 seconds = 5000ms)
    :param sample_rate: Sample rate in Hz
    :return: AudioSegment object
    """
    silent_segment = AudioSegment.silent(duration=duration_ms)
    return silent_segment
    # from pydub.generators import Silence
    # Silence().to_audio_segment(duration=duration_ms)

def generate_noise_audio(duration_ms=5000, sample_rate=44100):
    """
    Generate white noise audio.
    
    :param duration_ms: Duration in milliseconds (5 seconds = 5000ms)
    :param sample_rate: Sample rate in Hz
    :return: AudioSegment object
    """
    samples = np.random.uniform(-1, 1, int(sample_rate * duration_ms / 1000))
    return AudioSegment(
        samples.tobytes(), 
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )



def extract_log_mel_features(wav_path, n_mels=64, n_fft=1024, hop_length=512, num_frames=96, duration=5):
    """
    Extract log-mel spectrogram features from audio file.
    """
    y, sr = librosa.load(wav_path, duration=duration)
    
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))
    
    y_segments = np.array_split(y, 5)
    
    log_mel_segments = []
    for segment in y_segments:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel_spectrogram)

        if log_mel.std() > 0:
            log_mel = (log_mel - log_mel.mean()) / log_mel.std()

        if log_mel.shape[1] < num_frames:
            pad_width = num_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > num_frames:
            log_mel = log_mel[:, :num_frames]
        
        log_mel_segments.append(log_mel)
    
    log_mel_stack = np.stack(log_mel_segments)
    log_mel_tensor = torch.from_numpy(log_mel_stack).float().permute(0, 2, 1).unsqueeze(1)
    
    return log_mel_tensor

def save_audio_and_features(audio_segment, save_path, feature_save_path):
    """
    Save audio file and extract/save its features.
    """
    # Save audio
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    audio_segment.export(save_path, format="wav")
    
    # Extract and save features
    os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
    features = extract_log_mel_features(save_path)
    with open(feature_save_path, 'wb') as f:
        pickle.dump(features, f)


def get_semantic_category(dataset_type):
    if dataset_type == 's4':
        return {
            # Music / Musical Instruments
            'playing_acoustic_guitar': 'music',
            'playing_glockenspiel': 'music',
            'playing_piano': 'music',
            'playing_tabla': 'music',
            'playing_ukulele': 'music',
            'playing_violin': 'music',
            
            # Human Voice / Vocalizations
            'baby_laughter': 'human',
            'female_singing': 'human',
            'male_speech': 'human',
            
            # Animals
            'cat_meowing': 'animal',
            'coyote_howling': 'animal',
            'dog_barking': 'animal',
            'horse_clip-clop': 'animal',
            'lions_roaring': 'animal',
            'mynah_bird_singing': 'animal',
            
            # Devices / Machines / Vehicles
            'driving_buses': 'device',
            'helicopter': 'device',
            'race_car': 'device',            
            'ambulance_siren': 'device',
            'cap_gun_shooting': 'device',
            'chainsawing_trees': 'device',
            'lawn_mowing': 'device',
            'typing_on_computer_keyboard': 'device'
        }
    elif dataset_type == 'ms3':
        return {
            # Music / Musical Instruments
            'guitar': 'music',
            'tabla': 'music',
            'violin': 'music',
            'ukulele': 'music',
            'piano': 'music',
            'marimba': 'music',
            
            # Human Voice / Vocalizations
            'man': 'human',
            'woman': 'human',
            'baby': 'human',
            
            # Animals
            'dog': 'animal',
            'cat': 'animal',
            'lion': 'animal',
            'bird': 'animal',
            'wolf': 'animal',
            
            # Devices / Machines
            'mower': 'device',
            'background': 'device',
            'bus': 'device',
            'car': 'device',
            'gun': 'device',
            'saw': 'device',
            'emergency-car': 'device',
            'keyboard': 'device'
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")



def get_broad_categories(objects, semantic_map):
    """
    Get the broad categories for a set of objects.
    
    Args:
        objects (set): Set of specific object labels
        semantic_map (dict): Mapping from specific labels to broad categories
    
    Returns:
        set: Set of broad categories
    """
    categories = set()
    for obj in objects:
        if obj in semantic_map:
            categories.add(semantic_map[obj])
    return categories


def merge_wav_files(file1_path, file2_path):
    """
    Merge two WAV audio files into a single track.
    
    Args:
        file1_path (str): Path to the first WAV file
        file2_path (str): Path to the second WAV file
    """
    # Load audio files
    audio1 = AudioSegment.from_wav(file1_path)
    audio2 = AudioSegment.from_wav(file2_path)
    
    return audio1
    # # Overlay audio tracks
    # merged_audio = audio1.overlay(audio2)
    
    # # # Get durations in seconds
    # # duration1 = audio1.duration_seconds  # Duration of first audio
    # # duration2 = audio2.duration_seconds  # Duration of second audio
    # # merged_duration = merged_audio.duration_seconds
    # # print(f"Duration of audio1: {duration1:.2f} seconds")
    # # print(f"Duration of audio2: {duration2:.2f} seconds")  
    # # print(f"Duration of merged audio: {merged_duration:.2f} seconds")
    # return merged_audio

def process_ms3_split(base_dir, split, misaligned_wav_dir, misaligned_feature_dir,
                     noise_dir, noise_feature_dir):
    """
    Process a split of the ms3 dataset using semantic categories for misalignment.
    Random selection from suitable contrast categories.
    """
    # Read metadata
    metadata_file = 'path_to_metadata_file'
    df = pd.read_csv(metadata_file)
    df = df[df['label'] == 'v1m']  # Filter for ms3 dataset
    df = df[df['split'] == split]  # Filter for current split
    
    semantic_map = get_semantic_category('ms3')
    
    # Process object categories
    def get_objects(a_obj):
        return set(a_obj.split('_'))
    df['objects'] = df['a_obj'].apply(get_objects)
    
    # Add broad categories column
    df['broad_categories'] = df['objects'].apply(
        lambda x: get_broad_categories(x, semantic_map))
    
    # Create category groups for each video
    category_groups = {
        'music': [],
        'human': [],
        'animal': [],
        'device': []
    }
    
    # Group videos by their categories
    for idx, row in df.iterrows():
        for category in row['broad_categories']:
            if category in category_groups:
                category_groups[category].append(row['vid'])
    
    # Define least suitable contrast for each category (to avoid)
    least_suitable_contrast = {
        'music': 'human',  
        'human': 'music',  
        'animal': 'human', 
        'device': 'music'   
    }
    
    # Process each audio file
    split_dir = os.path.join(base_dir, split)
    audio_files = [f for f in os.listdir(split_dir) if f.endswith('.wav')]
    
    for audio_file in audio_files:
        vid = audio_file.replace('.wav', '')
        row = df[df['vid'] == vid]
        
        if not row.empty:
            current_categories = row['broad_categories'].iloc[0]
            
            if current_categories:
                # Get all available categories for contrast
                available_vids = []
                
                # For each current category, avoid its least suitable contrast
                unsuitable_categories = set(least_suitable_contrast.get(cat, '') 
                                         for cat in current_categories)
                
                # Find videos that don't share categories with current video
                # and aren't in unsuitable categories
                for other_idx, other_row in df.iterrows():
                    other_categories = other_row['broad_categories']
                    if (other_row['vid'] != vid and
                        len(other_categories.intersection(current_categories)) == 0 and  
                        len(other_categories.intersection(unsuitable_categories)) == 0): 
                        available_vids.append(other_row['vid'])
                
                if available_vids:
                    # Randomly select a contrasting video
                    mismatched_vid = random.choice(available_vids)
                    
                    # Create misaligned audio
                    src_path = os.path.join(split_dir, f"{mismatched_vid}.wav")
                    misaligned_wav_path = os.path.join(misaligned_wav_dir, split, audio_file)
                    misaligned_feature_path = os.path.join(misaligned_feature_dir, split,
                                                         audio_file.replace('.wav', '.pkl'))
                    
                    if os.path.exists(src_path):
                        # Get category information for logging
                        mismatched_row = df[df['vid'] == mismatched_vid]
                        mismatched_categories = mismatched_row['broad_categories'].iloc[0]
                        
                        print(f"Misaligning: {vid} ({' + '.join(sorted(current_categories))}) with " +
                              f"{mismatched_vid} ({' + '.join(sorted(mismatched_categories))})")
                        # print(f"  - Original objects: {row['a_obj'].iloc[0]}")
                        # print(f"  - Misaligned with objects: {mismatched_row['a_obj'].iloc[0]}")
                        
                        # Create directories if they don't exist
                        os.makedirs(os.path.dirname(misaligned_wav_path), exist_ok=True)
                        os.makedirs(os.path.dirname(misaligned_feature_path), exist_ok=True)
                        
                        # Copy and process audio
                        audio = AudioSegment.from_wav(src_path)
                        save_audio_and_features(audio, misaligned_wav_path, misaligned_feature_path)
                else:
                    print(f"Warning: No suitable contrast found for {vid} ({' + '.join(sorted(current_categories))})")
                    other_vids = [v for v in df['vid'].values if v != vid]
                    if other_vids:
                        mismatched_vid = random.choice(other_vids)
                        src_path = os.path.join(split_dir, f"{mismatched_vid}.wav")
                        misaligned_wav_path = os.path.join(misaligned_wav_dir, split, audio_file)
                        misaligned_feature_path = os.path.join(misaligned_feature_dir, split,
                                                             audio_file.replace('.wav', '.pkl'))
                        
                        if os.path.exists(src_path):
                            print(f"Falling back to random misalignment with {mismatched_vid}")
                            audio = AudioSegment.from_wav(src_path)
                            save_audio_and_features(audio, misaligned_wav_path, misaligned_feature_path)
        
        # Create noise audio
        noise_wav_path = os.path.join(noise_dir, split, audio_file)
        noise_feature_path = os.path.join(noise_feature_dir, split,
                                        audio_file.replace('.wav', '.pkl'))
        
        os.makedirs(os.path.dirname(noise_wav_path), exist_ok=True)
        os.makedirs(os.path.dirname(noise_feature_path), exist_ok=True)
        
        noise_audio = generate_noise_audio(duration_ms=5000)
        save_audio_and_features(noise_audio, noise_wav_path, noise_feature_path)

def merge_process_ms3_split(base_dir, split, merged_wav_dir, merged_feature_dir, neg_audio_dir):
    split_dir = os.path.join(base_dir, split)
    audio_files = [f for f in os.listdir(split_dir) if f.endswith('.wav')]

    for audio_file in audio_files:

        merged_wav_path = os.path.join(merged_wav_dir, split, audio_file)
        merged_feature_path = os.path.join(merged_feature_dir, split, audio_file.replace('.wav', '.pkl'))
        outlier_audio_path = os.path.join(neg_audio_dir, split, audio_file)
        os.makedirs(os.path.dirname(merged_wav_path), exist_ok=True)
        os.makedirs(os.path.dirname(merged_feature_path), exist_ok=True)

        current_wav_path = os.path.join(split_dir, audio_file)
        merged_audio = merge_wav_files(current_wav_path, outlier_audio_path)

        save_audio_and_features(merged_audio, merged_wav_path, merged_feature_path)                   
    
def process_s4_split(base_dir, split, misaligned_wav_dir, misaligned_feature_dir,
                    noise_dir, noise_feature_dir):
    """
    Process a split of the s4 dataset using exact label matching and semantic categories for misalignment.
    Random selection from suitable contrast categories.
    """
    split_dir = os.path.join(base_dir, split)
    semantic_map = get_semantic_category('s4')
    
    # Get all categories in the split directory
    categories = [c for c in os.listdir(split_dir) if '.DS_Store' not in c]
    
    # Create semantic category groups for easier misalignment
    category_groups = {
        'music': [],
        'human': [],
        'animal': [],
        'vehicle': [],
        'device': []
    }
    
    # Map directories to their semantic categories
    for category in categories:
        if category in semantic_map:
            broad_category = semantic_map[category]
            category_groups[broad_category].append(category)
        else:
            print(f"Warning: Category '{category}' not found in semantic mapping")
    
    # Define least suitable contrast for each category (to avoid)
    least_suitable_contrast = {
        'music': 'human',     
        'human': 'music',     
        'animal': 'human',    
        'vehicle': 'device',  
        'device': 'vehicle'   
    }
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        category_dir = os.path.join(split_dir, category)
        audio_files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
        
        # Get the semantic category for current directory
        current_semantic_category = semantic_map.get(category)
        
        for audio_file in audio_files:
            # Find a contrasting category for misalignment
            contrasting_category = None
            
            if current_semantic_category:
                # Get all categories except the current one and the least suitable contrast
                least_suitable = least_suitable_contrast.get(current_semantic_category)
                available_categories = []
                
                for cat_type, cat_list in category_groups.items():
                    if (cat_type != current_semantic_category and 
                        cat_type != least_suitable and 
                        cat_list): 
                        available_categories.extend(cat_list)
                
                if available_categories:
                    contrasting_category = random.choice(available_categories)
            
            # If no contrasting category found, choose random category different from current
            if not contrasting_category:
                other_categories = [c for c in categories if c != category]
                if other_categories:
                    contrasting_category = random.choice(other_categories)
            
            if contrasting_category:
                # Create misaligned audio
                contrast_dir = os.path.join(split_dir, contrasting_category)
                contrast_audio_files = [f for f in os.listdir(contrast_dir) if f.endswith('.wav')]
                
                if contrast_audio_files:
                    random_audio = random.choice(contrast_audio_files)
                    src_path = os.path.join(contrast_dir, random_audio)
                    
                    misaligned_wav_path = os.path.join(misaligned_wav_dir, split, category, audio_file)
                    misaligned_feature_path = os.path.join(misaligned_feature_dir, split, category,
                                                         audio_file.replace('.wav', '.pkl'))
                    
                    if os.path.exists(src_path):
                        # Log the misalignment with semantic categories
                        current_type = current_semantic_category if current_semantic_category else "uncategorized"
                        contrast_type = semantic_map.get(contrasting_category, "uncategorized")
                        
                        print(f"Misaligning: {category} ({current_type}) with {contrasting_category} ({contrast_type})")
                        # print(f"  - Source file: {audio_file}")
                        # print(f"  - Misaligned with: {random_audio}")
                        
                        # Create output directories
                        os.makedirs(os.path.dirname(misaligned_wav_path), exist_ok=True)
                        os.makedirs(os.path.dirname(misaligned_feature_path), exist_ok=True)
                        
                        # Process audio
                        audio = AudioSegment.from_wav(src_path)
                        save_audio_and_features(audio, misaligned_wav_path, misaligned_feature_path)
            
            # Create noise audio
            noise_wav_path = os.path.join(noise_dir, split, category, audio_file)
            noise_feature_path = os.path.join(noise_feature_dir, split, category,
                                            audio_file.replace('.wav', '.pkl'))
            
            os.makedirs(os.path.dirname(noise_wav_path), exist_ok=True)
            os.makedirs(os.path.dirname(noise_feature_path), exist_ok=True)
            
            noise_audio = generate_noise_audio(duration_ms=5000)
            save_audio_and_features(noise_audio, noise_wav_path, noise_feature_path)

def merge_process_s4_split(base_dir, split, merged_wav_dir, merged_feature_dir, neg_audio_dir):
    split_dir = os.path.join(base_dir, split)
    
    # Get all categories in the split directory
    categories = [c for c in os.listdir(split_dir) if '.DS_Store' not in c]
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        category_dir = os.path.join(split_dir, category)
        audio_files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]

        for audio_file in audio_files:

            merged_wav_path = os.path.join(merged_wav_dir, split, category, audio_file)
            merged_feature_path = os.path.join(merged_feature_dir, split, category,
                                                    audio_file.replace('.wav', '.pkl'))
            outlier_audio_path = os.path.join(neg_audio_dir, split, category, audio_file)

            os.makedirs(os.path.dirname(merged_wav_path), exist_ok=True)
            os.makedirs(os.path.dirname(merged_feature_path), exist_ok=True)

            current_wav_path = os.path.join(split_dir, category, audio_file)
            merged_audio = merge_wav_files(current_wav_path, outlier_audio_path)

            save_audio_and_features(merged_audio, merged_wav_path, merged_feature_path)   

                             
def process_dataset(dataset_type, base_path, output_base):
    """
    Process either s4 or ms3 dataset.
    """

    if dataset_type == 's4':
        base_dir = f'{base_path}/Single-source/s4_data/audio_wav'
        output_dir = f'{output_base}/Single-source/s4_data'
    else:  # ms3
        base_dir = f'{base_path}/Multi-sources/ms3_data/audio_wav'
        output_dir = f'{output_base}/Multi-sources/ms3_data'

    # Generate and save one silent audio for the whole dataset
    silent_audio = generate_silent_audio(duration_ms=5000)
    silent_wav_path = os.path.join(output_base, 'silent.wav')
    silent_feature_path = os.path.join(output_base, 'silent.pkl')
    save_audio_and_features(silent_audio, silent_wav_path, silent_feature_path)
    
    # Create output directories
    misaligned_wav_dir = os.path.join(output_dir, 'audio_wav_misaligned')
    misaligned_feature_dir = os.path.join(output_dir, 'audio_log_mel_misaligned')
    noise_dir = os.path.join(output_dir, 'audio_wav_noise')
    noise_feature_dir = os.path.join(output_dir, 'audio_log_mel_noise')

    for split in ['test', 'train', 'val']:
        print(f"Processing {split} split...")
        
        if dataset_type == 's4':
            process_s4_split(
                base_dir, split,
                misaligned_wav_dir, misaligned_feature_dir,
                noise_dir, noise_feature_dir
            )
        else:
            process_ms3_split(
                base_dir, split,
                misaligned_wav_dir, misaligned_feature_dir,
                noise_dir, noise_feature_dir
            )


def merge_audio_gen(dataset_type, base_path, output_base):
    if dataset_type == 's4':
        base_dir = f'{base_path}/Single-source/s4_data/audio_wav'
        output_dir = f'{output_base}/Single-source/s4_data'
    else:  # ms3
        base_dir = f'{base_path}/Multi-sources/ms3_data/audio_wav'
        output_dir = f'{output_base}/Multi-sources/ms3_data'
    # Create output directories
    merged_wav_dir = os.path.join(output_dir, 'audio_wav_merged')
    merged_feature_dir = os.path.join(output_dir, 'audio_log_mel_merged')
    neg_audio_dir = os.path.join(output_dir, 'audio_wav_misaligned')
    
    for split in ['test', 'train', 'val']:
        print(f"Processing {split} split...")
        
        if dataset_type == 's4':
            merge_process_s4_split(
                base_dir, split,
                merged_wav_dir, merged_feature_dir, neg_audio_dir
            )
        else:
            merge_process_ms3_split(
                base_dir, split,
                merged_wav_dir, merged_feature_dir, neg_audio_dir
            )

def main():
    parser = argparse.ArgumentParser(description='Process audio datasets')
    parser.add_argument('--dataset', type=str, choices=['s4', 'ms3'], required=True,
                      help='Dataset to process (s4 or ms3)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_path = 'path_to_dataset'
    output_base = 'path_to_output_directory'
    
    # process_dataset(args.dataset, base_path, output_base)
    merge_audio_gen(args.dataset, base_path, output_base)


if __name__ == "__main__":
    main()