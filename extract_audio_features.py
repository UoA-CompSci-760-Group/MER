import os
import glob
import numpy as np
import pandas as pd
import librosa
import warnings
import time
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

warnings.filterwarnings('ignore')

def extract_features_from_file(file_path, sr=22050, duration=30, n_mfcc=13, n_chroma=12, n_bands=6):

    file_name = os.path.basename(file_path)
    audio_id = os.path.splitext(file_name)[0]
    
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        features = {'audio_id': audio_id}

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = mfcc_means[i]
            features[f'mfcc_{i+1}_std'] = mfcc_stds[i]
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_means = np.mean(chroma, axis=1)
        chroma_stds = np.std(chroma, axis=1)
        
        for i in range(n_chroma):
            features[f'chroma_{i+1}_mean'] = chroma_means[i]
            features[f'chroma_{i+1}_std'] = chroma_stds[i]
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_std'] = np.std(spec_cent)
        
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['rolloff_mean'] = np.mean(rolloff)
        
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['flatness_mean'] = np.mean(flatness)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
        features['contrast_mean'] = np.mean(contrast)
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['beat_strength'] = np.mean(onset_env)
        
        spec = np.abs(librosa.stft(y))
        total_energy = np.sum(spec)
        if total_energy > 0:
            low_freq_limit = int(spec.shape[0] * 0.15)
            low_energy = np.sum(spec[:low_freq_limit, :])
            features['low_energy_ratio'] = low_energy / total_energy
        else:
            features['low_energy_ratio'] = 0
            
        if len(rms) > 1:
            rms_sum = np.sum(rms)
            if rms_sum > 0:
                rms_norm = rms / rms_sum
                features['energy_entropy'] = -np.sum(rms_norm * np.log2(rms_norm + 1e-10))
            else:
                features['energy_entropy'] = 0
        else:
            features['energy_entropy'] = 0
            
        if total_energy > 0:
            bright_boundary = int(spec.shape[0] * 0.5)
            bright_energy = np.sum(spec[bright_boundary:, :])
            features['brightness'] = bright_energy / total_energy
        else:
            features['brightness'] = 0
            
        if total_energy > 0:
            warm_low = int(spec.shape[0] * 0.1)  # 10%
            warm_high = int(spec.shape[0] * 0.4)  # 40%
            warm_energy = np.sum(spec[warm_low:warm_high, :])
            features['warmth'] = warm_energy / total_energy
        else:
            features['warmth'] = 0
            
        features['activity'] = np.mean(rms) * (tempo / 120.0 if tempo > 0 else 1.0)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y_power = np.sum(y**2)
        if y_power > 0:
            features['harmonic_energy_ratio'] = np.sum(y_harmonic**2) / y_power
        else:
            features['harmonic_energy_ratio'] = 0
            
        y_percussive_power = np.sum(y_percussive**2)
        if y_percussive_power > 0:
            features['harmonicity'] = np.sum(y_harmonic**2) / y_percussive_power
        else:
            features['harmonicity'] = 0
        
        return features
        
    except Exception as e:
        print(f"error processing file {file_path}: {str(e)}")
        return {'audio_id': audio_id, 'error': str(e)}

def process_audio_batch(audio_dir, output_file, max_workers=None, batch_size=None):
    all_audio_files = []
    for quadrant_dir in ['Q1', 'Q2', 'Q3', 'Q4']:
        quadrant_path = os.path.join(audio_dir, quadrant_dir)
        if os.path.exists(quadrant_path):
            audio_files = glob.glob(os.path.join(quadrant_path, "*.mp3"))
            all_audio_files.extend(audio_files)
    
    if batch_size and batch_size > 0:
        all_audio_files = all_audio_files[:batch_size]
    
    print(f"processing {len(all_audio_files)} audio files...")
    start_time = time.time()
    
    all_features = []
    
    if max_workers and max_workers > 1:
        print(f"using {max_workers} processes for parallel processing...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_features_from_file, file_path) for file_path in all_audio_files]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                features = future.result()
                if features:
                    all_features.append(features)
    else:
        print("using single process...")
        for file_path in tqdm(all_audio_files):
            features = extract_features_from_file(file_path)
            if features:
                all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"processing time: {duration:.2f} seconds")
    print(f"output file: {output_file}")
    
    print(f"\nfeatures: {len(features_df.columns) - 1} dimensions")
    print(f"total files: {len(features_df)}")
    
    error_count = 'error' in features_df.columns and features_df['error'].notna().sum()
    if error_count and error_count > 0:
        print(f"warning: {error_count} files failed to process")
    
    return features_df

def main():
    audio_dir = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\MERGE_Bimodal_Complete\audio"
    output_dir = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\processed_data"
    output_file = os.path.join(output_dir, "audio_features.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    max_workers = 3  
    batch_size = None
    
    process_audio_batch(audio_dir, output_file, max_workers, batch_size)

if __name__ == "__main__":
    main()
