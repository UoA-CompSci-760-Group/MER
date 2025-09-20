import os
import re
import pandas as pd
import numpy as np
import glob
from collections import Counter
import time

POSITIVE_WORDS = {
    'love', 'happy', 'joy', 'good', 'beautiful', 'nice', 'best', 'better', 
    'great', 'fun', 'sweet', 'warm', 'bright', 'smile', 'laugh', 'dream', 
    'hope', 'light', 'perfect', 'gentle', 'peace'
}

NEGATIVE_WORDS = {
    'hate', 'sad', 'bad', 'worst', 'pain', 'cry', 'hurt', 'lonely', 'sorry', 
    'fear', 'dark', 'angry', 'lost', 'die', 'death', 'broken', 'cold', 
    'afraid', 'tears', 'worry', 'hard'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_basic_features(text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    word_count = len(words)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_basic_features(text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            'word_count': 0,
            'unique_word_count': 0,
            'lexical_diversity': 0,
            'avg_word_length': 0,
            'line_count': 0,
            'avg_words_per_line': 0,
            'positive_word_ratio': 0,
            'negative_word_ratio': 0,
            'sentiment_ratio': 0
        }
    
    unique_words = set(words)
    unique_word_count = len(unique_words)
    lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
    avg_word_length = sum(len(w) for w in words) / word_count
    
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    line_count = len(non_empty_lines)
    avg_words_per_line = word_count / line_count if line_count > 0 else 0
    
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    positive_word_ratio = positive_count / word_count
    negative_word_ratio = negative_count / word_count
    sentiment_ratio = positive_count / (negative_count + 1)
    
    return {
        'word_count': word_count,
        'unique_word_count': unique_word_count,
        'lexical_diversity': lexical_diversity,
        'avg_word_length': avg_word_length,
        'line_count': line_count,
        'avg_words_per_line': avg_words_per_line,
        'positive_word_ratio': positive_word_ratio,
        'negative_word_ratio': negative_word_ratio,
        'sentiment_ratio': sentiment_ratio
    }

def extract_top_words(text, top_n=5):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    common_stopwords = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'my', 'your', 
                        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
    filtered_words = [w for w in words if w not in common_stopwords and len(w) > 1]
    
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    
    result = {}
    for i, (word, count) in enumerate(top_words):
        result[f'top_word_{i+1}'] = word
        result[f'top_word_{i+1}_count'] = count
    
    for i in range(len(top_words), top_n):
        result[f'top_word_{i+1}'] = ''
        result[f'top_word_{i+1}_count'] = 0
        
    return result

def extract_repetition_features(text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    if len(words) == 0:
        return {
            'repetition_ratio': 0,
            'repeated_words_count': 0,
            'repeated_phrases_count': 0
        }
    
    word_counts = Counter(words)
    repeated_words = [word for word, count in word_counts.items() if count > 1]
    repeated_words_count = len(repeated_words)
    
    repeated_phrases_count = 0
    lines = text.lower().split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    line_counts = Counter(non_empty_lines)
    for line, count in line_counts.items():
        if count > 1 and len(line.split()) > 2:
            repeated_phrases_count += 1
    
    repetition_ratio = sum(count - 1 for count in word_counts.values()) / len(words) if len(words) > 0 else 0
    
    return {
        'repetition_ratio': repetition_ratio,
        'repeated_words_count': repeated_words_count,
        'repeated_phrases_count': repeated_phrases_count
    }

def process_lyrics_file(file_path):
    file_name = os.path.basename(file_path)
    lyric_id = os.path.splitext(file_name)[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lyrics = f.read()
        
        basic_features = extract_basic_features(lyrics)
        top_words = extract_top_words(lyrics)
        repetition_features = extract_repetition_features(lyrics)
        
        features = {'lyric_id': lyric_id}
        features.update(basic_features)
        features.update(top_words)
        features.update(repetition_features)
        
        return features
    except Exception as e:
        print(f"error processing file {file_path}: {str(e)}")
        return {'lyric_id': lyric_id, 'error': str(e)}

def process_lyrics_batch(lyrics_dir, output_file, max_files=None):
    all_lyric_files = []
    for quadrant_dir in ['Q1', 'Q2', 'Q3', 'Q4']:
        quadrant_path = os.path.join(lyrics_dir, quadrant_dir)
        if os.path.exists(quadrant_path):
            lyric_files = glob.glob(os.path.join(quadrant_path, "*.txt"))
            all_lyric_files.extend(lyric_files)
    
    if max_files and max_files > 0:
        all_lyric_files = all_lyric_files[:max_files]
    
    print(f"processing {len(all_lyric_files)} lyric files...")
    start_time = time.time()
    
    all_features = []
    for i, file_path in enumerate(all_lyric_files):
        if i % 100 == 0:
            print(f"processing {i}/{len(all_lyric_files)} files...")
        
        features = process_lyrics_file(file_path)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"processing completed! time taken: {duration:.2f} seconds")
    print(f"features saved to: {output_file}")
    
    return features_df

if __name__ == "__main__":
    lyrics_dir = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\MERGE_Bimodal_Complete\lyrics"
    output_dir = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\processed_data"
    output_file = os.path.join(output_dir, "lyrics_features.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    process_lyrics_batch(lyrics_dir, output_file)
