import os
import numpy as np
import pandas as pd

def va_to_single_func(arousal, valence):
    if arousal >= 0.5 and valence >= 0.5:
        quadrant = "Q1"
    elif arousal >= 0.5 and valence < 0.5:
        quadrant = "Q2"
    elif arousal < 0.5 and valence < 0.5:
        quadrant = "Q3"
    else:  # arousal < 0.5 and valence >= 0.5
        quadrant = "Q4"
    
    if quadrant == "Q1":
        if arousal > valence:
            emotion = "Excited"
        else:
            emotion = "Happy"
    
    elif quadrant == "Q2":
        if valence < 0.25:
            emotion = "Nervous"
        else:
            emotion = "Intense"
    
    elif quadrant == "Q3":
        if valence < 0.25:
            emotion = "Melancholic"
        else:
            emotion = "Sad"
    
    else:  # Q4
        if arousal < 0.25:
            emotion = "Relaxed"
        else:
            emotion = "Calm"
    
    return {
        "quadrant": quadrant,
        "emotion_label": emotion
    }
    


def apply_va_to_emotion(df):
    result_df = df.copy()
    
    required_cols = ["Arousal", "Valence"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    results = df.apply(
        lambda row: va_to_single_func(row["Arousal"], row["Valence"]),
        axis=1
    )
    
    result_df["Quadrant"] = [r["quadrant"] for r in results]
    result_df["Emotion"] = [r["emotion_label"] for r in results]
    
    return result_df


def load_data(data_path):
    print(f"loading data from {data_path}...")
    
    va_file = os.path.join(data_path, "merge_bimodal_complete_av_values.csv")
    va_df = pd.read_csv(va_file)
    print(f"loaded {len(va_df)} records")
    
    metadata_file = os.path.join(data_path, "merge_bimodal_complete_metadata.csv")
    metadata_df = pd.read_csv(metadata_file)
    print(f"loaded {len(metadata_df)} records")
    
    merged_df = pd.merge(va_df, metadata_df, on=["Audio_Song", "Lyric_Song"], how="inner")
    print(f"merged data has {len(merged_df)} records")
    
    return merged_df


def process_data(df):
    print("processing va to emotion...")
    
    result_df = apply_va_to_emotion(df)
    
    print("checking and handling invalid values...")
    va_range = (0, 1)
    invalid_va = ((result_df['Arousal'] < va_range[0]) | (result_df['Arousal'] > va_range[1]) | 
                  (result_df['Valence'] < va_range[0]) | (result_df['Valence'] > va_range[1]))
    
    if invalid_va.any():
        print(f"{invalid_va.sum()} records have invalid values, cliping them to range [0,1]")
        result_df.loc[invalid_va, 'Arousal'] = result_df.loc[invalid_va, 'Arousal'].clip(*va_range)
        result_df.loc[invalid_va, 'Valence'] = result_df.loc[invalid_va, 'Valence'].clip(*va_range)
    
    return result_df


def analyze_results(df):
    print("\nemotion label distribution:")

    if 'Quadrant' in df.columns:
        quadrant_counts = df['Quadrant'].value_counts()
        total = len(df)
        print("\nquadrant distribution:")
        for quadrant, count in quadrant_counts.items():
            percentage = count / total * 100
            print(f"{quadrant}: {count} records ({percentage:.2f}%)")
    
    if 'Emotion' in df.columns:
        emotion_counts = df['Emotion'].value_counts()
        print("\n   emotion label distribution:")
        for emotion, count in emotion_counts.items():
            percentage = count / total * 100
            print(f"{emotion}: {count} records ({percentage:.2f}%)")


def save_results(df, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, "merge_complete_with_single_emotion.csv")
    df.to_csv(output_file, index=False)
    print(f"\ncomplete data saved to: {output_file}")
    
    simplified_df = df[['Audio_Song', 'Lyric_Song', 'Arousal', 'Valence', 'Quadrant', 'Emotion']]
    simplified_output = os.path.join(output_path, "merge_single_emotion_simplified.csv")
    simplified_df.to_csv(simplified_output, index=False)
    print(f"simplified data saved to: {simplified_output}")
    
    if 'Quadrant' in df.columns:
        for quadrant in df['Quadrant'].unique():
            quadrant_df = df[df['Quadrant'] == quadrant]
            quadrant_file = os.path.join(output_path, f"merge_single_{quadrant}_samples.csv")
            quadrant_df.to_csv(quadrant_file, index=False)
            print(f"quadrant: {quadrant} has {len(quadrant_df)} records saved to: {quadrant_file}")


def main():
    data_path = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\MERGE_Bimodal_Complete"
    output_path = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\processed_data"
    
    print("processing ...")
    
    df = load_data(data_path)
    processed_df = process_data(df)
    analyze_results(processed_df)
    save_results(processed_df, output_path)
    
    print("\nprocessing completed!")


if __name__ == "__main__":
    main()
