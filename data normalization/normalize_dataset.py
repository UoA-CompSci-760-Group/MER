import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_normalized_dataset(input_file, output_file):

    print(f"\nloading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"loaded {len(df)} rows")

    id_cols = ['Audio_Song', 'Lyric_Song']
    label_cols = ['Arousal', 'Valence', 'Quadrant', 'Emotion']
    feature_cols = [col for col in df.columns if col not in id_cols + label_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"rows in the ID column: {len(id_cols)}")
    print(f"rows in the label column: {len(label_cols)}")
    print(f"rows in the feature column: {len(feature_cols)}")
    
    print("standardizing features...")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"standardized dataset saved to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("\nstandardized feature statistics:")
    for col in feature_cols[:5]:
        mean = df[col].mean()
        std = df[col].std()
        print(f"{col}: mean={mean:.6f}, std={std:.6f}")
    
    if len(feature_cols) > 5:
        print(f"{len(feature_cols) - 5} features not displayed")
    
    print(f"\nstandardized dataset saved to: {output_file}")
    
    return df

def main():
    data_dir = r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\processed_data"
    input_file = os.path.join(data_dir, "multimodal_dataset.csv")
    output_file = os.path.join(data_dir, "multimodal_dataset_normalized.csv")
    
    print("\nusing new VA to emotion model to update emotion labels...")
    
    normalized_df = create_normalized_dataset(input_file, output_file)
    
    try:
        import sys
        sys.path.append(r"D:\DESKTOP\auckland uni\CompSci 760\msc_emo_pred\datasets\convert VA 2 single")
        import va_to_emotion
        
        print("\nusing new VA to emotion model to update emotion labels...")
        updated_df = update_emotion_labels(normalized_df, va_to_emotion)
        
        updated_output = os.path.join(data_dir, "multimodal_dataset_normalized_updated.csv")
        updated_df.to_csv(updated_output, index=False)
        print(f"updated dataset: {updated_output}")
        
        print("\ndistribution before update:")
        before_counts = normalized_df['Emotion'].value_counts()
        after_counts = updated_df['Emotion'].value_counts()
        
        print("\ndistribution after update:")
        for emotion, count in before_counts.items():
            print(f"  {emotion}: {count} ({count/len(normalized_df)*100:.2f}%)")
        
        print("\ndistribution after update:")
        for emotion, count in after_counts.items():
            print(f"  {emotion}: {count} ({count/len(updated_df)*100:.2f}%)")
        
    except ImportError:
        print("\nVA to emotion module not found, skipping emotion label update")
    
    print("\nprocess completed!")

if __name__ == "__main__":
    main()
