"""
Train and Save Model - Video QoE Prediction Assignment

This script provides a basic template for training models.
You should modify this to improve your predictions.

SUBMISSION FILES:
- feature_extractor.py  (your feature extraction code)
- models.pkl            (dictionary of trained models)
- scalers.pkl           (dictionary of feature scalers, optional)
"""

import pandas as pd
import pickle
from pathlib import Path

# Import your feature extractors
from feature_extractor import (
    extract_features_resolution,
    extract_features_rebuffering,
    extract_features_startup,
    extract_features_switches,
)


# Configuration
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'

# QoE targets and their corresponding feature extractors
QOE_CONFIG = {
    'avg_resolution': extract_features_resolution,
    'rebuffering_ratio': extract_features_rebuffering,
    'startup_latency': extract_features_startup,
    'bitrate_switches_per_second': extract_features_switches,
}


def main():
    # Load labels and session list
    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv')
    with open(DATA_DIR / 'train_sessions.txt', 'r') as f:
        session_ids = [line.strip() for line in f if line.strip()]

    print(f"Training on {len(session_ids)} sessions...")

    models = {}
    scalers = {}

    for target, extractor in QOE_CONFIG.items():
        print(f"\n=== {target} ===")

        # Extract features
        all_features = []
        for session_id in session_ids:
            video_path = TRAIN_DIR / session_id / 'video_traffic.csv'
            if video_path.exists():
                try:
                    feats = extractor(str(video_path))
                    feats['session_id'] = session_id
                    all_features.append(feats)
                except:
                    pass

        features_df = pd.DataFrame(all_features)
        feat_cols = [c for c in features_df.columns if c != 'session_id']

        # Merge with labels
        data = features_df.merge(train_labels[['session_id', target]], on='session_id')
        X = data[feat_cols].values
        y = data[target].values

        print(f"  Features: {len(feat_cols)}, Samples: {len(X)}")

        # TODO: Add your model training code here
        # - Feature scaling/normalization
        # - Model selection
        # - Hyperparameter tuning
        # - Cross-validation

        # Example (replace with your own):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        models[target] = model
        # scalers[target] = your_scaler  # if you use scaling

    # Save models
    with open('models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("\nSaved models.pkl")

    # Save scalers (if used)
    if scalers:
        with open('scalers.pkl', 'wb') as f:
            pickle.dump(scalers, f)
        print("Saved scalers.pkl")

    print("\nSubmit: feature_extractor.py, models.pkl, scalers.pkl (if used)")


if __name__ == '__main__':
    main()
