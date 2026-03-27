"""
Local Evaluation Script

Use this to test your submission locally before uploading to Gradescope.
This uses a portion of your training data as a validation set.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from feature_extractor import (
    extract_features_resolution,
    extract_features_rebuffering,
    extract_features_startup,
    extract_features_switches,
)

DATA_DIR = Path('student_data')
TRAIN_DIR = DATA_DIR / 'train'

QOE_TARGETS = [
    'avg_resolution',
    'rebuffering_ratio',
    'startup_latency',
    'bitrate_switches_per_second'
]

TARGET_EXTRACTORS = {
    'avg_resolution': extract_features_resolution,
    'rebuffering_ratio': extract_features_rebuffering,
    'startup_latency': extract_features_startup,
    'bitrate_switches_per_second': extract_features_switches,
}

TARGET_WEIGHTS = {
    'avg_resolution': 0.25,
    'rebuffering_ratio': 0.30,
    'startup_latency': 0.25,
    'bitrate_switches_per_second': 0.20
}

NORMALIZATION_FACTORS = {
    'avg_resolution': 500.0,
    'rebuffering_ratio': 0.5,
    'startup_latency': 30.0,
    'bitrate_switches_per_second': 0.1
}


def main():
    print("=" * 60)
    print("Local Evaluation (using validation split)")
    print("=" * 60)

    # Load models and scalers
    print("\nLoading models.pkl...")
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)

    scalers = {}
    try:
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        print("Loaded scalers.pkl")
    except FileNotFoundError:
        print("No scalers.pkl found")

    # Load labels and create validation split
    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv')
    _, val_labels = train_test_split(train_labels, test_size=0.2, random_state=42)

    print(f"\nEvaluating on {len(val_labels)} validation sessions...")

    results = {}

    for target in QOE_TARGETS:
        extractor = TARGET_EXTRACTORS[target]
        y_true = []
        y_pred = []

        for _, row in val_labels.iterrows():
            session_id = row['session_id']
            video_path = TRAIN_DIR / session_id / 'video_traffic.csv'

            if not video_path.exists():
                continue

            try:
                features = extractor(str(video_path))
                feature_names = sorted(features.keys())
                X = np.array([[features.get(k, 0.0) for k in feature_names]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                if target in scalers:
                    X = scalers[target].transform(X)

                pred = models[target].predict(X)[0]
                y_pred.append(pred)
                y_true.append(row[target])
            except:
                pass

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

        norm_rmse = rmse / NORMALIZATION_FACTORS[target]
        score = max(0.0, 1.0 - norm_rmse) * TARGET_WEIGHTS[target] * 100

        results[target] = {'rmse': rmse, 'r2': r2, 'score': score}

    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    total_score = 0
    for target in QOE_TARGETS:
        r = results[target]
        print(f"\n{target}:")
        print(f"  RMSE: {r['rmse']:.4f}, R²: {r['r2']:.4f}")
        print(f"  Score: {r['score']:.2f} / {TARGET_WEIGHTS[target] * 100:.0f}")
        total_score += r['score']

    print("\n" + "-" * 60)
    print(f"TOTAL SCORE: {total_score:.2f} / 100")
    print("=" * 60)


if __name__ == '__main__':
    main()
