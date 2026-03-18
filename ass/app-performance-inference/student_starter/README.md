# Video QoE Prediction Assignment

## Overview

Build a machine learning model that predicts video streaming Quality of Experience (QoE) from network traffic measurements.

## Dataset

You are provided with **training data only**. The test data is hidden and will be used by the autograder to evaluate your submission.

### Directory Structure

```
data/
├── train/                    # Training sessions
│   ├── train_00000/
│   │   └── video_traffic.csv
│   ├── train_00001/
│   │   └── video_traffic.csv
│   └── ...
├── train_labels.csv          # QoE labels for training
└── train_sessions.txt        # List of training session IDs
```

The training set contains about 7,800 labeled sessions (see `student_data/train_labels.csv`). Each session is stored as a folder under `data/train/` containing a `video_traffic.csv` file with raw packet-level measurements for that session. Use `train_sessions.txt` to enumerate which session folders to process.

The `train_labels.csv` file has the following schema (header shown):

```
session_id,avg_resolution,rebuffering_ratio,startup_latency,bitrate_switches_per_second
```

All labels are numeric: `avg_resolution` is in pixels, `rebuffering_ratio` is in [0,1], and the other targets are non-negative floats. When writing extractors, ensure feature values are numeric and handle empty sessions or missing packets gracefully.


### Network Traffic Format (video_traffic.csv)

| Column | Description |
|--------|-------------|
| `timestamp` | Unix timestamp of the packet |
| `ipSrc`, `ipDst` | Source/destination IP addresses |
| `tcpPortSrc`, `tcpPortDst` | TCP ports (empty if UDP) |
| `udpPortSrc`, `udpPortDst` | UDP ports (empty if TCP) |
| `tcpLen`, `udpLen` | Payload length in bytes |
| `payloadProtocolNumber` | 6 for TCP, 17 for UDP |

### QoE Targets

| Metric | Description | Range |
|--------|-------------|-------|
| `avg_resolution` | Average resolution (pixels) | 144-2160 |
| `rebuffering_ratio` | Fraction of time buffering | 0-1 |
| `startup_latency` | Seconds to start playback | 0-inf |
| `bitrate_switches_per_second` | Quality changes per second | 0-inf |

## Submission Instructions

Train and submit a separate model for each QoE metric (per-feature / per-metric models). The provided `train_model.py` trains one model per metric by default: extract metric-specific features and fit a regressor for each target in turn.

Required submission files
- `feature_extractor.py` — must include metric-specific extractors (see below). Each extractor should return a dict of features; when used by the provided trainer the extractor's output will be merged on `session_id`.
- `models.pkl` — pickled dictionary mapping metric names to trained model objects, e.g. `{"avg_resolution": model_res, ...}`. The autograder will load this and use each model to predict its corresponding metric.
- `scalers.pkl` (optional) — pickled dictionary mapping metric names to scalers/transformers if you used them during training.

Notes
- The autograder expects per-metric model files and will validate that the submission CSV produced by those models contains the required columns and no missing values. Combined single-model submissions (one model that outputs all metrics) are not required and are not the expected grading format; prefer per-metric models.
- Make sure your `feature_extractor.py` functions are deterministic and robust to sessions with few or no packets.

## Feature Extraction

The `feature_extractor.py` file provides **separate functions** for each QoE metric:

```python
# For predicting average resolution
def extract_features_resolution(video_traffic_path: str) -> Dict[str, float]:
    ...

# For predicting rebuffering ratio
def extract_features_rebuffering(video_traffic_path: str) -> Dict[str, float]:
    ...

# For predicting startup latency
def extract_features_startup(video_traffic_path: str) -> Dict[str, float]:
    ...

# For predicting bitrate switches
def extract_features_switches(video_traffic_path: str) -> Dict[str, float]:
    ...



This design lets you use different features for different metrics!

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Implement Feature Extraction

Edit `feature_extractor.py` to add features. Start with the metric-specific functions:

```python
def extract_features_resolution(video_traffic_path: str) -> Dict[str, float]:
    df = pd.read_csv(video_traffic_path)
    features = {}

    # Resolution correlates with throughput
    df['packet_size'] = df['tcpLen'].fillna(0) + df['udpLen'].fillna(0)
    duration = df['timestamp'].max() - df['timestamp'].min()
    features['throughput'] = df['packet_size'].sum() / max(duration, 0.001)

    # Add more features...

    return features
```

### 3. Train Models

Run the trainer to build one model per QoE metric (default behavior of the starter `train_model.py`):

```bash
python train_model.py
```

### 4. Test Locally

```python
from feature_extractor import extract_features

features = extract_features('data/train/train_00000/video_traffic.csv')
print(features)
```

### 5. Submit to Gradescope

Upload the required files: `feature_extractor.py`, `models.pkl`, and (optionally) `scalers.pkl`.

## Feature Ideas by Metric

### avg_resolution
- Overall throughput (higher = higher resolution)
- Sustained bandwidth percentiles
- Large packet ratio

### rebuffering_ratio
- Idle periods (gaps > 1 second)
- Throughput drops
- Low bandwidth duration

### startup_latency
- Time to first large burst
- Bytes in first N seconds
- Initial throughput ramp-up

### bitrate_switches_per_second
- Throughput coefficient of variation
- Number of throughput changes
- Bandwidth stability metrics

## Scoring

Overall grade composition:

- Demo: **60%** 
- Relative evaluation (autograder): **40%**

The relative evaluation (the 40% autograded portion) is computed as a weighted
combination of the four QoE metrics.

**Note**: Evaluation uses a **random subset** of test data that changes each submission.

## Tips

- Start simple, iterate
- Use cross-validation
- Different features matter for different metrics
- Handle edge cases (empty sessions, NaN values)
- Test locally before submitting

Good luck!
