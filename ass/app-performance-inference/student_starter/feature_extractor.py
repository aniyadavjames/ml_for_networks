"""
Feature Extractor for Video QoE Prediction

This module contains feature extraction functions for predicting video QoE metrics.
You can implement SEPARATE feature extractors for each QoE metric, allowing you to
design specialized features for each prediction task.

The video_traffic.csv file contains packet-level information:
- timestamp: Unix timestamp of the packet
- ipSrc, ipDst: Source and destination IP addresses
- tcpPortSrc, tcpPortDst: TCP ports (empty if UDP)
- udpPortSrc, udpPortDst: UDP ports (empty if TCP)
- tcpLen, udpLen: Payload length for TCP/UDP packets
- payloadProtocolNumber: 6 for TCP, 17 for UDP

QoE Metrics to Predict:
- avg_resolution: Average video resolution in pixels (144-2160)
- rebuffering_ratio: Fraction of time spent rebuffering (0-1)
- startup_latency: Seconds until playback starts (0+)
- bitrate_switches_per_second: Quality changes per second (0+)

USAGE:
- Implement separate feature extractors for each metric (recommended)
- Or use a single shared extractor via extract_features()
- The autograder will call extract_features() which combines all extractors
"""

import pandas as pd
import numpy as np
from typing import Dict



# =============================================================================
# METRIC-SPECIFIC FEATURE EXTRACTORS
# Implement specialized features for each QoE metric below
# =============================================================================

def extract_features_resolution(video_traffic_path: str) -> Dict[str, float]:
    """
    Extract features for predicting AVERAGE RESOLUTION.

    Resolution is typically correlated with:
    - Overall throughput (higher throughput = higher resolution possible)
    - Sustained bandwidth over time
    - Packet sizes (larger packets often indicate higher quality video)

    Args:
        video_traffic_path: Path to the video_traffic.csv file

    Returns:
        Dictionary mapping feature names to float values

    """
    df = pd.read_csv(video_traffic_path)
    print(f"Loaded {len(df)} packets for resolution feature extraction.")
    print(f"Columns: {df.columns.tolist()}")

    features = {}

    # === TODO: Implement your features for resolution prediction ===


    #converting to only one source and destination port column
    df['portSrc'] = df['tcpPortSrc'].fillna(df['udpPortSrc']) 
    df['portDst'] = df['tcpPortDst'].fillna(df['udpPortDst']) 

    #get student and youtuber ip address
    student_ip = df[df['portDst'] == 443]['ipSrc'].unique()
    youtube_ip = df[df['portSrc'] == 443]['ipSrc'].unique()

    print(f"Student IP: {student_ip}")
    print(f"YouTuber IP: {youtube_ip}")


        #filtering only downloadpacket
    download_df = df[df['ipDst'] == student_ip[0]] 



    # Resolution correlates with throughput
    download_df['packet_size'] = download_df['tcpLen'].fillna(0) + download_df['udpLen'].fillna(0)
    duration = download_df['timestamp'].max() - download_df['timestamp'].min()
    features['throughput'] = download_df['packet_size'].sum() / max(duration, 0.001)
    features['avg_packet_size'] = download_df['packet_size'].mean()
    features['std_packet_size'] = download_df['packet_size'].std()
    features['num_packets'] = len(download_df)



    #calculating unique packet sizes and frequencies
    unique_sizes = download_df['packet_size'].unique()
    features['unique_packet_sizes'] = len(unique_sizes)

    #creating classes of pkts
   
    avg_size = download_df['packet_size'].mean()
    std_size = download_df['packet_size'].std()
    max_size = download_df['packet_size'].max()
    min_size = download_df['packet_size'].min()

    
    class_names = ['tiny_sz_pkt', 'small_sz_pkt', 'average_sz_pkt', 'medium_large_sz_pkt', 'large_sz_pkt', 'Jumbo_sz_pkt']

    # bins
    bins = [min_size - 1, avg_size - std_size, avg_size, avg_size + 0.5*std_size, avg_size + std_size, avg_size + 2*std_size, max_size + 1]
    bins = sorted(list(set(bins)))


    download_df['packet_class'] = pd.cut(
        download_df['packet_size'], 
        bins=bins, 
        labels=class_names, 
        include_lowest=True
    )


    packet_classes=download_df['packet_class'].value_counts().sort_index()
    print(f"Packet class distribution:\n{packet_classes}")

    for class_name, count in packet_classes.items():
        features[f'{class_name}_count'] = count

    

    return features


def extract_features_rebuffering(video_traffic_path: str) -> Dict[str, float]:
    """
    Extract features for predicting REBUFFERING RATIO.

    Rebuffering is typically correlated with:
    - Throughput variability and drops
    - Idle periods (gaps in packet arrivals)
    - Buffer depletion patterns
    - Periods of low bandwidth

    Args:
        video_traffic_path: Path to the video_traffic.csv file

    Returns:
        Dictionary mapping feature names to float values
    """
    df = pd.read_csv(video_traffic_path)
    features = {}

    # === TODO: Implement your features for rebuffering prediction ===


    return features


def extract_features_startup(video_traffic_path: str) -> Dict[str, float]:
    """
    Extract features for predicting STARTUP LATENCY.

    Startup latency is typically correlated with:
    - Time to receive initial data burst
    - Early packet timing patterns
    - Initial throughput ramp-up
    - Time to fill initial buffer

    Args:
        video_traffic_path: Path to the video_traffic.csv file

    Returns:
        Dictionary mapping feature names to float values
    """
    df = pd.read_csv(video_traffic_path)
    features = {}

    # === TODO: Implement your features for startup latency prediction ===


    return features


def extract_features_switches(video_traffic_path: str) -> Dict[str, float]:
    """
    Extract features for predicting BITRATE SWITCHES PER SECOND.

    Bitrate switches are typically correlated with:
    - Throughput variability over time
    - Coefficient of variation of bandwidth
    - Frequency of throughput changes
    - Network instability

    Args:
        video_traffic_path: Path to the video_traffic.csv file

    Returns:
        Dictionary mapping feature names to float values
    """
    df = pd.read_csv(video_traffic_path)
    features = {}

    # === TODO: Implement your features for bitrate switch prediction ===


    return features


# =============================================================================
# MAIN FEATURE EXTRACTOR
# This function combines all metric-specific extractors
# =============================================================================

def extract_features(video_traffic_path: str) -> Dict[str, float]:
    """
    Extract ALL features for a session (combines all metric-specific extractors).

    This function is called by the autograder. It combines features from all
    metric-specific extractors into a single feature dictionary.

    Args:
        video_traffic_path: Path to the video_traffic.csv file

    Returns:
        Dictionary mapping feature names to float values
    """
    features = {}

    # Combine features from all extractors with prefixes to avoid name collisions
    for prefix, extractor in [
        ('res', extract_features_resolution),
        ('rebuf', extract_features_rebuffering),
        ('start', extract_features_startup),
        ('switch', extract_features_switches),
    ]:
        try:
            print(f"Running {extractor.__name__} extractor...")
            metric_features = extractor(video_traffic_path)
            print(f"Extracted {len(metric_features)} features for {prefix} metric.")
            for name, value in metric_features.items():
                features[f'{prefix}_{name}'] = value
        except Exception as e:
            print(f"Warning: {prefix} extractor failed: {e}")

    return features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_features_for_all_sessions(
    data_dir: str,
    sessions_file: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Extract features for all sessions listed in sessions_file.

    Args:
        data_dir: Directory containing session subdirectories
        sessions_file: Path to file listing session IDs (one per line)
        output_path: Optional path to save features CSV

    Returns:
        DataFrame with session_id and extracted features
    """
    from pathlib import Path

    data_path = Path(data_dir)

    # Read session IDs
    with open(sessions_file, 'r') as f:
        session_ids = [line.strip() for line in f if line.strip()]

    all_features = []

    print(f"Extracting features for {len(session_ids)} sessions...")

    for i, session_id in enumerate(session_ids):
        video_traffic_path = data_path / session_id / 'video_traffic.csv'

        if not video_traffic_path.exists():
            print(f"Warning: {video_traffic_path} not found, skipping...")
            continue

        try:
            features = extract_features(str(video_traffic_path))
            features['session_id'] = session_id
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {session_id}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(session_ids)} sessions...")

    df = pd.DataFrame(all_features)

    # Move session_id to first column
    cols = ['session_id'] + [c for c in df.columns if c != 'session_id']
    df = df[cols]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved features to {output_path}")

    return df


if __name__ == '__main__':
    # Test on a single session
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = 'student_data/train/train_00000/video_traffic.csv'

    print(f"Testing feature extraction on: {test_path}")
    print("=" * 60)

    try:
        features = extract_features(test_path)
        print(f"\nExtracted {len(features)} features:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value:.4f}")
    except FileNotFoundError:
        print(f"File not found: {test_path}")
        print("Usage: python feature_extractor.py <path_to_video_traffic.csv>")
