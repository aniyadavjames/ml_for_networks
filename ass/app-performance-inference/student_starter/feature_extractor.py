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

def compute_rms(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else 0
def safe_div(a, b):
    return a / (b + 1e-6)

def preprocess(video_traffic_path):
    df = pd.read_csv(video_traffic_path)

    df['portSrc'] = df['tcpPortSrc'].fillna(df['udpPortSrc'])
    df['portDst'] = df['tcpPortDst'].fillna(df['udpPortDst'])

    student_ip = df[df['portDst'] == 443]['ipSrc'].unique()
    youtube_ip = df[df['portSrc'] == 443]['ipDst'].unique() 

    if len(student_ip) == 0 or len(youtube_ip) == 0:
        return None, None

    student_ip = student_ip[0]
    youtube_ip = youtube_ip[0]

    download_df = df[df['ipDst'] == student_ip].copy()
    upload_df = df[df['ipSrc'] == student_ip].copy()
    trace_start = df['timestamp'].min()
    
    if len(download_df) == 0:
        return None, None

    download_df['packet_size'] = download_df['tcpLen'].fillna(0) + download_df['udpLen'].fillna(0)

    t0 = download_df['timestamp'].min()
    download_df['time_bin'] = ((download_df['timestamp'] - t0)).astype(int)

    return download_df, upload_df,trace_start


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
    # print(f"Loaded {len(df)} packets for resolution feature extraction.")
    # print(f"Columns: {df.columns.tolist()}")

    features = {}
    

    # === TODO: Implement your features for resolution prediction ===

    download_df,upload_df,trace_start = preprocess(video_traffic_path)
    if download_df is None:
        return {}
    
    duration = download_df['timestamp'].max() - download_df['timestamp'].min()
    duration = max(duration, 0.1)

    
    download_df['packet_size'] = download_df['tcpLen'].fillna(0) + download_df['udpLen'].fillna(0)
    features['throughput'] = download_df['packet_size'].sum() / duration
    features['up_ds_ratio'] = len(upload_df) / max(len(download_df), 1)
    features['avg_packet_size'] = download_df['packet_size'].mean()

    
    download_df['time_bin'] = (download_df['timestamp'] - download_df['timestamp'].min()).astype(int)
    bin_stats = download_df.groupby('time_bin')['packet_size'].sum()
    
    features['p90_throughput'] = bin_stats.quantile(0.9) if not bin_stats.empty else 0
    features['throughput_var'] = bin_stats.std() / (bin_stats.mean() + 1e-6) # Coeff of Variation

    
    active_bins = (bin_stats > (bin_stats.mean() * 0.5)).sum()
    features['burst_duty_cycle'] = active_bins / len(bin_stats) if len(bin_stats) > 0 else 0

    
    if len(download_df) > 1:
        download_df['iat'] = download_df['timestamp'].diff()
        features['iat_jitter'] = download_df['iat'].std()
        features['max_gap'] = download_df['iat'].max()
    else:
        features['iat_jitter'] = 0
        features['max_gap'] = 0
        
        
        
    bin_tp = download_df.groupby('time_bin')['packet_size'].sum()

    features['tp_mean'] = bin_tp.mean()
    features['tp_p50'] = bin_tp.quantile(0.5)
    features['tp_p90'] = bin_tp.quantile(0.9)

    features['tp_rms'] = compute_rms(bin_tp.values)
    features['tp_cv'] = safe_div(bin_tp.std(), bin_tp.mean())

    features['burstiness'] = safe_div(bin_tp.max() - bin_tp.min(), bin_tp.mean())

    features['stable_tp_ratio'] = (bin_tp > bin_tp.mean()*0.7).sum() / max(len(bin_tp),1)

    threshold = bin_tp.mean() * 0.3
    features['low_tp_ratio'] = (bin_tp < threshold).sum() / max(len(bin_tp),1)
    
    

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
    # === TODO: Implement your features for rebuffering prediction ===
    features={}

    download_df,upload_df,trace_start = preprocess(video_traffic_path)
    if download_df is None:
        return {}

    
    duration = download_df['timestamp'].max() - download_df['timestamp'].min()
    duration = max(duration, 0.1)
    
    
    bin_tp = download_df.groupby('time_bin')['packet_size'].sum()

    # Throughput stability
    features['tp_rms'] = compute_rms(bin_tp.values)
    features['tp_min'] = bin_tp.min()
    features['tp_p10'] = bin_tp.quantile(0.1)

    threshold = bin_tp.mean() * 0.4
    features['low_tp_ratio'] = (bin_tp < threshold).sum() / max(len(bin_tp),1)

    # Throughput variation
    features['tp_diff_rms'] = compute_rms(np.diff(bin_tp.values)) if len(bin_tp) > 1 else 0

    # GAP FEATURES (CRITICAL)
    gaps = download_df['timestamp'].diff().dropna()

    features['gap_rms'] = compute_rms(gaps)
    features['gap_cv'] = safe_div(gaps.std(), gaps.mean())
    features['long_gap_ratio'] = (gaps > 1.0).sum() / max(len(gaps),1)

    # BUFFER PROXY (VERY POWERFUL)
    features['buffer_proxy'] = safe_div(bin_tp.mean(), features['gap_rms'])

    return features





    # 2. GAP ANALYSIS (The "Stall" Detector)
    gaps = download_df['timestamp'].diff().dropna()
    features['max_gap'] = gaps.max() if not gaps.empty else 0
    # Count "critical" gaps where the player likely ran out of buffer
    features['stall_freq_long'] = (gaps > 2.0).sum() 
    features['stall_freq_med'] = (gaps > 0.5).sum()

    # 3. ROLLING THROUGHPUT (Local Starvation)
    # Instead of global bins, use a rolling 2-second window to find the "worst" moment
    download_df['cum_bytes'] = download_df['packet_size'].cumsum()
    # Set index to timestamp to use rolling time windows
    download_df.index = pd.to_datetime(download_df['timestamp'], unit='s')
    rolling_tp = download_df['packet_size'].rolling('2s').sum()
    
    features['min_rolling_tp'] = rolling_tp.min()
    features['avg_rolling_tp'] = rolling_tp.mean()
    # If the worst 2 seconds is way lower than the average, it's a rebuffer
    features['tp_drop_severity'] = rolling_tp.min() / (rolling_tp.mean() + 1e-6)

    # 4. VOLATILITY
    features['cv_tp'] = rolling_tp.std() / (rolling_tp.mean() + 1e-6)

    # 5. RE-FILL SPEED
    # After a gap, how fast does the data come back? (Slow recovery = longer rebuffer)
    features['throughput'] = download_df['packet_size'].sum() / duration

    


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
    
    download_df,upload_df,trace_start = preprocess(video_traffic_path)
    features = {}
    if download_df is None:
        return {}


    download_df['cum_bytes'] = download_df['packet_size'].cumsum()

    
    start_time = download_df['timestamp'].min()
    
    features = {}

    # --- COMPLEX FEATURE 1: Time to X Megabytes ---
    # The most direct proxy for startup latency: how long to get the first 1MB or 2MB?

    target = 1 * 1024 * 1024
    reached = download_df[download_df['cum_bytes'] >= target]

    if not reached.empty:
        features['time_to_1mb'] = reached['timestamp'].iloc[0] - start_time
    else:
        features['time_to_1mb'] = download_df['timestamp'].max() - start_time

    # Initial burst (first 2 sec)
    first_2s = download_df[download_df['timestamp'] <= (start_time + 2)]

    features['initial_bytes'] = first_2s['packet_size'].sum()
    features['initial_rate'] = features['initial_bytes'] / 2.0
    features['initial_tp_rms'] = compute_rms(first_2s['packet_size'].values)

    # Delay features
    features['network_init_delay'] = start_time - trace_start
    features['first_packet_delay'] = start_time - trace_start

    # Packet timing
    if len(download_df) > 10:
        features['first_10_pkt_time'] = download_df.iloc[9]['timestamp'] - start_time
    else:
        features['first_10_pkt_time'] = 0

    return features




    
    target_bytes = 1 * 1024 * 1024  # 1 MB
    target_reached = download_df[download_df['cumulative_bytes'] >= target_bytes]
    
    if not target_reached.empty:
        features['time_to_1mb'] = target_reached['timestamp'].iloc[0] - start_time
    else:
        # If it never reached 1MB, it's a very slow session
        features['time_to_1mb'] = download_df['timestamp'].max() - start_time

    # --- COMPLEX FEATURE 2: Initial Burst Rate (First 2 Seconds) ---
    # Heavy initial throughput usually means a fast start.
    first_2s = download_df[download_df['timestamp'] <= (start_time + 2.0)]
    features['initial_burst_total'] = first_2s['packet_size'].sum()
    features['initial_burst_rate'] = features['initial_burst_total'] / 2.0

    # --- COMPLEX FEATURE 3: Transport Handshake / RTT Proxy ---
    # The time between the very first packet in the trace and the first download packet
    # captures the "network distance" or DNS/TLS handshake overhead.
    trace_start = df['timestamp'].min()
    features['network_init_delay'] = start_time - trace_start

    # --- COMPLEX FEATURE 4: Initial Packet Frequency ---
    # High frequency (low inter-arrival time) in the first 100 packets.
    if len(download_df) > 10:
        first_100 = download_df.head(100)
        features['initial_iat_avg'] = first_100['timestamp'].diff().mean()
    else:
        features['initial_iat_avg'] = 0

    return features


    features['throughput'] = download_df['packet_size'].sum() / max(duration, 0.001)
    features['avg_packet_size'] = download_df['packet_size'].mean()
    features['std_packet_size'] = download_df['packet_size'].std()
    features['num_packets'] = len(download_df)
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
    features = {}
    download_df,upload_df,trace_start = preprocess(video_traffic_path)
    if download_df is None:
        return {}
    
    
    duration = download_df['timestamp'].max() - download_df['timestamp'].min()
    duration = max(duration, 0.1)

    
    download_df['time_bin_2s'] = ((download_df['timestamp'] - download_df['timestamp'].min()) // 2)
    # bin_tp = download_df.groupby('time_bin_2s')['packet_size'].sum()


    # if len(bin_tp) > 1:
    #     tp_pct_change = bin_tp.pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    #     features['throughput_shocks'] = (tp_pct_change.abs() > 0.5).sum() / len(bin_tp)  # ✅ FIXED NORMALIZATION
    #     features['max_shock'] = tp_pct_change.abs().max()
    # else:
    #     features['throughput_shocks'] = 0
    #     features['max_shock'] = 0

    # upload_df = upload_df.sort_values(by='timestamp')

    # if len(upload_df) > 1:
    #     iat = upload_df['timestamp'].diff().dropna()
    #     features['avg_iat'] = iat.mean()
    #     features['iat_std_norm'] = iat.std() / (iat.mean() + 1e-6)
    # else:
    #     features['avg_iat'] = 0
    #     features['iat_std_norm'] = 0


    # if not bin_tp.empty:
    #     features['cv_throughput'] = bin_tp.std() / (bin_tp.mean() + 1e-6)
    # else:
    #     features['cv_throughput'] = 0


    # if len(bin_tp) > 2:
    #     diffs = np.diff(bin_tp.values)
    #     reversals = np.where(np.diff(np.sign(diffs)))[0]
    #     features['trend_reversals_rate'] = len(reversals) / len(bin_tp)  # ✅ FIXED
    # else:
    #     features['trend_reversals_rate'] = 0

    # features = {k: (0 if (np.isnan(v) or np.isinf(v)) else v) for k, v in features.items()}

    # return features



    
    
    
    upload_df_sorted = upload_df.sort_values(by='timestamp')
    upload_df_sorted['inter_arrival_time'] = upload_df_sorted['timestamp'].diff()
    

    
    duration = download_df['timestamp'].max() - download_df['timestamp'].min()
    duration = max(duration, 0.1)

    # --- COMPLEX FEATURE 1: Throughput Shocks ---
    # We bin throughput by 2-second windows and count "significant" changes
    download_df['time_bin'] = (download_df['timestamp'] - download_df['timestamp'].min()) // 2
    bin_tp = download_df.groupby('time_bin')['packet_size'].sum()
    
    
    if len(bin_tp) > 1:
        # Calculate percentage change between consecutive windows
        tp_pct_change = bin_tp.pct_change().dropna()
        tp_pct_change = bin_tp.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        # A "shock" is a change > 50% (either up or down)
        features['throughput_shocks'] = (tp_pct_change.abs() > 0.5).sum() / duration
        features['max_shock'] = tp_pct_change.abs().max()
    else:
        features['throughput_shocks'] = 0
        features['max_shock'] = 0

    # --- COMPLEX FEATURE 2: Request Frequency Stability (Your IAT logic) ---
    upload_df = upload_df.sort_values(by='timestamp')
    if len(upload_df) > 1:
        iat = upload_df['timestamp'].diff().dropna()
        features['avg_iat'] = iat.mean()
        features['iat_std_norm'] = iat.std() / (iat.mean() + 1e-6) # Normalized jitter
    else:
        features['avg_iat'] = 0
        features['iat_std_norm'] = 0

    # --- COMPLEX FEATURE 3: Coefficient of Variation (CV) ---
    # High CV = High instability = More switches
    if not bin_tp.empty:
        features['cv_throughput'] = bin_tp.std() / (bin_tp.mean() + 1e-6)
    else:
        features['cv_throughput'] = 0

    # --- COMPLEX FEATURE 4: Throughput Trend Reversals ---
    # Count how many times the throughput goes from increasing to decreasing
    if len(bin_tp) > 2:
        diffs = np.diff(bin_tp)
        # Check where the sign of the difference changes
        reversals = np.where(np.diff(np.sign(diffs)))[0]
        features['trend_reversals_rate'] = len(reversals) / duration
    else:
        features['trend_reversals_rate'] = 0
        
    bin_tp = download_df.groupby('time_bin')['packet_size'].sum()

    # Instability metrics
    features['tp_rms'] = compute_rms(bin_tp.values)
    features['tp_cv'] = safe_div(bin_tp.std(), bin_tp.mean())

    # Sudden changes
    if len(bin_tp) > 1:
        diff = np.diff(bin_tp.values)
        features['tp_diff_rms'] = compute_rms(diff)
    else:
        features['tp_diff_rms'] = 0

    # Shock detection
    if len(bin_tp) > 1:
        pct = pd.Series(bin_tp).pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        features['shock_rate'] = (pct.abs() > 0.5).sum()
        features['max_shock'] = pct.abs().max()
    else:
        features['shock_rate'] = 0
        features['max_shock'] = 0

    # Trend reversals
    if len(bin_tp) > 2:
        diffs = np.diff(bin_tp.values)
        features['sign_changes'] = np.sum(np.diff(np.sign(diffs)) != 0)
    else:
        features['sign_changes'] = 0

    return features

    
    
    
    
    
    # print(upload_df_sorted)
    features['avg_iat']=upload_df_sorted['inter_arrival_time'].mean()
    features['std_iat']=upload_df_sorted['inter_arrival_time'].std()
    
    
    
    return features




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
