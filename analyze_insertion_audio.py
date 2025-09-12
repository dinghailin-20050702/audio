#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版：音频 + 加速度插接分析（增加加速度时/频域高级特征）
- 递归扫描 recordings/ 下的 .wav（含 recordings/recordings 嵌套目录）
- 预处理：单声道、归一化、300 Hz 高通
- 事件检测：RMS + 谱通量 z-score 联合打分找峰值
- 特征提取：事件窗 vs 背景窗（FFT/谱统计/带通能量/MFCC/时域）
- 加速度分析：匹配时间戳，Welch PSD，统计效应量，复合评分
- 输出：每文件图像（波形+STFT），汇总 CSV（事件/背景/差值特征 + 加速度特征）

依赖：numpy scipy librosa soundfile matplotlib pandas python-docx
安装：pip install numpy scipy librosa soundfile matplotlib pandas python-docx
"""

import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, welch
from scipy import stats
import matplotlib.pyplot as plt

# --------------------
# 配置参数（可按需要调整）
# --------------------
SEARCH_DIRS = ["recordings"]          # 递归扫描这些目录
OUTPUT_DIR = "analysis_out"           # 输出文件夹
TARGET_SR = 44100                     # 统一采样率
HIGHPASS_CUTOFF = 300.0               # 高通截止频率(Hz)
EVENT_PRE_SEC = 0.15                  # 事件窗：峰值前时长
EVENT_POST_SEC = 0.40                 # 事件窗：峰值后时长
BG_SEC = 0.5                          # 背景窗长度（从事件前取）
MIN_EVENT_GAP_SEC = 0.2               # 峰间最小距离（帧）
PEAK_ZSCORE_TH = 3.5                  # 峰值 z-score 阈值
FIG_DPI = 140

# 加速度分析参数
ACC_FREQ_BANDS = [(0, 5), (5, 10), (10, 20), (20, 50)]  # 加速度频带分析
W_COEF = {                            # 复合评分权重
    'diff_acc_mag_rms': 0.30,
    'diff_acc_jerk_rms': 0.20,
    'cohen_d_mag': 0.25,
    'cohen_d_jerk': 0.15,
    'diff_rms_dbfs': 0.10
}

# 带通频带（用于能量比例）
BANDS = [(300, 1000), (1000, 3000), (3000, 8000)]
MFCC_N = 13

# --------------------
# 工具函数
# --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_wavs() -> list:
    files = []
    for d in SEARCH_DIRS:
        files += glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
        files += glob.glob(os.path.join(d, "**", "*.WAV"), recursive=True)
    return sorted(files)

def find_matching_accel_csv(wav_path: str) -> Optional[str]:
    """Find corresponding acceleration CSV file based on timestamp in filename"""
    wav_base = os.path.basename(wav_path)
    # Extract timestamp from filename like audio_20250911_143152.wav
    if "audio_" in wav_base:
        timestamp_part = wav_base.replace("audio_", "").replace(".wav", "").replace(".WAV", "")
        # Look for accel_data_[timestamp].csv in root directory
        accel_pattern = f"accel_data_{timestamp_part}.csv"
        accel_path = os.path.join(".", accel_pattern)
        if os.path.exists(accel_path):
            return accel_path
    return None

def load_accel_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess acceleration data"""
    try:
        df = pd.read_csv(csv_path)
        # Parse timestamp if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate magnitude and jerk
        if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            df['acc_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
            # Jerk (rate of change of acceleration)
            df['acc_jerk'] = np.sqrt(
                np.gradient(df['accel_x'])**2 + 
                np.gradient(df['accel_y'])**2 + 
                np.gradient(df['accel_z'])**2
            )
        return df
    except Exception as e:
        print(f"Error loading acceleration data {csv_path}: {e}")
        return None

def extract_accel_segment(df: pd.DataFrame, start_time: float, duration: float) -> Optional[pd.DataFrame]:
    """Extract acceleration data segment based on time offset from start"""
    if df is None or 'timestamp' not in df.columns:
        return None
    
    # Convert to relative time from first timestamp
    start_timestamp = df['timestamp'].iloc[0]
    df_rel = df.copy()
    df_rel['rel_time'] = (df_rel['timestamp'] - start_timestamp).dt.total_seconds()
    
    # Extract segment
    mask = (df_rel['rel_time'] >= start_time) & (df_rel['rel_time'] <= start_time + duration)
    segment = df_rel[mask].copy()
    
    return segment if len(segment) > 0 else None

def compute_accel_features(accel_df: pd.DataFrame) -> Dict[str, float]:
    """Compute acceleration features for a segment"""
    if accel_df is None or len(accel_df) == 0:
        return {}
    
    features = {}
    
    # Basic statistics for magnitude
    if 'acc_mag' in accel_df.columns:
        mag = accel_df['acc_mag'].values
        features.update({
            'acc_mag_mean': float(np.mean(mag)),
            'acc_mag_std': float(np.std(mag)),
            'acc_mag_rms': float(np.sqrt(np.mean(mag**2))),
            'acc_mag_peak': float(np.max(np.abs(mag))),
            'acc_mag_range': float(np.ptp(mag))
        })
        
        # PSD analysis using Welch method
        if len(mag) > 4:  # Need minimum samples for PSD
            try:
                # Estimate sampling rate from timestamps
                if len(accel_df) > 1:
                    dt = (accel_df['timestamp'].iloc[-1] - accel_df['timestamp'].iloc[0]).total_seconds() / (len(accel_df) - 1)
                    fs = 1.0 / dt if dt > 0 else 100.0  # fallback to 100 Hz
                else:
                    fs = 100.0
                
                f, psd = welch(mag, fs=fs, nperseg=min(len(mag), 16))
                
                # Band energy ratios
                for band_idx, (f_low, f_high) in enumerate(ACC_FREQ_BANDS):
                    band_mask = (f >= f_low) & (f < f_high)
                    if np.any(band_mask):
                        band_power = np.sum(psd[band_mask])
                        total_power = np.sum(psd) + 1e-12
                        features[f'acc_psd_band_{f_low}-{f_high}_ratio'] = float(band_power / total_power)
                
                # Peak frequency
                if len(psd) > 0:
                    peak_idx = np.argmax(psd)
                    features['acc_peak_freq'] = float(f[peak_idx]) if peak_idx < len(f) else 0.0
                    
            except Exception as e:
                print(f"Warning: PSD analysis failed: {e}")
    
    # Jerk features
    if 'acc_jerk' in accel_df.columns:
        jerk = accel_df['acc_jerk'].values
        features.update({
            'acc_jerk_mean': float(np.mean(jerk)),
            'acc_jerk_std': float(np.std(jerk)),
            'acc_jerk_rms': float(np.sqrt(np.mean(jerk**2))),
            'acc_jerk_peak': float(np.max(np.abs(jerk)))
        })
    
    return features

def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    if n1 == 1 and n2 == 1:
        return 0.0
    
    # Calculate pooled standard deviation
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def compute_effect_sizes(ev_accel: pd.DataFrame, bg_accel: pd.DataFrame) -> Dict[str, float]:
    """Compute statistical effect sizes between event and background acceleration"""
    effects = {}
    
    if ev_accel is None or bg_accel is None or len(ev_accel) == 0 or len(bg_accel) == 0:
        return effects
    
    # Cohen's d for magnitude and jerk
    if 'acc_mag' in ev_accel.columns and 'acc_mag' in bg_accel.columns:
        effects['cohen_d_mag'] = cohen_d(ev_accel['acc_mag'].values, bg_accel['acc_mag'].values)
        
        # t-test p-value
        try:
            _, p_val = stats.ttest_ind(ev_accel['acc_mag'].values, bg_accel['acc_mag'].values)
            effects['p_ttest_mag'] = float(p_val)
        except:
            effects['p_ttest_mag'] = 1.0
    
    if 'acc_jerk' in ev_accel.columns and 'acc_jerk' in bg_accel.columns:
        effects['cohen_d_jerk'] = cohen_d(ev_accel['acc_jerk'].values, bg_accel['acc_jerk'].values)
        
        try:
            _, p_val = stats.ttest_ind(ev_accel['acc_jerk'].values, bg_accel['acc_jerk'].values)
            effects['p_ttest_jerk'] = float(p_val)
        except:
            effects['p_ttest_jerk'] = 1.0
    
    return effects

def compute_accel_composite_score(features: Dict[str, float]) -> float:
    """Compute composite acceleration event score"""
    score = 0.0
    total_weight = 0.0
    
    for feature, weight in W_COEF.items():
        if feature in features:
            # Normalize features (absolute value for effect sizes)
            val = features[feature]
            if feature.startswith('cohen_d'):
                val = abs(val)
            elif feature.startswith('diff_'):
                val = abs(val)
            
            score += weight * val
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0

def read_wav_mono(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def butter_highpass_sos(cutoff: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    norm = cutoff / nyq
    return butter(order, norm, btype="highpass", output="sos")

def highpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0:
        return y
    sos = butter_highpass_sos(cutoff, sr)
    return sosfiltfilt(sos, y)

def frame_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    n_fft = 2048
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = librosa.feature.rms(S=S)[0]
    # 谱通量（仅正增量）
    flux = np.diff(S, axis=1)
    flux[flux < 0] = 0
    spec_flux = np.pad(flux.sum(axis=0), (1, 0), mode="edge")
    return dict(S=S, rms=rms, spec_flux=spec_flux, hop=hop, n_fft=n_fft)

def zscore_robust(x: np.ndarray) -> np.ndarray:
    mu = np.median(x)
    mad = np.median(np.abs(x - mu)) + 1e-12
    return (x - mu) / (1.4826 * mad)

def detect_event(y: np.ndarray, sr: int) -> Tuple[int, Dict[str, np.ndarray]]:
    feats = frame_features(y, sr)
    rms_z = zscore_robust(feats["rms"])
    flux_z = zscore_robust(feats["spec_flux"])
    score = 0.5 * rms_z + 0.5 * flux_z

    import scipy.signal as sps
    distance = int(MIN_EVENT_GAP_SEC * sr / feats["hop"])
    peaks, props = sps.find_peaks(score, height=PEAK_ZSCORE_TH, distance=max(distance,1))
    if len(peaks) == 0:
        peak_idx = int(np.argmax(score))
    else:
        peak_idx = int(peaks[np.argmax(props["peak_heights"])])

    event_sample = int(peak_idx * feats["hop"])
    return event_sample, dict(score=score, **feats)

def segment_bounds(center_sample: int, sr: int, pre_s: float, post_s: float, total_len: int) -> Tuple[int,int]:
    a = max(0, center_sample - int(pre_s * sr))
    b = min(total_len, center_sample + int(post_s * sr))
    return a, b

def take_bg_bounds(center_sample: int, sr: int, bg_s: float, total_len: int) -> Tuple[int,int]:
    b = max(0, center_sample - int(0.05 * sr))
    a = max(0, b - int(bg_s * sr))
    return a, b

def fft_spectrum(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    w = np.hanning(n)
    Y = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(n, 1/sr)
    mag = np.abs(Y)
    return f, mag

def band_energy(f: np.ndarray, mag: np.ndarray, f_lo: float, f_hi: float) -> float:
    m = (f >= f_lo) & (f < f_hi)
    if not np.any(m):
        return 0.0
    return float(np.sum(mag[m]**2))

def safe_db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-12))

def compute_frame_stats(y: np.ndarray, sr: int) -> Dict[str, float]:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]))
    rolloff85 = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]))
    rolloff95 = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    return dict(
        centroid_mean=centroid,
        bandwidth_mean=bandwidth,
        rolloff85_mean=rolloff85,
        rolloff95_mean=rolloff95,
        flatness_mean=flatness,
        zcr_mean=zcr,
    )

def compute_block_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    peak = float(np.max(np.abs(y)) + 1e-12)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    f, mag = fft_spectrum(y, sr)
    peak_bin = int(np.argmax(mag)) if len(mag) else 0
    peak_freq = float(f[peak_bin]) if len(f) else 0.0

    band_energies = {}
    for (lo, hi) in BANDS:
        band_energies[f"band_{lo}-{hi}_E"] = band_energy(f, mag, lo, hi)
    total_E = float(np.sum(mag**2) + 1e-12)
    band_ratios = {k.replace("_E","_ratio"): (v/total_E) for k, v in band_energies.items()}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
    mfcc_mean = {f"mfcc_{i+1}_mean": float(np.mean(mfcc[i])) for i in range(MFCC_N)}

    return {
        "duration_s": len(y)/sr,
        "peak_dbfs": safe_db(peak),
        "rms_dbfs": safe_db(rms),
        "crest_factor": float(peak / rms),
        "peak_freq": peak_freq,
        **compute_frame_stats(y, sr),
        **band_energies,
        **band_ratios,
        **mfcc_mean,
    }

def plot_file(y: np.ndarray, sr: int, event_sample: int, out_png: str, title: str):
    a, b = segment_bounds(event_sample, sr, EVENT_PRE_SEC, EVENT_POST_SEC, len(y))
    # STFT
    n_fft = 1024
    hop = 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    SdB = librosa.amplitude_to_db(S, ref=np.max)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    t_event = event_sample / sr

    plt.figure(figsize=(10,6), dpi=FIG_DPI)
    # 波形
    ax1 = plt.subplot(2,1,1)
    tt = np.arange(len(y))/sr
    ax1.plot(tt, y, lw=0.8, color='C0')
    ax1.axvline(t_event, color='C3', ls='--', lw=1.2, label='detected event')
    ax1.axvspan(a/sr, b/sr, color='C3', alpha=0.2, label='event window')
    ax1.set_title(f"{title} - waveform", fontsize=11)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 时频
    ax2 = plt.subplot(2,1,2)
    librosa.display.specshow(SdB, x_axis='time', y_axis='hz', sr=sr, hop_length=hop, cmap='magma')
    ax2.axvline(t_event, color='w', ls='--', lw=1.0)
    ax2.set_title("Spectrogram (dB)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    rows = []

    wavs = list_wavs()
    if not wavs:
        print("未在 recordings/ 下找到 .wav 文件。请确认路径。")
        return

    print(f"发现 {len(wavs)} 个音频文件")
    for path in wavs:
        try:
            y, sr = read_wav_mono(path, TARGET_SR)
            y = highpass(y, sr, HIGHPASS_CUTOFF)

            event_sample, det = detect_event(y, sr)
            ev_a, ev_b = segment_bounds(event_sample, sr, EVENT_PRE_SEC, EVENT_POST_SEC, len(y))
            bg_a, bg_b = take_bg_bounds(event_sample, sr, BG_SEC, len(y))
            y_event = y[ev_a:ev_b]
            y_bg = y[bg_a:bg_b]

            # 音频特征
            ev_feat = compute_block_features(y_event, sr)
            bg_feat = compute_block_features(y_bg, sr)

            # 差值/比值特征（凸显“和平时不同”）
            diff_feat = {}
            for k in ["centroid_mean","bandwidth_mean","rolloff85_mean","rolloff95_mean","flatness_mean",
                      "zcr_mean","peak_freq","peak_dbfs","rms_dbfs","crest_factor"]:
                diff_feat[f"diff_{k}"] = ev_feat.get(k,0.0) - bg_feat.get(k,0.0)
            for (lo,hi) in BANDS:
                key = f"band_{lo}-{hi}_ratio"
                diff_feat[f"diff_{key}"] = ev_feat.get(key,0.0) - bg_feat.get(key,0.0)

            # 加速度分析
            accel_csv = find_matching_accel_csv(path)
            accel_features = {}
            accel_diff_features = {}
            effect_features = {}
            
            if accel_csv:
                accel_df = load_accel_data(accel_csv)
                if accel_df is not None:
                    # Extract acceleration segments aligned with audio
                    event_time = event_sample / sr
                    ev_accel = extract_accel_segment(accel_df, event_time - EVENT_PRE_SEC, 
                                                   EVENT_PRE_SEC + EVENT_POST_SEC)
                    bg_accel = extract_accel_segment(accel_df, event_time - EVENT_PRE_SEC - BG_SEC, BG_SEC)
                    
                    # Compute acceleration features
                    if ev_accel is not None:
                        ev_accel_feat = compute_accel_features(ev_accel)
                        accel_features.update({f"ev_{k}": v for k, v in ev_accel_feat.items()})
                    
                    if bg_accel is not None:
                        bg_accel_feat = compute_accel_features(bg_accel)
                        accel_features.update({f"bg_{k}": v for k, v in bg_accel_feat.items()})
                    
                    # Compute differences
                    if ev_accel is not None and bg_accel is not None:
                        ev_accel_feat = compute_accel_features(ev_accel)
                        bg_accel_feat = compute_accel_features(bg_accel)
                        
                        for k in ["acc_mag_mean", "acc_mag_std", "acc_mag_rms", "acc_mag_peak", 
                                "acc_jerk_mean", "acc_jerk_std", "acc_jerk_rms", "acc_jerk_peak"]:
                            if k in ev_accel_feat and k in bg_accel_feat:
                                accel_diff_features[f"diff_{k}"] = ev_accel_feat[k] - bg_accel_feat[k]
                        
                        # Effect sizes
                        effect_features = compute_effect_sizes(ev_accel, bg_accel)
                        
                        # Add composite score
                        all_features = {**accel_diff_features, **effect_features, **diff_feat}
                        accel_features['accel_event_score'] = compute_accel_composite_score(all_features)

            # 保存图
            rel = os.path.relpath(path)
            base = os.path.splitext(os.path.basename(path))[0]
            out_png = os.path.join(OUTPUT_DIR, f"{base}.png")
            plot_file(y, sr, event_sample, out_png, title=rel)

            row = {
                "file": rel,
                "event_sample": event_sample,
                "event_t": event_sample/sr,
                **{f"ev_{k}": v for k,v in ev_feat.items()},
                **{f"bg_{k}": v for k,v in bg_feat.items()},
                **diff_feat,
                **accel_features,
                **accel_diff_features,
                **effect_features,
            }
            rows.append(row)
            print(f"完成: {rel}")
        except Exception as e:
            print(f"处理失败: {path} -> {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    # Generate markdown report
    md_path = os.path.join(OUTPUT_DIR, "report_audio_accel.md")
    generate_markdown_report(df, md_path)
    
    print(f"已输出: {csv_path}")
    print(f"已输出: {md_path}")
    print("建议后续：用 diff_* 指标做阈值选择与判定，运行 accel_detailed_word_report.py 生成详细报告。")

def generate_markdown_report(df: pd.DataFrame, output_path: str):
    """Generate markdown report with acceleration analysis"""
    lines = [
        "# 音频 + 加速度插接分析报告",
        "",
        f"## 总览",
        f"- 分析文件数: {len(df)}",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # Check for acceleration data
    accel_cols = [c for c in df.columns if c.startswith('ev_acc_') or c.startswith('cohen_d')]
    has_accel = len(accel_cols) > 0
    
    if has_accel:
        lines.extend([
            "## 加速度特征统计",
            ""
        ])
        
        # Top files by composite score
        if 'accel_event_score' in df.columns:
            top_scores = df.nlargest(5, 'accel_event_score')[['file', 'accel_event_score']]
            lines.extend([
                "### 复合评分最高的文件:",
                ""
            ])
            for _, row in top_scores.iterrows():
                lines.append(f"- {row['file']}: {row['accel_event_score']:.3f}")
            lines.append("")
        
        # Effect size summary
        effect_cols = [c for c in df.columns if c.startswith('cohen_d_')]
        if effect_cols:
            lines.extend([
                "### 效应量统计 (Cohen's d):",
                ""
            ])
            for col in effect_cols:
                if col in df.columns:
                    mean_effect = df[col].mean()
                    lines.append(f"- {col}: 平均 = {mean_effect:.3f}")
            lines.append("")
    
    # Audio features summary
    lines.extend([
        "## 音频特征统计",
        ""
    ])
    
    # Key difference features
    diff_cols = [c for c in df.columns if c.startswith('diff_') and 'acc_' not in c]
    if diff_cols:
        lines.extend([
            "### 主要差异特征 (事件 vs 背景):",
            ""
        ])
        for col in diff_cols[:10]:  # Show top 10
            if col in df.columns:
                mean_diff = df[col].mean()
                lines.append(f"- {col}: 平均差异 = {mean_diff:.3f}")
        lines.append("")
    
    lines.extend([
        "## 分析说明",
        "",
        "- `ev_*`: 事件窗口特征",
        "- `bg_*`: 背景窗口特征", 
        "- `diff_*`: 差值特征 (事件 - 背景)",
        "- `cohen_d_*`: Cohen's d 效应量",
        "- `p_ttest_*`: t检验 p值",
        "- `accel_event_score`: 复合加速度事件评分",
        "",
        "运行 `python accel_detailed_word_report.py` 生成详细 Word 报告。"
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    main()