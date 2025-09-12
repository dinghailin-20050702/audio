#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型插接声音+加速度特征分析（自动定位事件 + 频域/时频特征 + 加速度统计分析）
- 递归扫描 recordings/ 下的 .wav（含 recordings/recordings 嵌套目录）
- 匹配加速度数据：audio_YYYYMMDD_HHMMSS.wav <-> accel_data_YYYYMMDD_HHMMSS.csv
- 预处理：单声道、归一化、300 Hz 高通
- 事件检测：RMS + 谱通量 z-score 联合打分找峰值
- 特征提取：事件窗 vs 背景窗（FFT/谱统计/带通能量/MFCC/时域）
- 加速度分析：时域+频域特征、Cohen's d、t-test、复合评分
- 输出：每文件图像（波形+STFT），汇总 CSV（事件/背景/差值特征+加速度特征），Markdown报告

依赖：numpy scipy librosa soundfile matplotlib pandas
安装：pip install numpy scipy librosa soundfile matplotlib pandas
"""

import os
import glob
import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import ttest_ind
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

# 带通频带（用于能量比例）- 音频
BANDS = [(300, 1000), (1000, 3000), (3000, 8000)]
MFCC_N = 13

# 加速度频域分析频带
ACCEL_BANDS = [(0, 10), (10, 50), (50, 100), (100, 300), (300, 800)]

# 复合评分权重
W_COEF = {
    'diff_acc_mag_rms': 0.30,
    'diff_acc_jerk_rms': 0.20,
    'cohen_d_mag': 0.25,
    'cohen_d_jerk': 0.15,
    'diff_rms_dbfs': 0.10
}

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

# --------------------
# 加速度数据处理函数
# --------------------
def find_matching_accel_file(audio_path: str) -> Optional[str]:
    """根据音频文件名找到匹配的加速度文件"""
    # 提取时间戳：audio_YYYYMMDD_HHMMSS.wav -> YYYYMMDD_HHMMSS
    match = re.search(r'audio_(\d{8}_\d{6})\.wav', audio_path)
    if not match:
        return None
    
    timestamp = match.group(1)
    accel_pattern = f"accel_data_{timestamp}.csv"
    
    # 在同一目录和根目录查找
    audio_dir = os.path.dirname(audio_path)
    candidates = [
        os.path.join(audio_dir, accel_pattern),
        os.path.join(".", accel_pattern)
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def load_acceleration_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    加载加速度数据，返回时间序列、加速度幅值、采样率
    返回: (time_array, magnitude_array, sampling_rate)
    """
    try:
        df = pd.read_csv(csv_path)
        if 'timestamp' not in df.columns:
            raise ValueError("CSV中缺少timestamp列")
        
        # 计算采样率
        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dropna().dt.total_seconds()
        sampling_rate = 1.0 / time_diffs.median() if len(time_diffs) > 0 else 100.0
        
        # 计算加速度幅值
        accel_cols = ['accel_x', 'accel_y', 'accel_z']
        missing_cols = [col for col in accel_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV中缺少加速度列: {missing_cols}")
        
        accel_data = df[accel_cols].values
        magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        
        # 时间数组（相对时间，秒）
        time_array = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        
        return time_array, magnitude, sampling_rate
    except Exception as e:
        raise ValueError(f"加载加速度数据失败: {e}")

def segment_accel_data(time_array: np.ndarray, magnitude: np.ndarray, 
                      audio_event_time: float, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据音频事件时间分割加速度数据为事件窗和背景窗
    """
    # 事件窗
    event_start = audio_event_time - EVENT_PRE_SEC
    event_end = audio_event_time + EVENT_POST_SEC
    event_mask = (time_array >= event_start) & (time_array <= event_end)
    
    # 背景窗
    bg_end = max(0, audio_event_time - 0.05)  # 事件前0.05秒结束
    bg_start = max(0, bg_end - BG_SEC)
    bg_mask = (time_array >= bg_start) & (time_array <= bg_end)
    
    event_data = magnitude[event_mask]
    bg_data = magnitude[bg_mask]
    
    return event_data, bg_data

def compute_acceleration_features(data: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """计算加速度特征"""
    if len(data) == 0:
        return {f"{prefix}acc_mag_rms": 0.0, f"{prefix}acc_mag_mean": 0.0, 
                f"{prefix}acc_mag_std": 0.0, f"{prefix}acc_jerk_rms": 0.0}
    
    # 基础时域特征
    mag_rms = float(np.sqrt(np.mean(data**2)))
    mag_mean = float(np.mean(data))
    mag_std = float(np.std(data))
    
    # 加加速度（jerk）
    if len(data) > 1:
        jerk = np.diff(data) * sampling_rate
        jerk_rms = float(np.sqrt(np.mean(jerk**2))) if len(jerk) > 0 else 0.0
    else:
        jerk_rms = 0.0
    
    # 高级时域特征
    mag_peak = float(np.max(np.abs(data)))
    mag_p95 = float(np.percentile(np.abs(data), 95)) if len(data) > 0 else 0.0
    mag_crest = mag_peak / (mag_rms + 1e-12)
    mag_iabs = float(np.trapezoid(np.abs(data))) if len(data) > 1 else float(np.sum(np.abs(data)))
    
    features = {
        f"{prefix}acc_mag_rms": mag_rms,
        f"{prefix}acc_mag_mean": mag_mean,
        f"{prefix}acc_mag_std": mag_std,
        f"{prefix}acc_jerk_rms": jerk_rms,
        f"{prefix}mag_peak": mag_peak,
        f"{prefix}mag_p95": mag_p95,
        f"{prefix}mag_crest": mag_crest,
        f"{prefix}mag_iabs": mag_iabs,
    }
    
    # 偏度和峰度
    if len(data) > 3:
        from scipy.stats import skew, kurtosis
        features[f"{prefix}mag_skew"] = float(skew(data))
        features[f"{prefix}mag_kurt"] = float(kurtosis(data))
        
        if len(data) > 1:
            jerk = np.diff(data) * sampling_rate
            if len(jerk) > 3:
                jerk_peak = float(np.max(np.abs(jerk)))
                jerk_p95 = float(np.percentile(np.abs(jerk), 95))
                features[f"{prefix}jerk_peak"] = jerk_peak
                features[f"{prefix}jerk_p95"] = jerk_p95
                features[f"{prefix}jerk_skew"] = float(skew(jerk))
                features[f"{prefix}jerk_kurt"] = float(kurtosis(jerk))
    
    # 频域特征（使用Welch方法）
    if len(data) > 8:  # 需要足够的数据点进行频域分析
        try:
            freqs, psd = welch(data, fs=sampling_rate, nperseg=min(len(data)//2, 256))
            
            # 主频率
            dom_freq_idx = np.argmax(psd)
            psd_dom_freq = float(freqs[dom_freq_idx])
            
            # 谱质心
            psd_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
            
            features[f"{prefix}psd_dom_freq"] = psd_dom_freq
            features[f"{prefix}psd_centroid"] = psd_centroid
            
            # 频带能量比例
            total_energy = np.sum(psd) + 1e-12
            for lo, hi in ACCEL_BANDS:
                band_mask = (freqs >= lo) & (freqs < hi)
                band_energy = np.sum(psd[band_mask])
                features[f"{prefix}psd_band_{lo}-{hi}_ratio"] = float(band_energy / total_energy)
                
        except Exception:
            # 如果频域分析失败，使用默认值
            features[f"{prefix}psd_dom_freq"] = 0.0
            features[f"{prefix}psd_centroid"] = 0.0
            for lo, hi in ACCEL_BANDS:
                features[f"{prefix}psd_band_{lo}-{hi}_ratio"] = 0.0
    
    return features

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    if n1 == 1 and n2 == 1:
        return 0.0
    
    # 合并标准差
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)

def compute_ttest_pvalue(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算独立样本t检验p值"""
    if len(group1) < 10 or len(group2) < 10:
        return 1.0  # 样本量太小，返回默认值
    
    try:
        _, p_value = ttest_ind(group1, group2, equal_var=False)
        return float(p_value)
    except Exception:
        return 1.0

def compute_accel_event_score(features: Dict[str, float]) -> float:
    """计算复合加速度事件评分"""
    score = 0.0
    for metric, weight in W_COEF.items():
        if metric in features:
            # 对于Cohen's d，使用绝对值
            if 'cohen_d' in metric:
                score += weight * abs(features[metric])
            else:
                score += weight * features[metric]
    return score

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

def generate_markdown_report(df: pd.DataFrame, output_dir: str):
    """生成Markdown摘要报告"""
    md_path = os.path.join(output_dir, "report_audio_accel.md")
    
    # 按accel_event_score排序，如果不存在则按diff_rms_dbfs排序
    sort_key = 'accel_event_score' if 'accel_event_score' in df.columns else 'diff_rms_dbfs'
    df_sorted = df.sort_values(sort_key, ascending=False)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 音频+加速度插接事件分析报告\n\n")
        f.write(f"- 分析文件数量: {len(df)}\n")
        f.write(f"- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Top 10 统计
        f.write("## Top 10 事件排名\n\n")
        f.write("| 排名 | 文件 | 事件评分 | 加速度RMS差值 | 加速度Jerk差值 | Cohen's d (幅值) |\n")
        f.write("|------|------|----------|---------------|----------------|------------------|\n")
        
        top10 = df_sorted.head(10)
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            score = row.get('accel_event_score', 0.0)
            mag_diff = row.get('diff_acc_mag_rms', 0.0)
            jerk_diff = row.get('diff_acc_jerk_rms', 0.0)
            cohen_d = row.get('cohen_d_mag', 0.0)
            f.write(f"| {i} | {row['file']} | {score:.3f} | {mag_diff:.3f} | {jerk_diff:.3f} | {cohen_d:.3f} |\n")
        
        f.write("\n## 关键指标术语解释\n\n")
        f.write("- **accel_event_score**: 综合评分（权重加权：加速度RMS差值×0.3 + Jerk差值×0.2 + Cohen's d×0.4 + 音频RMS差值×0.1）\n")
        f.write("- **diff_acc_mag_rms**: 事件窗vs背景窗加速度RMS差值\n")
        f.write("- **diff_acc_jerk_rms**: 事件窗vs背景窗加加速度RMS差值\n")
        f.write("- **cohen_d_mag**: 加速度幅值的Cohen's d效应量（衡量事件vs背景差异显著性）\n")
        f.write("- **cohen_d_jerk**: 加加速度的Cohen's d效应量\n")
        f.write("- **p_ttest_mag/jerk**: 独立样本t检验p值（<0.05表示显著差异）\n")
        f.write("- **mag_peak**: 加速度幅值峰值\n")
        f.write("- **mag_p95**: 加速度幅值95百分位数\n")
        f.write("- **jerk_p95**: 加加速度95百分位数\n")
        f.write("- **psd_band_X-Y_ratio**: 频域X-Y Hz频带能量占比\n\n")
        
        f.write("## 参数调整说明\n\n")
        f.write("- **敏感性调整**: 修改PEAK_ZSCORE_TH (默认3.5)，值越小越敏感\n")
        f.write("- **时窗调整**: EVENT_PRE_SEC/POST_SEC (默认0.15/0.40s)，BG_SEC (默认0.5s)\n")
        f.write("- **评分权重**: 修改W_COEF字典中的权重值\n")
        f.write("- **报告阈值**: 在accel_detailed_word_report.py中调整SCORE_THRESHOLD\n")
    
    print(f"已生成Markdown报告: {md_path}")

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

            # 特征
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
            accel_path = find_matching_accel_file(path)
            if accel_path:
                try:
                    time_array, magnitude, accel_sr = load_acceleration_data(accel_path)
                    event_time = event_sample / sr
                    
                    accel_event, accel_bg = segment_accel_data(time_array, magnitude, event_time, accel_sr)
                    
                    # 计算加速度特征
                    ev_accel_feat = compute_acceleration_features(accel_event, accel_sr, "ev_")
                    bg_accel_feat = compute_acceleration_features(accel_bg, accel_sr, "bg_")
                    
                    # 加速度差值特征
                    accel_diff_feat = {}
                    for key in ev_accel_feat:
                        if key.startswith("ev_"):
                            base_key = key[3:]  # 去掉"ev_"前缀
                            bg_key = f"bg_{base_key}"
                            if bg_key in bg_accel_feat:
                                accel_diff_feat[f"diff_{base_key}"] = ev_accel_feat[key] - bg_accel_feat[bg_key]
                    
                    # 统计分析
                    cohen_d_mag = compute_cohens_d(accel_event, accel_bg)
                    cohen_d_jerk = 0.0
                    p_ttest_mag = compute_ttest_pvalue(accel_event, accel_bg)
                    p_ttest_jerk = 1.0
                    
                    if len(accel_event) > 1 and len(accel_bg) > 1:
                        jerk_event = np.diff(accel_event) * accel_sr
                        jerk_bg = np.diff(accel_bg) * accel_sr
                        cohen_d_jerk = compute_cohens_d(jerk_event, jerk_bg)
                        p_ttest_jerk = compute_ttest_pvalue(jerk_event, jerk_bg)
                    
                    stat_feat = {
                        'cohen_d_mag': cohen_d_mag,
                        'cohen_d_jerk': cohen_d_jerk,
                        'p_ttest_mag': p_ttest_mag,
                        'p_ttest_jerk': p_ttest_jerk
                    }
                    
                    # 复合评分
                    combined_feat = {**diff_feat, **accel_diff_feat, **stat_feat}
                    accel_event_score = compute_accel_event_score(combined_feat)
                    
                except Exception as e:
                    print(f"加速度分析失败 {accel_path}: {e}")
                    # 使用默认值
                    ev_accel_feat = {}
                    bg_accel_feat = {}
                    accel_diff_feat = {}
                    stat_feat = {'cohen_d_mag': 0.0, 'cohen_d_jerk': 0.0, 'p_ttest_mag': 1.0, 'p_ttest_jerk': 1.0}
                    accel_event_score = 0.0
            else:
                # 未找到匹配的加速度文件，使用默认值
                ev_accel_feat = {}
                bg_accel_feat = {}
                accel_diff_feat = {}
                stat_feat = {'cohen_d_mag': 0.0, 'cohen_d_jerk': 0.0, 'p_ttest_mag': 1.0, 'p_ttest_jerk': 1.0}
                accel_event_score = 0.0

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
                **ev_accel_feat,
                **bg_accel_feat,
                **accel_diff_feat,
                **stat_feat,
                "accel_event_score": accel_event_score,
            }
            rows.append(row)
            print(f"完成: {rel}" + (f" (匹配加速度数据)" if accel_path else " (无加速度数据)"))
        except Exception as e:
            print(f"处理失败: {path} -> {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    # 生成Markdown报告
    generate_markdown_report(df, OUTPUT_DIR)
    
    print(f"\n已输出: {csv_path}")
    print(f"处理音频文件: {len([r for r in rows if r])}/{len(wavs)}")
    print(f"匹配加速度数据: {len([r for r in rows if r.get('cohen_d_mag', 0) != 0])}/{len(wavs)}")
    print("建议后续：运行 python accel_detailed_word_report.py 生成详细Word报告。")

if __name__ == "__main__":
    main()