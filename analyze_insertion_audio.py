#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插接声音特征分析（自动定位事件 + 频域/时频特征）
- 递归扫描 recordings/ 下的 .wav（含 recordings/recordings 嵌套目录）
- 预处理：单声道、归一化、300 Hz 高通
- 事件检测：RMS + 谱通量 z-score 联合打分找峰值
- 特征提取：事件窗 vs 背景窗（FFT/谱统计/带通能量/MFCC/时域）
- 输出：每文件图像（波形+STFT），汇总 CSV（事件/背景/差值特征）

依赖：numpy scipy librosa soundfile matplotlib pandas
安装：pip install numpy scipy librosa soundfile matplotlib pandas
"""

import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
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
            }
            rows.append(row)
            print(f"完成: {rel}")
        except Exception as e:
            print(f"处理失败: {path} -> {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"已输出: {csv_path}")
    print("建议后续：用 diff_* 指标做阈值选择与判定。")

if __name__ == "__main__":
    main()