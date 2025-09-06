#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("analysis_out","features.csv")
OUT_MD = os.path.join("analysis_out","summary.md")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"未找到 {CSV_PATH}，请先运行分析脚本生成 features.csv")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV 为空")
        sys.exit(1)

    # 关心的字段（若不存在会自动跳过）
    want = [
        "diff_centroid_mean",
        "diff_flatness_mean",
        "diff_rolloff95_mean",
        "diff_band_300-1000_ratio",
        "diff_band_1000-3000_ratio",
        "diff_band_3000-8000_ratio",
        "ev_crest_factor","bg_crest_factor",
        "ev_rms_dbfs","bg_rms_dbfs",
        "ev_peak_dbfs","bg_peak_dbfs",
        "ev_duration_s","bg_duration_s"
    ]
    have = [c for c in want if c in df.columns]
    desc = df[have].describe(percentiles=[0.25,0.5,0.75]).T

    # 简单打分（仅示例）：高频占比+谱质心提升+平坦度提升
    score_terms = []
    if "diff_band_3000-8000_ratio" in df: score_terms.append(df["diff_band_3000-8000_ratio"].clip(lower=0))
    if "diff_band_1000-3000_ratio" in df: score_terms.append(df["diff_band_1000-3000_ratio"].clip(lower=0)*0.5)
    if "diff_centroid_mean" in df: score_terms.append((df["diff_centroid_mean"] / 2000.0).clip(lower=0, upper=1))
    if "diff_flatness_mean" in df: score_terms.append((df["diff_flatness_mean"] * 2).clip(lower=0, upper=1))
    if score_terms:
        score = sum(score_terms)
        df["_suspect_plug_score"] = (score - score.min())/(score.max()-score.min() + 1e-9)
    else:
        df["_suspect_plug_score"] = 0.0

    top = df.sort_values("_suspect_plug_score", ascending=False).head(5)[["file","_suspect_plug_score"]]

    # 生成文字摘要
    lines = []
    lines.append(f"样本数: {len(df)}")

    def med(name):
        return desc.loc[name,"50%"] if name in desc.index else np.nan

    centroid = med("diff_centroid_mean")
    flatness = med("diff_flatness_mean")
    roll95 = med("diff_rolloff95_mean")
    b_hf = med("diff_band_3000-8000_ratio")
    b_mf = med("diff_band_1000-3000_ratio")

    lines.append("总体特征（中位数层面）:")
    if pd.notna(centroid): lines.append(f"- 事件相对背景的谱质心上升: {centroid:.0f} Hz")
    if pd.notna(flatness): lines.append(f"- 谱平坦度提升: {flatness:.3f}")
    if pd.notna(roll95):   lines.append(f"- 95% 能量滚降点上移: {roll95:.0f} Hz")
    if pd.notna(b_mf):     lines.append(f"- 中频(1–3 kHz)能量占比提升: {b_mf:.3f}")
    if pd.notna(b_hf):     lines.append(f"- 高频(3–8 kHz)能量占比提升: {b_hf:.3f}")

    if "ev_crest_factor" in df.columns and "bg_crest_factor" in df.columns:
        crest_med = (df["ev_crest_factor"] - df["bg_crest_factor"]).median()
        lines.append(f"- 峰均比(crest factor)提升: {crest_med:.2f} (事件更尖锐)")

    # 简要判定覆盖率
    pos_counts = []
    if "diff_band_3000-8000_ratio" in df:
        pos_counts.append(("高频占比提升>0.05", (df["diff_band_3000-8000_ratio"] > 0.05).mean()))
    if "diff_centroid_mean" in df:
        pos_counts.append(("谱质心上升>1000 Hz", (df["diff_centroid_mean"] > 1000).mean()))
    if "diff_flatness_mean" in df:
        pos_counts.append(("谱平坦度提升>0.05", (df["diff_flatness_mean"] > 0.05).mean()))
    if pos_counts:
        lines.append("满足典型‘插接瞬态+高频宽带’特征的比例（粗略阈值）:")
        for name, frac in pos_counts:
            lines.append(f"- {name}: {int(frac*100)}% 的文件")

    # Top5
    lines.append("最像‘插接事件’的前5个（综合打分，仅供参考）:")
    for _, r in top.iterrows():
        lines.append(f"- {r['file']}: score={r['_suspect_plug_score']:.2f}")

    txt = "\n".join(lines)
    print(txt)

    # 写入 Markdown 摘要
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# 插接声特征摘要\n\n")
        f.write(txt.replace("\n","\n\n"))
    print(f"\n已生成 {OUT_MD}")

if __name__ == "__main__":
    main()