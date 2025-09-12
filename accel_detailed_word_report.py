#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加速度事件详细Word报告生成器
- 读取 analysis_out/features.csv
- 按 accel_event_score 排序并筛选
- 生成包含统计图表的详细Word文档报告

依赖：python-docx pandas matplotlib numpy
"""

import os
import sys
from typing import List, Dict, Optional
import tempfile
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

# --------------------
# 配置参数
# --------------------
CSV_PATH = "analysis_out/features.csv"
OUTPUT_DOC = "analysis_out/accel_detailed_report.docx"
TMP_PLOTS_DIR = "analysis_out/tmp_plots"
SCORE_THRESHOLD = None  # 评分阈值，None表示不过滤
MAX_DETAILED_PLOTS = 15  # 最大详细分析图表数量

# 关键列名
KEY_COLUMNS = [
    'file', 'accel_event_score', 'diff_acc_mag_rms', 'diff_acc_jerk_rms', 
    'cohen_d_mag', 'cohen_d_jerk', 'ev_mag_peak', 'bg_mag_peak', 'ev_mag_p95', 'bg_mag_p95'
]

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def clean_filename(filename: str) -> str:
    """清理文件名，移除路径分隔符"""
    return filename.replace('/', '_').replace('\\', '_')

def create_comparison_bar_chart(data: Dict[str, float], title: str, 
                               ylabel: str, save_path: str) -> str:
    """创建事件vs背景对比柱状图"""
    plt.figure(figsize=(8, 6))
    
    # 分离事件和背景数据
    event_data = {k.replace('ev_', ''): v for k, v in data.items() if k.startswith('ev_')}
    bg_data = {k.replace('bg_', ''): v for k, v in data.items() if k.startswith('bg_')}
    
    # 获取共同的指标名称
    metrics = list(set(event_data.keys()) & set(bg_data.keys()))
    if not metrics:
        return None
    
    x = np.arange(len(metrics))
    width = 0.35
    
    event_values = [event_data.get(m, 0) for m in metrics]
    bg_values = [bg_data.get(m, 0) for m in metrics]
    
    plt.bar(x - width/2, event_values, width, label='Event Window', alpha=0.8, color='orange')
    plt.bar(x + width/2, bg_values, width, label='Background Window', alpha=0.8, color='lightblue')
    
    plt.xlabel('Metrics')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_frequency_band_chart(data: Dict[str, float], title: str, save_path: str) -> str:
    """创建频域频带比例图"""
    plt.figure(figsize=(10, 6))
    
    # 提取频带比例数据
    band_data = {}
    for key, value in data.items():
        if 'psd_band_' in key and '_ratio' in key and key.startswith('ev_'):
            band_name = key.replace('ev_psd_band_', '').replace('_ratio', '') + ' Hz'
            band_data[band_name] = value
    
    if not band_data:
        return None
    
    bands = list(band_data.keys())
    values = list(band_data.values())
    
    plt.bar(bands, values, alpha=0.7, color='steelblue')
    plt.xlabel('Frequency Bands')
    plt.ylabel('Energy Ratio')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def add_table_to_doc(doc: Document, data: pd.DataFrame, title: str, max_rows: int = 20):
    """向文档添加表格"""
    doc.add_heading(title, level=2)
    
    # 选择要显示的列
    display_cols = [col for col in KEY_COLUMNS if col in data.columns]
    table_data = data[display_cols].head(max_rows)
    
    # 创建表格
    table = doc.add_table(rows=1, cols=len(display_cols))
    table.style = 'Light Shading Accent 1'
    
    # 添加表头
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(display_cols):
        hdr_cells[i].text = col
    
    # 添加数据行
    for _, row in table_data.iterrows():
        row_cells = table.add_row().cells
        for i, col in enumerate(display_cols):
            value = row[col]
            if isinstance(value, (int, float)) and not np.isnan(value):
                row_cells[i].text = f"{value:.3f}"
            else:
                row_cells[i].text = str(value)

def generate_word_report(df: pd.DataFrame):
    """生成Word详细报告"""
    # 确保输出目录存在
    ensure_dir(TMP_PLOTS_DIR)
    ensure_dir(os.path.dirname(OUTPUT_DOC))
    
    # 创建Word文档
    doc = Document()
    
    # 标题
    title = doc.add_heading('音频+加速度插接事件详细分析报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 概要信息
    doc.add_heading('1. 分析概要', level=1)
    total_files = len(df)
    accel_files = len(df[df['accel_event_score'] > 0])
    
    summary_p = doc.add_paragraph()
    summary_p.add_run(f"• 总文件数: {total_files}\n")
    summary_p.add_run(f"• 有加速度数据的文件: {accel_files}\n")
    summary_p.add_run(f"• 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 筛选和排序数据
    sort_col = 'accel_event_score'
    if sort_col not in df.columns:
        sort_col = 'diff_rms_dbfs'
    
    df_sorted = df.sort_values(sort_col, ascending=False)
    
    if SCORE_THRESHOLD is not None:
        df_filtered = df_sorted[df_sorted[sort_col] >= SCORE_THRESHOLD]
    else:
        df_filtered = df_sorted
    
    # 整体统计
    doc.add_heading('2. 整体统计', level=1)
    stats_cols = ['diff_acc_mag_rms', 'diff_acc_jerk_rms', 'cohen_d_mag']
    valid_stats = [col for col in stats_cols if col in df.columns]
    
    if valid_stats:
        stats_data = df[valid_stats].describe()
        stats_p = doc.add_paragraph("关键指标统计摘要：\n")
        for col in valid_stats:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                stats_p.add_run(f"• {col}: 平均 {mean_val:.3f} ± {std_val:.3f}\n")
    
    # 排名概览表
    add_table_to_doc(doc, df_filtered, '3. Top 20 事件排名概览', max_rows=20)
    
    # 详细分析部分
    doc.add_heading('4. 详细事件分析', level=1)
    doc.add_paragraph("以下为评分最高的事件详细分析图表：")
    
    # 为前N个事件生成详细图表
    detailed_count = 0
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        if detailed_count >= MAX_DETAILED_PLOTS:
            break
        
        if row.get('accel_event_score', 0) <= 0:
            continue  # 跳过没有加速度数据的文件
        
        detailed_count += 1
        filename = row['file']
        clean_name = clean_filename(filename)
        
        # 添加小标题
        doc.add_heading(f'4.{detailed_count} {filename}', level=3)
        
        # 基本信息
        info_p = doc.add_paragraph()
        info_p.add_run(f"评分: {row.get('accel_event_score', 0):.3f}\n")
        info_p.add_run(f"Cohen's d (幅值): {row.get('cohen_d_mag', 0):.3f}\n")
        info_p.add_run(f"加速度RMS差值: {row.get('diff_acc_mag_rms', 0):.3f}\n")
        
        # 生成对比图表
        try:
            # 时域特征对比
            time_domain_data = {
                f'ev_{key}': row.get(f'ev_{key}', 0) for key in ['acc_mag_rms', 'mag_peak', 'mag_p95']
            }
            time_domain_data.update({
                f'bg_{key}': row.get(f'bg_{key}', 0) for key in ['acc_mag_rms', 'mag_peak', 'mag_p95']
            })
            
            chart_path = os.path.join(TMP_PLOTS_DIR, f"{clean_name}_time_domain.png")
            if create_comparison_bar_chart(
                time_domain_data, 
                f"时域特征对比 - {filename}", 
                "加速度值", 
                chart_path
            ):
                doc.add_picture(chart_path, width=Inches(6))
            
            # 频域频带比例图
            freq_chart_path = os.path.join(TMP_PLOTS_DIR, f"{clean_name}_frequency.png")
            if create_frequency_band_chart(
                row.to_dict(), 
                f"频域频带分布 - {filename}", 
                freq_chart_path
            ):
                doc.add_picture(freq_chart_path, width=Inches(6))
                
        except Exception as e:
            doc.add_paragraph(f"图表生成失败: {str(e)}")
        
        # 添加分页符（除了最后一个）
        if detailed_count < min(MAX_DETAILED_PLOTS, len(df_filtered)):
            doc.add_page_break()
    
    # 术语说明
    doc.add_heading('5. 术语说明', level=1)
    glossary = [
        ("accel_event_score", "综合评分，基于加速度RMS差值、Jerk差值、Cohen's d和音频RMS差值的加权平均"),
        ("diff_acc_mag_rms", "事件窗与背景窗加速度RMS的差值，正值表示事件期间加速度更大"),
        ("diff_acc_jerk_rms", "事件窗与背景窗加加速度RMS的差值，衡量运动变化的激烈程度"),
        ("cohen_d_mag", "加速度幅值的Cohen's d效应量，衡量事件与背景的差异显著性"),
        ("mag_peak", "加速度幅值的峰值"),
        ("mag_p95", "加速度幅值的95百分位数"),
        ("jerk_p95", "加加速度的95百分位数"),
        ("psd_band_ratio", "功率谱密度在特定频带的能量比例")
    ]
    
    for term, definition in glossary:
        p = doc.add_paragraph()
        p.add_run(f"• {term}: ").bold = True
        p.add_run(definition)
    
    # 保存文档
    doc.save(OUTPUT_DOC)
    print(f"Word报告已生成: {OUTPUT_DOC}")
    
    # 清理临时文件
    if os.path.exists(TMP_PLOTS_DIR):
        shutil.rmtree(TMP_PLOTS_DIR)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"错误: 未找到特征文件 {CSV_PATH}")
        print("请先运行 python analyze_insertion_audio.py 生成特征数据")
        sys.exit(1)
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"读取到 {len(df)} 条记录")
        
        if df.empty:
            print("警告: CSV文件为空")
            sys.exit(1)
        
        generate_word_report(df)
        
    except Exception as e:
        print(f"生成报告时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()