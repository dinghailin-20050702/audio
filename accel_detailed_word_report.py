#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 analysis_out/features.csv 生成详细 Word 报告

功能特性：
1. 从 features.csv 读取音频 + 加速度综合分析结果
2. 筛选有加速度数据的样本，按事件评分排序
3. 生成包含以下内容的 Word 文档：
   - 执行摘要：整体统计和关键发现
   - 详细分析：每个文件的综合评分和特征对比
   - 推荐样本：高评分样本的深度分析
   - 技术指标：统计显著性和效应量分析

输出：analysis_out/accel_detailed_report.docx

依赖：pandas python-docx numpy
使用：python accel_detailed_word_report.py
"""

import os
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

def load_features_data(csv_path: str = "analysis_out/features.csv") -> pd.DataFrame:
    """加载特征数据并筛选有加速度数据的样本"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到特征文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # 筛选有加速度数据的样本
    accel_df = df.dropna(subset=['diff_acc_mag_rms', 'cohen_d_mag', 'accel_event_score'])
    return df, accel_df

def add_heading_with_style(doc, text: str, level: int = 1):
    """添加带样式的标题"""
    heading = doc.add_heading(text, level=level)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return heading

def add_table_with_data(doc, headers: list, data: list, title: str = None):
    """添加数据表格"""
    if title:
        doc.add_paragraph(title, style='Heading 3')
    
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    
    # 添加表头
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        # 设置表头样式为粗体
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
    
    # 添加数据行
    for row_data in data:
        row_cells = table.add_row().cells
        for i, cell_data in enumerate(row_data):
            row_cells[i].text = str(cell_data)
    
    return table

def format_number(val, decimals: int = 3):
    """格式化数字显示"""
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"

def get_significance_level(p_value: float) -> str:
    """获取统计显著性级别"""
    if pd.isna(p_value):
        return "N/A"
    if p_value < 0.001:
        return "*** (p < 0.001)"
    elif p_value < 0.01:
        return "** (p < 0.01)"
    elif p_value < 0.05:
        return "* (p < 0.05)"
    elif p_value < 0.1:
        return "• (p < 0.1)"
    else:
        return "n.s."

def generate_summary_stats(df: pd.DataFrame, accel_df: pd.DataFrame) -> dict:
    """生成汇总统计"""
    stats = {
        'total_files': len(df),
        'accel_files': len(accel_df),
        'coverage_rate': len(accel_df) / len(df) * 100,
    }
    
    if len(accel_df) > 0:
        stats.update({
            'avg_accel_score': accel_df['accel_event_score'].mean(),
            'max_accel_score': accel_df['accel_event_score'].max(),
            'avg_cohen_d': accel_df['cohen_d_mag'].mean(),
            'max_cohen_d': accel_df['cohen_d_mag'].max(),
            'significant_count': len(accel_df[accel_df['p_value_acc'] < 0.05]),
            'high_score_count': len(accel_df[accel_df['accel_event_score'] > 1.0]),
        })
    
    return stats

def create_detailed_report(output_path: str = "analysis_out/accel_detailed_report.docx"):
    """生成详细的 Word 报告"""
    
    # 加载数据
    try:
        df, accel_df = load_features_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 python analyze_insertion_audio.py 生成特征文件")
        return
    
    # 创建文档
    doc = Document()
    
    # 标题
    title = doc.add_heading('音频 + 加速度插接事件综合分析报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 基本信息
    doc.add_paragraph(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph()
    
    # 生成统计摘要
    stats = generate_summary_stats(df, accel_df)
    
    # 1. 执行摘要
    add_heading_with_style(doc, "1. 执行摘要", 1)
    
    summary_text = f"""
本次分析共处理 {stats['total_files']} 个音频文件，其中 {stats['accel_files']} 个文件匹配到对应的加速度数据（覆盖率 {stats['coverage_rate']:.1f}%）。

关键发现：
"""
    
    if stats['accel_files'] > 0:
        summary_text += f"""
• 平均加速度事件评分: {stats['avg_accel_score']:.3f}
• 最高加速度事件评分: {stats['max_accel_score']:.3f}  
• 平均 Cohen's d 效应量: {stats['avg_cohen_d']:.3f}
• 最大 Cohen's d 效应量: {stats['max_cohen_d']:.3f}
• 统计显著差异样本数 (p < 0.05): {stats['significant_count']} 个
• 高评分样本数 (评分 > 1.0): {stats['high_score_count']} 个
"""
    else:
        summary_text += "\n• 未发现匹配的加速度数据，建议检查文件命名和路径"
    
    doc.add_paragraph(summary_text)
    
    if len(accel_df) == 0:
        doc.add_paragraph("⚠️ 由于没有加速度数据，无法生成详细分析。请确保 accel_data_YYYYMMDD_HHMMSS.csv 文件存在且命名正确。")
        # 保存文档
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.save(output_path)
        print(f"报告已生成: {output_path}")
        return
    
    # 2. 样本概览
    add_heading_with_style(doc, "2. 样本概览", 1)
    
    # 按评分排序
    accel_df_sorted = accel_df.sort_values('accel_event_score', ascending=False)
    
    # 创建概览表格
    overview_headers = ["文件名", "事件评分", "Cohen's d", "p值", "显著性", "事件时间(s)"]
    overview_data = []
    
    for _, row in accel_df_sorted.head(10).iterrows():  # 显示前10个
        filename = os.path.basename(row['file'])
        overview_data.append([
            filename,
            format_number(row['accel_event_score'], 3),
            format_number(row['cohen_d_mag'], 3),
            format_number(row['p_value_acc'], 4),
            get_significance_level(row['p_value_acc']),
            format_number(row['event_t'], 2)
        ])
    
    add_table_with_data(doc, overview_headers, overview_data, "按加速度事件评分排序的样本 (前10名)")
    
    # 3. 统计分析
    add_heading_with_style(doc, "3. 统计分析", 1)
    
    # 3.1 评分分布统计
    doc.add_heading("3.1 评分分布统计", 2)
    
    score_stats = accel_df['accel_event_score'].describe()
    cohen_stats = accel_df['cohen_d_mag'].describe()
    
    stats_headers = ["指标", "均值", "标准差", "最小值", "25%", "中位数", "75%", "最大值"]
    stats_data = [
        ["加速度事件评分"] + [format_number(score_stats[col], 3) for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']],
        ["Cohen's d 效应量"] + [format_number(cohen_stats[col], 3) for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    ]
    
    add_table_with_data(doc, stats_headers, stats_data)
    
    # 3.2 显著性分析
    doc.add_heading("3.2 统计显著性分析", 2)
    
    sig_counts = {
        'p < 0.001': len(accel_df[accel_df['p_value_acc'] < 0.001]),
        'p < 0.01': len(accel_df[(accel_df['p_value_acc'] >= 0.001) & (accel_df['p_value_acc'] < 0.01)]),
        'p < 0.05': len(accel_df[(accel_df['p_value_acc'] >= 0.01) & (accel_df['p_value_acc'] < 0.05)]),
        'p < 0.1': len(accel_df[(accel_df['p_value_acc'] >= 0.05) & (accel_df['p_value_acc'] < 0.1)]),
        'p ≥ 0.1': len(accel_df[accel_df['p_value_acc'] >= 0.1])
    }
    
    sig_headers = ["显著性水平", "样本数", "百分比"]
    sig_data = []
    total_samples = len(accel_df)
    
    for level, count in sig_counts.items():
        percentage = count / total_samples * 100
        sig_data.append([level, str(count), f"{percentage:.1f}%"])
    
    add_table_with_data(doc, sig_headers, sig_data)
    
    # 4. 高评分样本详细分析
    add_heading_with_style(doc, "4. 高评分样本详细分析", 1)
    
    # 选择评分最高的前5个样本
    top_samples = accel_df_sorted.head(5)
    
    doc.add_paragraph("以下为加速度事件评分最高的5个样本的详细分析：")
    doc.add_paragraph()
    
    for i, (_, row) in enumerate(top_samples.iterrows(), 1):
        doc.add_heading(f"4.{i} {os.path.basename(row['file'])}", 3)
        
        # 基本信息
        basic_info = f"""
事件检测时间: {row['event_t']:.2f} 秒
加速度事件评分: {row['accel_event_score']:.4f}
Cohen's d 效应量: {row['cohen_d_mag']:.4f}
统计显著性: {get_significance_level(row['p_value_acc'])} (p = {row['p_value_acc']:.4f})
"""
        doc.add_paragraph(basic_info)
        
        # 加速度特征对比
        accel_comparison = f"""
加速度特征对比：
• 事件期间均值: {row['event_acc_mean']:.3f} m/s²
• 背景期间均值: {row['bg_acc_mean']:.3f} m/s²
• RMS 差值: {row['diff_acc_mag_rms']:.3f} m/s²
• 数据点数 (事件/背景): {int(row['accel_data_points_event'])}/{int(row['accel_data_points_bg'])}
"""
        doc.add_paragraph(accel_comparison)
        
        # 音频特征亮点
        audio_highlights = f"""
音频特征亮点：
• 峰值频率差值: {row['diff_peak_freq']:.1f} Hz
• RMS 差值: {row['diff_rms_dbfs']:.2f} dB
• 谱心率差值: {row['diff_centroid_mean']:.1f} Hz
"""
        doc.add_paragraph(audio_highlights)
        doc.add_paragraph()
    
    # 5. 技术说明
    add_heading_with_style(doc, "5. 技术说明", 1)
    
    tech_explanation = """
5.1 评分计算方法

加速度事件评分 = 0.6 × 幅值变化评分 + 0.4 × 统计显著性评分

其中：
• 幅值变化评分 = |RMS差值| / 背景RMS（相对变化量）
• 统计显著性评分 = |Cohen's d|（标准化效应量）

5.2 统计指标说明

• Cohen's d: 标准化效应量，衡量两组数据差异的大小
  - d < 0.2: 小效应
  - 0.2 ≤ d < 0.5: 中等效应  
  - 0.5 ≤ d < 0.8: 大效应
  - d ≥ 0.8: 非常大效应

• p值: 统计显著性检验结果（t-test）
  - p < 0.05: 统计显著差异
  - p < 0.01: 高度显著差异
  - p < 0.001: 极显著差异

5.3 数据处理流程

1. 音频事件检测：RMS + 谱通量联合 z-score 打分
2. 时间窗口对齐：事件前0.15s至事件后0.40s
3. 加速度特征计算：矢量幅值统计对比
4. 综合评分：结合幅值变化和统计显著性
"""
    
    doc.add_paragraph(tech_explanation)
    
    # 6. 建议与结论
    add_heading_with_style(doc, "6. 建议与结论", 1)
    
    conclusion_text = f"""
基于当前分析结果，建议关注以下方面：

1. 重点样本筛选：
   • 优先分析加速度事件评分 > 1.0 的样本（共 {stats['high_score_count']} 个）
   • 重点关注 Cohen's d > 0.8 的样本（大效应量）
   • 验证 p < 0.01 的高显著性样本

2. 质量控制：
   • 检查加速度数据采样率和同步性
   • 验证时间戳匹配的准确性
   • 确认传感器校准状态

3. 进一步分析：
   • 结合频域特征进行多维度分析
   • 考虑环境因素对测量结果的影响
   • 建立插接事件的综合判定标准

本报告提供了初步的定量分析基础，建议结合专业知识进行深入解释和验证。
"""
    
    doc.add_paragraph(conclusion_text)
    
    # 保存文档
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"详细报告已生成: {output_path}")
    print(f"报告包含 {len(accel_df)} 个有加速度数据的样本分析")

if __name__ == "__main__":
    create_detailed_report()