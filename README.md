# 插接事件 音频 + 加速度 综合分析

本项目提供音频信号与加速度传感器数据的综合分析管道，专门用于插接操作事件的自动检测与特征提取。

## 功能特性

### 🎵 音频分析
- **事件检测**: RMS + 谱通量联合 z-score 打分自动定位事件峰值
- **预处理**: 单声道转换、归一化、300Hz 高通滤波去除低频噪音
- **特征提取**: 
  - 时域特征：RMS、峰值、波峰因子、过零率
  - 频域特征：谱心率、带宽、谱滚降、谱平坦度
  - 带通能量：300-1000Hz、1000-3000Hz、3000-8000Hz
  - MFCC 系数：13维梅尔频率倒谱系数

### 📊 加速度分析
- **自动匹配**: 基于时间戳自动匹配对应的加速度 CSV 文件
- **矢量分析**: 计算三轴加速度的合成矢量幅值
- **统计对比**: 事件期间 vs 背景期间的统计特征对比
- **效应量计算**: Cohen's d 标准化效应量评估
- **显著性检验**: t-test 统计检验评估差异显著性

### 📈 综合评分
- **事件评分**: 结合幅值变化和统计显著性的综合评分
- **多维度可视化**: 波形、时频谱、加速度时序的联合展示
- **详细报告**: 自动生成 Word 格式的专业分析报告

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包包括：
- `numpy`, `scipy`: 数值计算和信号处理
- `librosa`, `soundfile`: 音频处理和 I/O
- `matplotlib`: 数据可视化
- `pandas`: 数据处理和分析
- `python-docx`: Word 文档生成

## 数据格式要求

### 音频文件
- 格式：WAV 文件
- 命名：`audio_YYYYMMDD_HHMMSS.wav`
- 位置：`recordings/` 目录下（支持嵌套子目录）

### 加速度文件
- 格式：CSV 文件
- 命名：`accel_data_YYYYMMDD_HHMMSS.csv`
- 位置：项目根目录
- 必需列：`timestamp`, `accel_x`, `accel_y`, `accel_z`

时间戳格式示例：
- 音频：`audio_20250911_143152.wav`
- 加速度：`accel_data_20250911_143152.csv`

## 使用方法

### 1. 运行综合分析

```bash
python analyze_insertion_audio.py
```

**输出文件：**
- `analysis_out/features.csv`: 详细特征数据表
- `analysis_out/*.png`: 每个文件的可视化图像（波形+时频+加速度）

**输出特征包括：**
- 音频特征：`ev_*`（事件窗）、`bg_*`（背景窗）、`diff_*`（差值特征）
- 加速度特征：
  - `diff_acc_mag_rms`: 加速度 RMS 差值
  - `cohen_d_mag`: Cohen's d 效应量
  - `accel_event_score`: 综合事件评分
  - `p_value_acc`: 统计显著性 p 值

### 2. 生成详细报告

```bash
python accel_detailed_word_report.py
```

**输出文件：**
- `analysis_out/accel_detailed_report.docx`: 专业 Word 分析报告

**报告内容：**
- 执行摘要：整体统计和关键发现
- 样本概览：按评分排序的前10名样本
- 统计分析：评分分布和显著性分析
- 高评分样本详细分析：前5名样本的深度解读
- 技术说明：算法原理和指标解释
- 建议与结论：后续分析方向

## 技术原理

### 事件检测算法
1. **预处理**: 音频归一化 + 高通滤波（300Hz）
2. **特征计算**: RMS 和谱通量逐帧计算
3. **鲁棒 z-score**: 基于中位数和 MAD 的鲁棒标准化
4. **联合打分**: `score = 0.5 × rms_zscore + 0.5 × flux_zscore`
5. **峰值检测**: 寻找超过阈值的局部最大值

### 加速度事件评分
```
事件评分 = 0.6 × 幅值变化评分 + 0.4 × 统计显著性评分
```

- **幅值变化评分** = |RMS差值| / 背景RMS（相对变化）
- **统计显著性评分** = |Cohen's d|（标准化效应量）

### Cohen's d 效应量解释
- `d < 0.2`: 小效应
- `0.2 ≤ d < 0.5`: 中等效应
- `0.5 ≤ d < 0.8`: 大效应
- `d ≥ 0.8`: 非常大效应

## 输出文件结构

```
analysis_out/
├── features.csv                    # 综合特征数据表
├── accel_detailed_report.docx      # 详细分析报告
├── audio_YYYYMMDD_HHMMSS.png      # 各文件可视化图像
└── ...
```

## 配置参数

在 `analyze_insertion_audio.py` 中可调整的主要参数：

```python
TARGET_SR = 44100              # 目标采样率
HIGHPASS_CUTOFF = 300.0        # 高通滤波截止频率
EVENT_PRE_SEC = 0.15           # 事件窗：峰值前时长
EVENT_POST_SEC = 0.40          # 事件窗：峰值后时长
BG_SEC = 0.5                   # 背景窗长度
PEAK_ZSCORE_TH = 3.5           # 峰值检测阈值
```

## 质量控制建议

1. **数据质量检查**
   - 确认音频和加速度文件时间戳对应
   - 检查采样率和数据同步性
   - 验证传感器校准状态

2. **结果验证**
   - 人工检查高评分样本的合理性
   - 对比不同参数设置的结果稳定性
   - 结合专业知识解释异常值

3. **批量处理优化**
   - 使用 `.gitignore` 避免提交分析输出
   - 定期清理 `analysis_out/` 目录
   - 备份重要的分析结果

## 故障排除

### 常见问题

**Q: 运行提示 "未找到对应加速度文件"**
A: 检查文件命名格式，确保 `audio_YYYYMMDD_HHMMSS.wav` 与 `accel_data_YYYYMMDD_HHMMSS.csv` 时间戳一致

**Q: Word 报告生成失败**
A: 确认已安装 `python-docx`：`pip install python-docx`

**Q: 分析结果中加速度列为空**
A: 确认对应的 CSV 文件存在且包含必需的列（accel_x, accel_y, accel_z）

**Q: 图像显示不正常**
A: 检查 matplotlib 后端设置，可能需要 `plt.ioff()` 或调整 DPI 设置

## 扩展开发

项目采用模块化设计，便于扩展：

- **新增特征**: 在 `compute_block_features()` 中添加
- **自定义评分**: 修改 `compute_acceleration_features()` 中的评分公式
- **报告格式**: 在 `accel_detailed_word_report.py` 中自定义报告模板
- **数据源**: 支持其他格式的音频和传感器数据

## 许可证

本项目采用开源许可证，请根据实际需要选择合适的许可证类型。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系项目维护者。