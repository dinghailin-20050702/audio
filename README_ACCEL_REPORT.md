## 深度加速度特征 & Word 报告

新增脚本：
- analyze_insertion_audio.py （升级版，含高级加速度特征）
- accel_detailed_word_report.py （生成 Word 报告：analysis_out/accel_detailed_report.docx）

安装额外依赖：
```bash
pip install python-docx
```

运行顺序：
```bash
python analyze_insertion_audio.py
python accel_detailed_word_report.py
```

查看：
- analysis_out/features.csv （新增列：cohen_d_mag, p_ttest_mag, ev_mag_peak, diff_mag_peak, accel_event_score 等）
- analysis_out/accel_detailed_report.docx （Word 深度报告）

可调参数：
- analyze_insertion_audio.py 顶部：ACC_FREQ_BANDS, W_COEF, 事件/背景窗口
- accel_detailed_word_report.py：MAX_DETAILED_PLOTS, SCORE_THRESHOLD

复合评分 accel_event_score 默认权重：diff_acc_mag_rms(0.30), diff_acc_jerk_rms(0.20), cohen_d_mag(0.25), cohen_d_jerk(0.15), diff_rms_dbfs(0.10)

若要保留旧版本，可先重命名旧脚本再更新。

### 新增特征列说明

#### 事件/背景加速度特征
- `ev_acc_mag_*` / `bg_acc_mag_*`: 事件/背景窗口加速度幅值统计 (mean, std, rms, peak, range)
- `ev_acc_jerk_*` / `bg_acc_jerk_*`: 事件/背景窗口急动度统计 (mean, std, rms, peak)
- `ev_acc_psd_band_*` / `bg_acc_psd_band_*`: Welch PSD 频段能量比例 (0-5Hz, 5-10Hz, 10-20Hz, 20-50Hz)
- `ev_acc_peak_freq` / `bg_acc_peak_freq`: 功率谱密度峰值频率

#### 差值特征
- `diff_acc_mag_*`: 加速度幅值差异 (事件 - 背景)
- `diff_acc_jerk_*`: 急动度差异 (事件 - 背景)

#### 统计效应量
- `cohen_d_mag`: 加速度幅值的 Cohen's d 效应量
- `cohen_d_jerk`: 急动度的 Cohen's d 效应量  
- `p_ttest_mag`: 加速度幅值 t 检验 p 值
- `p_ttest_jerk`: 急动度 t 检验 p 值

#### 复合评分
- `accel_event_score`: 多维特征加权复合评分，用于排序和筛选关键事件

### Word 报告内容

生成的 `accel_detailed_report.docx` 包含：

1. **执行摘要**: 统计概览和关键指标
2. **分析方法**: 技术说明和参数配置
3. **事件排名**: 基于复合评分的前20个事件表格
4. **详细事件分析**: 前15个事件的四象限分析图表
   - 加速度幅值对比 (事件 vs 背景)
   - 急动度对比 (事件 vs 背景)  
   - PSD 频域能量分布
   - 统计效应量汇总
5. **结论与建议**: 基于数据的分析结论

### 使用场景

- **质量控制**: 识别插接过程中的异常振动事件
- **设备诊断**: 监测机械设备运行状态
- **工艺优化**: 分析不同操作参数对插接质量的影响
- **实时监测**: 建立阈值体系用于生产线自动化检测