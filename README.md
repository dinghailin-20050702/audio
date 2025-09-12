# audio

## Enhanced Audio + Acceleration Analysis

This repository provides enhanced audio and acceleration insertion analysis capabilities. The system pairs audio recordings with acceleration data to detect and analyze insertion events.

### Installation

```bash
pip install -r requirements.txt
```

### Required File Naming Convention

- Audio files: `audio_YYYYMMDD_HHMMSS.wav` (in `recordings/` directory)
- Acceleration files: `accel_data_YYYYMMDD_HHMMSS.csv` (in root or same directory as audio)

The system automatically pairs files based on matching timestamps.

### Basic Usage

```bash
# Analyze audio and acceleration data
python analyze_insertion_audio.py

# Generate detailed Word report
python accel_detailed_word_report.py
```

### Output Files

- `analysis_out/features.csv`: Comprehensive feature analysis results
- `analysis_out/report_audio_accel.md`: Markdown summary report  
- `analysis_out/accel_detailed_report.docx`: Detailed Word document with charts
- `analysis_out/*.png`: Audio waveform and spectrogram plots

### Key Analysis Features

The enhanced analysis provides:

#### Audio Analysis
- Event detection using RMS + spectral flux robust z-score
- Time-domain features (RMS, peak, crest factor, etc.)
- Frequency-domain features (spectral centroid, bandwidth, rolloffs, flatness, ZCR)
- Band energies and ratios for frequency bands: (300-1000), (1000-3000), (3000-8000) Hz
- MFCC features (13 coefficients)

#### Acceleration Analysis
- **Basic time-domain**: `acc_mag_rms`, `acc_mag_mean`, `acc_mag_std`, `acc_jerk_rms`
- **Advanced time-domain**: `mag_peak`, `mag_p95`, `mag_crest`, `mag_iabs`, `mag_skew`, `mag_kurt`, `jerk_peak`, `jerk_p95`, `jerk_skew`, `jerk_kurt`
- **Frequency-domain**: `psd_dom_freq`, `psd_centroid` using Welch PSD method
- **Band energies**: Frequency bands (0-10), (10-50), (50-100), (100-300), (300-800) Hz
- **Statistical metrics**: Cohen's d for magnitude and jerk, Welch t-test p-values
- **Composite score**: `accel_event_score` using weighted combination of key metrics

#### Key New Columns in features.csv

- `cohen_d_mag`: Cohen's d effect size for acceleration magnitude (event vs background)
- `cohen_d_jerk`: Cohen's d effect size for acceleration jerk
- `p_ttest_mag/jerk`: Independent samples t-test p-values
- `diff_acc_mag_rms`: Event vs background acceleration RMS difference
- `diff_acc_jerk_rms`: Event vs background jerk RMS difference
- `accel_event_score`: Composite score (weighted: diff_acc_mag_rms×0.3 + diff_acc_jerk_rms×0.2 + cohen_d_mag×0.25 + cohen_d_jerk×0.15 + diff_rms_dbfs×0.1)
- `ev_/bg_mag_peak`: Acceleration magnitude peak values
- `ev_/bg_mag_p95`: 95th percentile acceleration values
- `ev_/bg_jerk_p95`: 95th percentile jerk values
- `ev_/bg_psd_band_X-Y_ratio`: Power spectral density energy ratios in frequency bands

### Parameter Adjustment

#### Sensitivity Control
- `PEAK_ZSCORE_TH` (default: 3.5): Lower values increase sensitivity to events
- `EVENT_PRE_SEC/POST_SEC` (default: 0.15/0.40s): Event window timing
- `BG_SEC` (default: 0.5s): Background window duration

#### Scoring Weights
Modify `W_COEF` dictionary in `analyze_insertion_audio.py`:
```python
W_COEF = {
    'diff_acc_mag_rms': 0.30,    # Acceleration RMS difference weight
    'diff_acc_jerk_rms': 0.20,   # Jerk RMS difference weight  
    'cohen_d_mag': 0.25,         # Cohen's d magnitude weight
    'cohen_d_jerk': 0.15,        # Cohen's d jerk weight
    'diff_rms_dbfs': 0.10        # Audio RMS difference weight
}
```

#### Report Filtering
In `accel_detailed_word_report.py`:
- `SCORE_THRESHOLD`: Minimum score for inclusion (None = no filtering)
- `MAX_DETAILED_PLOTS`: Maximum detailed analysis charts (default: 15)

### Regenerating Reports

To regenerate only the Word report without re-analyzing:
```bash
python accel_detailed_word_report.py
```

### Quickstart

```bash
# Complete analysis workflow
pip install -r requirements.txt
python analyze_insertion_audio.py
python accel_detailed_word_report.py
open analysis_out/accel_detailed_report.docx  # macOS
# or 
start analysis_out/accel_detailed_report.docx  # Windows
```

### Troubleshooting

- Ensure audio and acceleration files follow the naming convention
- Audio files should be in `recordings/` directory
- Acceleration CSV files should have columns: `timestamp`, `accel_x`, `accel_y`, `accel_z`
- For files without matching acceleration data, analysis continues with audio-only features