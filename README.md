# Audio Analysis Reporting Pipeline

A comprehensive audio analysis system designed to detect and analyze audio insertion/connection events with advanced reporting capabilities.

## 🎯 Overview

This pipeline provides:
- **Audio Event Detection**: Automatic detection of insertion/connection events in audio recordings
- **Feature Extraction**: Comprehensive spectral, temporal, and statistical feature analysis
- **Advanced Visualizations**: Correlation heatmaps, feature distributions, comparative analysis, and interactive plots
- **Comprehensive Reporting**: HTML reports with embedded visualizations and analysis recommendations
- **Configurable Pipeline**: YAML-based configuration system for easy customization

## 📋 Features

### Core Analysis
- Recursive audio file scanning (supports nested directories)
- Audio preprocessing (mono conversion, normalization, high-pass filtering)
- Event detection using RMS + spectral flux analysis
- Detailed feature extraction (FFT, spectral statistics, MFCCs, band energies)

### Enhanced Visualizations
- **Individual Spectrograms**: Waveform and spectrogram plots for each audio file
- **Correlation Heatmaps**: Feature correlation analysis
- **Distribution Plots**: Statistical distributions of key features
- **Comparative Analysis**: Event vs background feature comparisons
- **Summary Dashboard**: Comprehensive overview with key metrics
- **Interactive 3D Plots**: Web-based interactive feature space visualization

### Reporting
- **HTML Reports**: Comprehensive web-based reports with embedded visualizations
- **Markdown Summaries**: Text-based analysis summaries
- **CSV Export**: Detailed feature data for further analysis
- **Configurable Thresholds**: Customizable detection and scoring parameters

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete pipeline with default settings
python pipeline.py

# Use custom configuration
python pipeline.py --config my_config.yaml

# Skip audio analysis and use existing features
python pipeline.py --skip-analysis

# Run with minimal output
python pipeline.py --quiet
```

### Individual Components

```bash
# Run just the audio analysis
python analyze_insertion_audio.py

# Generate feature summary
python summarize_features.py

# Run enhanced pipeline
python pipeline.py
```

## 📁 Project Structure

```
├── analyze_insertion_audio.py    # Core audio analysis script
├── summarize_features.py         # Feature summarization
├── pipeline.py                   # Main pipeline orchestrator
├── visualization.py              # Enhanced visualization functions
├── report_generator.py           # HTML report generation
├── config.yaml                   # Configuration file
├── requirements.txt               # Python dependencies
├── recordings/                    # Audio files directory
├── analysis_out/                  # Analysis outputs
└── reports/                       # Generated reports
```

## ⚙️ Configuration

The pipeline uses a YAML configuration file (`config.yaml`) for customization:

```yaml
# Input/Output Paths
search_dirs: ["recordings"]
output_dir: "analysis_out"
reports_dir: "reports"

# Audio Processing Parameters
audio:
  target_sample_rate: 44100
  highpass_cutoff: 300.0

# Event Detection Parameters
event_detection:
  pre_window_sec: 0.15
  post_window_sec: 0.40
  peak_zscore_threshold: 3.5

# Analysis Thresholds
thresholds:
  high_frequency_increase: 0.05
  spectral_centroid_increase: 1000
  suspicious_score_threshold: 0.7
```

## 📊 Output Files

### Analysis Directory (`analysis_out/`)
- `features.csv` - Detailed feature data for all audio files
- `summary.md` - Text-based analysis summary
- Individual PNG files - Spectrogram plots for each audio file

### Reports Directory (`reports/`)
- `audio_analysis_report.html` - Comprehensive HTML report
- `correlation_heatmap.png` - Feature correlation visualization
- `feature_distributions.png` - Statistical distribution plots
- `comparative_analysis.png` - Event vs background comparisons
- `summary_dashboard.png` - Overview dashboard
- `interactive_3d_analysis.html` - Interactive 3D feature space plot

## 🔍 Key Features Analyzed

### Spectral Features
- **Spectral Centroid**: Center frequency of the spectrum
- **Spectral Bandwidth**: Spread of the spectrum
- **Spectral Rolloff**: Frequency below which X% of energy is contained
- **Spectral Flatness**: Measure of spectral shape (noise-like vs tonal)

### Temporal Features
- **Zero Crossing Rate**: Rate of sign changes in the signal
- **Crest Factor**: Ratio of peak to RMS amplitude
- **RMS Energy**: Root mean square energy level

### Band-specific Features
- **Low Band (300-1000 Hz)**: Energy and ratios
- **Mid Band (1000-3000 Hz)**: Energy and ratios
- **High Band (3000-8000 Hz)**: Energy and ratios

### MFCCs
- **13 Mel-frequency Cepstral Coefficients**: Compact representation of spectral shape

## 📈 Analysis Process

1. **Audio Loading**: Files are loaded, converted to mono, and normalized
2. **Preprocessing**: High-pass filtering to remove low-frequency noise
3. **Event Detection**: RMS and spectral flux analysis to locate events
4. **Windowing**: Extract event and background windows
5. **Feature Extraction**: Compute comprehensive features for both windows
6. **Differential Analysis**: Calculate event-background differences
7. **Scoring**: Generate suspect scores based on configurable thresholds
8. **Visualization**: Create multiple plot types for analysis
9. **Reporting**: Generate comprehensive HTML reports

## 🎯 Use Cases

- **Audio Quality Control**: Detect insertion/connection events in recordings
- **Manufacturing Testing**: Analyze mechanical connection processes
- **Research**: Study acoustic signatures of physical events
- **Batch Processing**: Analyze large collections of audio files
- **Comparative Analysis**: Compare different recording conditions or setups

## 🛠️ Command Line Options

```bash
python pipeline.py --help

Options:
  -c, --config PATH        Path to configuration YAML file
  -s, --skip-analysis      Skip audio analysis, use existing features
  -q, --quiet             Run with minimal output
  -o, --output-dir PATH   Override output directory
  -r, --reports-dir PATH  Override reports directory
```

## 📋 Requirements

- Python 3.8+
- NumPy, SciPy - Numerical computing
- Librosa - Audio analysis
- Matplotlib, Seaborn - Plotting
- Pandas - Data manipulation
- PyYAML - Configuration files
- Plotly - Interactive visualizations
- Jinja2 - HTML templating

## 🤝 Contributing

This pipeline is designed to be extensible. Key areas for enhancement:
- Additional feature extraction methods
- New visualization types
- Alternative event detection algorithms
- Export formats (PDF, Excel, etc.)
- Real-time analysis capabilities

## 📄 License

[Your license information here]

## 🔗 Dependencies

See `requirements.txt` for complete dependency list.