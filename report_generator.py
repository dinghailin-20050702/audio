#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML Report Generator for Audio Analysis Pipeline

Generates comprehensive HTML reports with embedded visualizations, statistics,
and recommendations based on audio analysis results.
"""

import os
import base64
import datetime
from typing import Dict, List, Optional
import pandas as pd
import yaml
from jinja2 import Template

class HTMLReportGenerator:
    """Generate comprehensive HTML reports for audio analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reports_dir = config.get('reports_dir', 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 for embedding in HTML"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not encode image {image_path}: {e}")
            return ""
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the report"""
        stats = {
            'total_files': len(df),
            'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Key feature statistics
        if 'diff_centroid_mean' in df.columns:
            stats['centroid_change_median'] = df['diff_centroid_mean'].median()
            stats['centroid_change_std'] = df['diff_centroid_mean'].std()
            stats['centroid_increase_pct'] = (df['diff_centroid_mean'] > 1000).mean() * 100
        
        if 'diff_band_3000-8000_ratio' in df.columns:
            stats['hf_ratio_change_median'] = df['diff_band_3000-8000_ratio'].median()
            stats['hf_ratio_change_std'] = df['diff_band_3000-8000_ratio'].std()
            stats['hf_increase_pct'] = (df['diff_band_3000-8000_ratio'] > 0.05).mean() * 100
        
        if 'diff_flatness_mean' in df.columns:
            stats['flatness_change_median'] = df['diff_flatness_mean'].median()
            stats['flatness_increase_pct'] = (df['diff_flatness_mean'] > 0.05).mean() * 100
        
        if '_suspect_plug_score' in df.columns:
            stats['suspect_score_median'] = df['_suspect_plug_score'].median()
            stats['suspect_score_mean'] = df['_suspect_plug_score'].mean()
            stats['high_suspect_pct'] = (df['_suspect_plug_score'] > 0.7).mean() * 100
        
        if 'ev_crest_factor' in df.columns:
            stats['crest_factor_median'] = df['ev_crest_factor'].median()
            stats['crest_factor_mean'] = df['ev_crest_factor'].mean()
        
        return stats
    
    def _get_top_suspicious_files(self, df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """Get top suspicious files with details"""
        if '_suspect_plug_score' not in df.columns:
            return []
        
        top_files = df.nlargest(n, '_suspect_plug_score')
        
        results = []
        for idx, row in top_files.iterrows():
            file_info = {
                'filename': os.path.basename(row.get('file', f'Sample_{idx}')),
                'full_path': row.get('file', f'Sample_{idx}'),
                'suspect_score': row['_suspect_plug_score'],
                'event_time': row.get('event_t', 0),
                'features': {}
            }
            
            # Add key features
            feature_keys = [
                ('diff_centroid_mean', 'Spectral Centroid Change (Hz)', '{:.0f}'),
                ('diff_band_3000-8000_ratio', 'High Freq Ratio Change', '{:.3f}'),
                ('diff_band_1000-3000_ratio', 'Mid Freq Ratio Change', '{:.3f}'),
                ('diff_flatness_mean', 'Spectral Flatness Change', '{:.4f}'),
                ('ev_crest_factor', 'Event Crest Factor', '{:.2f}'),
                ('ev_rms_dbfs', 'Event RMS Level (dBFS)', '{:.1f}'),
                ('bg_rms_dbfs', 'Background RMS Level (dBFS)', '{:.1f}')
            ]
            
            for key, name, fmt in feature_keys:
                if key in row and pd.notna(row[key]):
                    file_info['features'][name] = fmt.format(row[key])
            
            results.append(file_info)
        
        return results
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate analysis recommendations based on statistics"""
        recommendations = []
        
        # High frequency energy analysis
        if 'hf_increase_pct' in stats:
            if stats['hf_increase_pct'] > 80:
                recommendations.append(
                    "🔴 HIGH ALERT: Over 80% of samples show significant high-frequency energy increases, "
                    "strongly suggesting insertion/connection events."
                )
            elif stats['hf_increase_pct'] > 50:
                recommendations.append(
                    "🟡 MODERATE: Over 50% of samples show high-frequency increases, "
                    "indicating possible insertion events."
                )
            else:
                recommendations.append(
                    "🟢 LOW: Less than 50% show significant high-frequency increases."
                )
        
        # Spectral centroid analysis
        if 'centroid_increase_pct' in stats:
            if stats['centroid_increase_pct'] > 30:
                recommendations.append(
                    "🔴 Significant spectral centroid shifts detected in over 30% of samples, "
                    "indicating frequency content changes consistent with mechanical events."
                )
            elif stats['centroid_increase_pct'] > 10:
                recommendations.append(
                    "🟡 Moderate spectral centroid shifts in some samples."
                )
        
        # Suspect score analysis
        if 'high_suspect_pct' in stats:
            if stats['high_suspect_pct'] > 20:
                recommendations.append(
                    f"🔴 {stats['high_suspect_pct']:.1f}% of files have high suspicion scores (>0.7). "
                    "Manual review of top-scoring files is recommended."
                )
        
        # Crest factor analysis
        if 'crest_factor_median' in stats:
            if stats['crest_factor_median'] > 20:
                recommendations.append(
                    "⚡ High crest factors detected, indicating sharp transient events typical of "
                    "mechanical insertions or connections."
                )
        
        # General recommendations
        recommendations.extend([
            "📊 Review the individual spectrograms for files with highest suspect scores.",
            "🔍 Compare event windows with background windows in suspicious files.",
            "📈 Monitor trends if this analysis is run regularly.",
            "⚙️ Consider adjusting detection thresholds based on false positive rates."
        ])
        
        return recommendations
    
    def generate_html_report(self, df: pd.DataFrame, plot_paths: Dict[str, str], 
                           individual_plots_dir: str = None) -> str:
        """Generate comprehensive HTML report"""
        
        # Calculate statistics
        stats = self._calculate_statistics(df)
        
        # Get top suspicious files
        top_files = self._get_top_suspicious_files(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats)
        
        # Encode plot images
        encoded_plots = {}
        for plot_name, plot_path in plot_paths.items():
            if plot_path and os.path.exists(plot_path) and plot_path.endswith('.png'):
                encoded_plots[plot_name] = self._encode_image(plot_path)
        
        # Get individual plot files if directory provided
        individual_plots = []
        if individual_plots_dir and os.path.exists(individual_plots_dir):
            for filename in sorted(os.listdir(individual_plots_dir)):
                if filename.endswith('.png'):
                    plot_path = os.path.join(individual_plots_dir, filename)
                    encoded_image = self._encode_image(plot_path)
                    if encoded_image:
                        individual_plots.append({
                            'filename': filename,
                            'basename': os.path.splitext(filename)[0],
                            'encoded_image': encoded_image
                        })
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .recommendations {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 20px;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin-bottom: 10px;
            line-height: 1.6;
        }
        .suspicious-files {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 8px;
            padding: 20px;
        }
        .file-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .filename {
            font-weight: bold;
            color: #2d3748;
        }
        .score-badge {
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            font-size: 0.9em;
        }
        .feature-item {
            padding: 5px;
            background: #f7fafc;
            border-radius: 4px;
        }
        .individual-plots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .individual-plot {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .individual-plot h4 {
            margin: 0 0 15px 0;
            color: #2d3748;
        }
        .individual-plot img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .footer {
            background: #2d3748;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }
        @media (max-width: 768px) {
            body { padding: 10px; }
            .content { padding: 15px; }
            .stats-grid { grid-template-columns: 1fr; }
            .individual-plots { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Audio Analysis Report</h1>
            <p>Generated on {{ stats.analysis_date }}</p>
        </div>
        
        <div class="content">
            <!-- Summary Statistics -->
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{{ stats.total_files }}</div>
                        <div class="stat-label">Files Analyzed</div>
                    </div>
                    {% if stats.centroid_change_median is defined %}
                    <div class="stat-card">
                        <div class="stat-number">{{ "%.0f"|format(stats.centroid_change_median) }} Hz</div>
                        <div class="stat-label">Median Spectral Centroid Change</div>
                    </div>
                    {% endif %}
                    {% if stats.hf_ratio_change_median is defined %}
                    <div class="stat-card">
                        <div class="stat-number">{{ "%.3f"|format(stats.hf_ratio_change_median) }}</div>
                        <div class="stat-label">Median High-Freq Ratio Change</div>
                    </div>
                    {% endif %}
                    {% if stats.suspect_score_median is defined %}
                    <div class="stat-card">
                        <div class="stat-number">{{ "%.2f"|format(stats.suspect_score_median) }}</div>
                        <div class="stat-label">Median Suspect Score</div>
                    </div>
                    {% endif %}
                    {% if stats.hf_increase_pct is defined %}
                    <div class="stat-card">
                        <div class="stat-number">{{ "%.0f"|format(stats.hf_increase_pct) }}%</div>
                        <div class="stat-label">Files with High-Freq Increases</div>
                    </div>
                    {% endif %}
                    {% if stats.crest_factor_median is defined %}
                    <div class="stat-card">
                        <div class="stat-number">{{ "%.1f"|format(stats.crest_factor_median) }}</div>
                        <div class="stat-label">Median Crest Factor</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <!-- Analysis Results -->
            {% if encoded_plots.summary_dashboard %}
            <div class="section">
                <h2>🎯 Analysis Dashboard</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ encoded_plots.summary_dashboard }}" alt="Summary Dashboard">
                </div>
            </div>
            {% endif %}
            
            {% if encoded_plots.correlation_heatmap %}
            <div class="section">
                <h2>🔗 Feature Correlations</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ encoded_plots.correlation_heatmap }}" alt="Correlation Heatmap">
                </div>
            </div>
            {% endif %}
            
            {% if encoded_plots.feature_distributions %}
            <div class="section">
                <h2>📈 Feature Distributions</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ encoded_plots.feature_distributions }}" alt="Feature Distributions">
                </div>
            </div>
            {% endif %}
            
            {% if encoded_plots.comparative_analysis %}
            <div class="section">
                <h2>⚖️ Event vs Background Comparison</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ encoded_plots.comparative_analysis }}" alt="Comparative Analysis">
                </div>
            </div>
            {% endif %}
            
            <!-- Recommendations -->
            <div class="section">
                <h2>💡 Analysis Recommendations</h2>
                <div class="recommendations">
                    <ul>
                    {% for recommendation in recommendations %}
                        <li>{{ recommendation }}</li>
                    {% endfor %}
                    </ul>
                </div>
            </div>
            
            <!-- Top Suspicious Files -->
            {% if top_files %}
            <div class="section">
                <h2>🚨 Most Suspicious Files</h2>
                <div class="suspicious-files">
                    {% for file in top_files %}
                    <div class="file-card">
                        <div class="file-header">
                            <span class="filename">{{ file.filename }}</span>
                            <span class="score-badge">Score: {{ "%.3f"|format(file.suspect_score) }}</span>
                        </div>
                        <div class="features-grid">
                            {% for feature_name, feature_value in file.features.items() %}
                            <div class="feature-item">
                                <strong>{{ feature_name }}:</strong> {{ feature_value }}
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            <!-- Individual Plots -->
            {% if individual_plots %}
            <div class="section">
                <h2>🎵 Individual Audio Analysis</h2>
                <div class="individual-plots">
                    {% for plot in individual_plots %}
                    <div class="individual-plot">
                        <h4>{{ plot.basename }}</h4>
                        <img src="data:image/png;base64,{{ plot.encoded_image }}" alt="Analysis for {{ plot.basename }}">
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            Audio Analysis Pipeline • Generated with Python & Scientific Libraries
        </div>
    </div>
</body>
</html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            stats=stats,
            encoded_plots=encoded_plots,
            top_files=top_files,
            recommendations=recommendations,
            individual_plots=individual_plots
        )
        
        # Save HTML report
        report_path = os.path.join(self.reports_dir, 'audio_analysis_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path