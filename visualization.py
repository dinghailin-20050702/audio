#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Audio Analysis Visualization Module

Provides comprehensive visualization functions for audio analysis reporting pipeline,
including correlation heatmaps, distribution plots, comparative analysis, and dashboard views.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Tuple, Optional

# Set default plotly template
pio.templates.default = "plotly_white"

class AudioVisualization:
    """Enhanced visualization functions for audio analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config.get('output_dir', 'analysis_out')
        self.reports_dir = config.get('reports_dir', 'reports')
        self.fig_dpi = config.get('visualization', {}).get('figure_dpi', 140)
        self.color_scheme = config.get('visualization', {}).get('color_scheme', 'viridis')
        self.font_size = config.get('visualization', {}).get('font_size', 10)
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set matplotlib defaults
        plt.rcParams.update({'font.size': self.font_size})
    
    def create_feature_correlation_heatmap(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create correlation heatmap of audio features"""
        
        # Select numeric diff_ columns for correlation analysis
        diff_cols = [col for col in df.columns if col.startswith('diff_') and df[col].dtype in ['float64', 'int64']]
        
        if len(diff_cols) < 2:
            print("Warning: Not enough diff columns for correlation analysis")
            return None
            
        corr_data = df[diff_cols].corr()
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.fig_dpi)
        
        # Create heatmap
        sns.heatmap(corr_data, annot=True, cmap=self.color_scheme, center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix (Event vs Background Differences)', 
                    fontsize=self.font_size + 2, pad=20)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.reports_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_feature_distributions(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create distribution plots for key features"""
        
        # Key features to plot
        key_features = [
            'diff_centroid_mean',
            'diff_band_3000-8000_ratio',
            'diff_band_1000-3000_ratio',
            'diff_flatness_mean',
            'ev_crest_factor',
            '_suspect_plug_score'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [f for f in key_features if f in df.columns]
        
        if not available_features:
            print("Warning: No key features found for distribution plots")
            return None
        
        # Create subplots
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), dpi=self.fig_dpi)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(available_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Create histogram with kde
            data = df[feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add kde if seaborn is available
                try:
                    sns.kdeplot(data=data, ax=ax, color='red', linewidth=2)
                except:
                    pass
                
                ax.set_title(f'Distribution of {feature}', fontsize=self.font_size)
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f'Mean: {data.mean():.3f}\nStd: {data.std():.3f}\nMedian: {data.median():.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(available_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Feature Distributions', fontsize=self.font_size + 4, y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.reports_dir, 'feature_distributions.png')
        plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_comparative_analysis(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create comparative analysis plots (event vs background)"""
        
        # Key comparison pairs
        comparison_features = [
            ('ev_centroid_mean', 'bg_centroid_mean', 'Spectral Centroid (Hz)'),
            ('ev_rms_dbfs', 'bg_rms_dbfs', 'RMS Level (dBFS)'),
            ('ev_crest_factor', 'bg_crest_factor', 'Crest Factor'),
            ('ev_band_3000-8000_ratio', 'bg_band_3000-8000_ratio', 'High Freq Ratio (3-8 kHz)')
        ]
        
        # Filter available comparisons
        available_comparisons = []
        for ev_col, bg_col, title in comparison_features:
            if ev_col in df.columns and bg_col in df.columns:
                available_comparisons.append((ev_col, bg_col, title))
        
        if not available_comparisons:
            print("Warning: No comparison features found")
            return None
        
        n_comparisons = len(available_comparisons)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.fig_dpi)
        axes = axes.flatten()
        
        for i, (ev_col, bg_col, title) in enumerate(available_comparisons[:4]):
            ax = axes[i]
            
            # Create scatter plot
            ax.scatter(df[bg_col], df[ev_col], alpha=0.7, s=50)
            
            # Add diagonal line (y=x)
            min_val = min(df[bg_col].min(), df[ev_col].min())
            max_val = max(df[bg_col].max(), df[ev_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Equal values')
            
            ax.set_xlabel(f'Background {title}')
            ax.set_ylabel(f'Event {title}')
            ax.set_title(f'{title}: Event vs Background')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add correlation coefficient
            corr = df[ev_col].corr(df[bg_col])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(available_comparisons), 4):
            axes[i].axis('off')
        
        plt.suptitle('Event vs Background Feature Comparison', fontsize=self.font_size + 4)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.reports_dir, 'comparative_analysis.png')
        plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_dashboard(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create a summary dashboard with key metrics"""
        
        fig = plt.figure(figsize=(20, 12), dpi=self.fig_dpi)
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Suspect Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if '_suspect_plug_score' in df.columns:
            score_data = df['_suspect_plug_score'].dropna()
            ax1.hist(score_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax1.axvline(score_data.median(), color='red', linestyle='--', label=f'Median: {score_data.median():.2f}')
            ax1.set_title('Suspect Score Distribution')
            ax1.set_xlabel('Suspect Score')
            ax1.set_ylabel('Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. High Frequency Energy Changes
        ax2 = fig.add_subplot(gs[0, 1])
        if 'diff_band_3000-8000_ratio' in df.columns:
            hf_data = df['diff_band_3000-8000_ratio'].dropna()
            colors = ['red' if x > 0.05 else 'blue' for x in hf_data]
            ax2.scatter(range(len(hf_data)), hf_data, c=colors, alpha=0.7)
            ax2.axhline(0.05, color='red', linestyle='--', label='Threshold (0.05)')
            ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('High Freq Energy Change')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Ratio Change')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Spectral Centroid Changes
        ax3 = fig.add_subplot(gs[0, 2])
        if 'diff_centroid_mean' in df.columns:
            centroid_data = df['diff_centroid_mean'].dropna()
            colors = ['red' if x > 1000 else 'blue' for x in centroid_data]
            ax3.scatter(range(len(centroid_data)), centroid_data, c=colors, alpha=0.7)
            ax3.axhline(1000, color='red', linestyle='--', label='Threshold (1000 Hz)')
            ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Spectral Centroid Change')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Frequency Change (Hz)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Feature Summary Statistics (text)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        stats_text = f"""
        Dataset Summary:
        • Total samples: {len(df)}
        • Audio files analyzed: {len(df)}
        
        Key Metrics (Median):
        """
        
        if 'diff_centroid_mean' in df.columns:
            stats_text += f"• Spectral centroid change: {df['diff_centroid_mean'].median():.0f} Hz\n"
        if 'diff_band_3000-8000_ratio' in df.columns:
            stats_text += f"• High freq energy change: {df['diff_band_3000-8000_ratio'].median():.3f}\n"
        if 'ev_crest_factor' in df.columns:
            stats_text += f"• Event crest factor: {df['ev_crest_factor'].median():.2f}\n"
        
        # Detection rates
        if 'diff_band_3000-8000_ratio' in df.columns:
            high_freq_rate = (df['diff_band_3000-8000_ratio'] > 0.05).mean() * 100
            stats_text += f"\nDetection Rates:\n• High freq increase >5%: {high_freq_rate:.0f}%\n"
        
        if 'diff_centroid_mean' in df.columns:
            centroid_rate = (df['diff_centroid_mean'] > 1000).mean() * 100
            stats_text += f"• Centroid increase >1kHz: {centroid_rate:.0f}%\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=self.font_size,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 5-8. Top suspicious files
        if '_suspect_plug_score' in df.columns:
            top_files = df.nlargest(4, '_suspect_plug_score')
            for i, (idx, row) in enumerate(top_files.iterrows()):
                ax = fig.add_subplot(gs[1, i])
                ax.axis('off')
                
                file_name = os.path.basename(row['file']) if 'file' in row else f"Sample {idx}"
                score = row['_suspect_plug_score']
                
                info_text = f"File: {file_name}\nScore: {score:.3f}\n"
                
                if 'diff_centroid_mean' in row:
                    info_text += f"Centroid Δ: {row['diff_centroid_mean']:.0f} Hz\n"
                if 'diff_band_3000-8000_ratio' in row:
                    info_text += f"HF Ratio Δ: {row['diff_band_3000-8000_ratio']:.3f}\n"
                if 'ev_crest_factor' in row:
                    info_text += f"Crest Factor: {row['ev_crest_factor']:.2f}\n"
                
                ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=self.font_size-1,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                ax.set_title(f'Top Suspicious #{i+1}', fontsize=self.font_size)
        
        # 9. Feature Correlation Mini-Heatmap
        ax9 = fig.add_subplot(gs[2, :2])
        diff_cols = [col for col in df.columns if col.startswith('diff_') and df[col].dtype in ['float64', 'int64']][:6]
        if len(diff_cols) > 1:
            corr_data = df[diff_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, square=True, 
                       fmt='.2f', cbar_kws={"shrink": .6}, ax=ax9)
            ax9.set_title('Feature Correlation (Top 6)')
        
        # 10. Event Detection Timeline
        ax10 = fig.add_subplot(gs[2, 2:])
        if 'event_t' in df.columns and 'file' in df.columns:
            # Extract timestamps from filenames if possible
            try:
                file_times = []
                for f in df['file']:
                    # Try to extract timestamp from filename
                    import re
                    match = re.search(r'(\d{8}_\d{6})', f)
                    if match:
                        file_times.append(match.group(1))
                    else:
                        file_times.append(f"File_{len(file_times)}")
                
                y_pos = range(len(df))
                colors = plt.cm.viridis(df['_suspect_plug_score'] if '_suspect_plug_score' in df.columns else [0.5]*len(df))
                
                scatter = ax10.scatter(df['event_t'], y_pos, c=colors, s=60, alpha=0.8)
                ax10.set_xlabel('Event Time (seconds)')
                ax10.set_ylabel('File Index')
                ax10.set_title('Event Detection Timeline')
                ax10.grid(True, alpha=0.3)
                
                # Add colorbar
                if '_suspect_plug_score' in df.columns:
                    plt.colorbar(scatter, ax=ax10, label='Suspect Score')
            except:
                ax10.text(0.5, 0.5, 'Timeline visualization\nnot available', 
                         transform=ax10.transAxes, ha='center', va='center')
                ax10.set_title('Event Detection Timeline')
        
        plt.suptitle('Audio Analysis Dashboard', fontsize=self.font_size + 6, y=0.95)
        
        if save_path is None:
            save_path = os.path.join(self.reports_dir, 'summary_dashboard.png')
        plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_plots(self, df: pd.DataFrame) -> List[str]:
        """Create interactive Plotly visualizations"""
        
        plot_paths = []
        
        # 1. Interactive 3D scatter plot
        if all(col in df.columns for col in ['diff_centroid_mean', 'diff_band_3000-8000_ratio', 'ev_crest_factor']):
            fig_3d = go.Figure(data=go.Scatter3d(
                x=df['diff_centroid_mean'],
                y=df['diff_band_3000-8000_ratio'],
                z=df['ev_crest_factor'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['_suspect_plug_score'] if '_suspect_plug_score' in df.columns else 'blue',
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Suspect Score")
                ),
                text=[os.path.basename(f) if isinstance(f, str) else f"Sample {i}" 
                      for i, f in enumerate(df.get('file', range(len(df))))],
                hovertemplate='<b>%{text}</b><br>' +
                             'Centroid Change: %{x:.0f} Hz<br>' +
                             'HF Ratio Change: %{y:.3f}<br>' +
                             'Crest Factor: %{z:.2f}<extra></extra>'
            ))
            
            fig_3d.update_layout(
                title='3D Feature Space Analysis',
                scene=dict(
                    xaxis_title='Spectral Centroid Change (Hz)',
                    yaxis_title='High Freq Ratio Change',
                    zaxis_title='Crest Factor'
                ),
                width=800,
                height=600
            )
            
            path_3d = os.path.join(self.reports_dir, 'interactive_3d_analysis.html')
            fig_3d.write_html(path_3d)
            plot_paths.append(path_3d)
        
        return plot_paths
    
    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate all visualization plots"""
        
        plot_paths = {}
        
        print("Generating correlation heatmap...")
        corr_path = self.create_feature_correlation_heatmap(df)
        if corr_path:
            plot_paths['correlation_heatmap'] = corr_path
        
        print("Generating feature distributions...")
        dist_path = self.create_feature_distributions(df)
        if dist_path:
            plot_paths['feature_distributions'] = dist_path
        
        print("Generating comparative analysis...")
        comp_path = self.create_comparative_analysis(df)
        if comp_path:
            plot_paths['comparative_analysis'] = comp_path
        
        print("Generating summary dashboard...")
        dash_path = self.create_summary_dashboard(df)
        if dash_path:
            plot_paths['summary_dashboard'] = dash_path
        
        print("Generating interactive plots...")
        interactive_paths = self.create_interactive_plots(df)
        for i, path in enumerate(interactive_paths):
            plot_paths[f'interactive_{i}'] = path
        
        return plot_paths