#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analysis Reporting Pipeline

Main orchestrator script that runs the complete audio analysis pipeline
with enhanced visualizations and comprehensive reporting capabilities.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Import existing modules
from analyze_insertion_audio import main as run_audio_analysis
from summarize_features import main as run_summarize_features
from visualization import AudioVisualization
from report_generator import HTMLReportGenerator

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default settings.")
        return {
            'output_dir': 'analysis_out',
            'reports_dir': 'reports',
            'visualization': {'figure_dpi': 140, 'color_scheme': 'viridis', 'font_size': 10}
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def ensure_directories(config: Dict):
    """Ensure all required directories exist"""
    dirs_to_create = [
        config.get('output_dir', 'analysis_out'),
        config.get('reports_dir', 'reports')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def add_suspect_scoring(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add suspect scoring to the dataframe using config thresholds"""
    thresholds = config.get('thresholds', {})
    
    # Calculate suspect score if not already present
    if '_suspect_plug_score' not in df.columns:
        score_terms = []
        
        # High frequency energy increase
        if 'diff_band_3000-8000_ratio' in df.columns:
            hf_threshold = thresholds.get('high_frequency_increase', 0.05)
            hf_term = df['diff_band_3000-8000_ratio'].clip(lower=0)
            score_terms.append(hf_term)
        
        # Mid frequency energy increase
        if 'diff_band_1000-3000_ratio' in df.columns:
            mf_term = df['diff_band_1000-3000_ratio'].clip(lower=0) * 0.5
            score_terms.append(mf_term)
        
        # Spectral centroid increase
        if 'diff_centroid_mean' in df.columns:
            centroid_threshold = thresholds.get('spectral_centroid_increase', 1000)
            centroid_term = (df['diff_centroid_mean'] / (2 * centroid_threshold)).clip(lower=0, upper=1)
            score_terms.append(centroid_term)
        
        # Spectral flatness increase
        if 'diff_flatness_mean' in df.columns:
            flatness_threshold = thresholds.get('spectral_flatness_increase', 0.05)
            flatness_term = (df['diff_flatness_mean'] / flatness_threshold).clip(lower=0, upper=1)
            score_terms.append(flatness_term)
        
        # Combine scores
        if score_terms:
            total_score = sum(score_terms)
            # Normalize to 0-1 range
            min_score = total_score.min()
            max_score = total_score.max()
            if max_score > min_score:
                df['_suspect_plug_score'] = (total_score - min_score) / (max_score - min_score)
            else:
                df['_suspect_plug_score'] = 0.5  # Default middle score if all equal
        else:
            df['_suspect_plug_score'] = 0.5  # Default if no scoring features available
    
    return df

def run_pipeline(config_path: str = 'config.yaml', skip_analysis: bool = False, 
                verbose: bool = True) -> Dict[str, str]:
    """Run the complete audio analysis pipeline"""
    
    # Load configuration
    config = load_config(config_path)
    ensure_directories(config)
    
    output_dir = config.get('output_dir', 'analysis_out')
    reports_dir = config.get('reports_dir', 'reports')
    
    if verbose:
        print("🎵 Audio Analysis Reporting Pipeline")
        print("=" * 50)
    
    # Step 1: Run audio analysis (unless skipping)
    if not skip_analysis:
        if verbose:
            print("Step 1/5: Running audio analysis...")
        
        # Check if we need to run analysis
        features_csv = os.path.join(output_dir, 'features.csv')
        if not os.path.exists(features_csv):
            if verbose:
                print("  Running audio feature extraction...")
            run_audio_analysis()
        else:
            if verbose:
                print("  Features file already exists, skipping analysis...")
    else:
        if verbose:
            print("Step 1/5: Skipping audio analysis (--skip-analysis)")
    
    # Step 2: Load and validate features data
    if verbose:
        print("Step 2/5: Loading feature data...")
    
    features_csv = os.path.join(output_dir, 'features.csv')
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file not found: {features_csv}. Run audio analysis first.")
    
    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError("Features dataframe is empty")
    
    # Add suspect scoring using config
    df = add_suspect_scoring(df, config)
    
    if verbose:
        print(f"  Loaded {len(df)} audio samples")
    
    # Step 3: Generate enhanced visualizations
    if verbose:
        print("Step 3/5: Generating enhanced visualizations...")
    
    viz = AudioVisualization(config)
    plot_paths = viz.generate_all_plots(df)
    
    if verbose:
        for plot_type, path in plot_paths.items():
            if path:
                print(f"  ✓ Generated {plot_type}: {os.path.basename(path)}")
    
    # Step 4: Generate summary
    if verbose:
        print("Step 4/5: Generating feature summary...")
    
    summary_md = os.path.join(output_dir, 'summary.md')
    if not os.path.exists(summary_md):
        run_summarize_features()
    
    # Step 5: Generate comprehensive HTML report
    if verbose:
        print("Step 5/5: Generating HTML report...")
    
    report_gen = HTMLReportGenerator(config)
    report_path = report_gen.generate_html_report(
        df=df,
        plot_paths=plot_paths,
        individual_plots_dir=output_dir
    )
    
    if verbose:
        print(f"  ✓ HTML report generated: {report_path}")
    
    # Create results summary
    results = {
        'features_csv': features_csv,
        'summary_md': summary_md,
        'html_report': report_path,
        'total_files': len(df),
        'output_dir': output_dir,
        'reports_dir': reports_dir
    }
    
    # Add plot paths to results
    results.update(plot_paths)
    
    if verbose:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Analyzed {len(df)} audio files")
        print(f"📋 Results available in: {reports_dir}/")
        print(f"🌐 Open HTML report: {report_path}")
    
    return results

def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(
        description='Audio Analysis Reporting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run complete pipeline with default config
  %(prog)s --config my_config.yaml  # Use custom configuration
  %(prog)s --skip-analysis          # Skip audio analysis, use existing features
  %(prog)s --quiet                  # Run with minimal output
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--skip-analysis', '-s',
        action='store_true',
        help='Skip audio analysis step, use existing features.csv'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run with minimal output'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--reports-dir', '-r',
        type=str,
        help='Override reports directory from config'
    )
    
    args = parser.parse_args()
    
    try:
        # Load config and apply overrides
        config = load_config(args.config)
        
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.reports_dir:
            config['reports_dir'] = args.reports_dir
        
        # Save updated config temporarily
        if args.output_dir or args.reports_dir:
            temp_config_path = '/tmp/temp_config.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            config_path = temp_config_path
        else:
            config_path = args.config
        
        # Run pipeline
        results = run_pipeline(
            config_path=config_path,
            skip_analysis=args.skip_analysis,
            verbose=not args.quiet
        )
        
        # Clean up temp config
        if args.output_dir or args.reports_dir:
            os.unlink(temp_config_path)
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())