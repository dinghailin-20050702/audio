#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analysis Pipeline - Usage Examples

This script demonstrates various ways to use the audio analysis pipeline.
"""

import os
import sys
from pipeline import run_pipeline, load_config

def example_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("Example 1: Basic Pipeline Usage")
    print("-" * 40)
    
    # Run the complete pipeline with default settings
    results = run_pipeline(verbose=True)
    
    print(f"Analysis completed!")
    print(f"- Processed {results['total_files']} audio files")
    print(f"- HTML report: {results['html_report']}")
    print(f"- CSV data: {results['features_csv']}")
    print()

def example_custom_config():
    """Example 2: Using custom configuration"""
    print("Example 2: Custom Configuration")
    print("-" * 40)
    
    # Load default config and modify
    config = load_config('config.yaml')
    
    # Modify some parameters
    config['thresholds']['high_frequency_increase'] = 0.08  # More strict threshold
    config['visualization']['color_scheme'] = 'plasma'      # Different color scheme
    config['reports_dir'] = 'custom_reports'               # Custom output directory
    
    # Save custom config
    import yaml
    with open('custom_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run with custom config
    results = run_pipeline(config_path='custom_config.yaml', verbose=True)
    
    print(f"Custom analysis completed!")
    print(f"- Results in: {results['reports_dir']}")
    print()

def example_programmatic_usage():
    """Example 3: Programmatic usage for integration"""
    print("Example 3: Programmatic Usage")
    print("-" * 40)
    
    try:
        # Run pipeline programmatically
        results = run_pipeline(verbose=False)  # Quiet mode
        
        # Access results programmatically
        if results:
            print("✅ Pipeline completed successfully")
            
            # Load the features data for further processing
            import pandas as pd
            df = pd.read_csv(results['features_csv'])
            
            # Example: Find files with highest suspect scores
            if '_suspect_plug_score' in df.columns:
                top_suspicious = df.nlargest(3, '_suspect_plug_score')
                print("Top 3 most suspicious files:")
                for _, row in top_suspicious.iterrows():
                    filename = os.path.basename(row.get('file', 'unknown'))
                    score = row['_suspect_plug_score']
                    print(f"  - {filename}: score={score:.3f}")
            
            # Example: Get summary statistics
            total_files = len(df)
            high_freq_increases = (df.get('diff_band_3000-8000_ratio', pd.Series()) > 0.05).sum()
            
            print(f"\nSummary Statistics:")
            print(f"  - Total files analyzed: {total_files}")
            print(f"  - Files with high-freq increases: {high_freq_increases}")
            print(f"  - Percentage with increases: {high_freq_increases/total_files*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def example_batch_processing():
    """Example 4: Batch processing multiple directories"""
    print("Example 4: Batch Processing")
    print("-" * 40)
    
    # Example of processing multiple recording directories
    recording_dirs = ['recordings', 'recordings2', 'archive']  # Example directories
    
    for i, rec_dir in enumerate(recording_dirs):
        if not os.path.exists(rec_dir):
            print(f"  Directory {rec_dir} not found, skipping...")
            continue
        
        print(f"  Processing directory: {rec_dir}")
        
        # Create custom config for this directory
        config = load_config('config.yaml')
        config['search_dirs'] = [rec_dir]
        config['output_dir'] = f'analysis_out_{i}'
        config['reports_dir'] = f'reports_{i}'
        
        # Save temporary config
        temp_config = f'temp_config_{i}.yaml'
        import yaml
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run analysis for this directory
            results = run_pipeline(config_path=temp_config, verbose=False)
            print(f"    ✅ Completed: {results['total_files']} files")
            
            # Cleanup temp config
            os.unlink(temp_config)
            
        except Exception as e:
            print(f"    ❌ Error processing {rec_dir}: {e}")
    
    print()

def example_analysis_only():
    """Example 5: Run only specific components"""
    print("Example 5: Component-specific Usage")
    print("-" * 40)
    
    # Example: Only run visualization on existing data
    features_csv = 'analysis_out/features.csv'
    
    if os.path.exists(features_csv):
        print("  Running only visualization and reporting...")
        
        # Run pipeline but skip analysis
        results = run_pipeline(skip_analysis=True, verbose=False)
        
        print(f"    ✅ Reports generated in: {results['reports_dir']}")
    else:
        print("  No existing features.csv found, run full analysis first")
    
    print()

def main():
    """Run all examples"""
    print("🎵 Audio Analysis Pipeline - Usage Examples")
    print("=" * 60)
    print()
    
    try:
        # Only run basic example to avoid cluttering output
        example_basic_usage()
        
        # Uncomment to run other examples:
        # example_custom_config()
        # example_programmatic_usage()
        # example_batch_processing()
        # example_analysis_only()
        
        print("📋 For more examples, edit this script and uncomment the desired examples.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()