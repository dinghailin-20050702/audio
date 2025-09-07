#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Audio Analysis Pipeline

Validates that all components work correctly and produce expected outputs.
"""

import os
import tempfile
import shutil
import pandas as pd
from pipeline import run_pipeline, load_config

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    # Test default config
    config = load_config('config.yaml')
    assert isinstance(config, dict), "Config should be a dictionary"
    assert 'output_dir' in config, "Config should have output_dir"
    
    # Test non-existent config (should return default)
    config_default = load_config('non_existent.yaml')
    assert isinstance(config_default, dict), "Should return default config for non-existent file"
    
    print("  ✅ Configuration loading tests passed")

def test_pipeline_with_existing_data():
    """Test pipeline with existing feature data"""
    print("Testing pipeline with existing data...")
    
    # Check if we have existing data
    if not os.path.exists('analysis_out/features.csv'):
        print("  ⚠️  No existing features.csv found, skipping this test")
        return
    
    # Run pipeline skipping analysis
    results = run_pipeline(skip_analysis=True, verbose=False)
    
    # Validate results
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'html_report' in results, "Results should include html_report"
    assert 'total_files' in results, "Results should include total_files"
    assert results['total_files'] > 0, "Should have processed some files"
    
    # Check that HTML report exists
    assert os.path.exists(results['html_report']), "HTML report should exist"
    
    # Check that report has reasonable size (not empty)
    report_size = os.path.getsize(results['html_report'])
    assert report_size > 10000, "HTML report should be substantial (>10KB)"
    
    print(f"  ✅ Pipeline test passed - processed {results['total_files']} files")

def test_feature_data_integrity():
    """Test that feature data has expected structure"""
    print("Testing feature data integrity...")
    
    features_csv = 'analysis_out/features.csv'
    if not os.path.exists(features_csv):
        print("  ⚠️  No features.csv found, skipping this test")
        return
    
    # Load and validate feature data
    df = pd.read_csv(features_csv)
    
    # Basic structure tests
    assert len(df) > 0, "Features dataframe should not be empty"
    assert 'file' in df.columns, "Should have 'file' column"
    assert 'event_t' in df.columns, "Should have 'event_t' column"
    
    # Test for key feature columns
    expected_prefixes = ['ev_', 'bg_', 'diff_']
    for prefix in expected_prefixes:
        cols_with_prefix = [col for col in df.columns if col.startswith(prefix)]
        assert len(cols_with_prefix) > 0, f"Should have columns starting with '{prefix}'"
    
    # Test that numeric columns are actually numeric
    numeric_cols = [col for col in df.columns if col.startswith(('ev_', 'bg_', 'diff_'))]
    for col in numeric_cols[:5]:  # Test first 5 to avoid excessive output
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
    
    print(f"  ✅ Feature data integrity test passed - {len(df)} rows, {len(df.columns)} columns")

def test_visualization_outputs():
    """Test that visualization outputs are generated correctly"""
    print("Testing visualization outputs...")
    
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        print("  ⚠️  No reports directory found, skipping this test")
        return
    
    # Expected output files
    expected_files = [
        'audio_analysis_report.html',
        'correlation_heatmap.png',
        'feature_distributions.png',
        'comparative_analysis.png',
        'summary_dashboard.png'
    ]
    
    missing_files = []
    for filename in expected_files:
        filepath = os.path.join(reports_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        else:
            # Check file is not empty
            size = os.path.getsize(filepath)
            assert size > 1000, f"{filename} should be substantial (>1KB), got {size} bytes"
    
    if missing_files:
        print(f"  ⚠️  Missing files: {missing_files}")
    else:
        print("  ✅ All expected visualization files present and non-empty")

def test_html_report_structure():
    """Test that HTML report has expected structure"""
    print("Testing HTML report structure...")
    
    html_report = 'reports/audio_analysis_report.html'
    if not os.path.exists(html_report):
        print("  ⚠️  No HTML report found, skipping this test")
        return
    
    # Read HTML content
    with open(html_report, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Test for key sections
    expected_sections = [
        '<title>Audio Analysis Report</title>',
        'Summary Statistics',
        'Most Suspicious Files',
        'Analysis Dashboard',
        'Individual Audio Analysis'
    ]
    
    missing_sections = []
    for section in expected_sections:
        if section not in html_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"  ⚠️  Missing HTML sections: {missing_sections}")
    else:
        print("  ✅ HTML report structure test passed")
    
    # Test that images are embedded (base64)
    assert 'data:image/png;base64,' in html_content, "Should contain embedded images"

def test_interactive_plots():
    """Test that interactive plots are generated"""
    print("Testing interactive plots...")
    
    interactive_file = 'reports/interactive_3d_analysis.html'
    if not os.path.exists(interactive_file):
        print("  ⚠️  No interactive plot found, skipping this test")
        return
    
    # Check file size (Plotly files are typically large)
    size = os.path.getsize(interactive_file)
    assert size > 100000, f"Interactive plot should be substantial (>100KB), got {size} bytes"
    
    # Check for Plotly content
    with open(interactive_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert 'plotly' in content.lower(), "Should contain Plotly content"
    assert 'scatter3d' in content.lower(), "Should contain 3D scatter plot"
    
    print("  ✅ Interactive plots test passed")

def run_all_tests():
    """Run all tests"""
    print("🧪 Audio Analysis Pipeline - Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_pipeline_with_existing_data,
        test_feature_data_integrity,
        test_visualization_outputs,
        test_html_report_structure,
        test_interactive_plots
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)