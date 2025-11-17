#!/usr/bin/env python3
"""
Demo Workflow Script for X Community Notes Analysis

This script demonstrates the complete workflow of the data science project,
showcasing topic classification, analysis, and visualization capabilities.

Usage:
    python code/demo_workflow.py [--quick] [--full]
    
    --quick: Run with 10K notes (faster, for demonstration)
    --full: Run with full dataset (slower, complete analysis)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.classification.topic_classifier import CustomTopicClassifier
from code.classification.analyze_classification_results import ClassificationAnalyzer


def demo_classification(max_notes: int | None = None) -> dict:
    """
    Demonstrate the topic classification workflow.
    
    Args:
        max_notes: Maximum number of notes to process (None for all)
        
    Returns:
        Dictionary containing classification results
    """
    print("\n" + "="*70)
    print("DEMO: TOPIC CLASSIFICATION WORKFLOW")
    print("="*70)
    
    # Initialize classifier
    print("\n📦 Step 1: Initializing Topic Classifier")
    print("-" * 50)
    classifier = CustomTopicClassifier(
        data_path="data",
        output_dir="demo_results/classification"
    )
    
    # Show available topics
    print("\n📋 Step 2: Available Topic Categories")
    print("-" * 50)
    classifier.show_seed_term_examples()
    
    # Run classification pipeline
    print("\n🚀 Step 3: Running Classification Pipeline")
    print("-" * 50)
    if max_notes:
        print(f"   Processing {max_notes:,} notes (demo mode)")
    else:
        print("   Processing full dataset")
    
    results = classifier.run_complete_pipeline(
        max_notes=max_notes,
        english_only=True,
        force_refilter=False
    )
    
    print("\n✅ Classification Complete!")
    print(f"   Results saved in: {classifier.output_dir}")
    
    return results


def demo_analysis(results_dir: str | Path = "demo_results/classification") -> dict:
    """
    Demonstrate the analysis and visualization workflow.
    
    Args:
        results_dir: Directory containing classification results
        
    Returns:
        Dictionary containing analysis results
    """
    print("\n" + "="*70)
    print("DEMO: RESULTS ANALYSIS & VISUALIZATION")
    print("="*70)
    
    # Initialize analyzer
    print("\n📊 Step 1: Initializing Results Analyzer")
    print("-" * 50)
    analyzer = ClassificationAnalyzer(
        results_dir=results_dir,
        output_dir="demo_results/analytics"
    )
    
    # Run complete analysis
    print("\n🎯 Step 2: Running Comprehensive Analysis")
    print("-" * 50)
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Analysis Complete!")
    print(f"   Visualizations saved in: {analyzer.output_dir}")
    
    return results


def print_summary(classification_results: dict, analysis_results: dict) -> None:
    """
    Print a summary of the demo results.
    
    Args:
        classification_results: Results from classification
        analysis_results: Results from analysis
    """
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    if classification_results:
        print("\n📊 Classification Results:")
        print("-" * 50)
        if 'analysis_results' in classification_results:
            analysis = classification_results['analysis_results']
            print(f"   Total Notes Processed: {analysis.get('total_notes', 'N/A'):,}")
            print(f"   Topics Assigned: {analysis.get('topics_assigned', 'N/A')}")
            if 'confidence_stats' in analysis:
                conf = analysis['confidence_stats']
                print(f"   Mean Confidence: {conf.get('mean', 0):.3f}")
    
    if analysis_results:
        print("\n📈 Analysis Results:")
        print("-" * 50)
        if 'summary_statistics' in analysis_results:
            stats = analysis_results['summary_statistics']
            print(f"   Total Notes: {stats.get('total_notes', 'N/A'):,}")
            print(f"   Unique Topics: {stats.get('unique_topics', 'N/A')}")
            print(f"   Mean Confidence: {stats.get('mean_confidence', 0):.3f}")
    
    print("\n📁 Output Files:")
    print("-" * 50)
    print("   Classification results: demo_results/classification/")
    print("   Analysis visualizations: demo_results/analytics/")
    print("\n✅ Demo completed successfully!")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demo workflow for X Community Notes Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/demo_workflow.py --quick    # Quick demo with 10K notes
  python code/demo_workflow.py --full     # Full demo with all notes
  python code/demo_workflow.py            # Default: 10K notes
        """
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick demo with 10K notes (default)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full demo with all notes (slower)'
    )
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Skip classification, only run analysis on existing results'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("X COMMUNITY NOTES ANALYSIS - DEMO WORKFLOW")
    print("="*70)
    print("\nThis demo showcases the complete data science workflow:")
    print("  1. Topic Classification (15-topic system)")
    print("  2. Results Analysis & Visualization")
    print("  3. Summary Report Generation")
    
    try:
        # Determine number of notes to process
        max_notes = None if args.full else 10000
        
        if args.analysis_only:
            # Only run analysis
            analysis_results = demo_analysis()
            print_summary({}, analysis_results)
        else:
            # Run full workflow
            classification_results = demo_classification(max_notes=max_notes)
            analysis_results = demo_analysis(results_dir=classification_results.get('file_paths', {}).get('classified_notes', 'demo_results/classification'))
            print_summary(classification_results, analysis_results)
        
        print("\n" + "="*70)
        print("🎉 DEMO COMPLETE - Ready for Presentation!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

