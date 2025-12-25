#!/usr/bin/env python
"""
Quick comparison script for analyzing experiment results.

Usage:
    python compare_experiments.py
    python compare_experiments.py --results_dir ./results --output_dir ./comparison_report
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.experiment_comparator import ExperimentComparator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare time series forecasting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all experiments in default results directory
  python compare_experiments.py
  
  # Specify custom results directory
  python compare_experiments.py --results_dir ./my_results
  
  # Specify custom output directory
  python compare_experiments.py --output_dir ./my_comparison
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Results directory to analyze (default: ./results)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for comparison report (default: ./results/_comparison_report)'
    )
    
    args = parser.parse_args()
    
    # Create comparator and generate report
    comparator = ExperimentComparator(results_dir=args.results_dir)
    output_path = comparator.generate_full_comparison_report(output_dir=args.output_dir)
    
    print(f"\n🎉 Done! Open {os.path.join(output_path, 'SUMMARY.md')} to see the summary.\n")


if __name__ == '__main__':
    main()
