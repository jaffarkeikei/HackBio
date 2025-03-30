#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis Pipeline Status Check

This script checks the status of each step in the analysis pipeline by
looking for the relevant output files and directories.

Usage:
    python check_status.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

def check_status():
    """Check the status of each step in the analysis pipeline"""
    
    # Define expected output files for each step
    status = {
        "Step 1: Molecular Analysis": {
            "files": [
                "results/molecular_descriptors.csv",
                "results/drug_likeness.csv",
                "plots/descriptor_distributions.png",
                "plots/molecule_clusters.png",
                "plots/molecule_examples.png"
            ],
            "completed": False
        },
        "Step 2: Data Preprocessing": {
            "files": [
                "results/processed_data.parquet",
                "models/svd_model.pkl",
                "models/encoders.pkl"
            ],
            "completed": False
        },
        "Step 3: Model Training": {
            "files": [
                "models/ensemble_model.pkl",
                "models/top_genes_used.pkl",
                "results/gene_performance_metrics.csv",
                "results/ensemble_model_evaluation.csv",
                "plots/gene_predictions.png"
            ],
            "completed": False
        },
        "Step 4: Correlation Analysis": {
            "files": [
                "results/property_gene_correlations.csv",
                "results/correlation_p_values.csv",
                "results/top_property_gene_relationships.csv",
                "plots/property_gene_correlations.png",
                "plots/clustered_property_gene_correlations.png"
            ],
            "completed": False
        },
        "Step 5: Model Interpretation": {
            "files": [
                "results/model_evaluation.csv",
                "results/gene_performance.csv",
                "results/cell_type_performance.csv",
                "results/drug_performance.csv"
            ],
            "completed": False
        }
    }
    
    # Check if each file exists
    for step, info in status.items():
        files_found = 0
        for file in info["files"]:
            if os.path.exists(file):
                files_found += 1
        
        # Mark as completed if all expected files are found
        if files_found == len(info["files"]):
            status[step]["completed"] = True
        elif files_found > 0:
            status[step]["completed"] = "Partial"
        
        status[step]["files_found"] = files_found
        status[step]["total_files"] = len(info["files"])
    
    return status

def print_status(status):
    """Print a formatted status report"""
    print("\n" + "="*80)
    print(f"HACKBIO PIPELINE STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    overall_progress = 0
    total_files = 0
    files_found = 0
    
    for step, info in status.items():
        total_files += info["total_files"]
        files_found += info["files_found"]
        
        if info["completed"] is True:
            step_status = "✅ COMPLETE"
            overall_progress += 1
        elif info["completed"] == "Partial":
            step_status = f"⚠️ PARTIAL ({info['files_found']}/{info['total_files']} files)"
            overall_progress += info["files_found"] / info["total_files"]
        else:
            step_status = "❌ NOT STARTED"
        
        print(f"\n{step}")
        print("-" * len(step))
        print(f"Status: {step_status}")
        
        # Print missing files for partial steps
        if info["completed"] == "Partial":
            print("Missing files:")
            for file in info["files"]:
                if not os.path.exists(file):
                    print(f"  - {file}")
    
    # Calculate overall completion percentage
    overall_percent = (files_found / total_files) * 100 if total_files > 0 else 0
    
    print("\n" + "="*80)
    print(f"OVERALL PROGRESS: {overall_percent:.1f}% ({files_found}/{total_files} files generated)")
    print("="*80 + "\n")
    
    return overall_percent

def check_log_file():
    """Check the complete_analysis.log file for any errors"""
    if not os.path.exists("complete_analysis.log"):
        print("No log file found. Pipeline may not have started.")
        return
    
    print("Recent log entries:")
    print("-" * 40)
    
    # Get the last 10 log entries
    with open("complete_analysis.log", "r") as f:
        lines = f.readlines()
        for line in lines[-10:]:
            print(line.strip())
    
    # Check for errors in the log
    error_count = 0
    with open("complete_analysis.log", "r") as f:
        for line in f:
            if "ERROR" in line or "FAILED" in line:
                error_count += 1
    
    if error_count > 0:
        print(f"\nFound {error_count} errors in the log file.")
        print("Check complete_analysis.log for details.")

def main():
    """Main function to run the status check"""
    status = check_status()
    overall_percent = print_status(status)
    
    # Check log file for errors
    check_log_file()
    
    # Provide recommendations
    print("RECOMMENDATIONS:")
    if overall_percent < 20:
        print("- Pipeline has barely started. Check for errors in the log.")
    elif overall_percent < 50:
        print("- Pipeline is in early stages. Continue monitoring progress.")
    elif overall_percent < 80:
        print("- Pipeline is making good progress. Check specific missing files.")
    else:
        print("- Pipeline is almost complete. Verify final results and interpretations.")
    
    # Next step recommendation
    incomplete_steps = [step for step, info in status.items() if info["completed"] is not True]
    if incomplete_steps:
        next_step = incomplete_steps[0].split(":")[0].split(" ")[1]
        print(f"- To continue, run: python complete_analysis.py --step {next_step}")

if __name__ == "__main__":
    main() 