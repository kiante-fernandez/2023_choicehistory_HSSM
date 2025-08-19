"""
Figure 2: Model Comparison plot

This script creates bar plot showing the relationship between model performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re
from datetime import datetime

def find_latest_model_files(comparison_dir):
    """
    Find the most recent timestamped model comparison files for each model type.
    
    Parameters:
    -----------
    comparison_dir : str
        Directory containing the model comparison CSV files
        
    Returns:
    --------
    dict
        Dictionary mapping model names to their most recent file paths
    """
    # Pattern to match timestamped files: modelname_YYYYMMDD_HHMMSS_model_comparison.csv
    pattern = r'^(\w+)_(\d{8}_\d{6})_model_comparison\.csv$'
    
    model_files = {}
    
    # Scan all files in the directory
    for filename in os.listdir(comparison_dir):
        match = re.match(pattern, filename)
        if match:
            model_name = match.group(1)
            timestamp_str = match.group(2)
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                # Keep track of the most recent file for each model
                if model_name not in model_files or timestamp > model_files[model_name]['timestamp']:
                    model_files[model_name] = {
                        'filename': filename,
                        'filepath': os.path.join(comparison_dir, filename),
                        'timestamp': timestamp
                    }
            except ValueError:
                print(f"Warning: Could not parse timestamp from {filename}")
                continue
    
    # Return just the file paths
    return {model: info['filepath'] for model, info in model_files.items()}

def combine_model_comparison_csvs(comparison_dir, output_dir, model_types=None, output_suffix=""):
    """
    Combine individual model comparison CSV files into a single file using most recent timestamped files.
    
    Parameters:
    -----------
    comparison_dir : str
        Directory containing the individual model comparison CSV files
    output_dir : str
        Directory to save the combined CSV file
    model_types : list, optional
        List of model types to include (e.g., ['ddm_nohist', 'ddm_prevresp_v'])
        If None, includes all found models
    output_suffix : str
        Suffix to add to output filename (e.g., "_ddm_only")
    """
    # Find the most recent files for each model
    latest_files = find_latest_model_files(comparison_dir)
    
    if not latest_files:
        print("No timestamped model comparison files found")
        return None
    
    print(f"Found latest model files:")
    for model, filepath in latest_files.items():
        print(f"  {model}: {os.path.basename(filepath)}")
    
    # Filter by model types if specified
    if model_types:
        latest_files = {k: v for k, v in latest_files.items() if k in model_types}
        if not latest_files:
            print(f"Warning: No files found for specified model types: {model_types}")
            return None
    
    combined_data = []
    
    for model_name, file_path in latest_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df['Model'] = model_name
            combined_data.append(df)
        else:
            print(f"Warning: {file_path} not found")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Pivot to wide format for easier analysis
        result_df = combined_df.pivot(index='Model', columns='Metric', values='Value').reset_index()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'combined_model_comparison{output_suffix}.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Combined CSV file saved to: {output_path}")
        return result_df
    else:
        print("No model comparison files found")
        return None

def plot_loo_comparison(comparison_dir, output_dir, model_types=None, reference_model=None, 
                       output_suffix="", plot_title="Model Comparison"):
    """
    Create a publication-ready plot comparing LOO values across models.
    
    Parameters:
    -----------
    comparison_dir : str
        Directory containing the model comparison CSV files
    output_dir : str
        Directory to save the plot
    model_types : list, optional
        List of model types to include. If None, includes all found models
    reference_model : str, optional
        Model to use as reference. If None, uses appropriate default
    output_suffix : str
        Suffix for output filenames
    plot_title : str
        Title for the plot
    """
    # Load the combined data
    combined_file = os.path.join(output_dir, f'combined_model_comparison{output_suffix}.csv')
    
    if not os.path.exists(combined_file):
        print(f"Combined file not found at {combined_file}. Running combine_model_comparison_csvs first.")
        df = combine_model_comparison_csvs(comparison_dir, output_dir, model_types, output_suffix)
    else:
        df = pd.read_csv(combined_file)
    
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Determine reference model if not specified
    if reference_model is None:
        if 'ddm_nohist' in df['Model'].values:
            reference_model = 'ddm_nohist'
        elif 'angle_nohist' in df['Model'].values:
            reference_model = 'angle_nohist'
        else:
            print("Warning: No obvious reference model found. Using first model as reference.")
            reference_model = df['Model'].iloc[0]
    
    if reference_model not in df['Model'].values:
        print(f"Error: Reference model '{reference_model}' not found in data")
        return
    
    # Set up plot style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'figure.dpi': 300
    })
    
    # Get reference value
    reference_loo = df[df['Model'] == reference_model]['loo'].iloc[0]
    
    # Calculate differences from reference model
    df_plot = df.copy()
    df_plot['loo_diff'] = df_plot['loo'] - reference_loo
    
    # Remove reference model from plot (difference would be 0)
    df_plot = df_plot[df_plot['Model'] != reference_model]
    
    # Create prettier model names
    model_name_map = {
        'ddm_prevresp_v': 'Previous Response → v',
        'ddm_prevresp_z': 'Previous Response → z', 
        'ddm_prevresp_zv': 'Previous Response → z,v',
        'angle_prevresp_v': 'Previous Response → v (angle)',
        'angle_prevresp_z': 'Previous Response → z (angle)', 
        'angle_prevresp_zv': 'Previous Response → z,v (angle)',
        'ddm_nohist': 'DDM (no history)',
        'angle_nohist': 'Angle (no history)'
    }
    
    df_plot['Model_pretty'] = df_plot['Model'].map(lambda x: model_name_map.get(x, x))
    
    if len(df_plot) == 0:
        print("No comparison models to plot (only reference model found)")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use custom color scheme
    if 'angle' in output_suffix:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, teal, blue for angle models
    elif 'combined' in output_suffix:
        # Different colors for DDM vs angle models
        colors = []
        for model in df_plot['Model']:
            if 'ddm' in model:
                if 'v' in model and 'z' not in model:
                    colors.append('#2E86AB')  # Blue for DDM v
                elif 'z' in model and 'v' not in model:
                    colors.append('#28A745')  # Green for DDM z
                elif 'zv' in model:
                    colors.append('#17A2B8')  # Cyan for DDM zv
            else:  # angle models
                if 'v' in model and 'z' not in model:
                    colors.append('#FF6B6B')  # Red for angle v
                elif 'z' in model and 'v' not in model:
                    colors.append('#4ECDC4')  # Teal for angle z
                elif 'zv' in model:
                    colors.append('#45B7D1')  # Light blue for angle zv
                else:
                    colors.append('#FFA07A')  # Light salmon for angle nohist
    else:
        colors = ['#2E86AB', '#28A745', '#17A2B8']  # Default DDM colors
    
    # Create bar plot
    bars = ax.bar(range(len(df_plot)), df_plot['loo_diff'], 
                  color=colors[:len(df_plot)], alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Customize the plot
    ax.set_ylabel('ΔLOO', fontsize=14, fontweight='bold')
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot['Model_pretty'], fontsize=11, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(i, height + abs(height) * 0.02 if height >= 0 else height - abs(height) * 0.02,
                f'{height:.0f}', ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=11)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Remove grid and customize spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set y-axis limits with some padding
    if len(df_plot['loo_diff']) > 0:
        y_max = max(df_plot['loo_diff']) * 1.15 if max(df_plot['loo_diff']) > 0 else 50
        y_min = min(df_plot['loo_diff']) * 1.15 if min(df_plot['loo_diff']) < 0 else -50
    else:
        y_max, y_min = 100, -100
    ax.set_ylim(y_min, y_max)
    
    # Add reference model annotation
    ref_name = model_name_map.get(reference_model, reference_model)
    ax.text(0.98, 0.98, f'Reference: {ref_name}', 
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'loo_model_comparison{output_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as PDF for publications
    output_path_pdf = os.path.join(output_dir, f'loo_model_comparison{output_suffix}.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"Plot saved to: {output_path}")
    print(f"PDF version saved to: {output_path_pdf}")
    plt.close()  # Close instead of show to prevent blocking
    
    # Print summary statistics
    print(f"\nSummary ({plot_title}):")
    print(f"Reference model ({ref_name}) LOO: {reference_loo:.2f}")
    print("\nModel comparisons (ΔLOO from reference):")
    for _, row in df_plot.iterrows():
        print(f"  {row['Model_pretty']}: {row['loo_diff']:.2f}")
    print("\nNote: Positive values indicate worse performance than reference model")

def plot_ddm_only_comparison(comparison_dir, output_dir):
    """
    Create comparison plot for DDM models only, using ddm_nohist as reference.
    """
    print("\n=== DDM-Only Model Comparison ===")
    ddm_models = ['ddm_nohist', 'ddm_prevresp_v', 'ddm_prevresp_z', 'ddm_prevresp_zv']
    
    # Combine CSV files for DDM models only
    df = combine_model_comparison_csvs(comparison_dir, output_dir, ddm_models, "_ddm_only")
    
    if df is not None:
        # Create plot
        plot_loo_comparison(comparison_dir, output_dir, ddm_models, 'ddm_nohist', 
                          "_ddm_only", "DDM Model Comparison")
    
def plot_angle_only_comparison(comparison_dir, output_dir):
    """
    Create comparison plot for angle models only, using angle_nohist as reference.
    """
    print("\n=== Angle-Only Model Comparison ===")
    angle_models = ['angle_nohist', 'angle_prevresp_v', 'angle_prevresp_z', 'angle_prevresp_zv']
    
    # Combine CSV files for angle models only
    df = combine_model_comparison_csvs(comparison_dir, output_dir, angle_models, "_angle_only")
    
    if df is not None:
        # Create plot
        plot_loo_comparison(comparison_dir, output_dir, angle_models, 'angle_nohist', 
                          "_angle_only", "Angle Model Comparison")

def plot_combined_comparison(comparison_dir, output_dir):
    """
    Create comparison plot for all models combined, using ddm_nohist as reference.
    """
    print("\n=== Combined Model Comparison (All Models) ===")
    
    # Let it find all available models
    df = combine_model_comparison_csvs(comparison_dir, output_dir, None, "_combined")
    
    if df is not None:
        # Create plot with ddm_nohist as reference
        plot_loo_comparison(comparison_dir, output_dir, None, 'ddm_nohist', 
                          "_combined", "Combined Model Comparison (DDM + Angle)")

def main():
    """
    Main function to run all model comparison analyses.
    """
    # Define directories
    comparison_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/model_comparisons"
    output_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures"
    
    print("Model Comparison Analysis - Using Most Recent Timestamped Files")
    print(f"Scanning directory: {comparison_dir}")
    
    # Show available files
    latest_files = find_latest_model_files(comparison_dir)
    if not latest_files:
        print("No timestamped model comparison files found!")
        return
        
    print(f"\nFound {len(latest_files)} model types with timestamped files")
    
    # Run DDM-only comparison
    plot_ddm_only_comparison(comparison_dir, output_dir)
    
    # Run angle-only comparison
    plot_angle_only_comparison(comparison_dir, output_dir)
    
    # Run combined comparison
    plot_combined_comparison(comparison_dir, output_dir)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("  - combined_model_comparison_ddm_only.csv + plots")
    print("  - combined_model_comparison_angle_only.csv + plots") 
    print("  - combined_model_comparison_combined.csv + plots")

if __name__ == "__main__":
    main()
    
    