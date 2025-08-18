"""
Figure 2: Model Comparison plot

This script creates bar plot showing the relationship between model performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def combine_model_comparison_csvs(comparison_dir, output_dir):
    """
    Combine individual model comparison CSV files into a single file.
    
    Parameters:
    -----------
    comparison_dir : str
        Directory containing the individual model comparison CSV files
    output_dir : str
        Directory to save the combined CSV file
    """
    model_files = {
        'ddm_nohist': 'ddm_nohist_model_comparison.csv',
        'ddm_prevresp_v': 'ddm_prevresp_v_model_comparison.csv',
        'ddm_prevresp_z': 'ddm_prevresp_z_model_comparison.csv',
        'ddm_prevresp_zv': 'ddm_prevresp_zv_model_comparison.csv'
    }
    
    combined_data = []
    
    for model_name, file_name in model_files.items():
        file_path = os.path.join(comparison_dir, file_name)
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
        output_path = os.path.join(output_dir, 'combined_model_comparison.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Combined CSV file saved to: {output_path}")
        return result_df
    else:
        print("No model comparison files found")
        return None

def plot_loo_comparison(comparison_dir, output_dir):
    """
    Create a publication-ready plot comparing LOO values across models, relative to ddm_nohist.
    
    Parameters:
    -----------
    comparison_dir : str
        Directory containing the combined model comparison CSV file
    output_dir : str
        Directory to save the plot
    """
    # Load the combined data
    combined_file = os.path.join(output_dir, 'combined_model_comparison.csv')
    
    if not os.path.exists(combined_file):
        print(f"Combined file not found at {combined_file}. Running combine_model_comparison_csvs first.")
        df = combine_model_comparison_csvs(comparison_dir, output_dir)
    else:
        df = pd.read_csv(combined_file)
    
    if df is None or df.empty:
        print("No data available for plotting")
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
    
    # Get reference value (ddm_nohist)
    reference_loo = df[df['Model'] == 'ddm_nohist']['loo'].iloc[0]
    
    # Calculate differences from reference model
    df_plot = df.copy()
    df_plot['loo_diff'] = df_plot['loo'] - reference_loo
    
    # Remove reference model from plot (difference would be 0)
    df_plot = df_plot[df_plot['Model'] != 'ddm_nohist']
    
    # Create prettier model names
    model_name_map = {
        'ddm_prevresp_v': 'Previous Response → v',
        'ddm_prevresp_z': 'Previous Response → z', 
        'ddm_prevresp_zv': 'Previous Response → z,v'
    }
    df_plot['Model_pretty'] = df_plot['Model'].map(model_name_map)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Use custom color scheme: blue, green, cyan. Match the colors from the old paper
    colors = ['#2E86AB', '#28A745', '#17A2B8']  # Blue for v, Green for z, Cyan for zv
    
    # Create bar plot
    bars = ax.bar(range(len(df_plot)), df_plot['loo_diff'], 
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Customize the plot
    ax.set_ylabel('ΔLOO', fontsize=14, fontweight='bold')
    ax.set_xlabel('')  # Remove x-axis label
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot['Model_pretty'], fontsize=11)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(i, height + 10,
                f'{height:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Remove grid and customize spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set y-axis limits with some padding
    y_max = max(df_plot['loo_diff']) * 1.15
    y_min = min(df_plot['loo_diff']) * 0.1 if min(df_plot['loo_diff']) < 0 else -50
    ax.set_ylim(y_min, y_max)
    
    # Add subtle annotation in top corner
    ax.text(0.98, 0.98, 'Reference: DDM (no history)', 
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'loo_model_comparison_publication.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as PDF for publications
    output_path_pdf = os.path.join(output_dir, 'loo_model_comparison_publication.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"Publication-ready plot saved to: {output_path}")
    print(f"PDF version saved to: {output_path_pdf}")
    plt.show()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Reference model (DDM no history) LOO: {reference_loo:.2f}")
    print("\nModel comparisons (ΔLOO from reference):")
    for _, row in df_plot.iterrows():
        print(f"  {row['Model_pretty']}: {row['loo_diff']:.2f}")
    print("\nNote: Positive values indicate worse performance than reference model")

def main():
    """
    Main function to run the model comparison analysis.
    """
    # Define directories
    comparison_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/model_comparisons"
    output_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures"
    
    # Combine CSV files
    print("Combining model comparison CSV files...")
    combine_model_comparison_csvs(comparison_dir, output_dir)
    
    # Create plot
    print("\nCreating LOO comparison plot...")
    plot_loo_comparison(comparison_dir, output_dir)

if __name__ == "__main__":
    main()
    
    