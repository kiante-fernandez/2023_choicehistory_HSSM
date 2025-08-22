"""
Conditional bias function plot for IBL training data.

This script creates a publication-ready conditional bias function plot for the 
IBL trainingChoiceWorld dataset, showing how repetition behavior varies across 
reaction time quantiles. Based on Urai et al. 2019 methodology.

Features:
- Loads IBL 2025 dataset (ibl_trainingChoiceWorld_20250819.csv)
- Calculates repeat behavior for each subject
- Creates RT quantiles and computes conditional bias function
- Plots individual subject lines + population mean with error bars
- Publication-ready styling and saves PNG + PDF formats

No model fitting - pure behavioral data analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Configuration
DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv'
OUTPUT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'

def load_and_preprocess_data(file_path):
    """
    Load and preprocess IBL behavioral data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV data file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data with repeat variable
    """
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    # Recode prevresp from {-1, 1} to {0, 1} to match response coding
    data['prevresp_recoded'] = data['prevresp'].map({-1.0: 0.0, 1.0: 1.0})
    
    # Create repetition variable (1 if current response equals previous response)
    data['repeat'] = np.where(data['response'] == data['prevresp_recoded'], 1, 0)
    
    # Create participant IDs for grouping
    data['participant_id'] = pd.factorize(data['subj_idx'])[0] + 1
    
    # Clean data - remove NaN values and negative RTs
    data = data.dropna(subset=['rt', 'response', 'prevresp'])
    data = data[data['rt'] >= 0]
    
    print(f"Loaded data for {data['subj_idx'].nunique()} subjects")
    print(f"Total trials: {len(data)}")
    print(f"Mean repeat rate: {data['repeat'].mean():.3f}")
    
    return data

def compute_conditional_bias_function(data, n_quantiles=5):
    """
    Compute conditional bias function following Urai et al. 2019.
    
    For each participant:
    1. Divide trials into RT quantiles 
    2. Calculate fraction of choices in direction of individual's history bias per quantile
    3. Aggregate across participants (mean ± SEM)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with columns: participant_id, subj_idx, rt, repeat
    n_quantiles : int
        Number of RT quantiles (default=5)
        
    Returns:
    --------
    tuple: (summary_df, subject_df)
        summary_df: Summary statistics across participants
        subject_df: Individual subject data for plotting
    """
    subject_data = []
    
    # Process each participant separately
    for subj_id in data['participant_id'].unique():
        subj_data = data[data['participant_id'] == subj_id].copy()
        
        if len(subj_data) < n_quantiles:
            continue
            
        # Determine individual's overall bias direction (repeat vs alternate)
        overall_repeat_rate = subj_data['repeat'].mean()
        
        # Create RT quantiles for this participant
        try:
            subj_data['rt_quantile'] = pd.qcut(subj_data['rt'], q=n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            # If RT values are too similar, use rank-based quantiles
            subj_data['rt_quantile'] = pd.qcut(subj_data['rt'].rank(method='first'), q=n_quantiles, labels=False)
        
        # For each quantile, calculate bias in direction of individual's tendency
        for q in range(n_quantiles):
            q_data = subj_data[subj_data['rt_quantile'] == q]
            
            if len(q_data) > 0:
                # Choice bias = fraction of choices in direction of individual's bias
                if overall_repeat_rate > 0.5:  # This person tends to repeat
                    choice_bias = q_data['repeat'].mean()
                else:  # This person tends to alternate  
                    choice_bias = 1 - q_data['repeat'].mean()  # Fraction of alternations
                
                rt_mean = q_data['rt'].mean()
                
                subject_data.append({
                    'participant_id': subj_id,
                    'subj_idx': subj_data['subj_idx'].iloc[0],
                    'rt_quantile': q,
                    'choice_bias': choice_bias,
                    'repeat_rate': q_data['repeat'].mean(),
                    'rt_mean': rt_mean,
                    'n_trials': len(q_data),
                    'overall_bias': 'repeat' if overall_repeat_rate > 0.5 else 'alternate',
                    'overall_repeat_rate': overall_repeat_rate
                })
    
    subject_df = pd.DataFrame(subject_data)
    
    # Compute summary statistics across participants
    summary_stats = []
    for q in range(n_quantiles):
        q_data = subject_df[subject_df['rt_quantile'] == q]
        
        if len(q_data) > 0:
            summary_stats.append({
                'rt_quantile': q,
                'choice_bias_mean': q_data['choice_bias'].mean(),
                'choice_bias_sem': stats.sem(q_data['choice_bias']),
                'rt_mean': q_data['rt_mean'].mean(),
                'rt_sem': stats.sem(q_data['rt_mean']),
                'n_subjects': len(q_data)
            })
    
    summary = pd.DataFrame(summary_stats)
    
    print(f"Computed conditional bias function for {len(subject_df)} subject-quantile combinations")
    print(f"Bias range: {summary['choice_bias_mean'].min():.3f} to {summary['choice_bias_mean'].max():.3f}")
    
    return summary, subject_df

def plot_conditional_bias_function(summary, subject_df, title="", n_quantiles=5):
    """
    Create two-panel plot: conditional bias function + raincloud plot for Fast vs Slow RT.
    
    Parameters:
    -----------
    summary : pd.DataFrame
        Summary statistics across subjects
    subject_df : pd.DataFrame
        Individual subject data
    title : str
        Plot title
    n_quantiles : int
        Number of RT quantiles
        
    Returns:
    --------
    tuple: (fig, (ax1, ax2))
        Matplotlib figure and axes objects
    """
    # Set publication-ready style matching the individual scatter plots
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300
    })
    
    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== LEFT PANEL: Conditional bias function =====
    
    # Plot individual subject data points with connecting lines
    if subject_df is not None and len(subject_df) > 0:
        for subject in subject_df['participant_id'].unique():
            subject_data = subject_df[subject_df['participant_id'] == subject].sort_values('rt_quantile')
            
            if len(subject_data) == n_quantiles:  # Only plot if subject has all quantiles
                ax1.plot(subject_data['rt_quantile'], subject_data['choice_bias'], 
                        color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Plot main line with error bars (population mean)
    if len(summary) > 0:
        ax1.errorbar(
            summary['rt_quantile'], 
            summary['choice_bias_mean'], 
            yerr=summary['choice_bias_sem'],
            fmt='-o', 
            color='black', 
            ecolor='black', 
            capsize=5, 
            linewidth=4, 
            markersize=8,
            zorder=3,
            label='Population mean'
        )
    
    # Add reference line at 0.5 (no bias)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    # Customize left panel
    ax1.set_xlabel('RT Quantile', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Choice bias\n(fraction)', fontsize=13, fontweight='bold')
    ax1.set_title('A. Conditional Bias Function', fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits and ticks
    ax1.set_ylim(0.40, 0.75)
    ax1.set_xlim(-0.5, n_quantiles-0.5)
    
    # Set x-tick labels  
    ax1.set_xticks(range(n_quantiles))
    ax1.set_xticklabels([f'Q{i+1}' for i in range(n_quantiles)])
    
    # Set y-ticks with proper spacing
    ax1.set_yticks(np.arange(0.45, 0.76, 0.05))
    
    # ===== RIGHT PANEL: Raincloud plot for Fast vs Slow =====
    
    # Extract Fast (Q1) and Slow (Q5) data
    if subject_df is not None and len(subject_df) > 0:
        fast_slow_data = []
        
        for subject in subject_df['participant_id'].unique():
            subject_data = subject_df[subject_df['participant_id'] == subject]
            
            # Get Q1 (Fast) and Q4 (Slow) - using 0-based indexing
            fast_data = subject_data[subject_data['rt_quantile'] == 0]  # Q1
            slow_data = subject_data[subject_data['rt_quantile'] == n_quantiles-1]  # Q5
            
            if len(fast_data) > 0:
                fast_slow_data.append({
                    'participant_id': subject,
                    'condition': 'Fast',
                    'choice_bias': fast_data['choice_bias'].iloc[0]
                })
            
            if len(slow_data) > 0:
                fast_slow_data.append({
                    'participant_id': subject,
                    'condition': 'Slow', 
                    'choice_bias': slow_data['choice_bias'].iloc[0]
                })
        
        raincloud_df = pd.DataFrame(fast_slow_data)
        
        if len(raincloud_df) > 0:
            # Colors for Fast and Slow
            colors = {'Fast': '#2E86AB', 'Slow': '#F24236'}  # Blue and Red
            
            # Create raincloud plot components
            positions = {'Fast': 0, 'Slow': 1}
            
            for condition in ['Fast', 'Slow']:
                condition_data = raincloud_df[raincloud_df['condition'] == condition]['choice_bias']
                pos = positions[condition]
                color = colors[condition]
                
                if len(condition_data) > 0:
                    # Violin plot (distribution shape)
                    violin_parts = ax2.violinplot([condition_data], positions=[pos], 
                                                 widths=0.6, showmeans=False, showmedians=False)
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.3)
                        pc.set_edgecolor(color)
                    
                    # Box plot (quartiles and median)
                    box_parts = ax2.boxplot([condition_data], positions=[pos], widths=0.3,
                                          patch_artist=True, showfliers=False)
                    box_parts['boxes'][0].set_facecolor(color)
                    box_parts['boxes'][0].set_alpha(0.75)
                    box_parts['medians'][0].set_color('black')
                    box_parts['medians'][0].set_linewidth(3)
                    
                    # Individual points (jittered)
                    jitter = np.random.normal(0, 0.04, len(condition_data))
                    ax2.scatter(pos + jitter, condition_data, color=color, alpha=0.6, s=30, zorder=3)
            
            # Connect paired points for each subject
            fast_data = raincloud_df[raincloud_df['condition'] == 'Fast']
            slow_data = raincloud_df[raincloud_df['condition'] == 'Slow']
            
            # Merge to get paired data
            paired_data = pd.merge(fast_data, slow_data, on='participant_id', suffixes=('_fast', '_slow'))
            
            for _, row in paired_data.iterrows():
                ax2.plot([0, 1], [row['choice_bias_fast'], row['choice_bias_slow']], 
                        color='gray', alpha=0.3, linewidth=0.5, zorder=1)
            
            # Statistical test
            if len(paired_data) > 0:
                fast_values = paired_data['choice_bias_fast'].values
                slow_values = paired_data['choice_bias_slow'].values
                
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(fast_values, slow_values)
                
                # Add statistical annotation
                y_max = max(raincloud_df['choice_bias'].max(), 0.7)
                ax2.plot([0, 1], [y_max + 0.01, y_max + 0.01], 'k-', linewidth=1)
                
                # Significance stars
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'
                
                ax2.text(0.5, y_max + 0.015, f't = {t_stat:.2f}\np = {p_val:.3f} {sig_text}', 
                        ha='center', va='bottom', fontsize=10)
    
    # Customize right panel
    ax2.set_xlabel('RT', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Choice bias\n(fraction)', fontsize=13, fontweight='bold')
    ax2.set_title('B. Fast vs Slow RT', fontsize=14, fontweight='bold', pad=15)
    
    # Set axis properties
    ax2.set_ylim(0.40, 0.75)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Fast', 'Slow'])
    ax2.set_yticks(np.arange(0.45, 0.76, 0.05))
    
    # Add reference line
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    # Apply styling to both panels
    for ax in [ax1, ax2]:
        # Remove grid lines
        ax.grid(False)
        
        # Create detached axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Enhance tick formatting
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
        # Set white background
        ax.set_facecolor('white')
    
    # Add sample size information to left panel
    n_subjects = len(subject_df['participant_id'].unique()) if subject_df is not None else 0
    ax1.text(0.02, 0.98, f'n = {n_subjects} subjects', 
            transform=ax1.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            verticalalignment='top')
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def main():
    """Main execution function."""
    print("="*60)
    print("IBL CONDITIONAL BIAS FUNCTION ANALYSIS")
    print("="*60)
    
    # Load and preprocess data
    data = load_and_preprocess_data(DATA_PATH)
    
    # Compute conditional bias function
    print("\nComputing conditional bias function...")
    summary, subject_df = compute_conditional_bias_function(data, n_quantiles=5)
    
    if len(summary) == 0:
        print("Error: No bias function data computed!")
        return
    
    # Create plot
    print("\nCreating plot...")
    dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    
    fig, (ax1, ax2) = plot_conditional_bias_function(
        summary, subject_df, 
        title="IBL Training Data: Conditional Bias Function"
    )
    
    # Save plots
    output_base = os.path.join(OUTPUT_DIR, f"{dataset_name}_conditional_bias_observed")
    
    png_path = f"{output_base}.png"
    pdf_path = f"{output_base}.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\nSaved plots:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Subjects: {len(subject_df['participant_id'].unique())}")
    print(f"Total subject-quantile observations: {len(subject_df)}")
    print(f"Choice bias range: {summary['choice_bias_mean'].min():.3f} to {summary['choice_bias_mean'].max():.3f}")
    
    # Test for significant deviation from chance across quantiles
    choice_bias_values = summary['choice_bias_mean'].values
    t_stat, p_val = stats.ttest_1samp(choice_bias_values, 0.5)
    print(f"One-sample t-test vs chance (0.5): t = {t_stat:.3f}, p = {p_val:.4f}")
    
    # Show plot
    plt.show()
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()