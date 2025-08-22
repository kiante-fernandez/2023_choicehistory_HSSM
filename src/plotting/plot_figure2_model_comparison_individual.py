"""
Figure 2: Individual Mouse Model Comparison Analysis

This script aggregates and visualizes model comparison results from individual mice,
showing distributions of performance differences and win frequencies across subjects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
import glob
from scipy import stats

# grab the utils that are already defined
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils_plot as tools
tools.seaborn_style()

def load_individual_comparisons(summaries_dir):
    """
    Load and aggregate model comparison data from individual mouse files.
    
    Parameters:
    -----------
    summaries_dir : str
        Directory containing individual mouse model comparison CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all individual mouse comparisons
    """
    # Find all model comparison files
    comparison_files = glob.glob(os.path.join(summaries_dir, "*_model_comparison.csv"))
    
    if not comparison_files:
        raise ValueError(f"No model comparison files found in {summaries_dir}")
    
    print(f"Found {len(comparison_files)} individual mouse comparison files")
    
    all_data = []
    
    for file_path in comparison_files:
        # Extract mouse ID from filename
        filename = os.path.basename(file_path)
        mouse_id = filename.replace("_model_comparison.csv", "")
        
        # Load the comparison data
        try:
            df = pd.read_csv(file_path, index_col=0)
            df['mouse_id'] = mouse_id
            df['model'] = df.index
            df = df.reset_index(drop=True)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid comparison files could be loaded")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Successfully loaded data for {len(combined_df['mouse_id'].unique())} mice")
    print(f"Models found: {sorted(combined_df['model'].unique())}")
    
    return combined_df

def calculate_model_statistics(df):
    """
    Calculate summary statistics for each model across mice.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with individual mouse comparisons
        
    Returns:
    --------
    dict
        Dictionary containing various statistics
    """
    stats_dict = {}
    
    # Calculate win frequencies
    wins = df[df['rank'] == 0]['model'].value_counts()
    total_mice = len(df['mouse_id'].unique())
    win_proportions = wins / total_mice
    
    stats_dict['wins'] = wins
    stats_dict['win_proportions'] = win_proportions
    stats_dict['total_mice'] = total_mice
    
    # Calculate ΔLOO from reference model (ddma = no history) for each mouse
    reference_model = 'ddma'
    history_models = ['ddmb', 'ddmc', 'ddmd']  # Only the history models
    
    # Calculate ΔLOO differences from reference for each mouse
    delta_loo_data = []
    for mouse_id in df['mouse_id'].unique():
        mouse_data = df[df['mouse_id'] == mouse_id]
        
        # Get reference model elpd_loo
        ref_data = mouse_data[mouse_data['model'] == reference_model]
        if len(ref_data) == 0:
            continue
        ref_loo = ref_data['elpd_loo'].iloc[0]
        
        # Calculate differences for history models
        for model in history_models:
            model_data = mouse_data[mouse_data['model'] == model]
            if len(model_data) > 0:
                model_loo = model_data['elpd_loo'].iloc[0]
                delta_loo = model_loo - ref_loo  # Positive = better than reference
                delta_loo_data.append({
                    'mouse_id': mouse_id,
                    'model': model,
                    'delta_loo': delta_loo
                })
    
    delta_loo_df = pd.DataFrame(delta_loo_data)
    stats_dict['delta_loo_df'] = delta_loo_df
    
    # Calculate summary statistics for ΔLOO
    model_stats = []
    for model in history_models:
        model_data = delta_loo_df[delta_loo_df['model'] == model]['delta_loo']
        stats = {
            'model': model,
            'mean_delta_loo': model_data.mean(),
            'std_delta_loo': model_data.std(),
            'sem_delta_loo': model_data.sem(),
            'median_delta_loo': model_data.median(),
            'n_mice': len(model_data)
        }
        model_stats.append(stats)
    
    stats_dict['model_summary'] = pd.DataFrame(model_stats)
    
    # Calculate average ranks (add 1 to convert 0-based to 1-based ranking)
    avg_ranks = df.groupby('model')['rank'].apply(lambda x: (x + 1).mean()).sort_values()
    stats_dict['avg_ranks'] = avg_ranks
    
    # Calculate pairwise model differences
    pairwise_matrix = calculate_pairwise_differences(df)
    stats_dict['pairwise_matrix'] = pairwise_matrix
    
    return stats_dict

def calculate_pairwise_differences(df):
    """
    Calculate pairwise ΔLOO differences between all models for each mouse.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with individual mouse comparisons
        
    Returns:
    --------
    pd.DataFrame
        Matrix of mean pairwise differences
    """
    models = ['ddma', 'ddmb', 'ddmc', 'ddmd']
    pairwise_data = []
    
    for mouse_id in df['mouse_id'].unique():
        mouse_data = df[df['mouse_id'] == mouse_id]
        
        # Create a dictionary of elpd_loo values for this mouse
        mouse_loo = {}
        for model in models:
            model_data = mouse_data[mouse_data['model'] == model]
            if len(model_data) > 0:
                mouse_loo[model] = model_data['elpd_loo'].iloc[0]
        
        # Calculate all pairwise differences for this mouse
        for model1 in models:
            for model2 in models:
                if model1 in mouse_loo and model2 in mouse_loo:
                    diff = mouse_loo[model1] - mouse_loo[model2]  # Positive = model1 better
                    pairwise_data.append({
                        'mouse_id': mouse_id,
                        'model1': model1,
                        'model2': model2,
                        'loo_diff': diff
                    })
    
    pairwise_df = pd.DataFrame(pairwise_data)
    
    # Calculate mean differences across mice
    mean_diffs = pairwise_df.groupby(['model1', 'model2'])['loo_diff'].mean().unstack()
    
    return mean_diffs

def create_model_comparison_plot(df, stats_dict, output_dir):
    """
    Create comprehensive model comparison visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with individual mouse comparisons
    stats_dict : dict
        Dictionary containing summary statistics
    output_dir : str
        Directory to save the plots
    """
    # # Set up plot style to match existing theme
    # plt.rcParams.update({
    #     'font.size': 12,
    #     'font.family': 'Arial',
    #     'axes.linewidth': 1.2,
    #     'axes.spines.top': False,
    #     'axes.spines.right': False,
    #     'xtick.major.size': 6,
    #     'ytick.major.size': 6,
    #     'xtick.major.width': 1.2,
    #     'ytick.major.width': 1.2,
    #     'figure.dpi': 300
    # })
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define model colors (consistent with DDM theme)
    model_colors = {
        'ddma': '#2E86AB',  # Blue
        'ddmb': '#28A745',  # Green  
        'ddmc': '#17A2B8',  # Cyan
        'ddmd': '#DC3545'   # Red
    }

    # Define prettier model names
    model_names = {
        'ddma': 'DDM (no history)',
        'ddmb': 'Previous Response → v',
        'ddmc': 'Previous Response → z',
        'ddmd': 'Previous Response → z,v'
    }

    ####################################
    # Panel A: Distribution of ΔLOO values (relative to no-history model)
    ax1 = plt.subplot(2, 2, 1)  # Top left
    
    # Use only the history models for the top panel
    delta_loo_df = stats_dict['delta_loo_df']
    history_models = ['ddmb', 'ddmc', 'ddmd']
    
    # Prepare data for violin plot
    data_for_plot = [delta_loo_df[delta_loo_df['model'] == model]['delta_loo'].values 
                     for model in history_models]
    colors = [model_colors[model] for model in history_models]
    
    # Create violin plot
    parts = ax1.violinplot(data_for_plot, positions=range(len(history_models)), 
                          showmeans=True, showmedians=True, widths=0.6)
    
    # Customize violin plot colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Customize other elements
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    
    # Add error bars (mean ± SEM)
    for i, model in enumerate(history_models):
        model_data = delta_loo_df[delta_loo_df['model'] == model]['delta_loo']
        mean_val = model_data.mean()
        sem_val = model_data.sem()
        ax1.errorbar(i, mean_val, yerr=sem_val, fmt='ko', capsize=5, 
                    capthick=2, markersize=8, linewidth=2)
    
    # Customize panel A
    ax1.set_ylabel('ΔLOO (vs. No History)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_title('Distribution of Model Performance vs. No-History Model', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(history_models)))
    ax1.set_xticklabels([model_names[m] for m in history_models], fontsize=11, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Panel B: Pairwise model comparison matrix
    ax2 = plt.subplot(2, 2, 2)  # Top right
    
    pairwise_matrix = stats_dict['pairwise_matrix']
    
    # Create heatmap
    im = ax2.imshow(pairwise_matrix.values, cmap='RdBu_r', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('ΔLOO (Row - Column)', fontsize=10, fontweight='bold')
    
    # Set ticks and labels
    ax2.set_xticks(range(len(pairwise_matrix.columns)))
    ax2.set_yticks(range(len(pairwise_matrix.index)))
    ax2.set_xticklabels([model_names[m] for m in pairwise_matrix.columns], 
                       fontsize=9, rotation=45, ha='right')
    ax2.set_yticklabels([model_names[m] for m in pairwise_matrix.index], fontsize=9)
    
    # Add text annotations
    for i in range(len(pairwise_matrix.index)):
        for j in range(len(pairwise_matrix.columns)):
            value = pairwise_matrix.iloc[i, j]
            color = 'white' if abs(value) > abs(pairwise_matrix.values).max() * 0.6 else 'black'
            ax2.text(j, i, f'{value:.1f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=9)
    
    ax2.set_title('Pairwise Model Differences', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model (Subtracted)', fontsize=10)
    ax2.set_ylabel('Model (Minuend)', fontsize=10)
    
    # Panel C: Model win frequency
    ax3 = plt.subplot(2, 2, 3)  # Bottom left
    
    wins = stats_dict['wins']
    win_props = stats_dict['win_proportions']
    total_mice = stats_dict['total_mice']
    
    # Ensure all models are represented
    all_models = sorted(df['model'].unique())
    win_counts = [wins.get(model, 0) for model in all_models]
    win_percentages = [win_props.get(model, 0) * 100 for model in all_models]
    
    # Calculate error bars for win frequencies (binomial confidence intervals)
    # Using Wilson score interval approximation
    win_errors = []
    for count in win_counts:
        p = count / total_mice
        # Standard error for binomial proportion
        se = np.sqrt(p * (1 - p) / total_mice)
        win_errors.append(se * total_mice)  # Convert back to count scale
    
    bars = ax3.bar(range(len(all_models)), win_counts, yerr=win_errors,
                   color=[model_colors[m] for m in all_models], 
                   alpha=0.85, edgecolor='black', linewidth=1.5,
                   capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, win_percentages)):
        height = bar.get_height()
        ax3.text(i, height + win_errors[i] + 1, f'{pct:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('Number of Wins (Rank 0)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_title('Model Win Frequency', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(all_models)))
    ax3.set_xticklabels([model_names[m] for m in all_models], fontsize=10, rotation=45, ha='right')
    ax3.set_ylim(0, max([count + err for count, err in zip(win_counts, win_errors)]) * 1.2)
    
    # Panel D: Average ranking
    ax4 = plt.subplot(2, 2, 4)  # Bottom right
    
    avg_ranks = stats_dict['avg_ranks']
    rank_models = avg_ranks.index.tolist()
    rank_values = avg_ranks.values
    
    # Calculate standard error for rankings
    rank_errors = []
    for model in rank_models:
        model_ranks = df[df['model'] == model]['rank'] + 1  # Convert to 1-based
        rank_errors.append(model_ranks.sem())
    
    bars = ax4.bar(range(len(rank_models)), rank_values, yerr=rank_errors,
                   color=[model_colors[m] for m in rank_models],
                   alpha=0.85, edgecolor='black', linewidth=1.5,
                   capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, rank_values, rank_errors)):
        height = bar.get_height()
        ax4.text(i, height + err + 0.05, f'{val:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_title('Average Model Ranking', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(rank_models)))
    ax4.set_xticklabels([model_names[m] for m in rank_models], fontsize=10, rotation=45, ha='right')
    
    # Set y-axis limits (don't invert since we want rank 1 at bottom, rank 4 at top)
    max_val_with_error = max([val + err for val, err in zip(rank_values, rank_errors)])
    ax4.set_ylim(0.8, max_val_with_error * 1.1)
    
    # Add sample size annotation
    n_mice = stats_dict['total_mice']
    fig.text(0.02, 0.02, f'N = {n_mice} mice', fontsize=10, style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plots
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_png = os.path.join(output_dir, 'individual_mouse_model_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    
    output_path_pdf = os.path.join(output_dir, 'individual_mouse_model_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"Plot saved to: {output_path_png}")
    print(f"PDF version saved to: {output_path_pdf}")
    plt.close()

    ##############################
    # overwrite: use same colors as in previous work
    # see https://github.com/anne-urai/2019_Urai_choice-history-ddm/blob/master/plot_all.m#L46
    model_colors, model_names = tools.get_colors()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,3))
    sns.barplot(x="model", y='delta_loo', order=['ddmc', 'ddmb', 'ddmd'],
                ax=ax, data=stats_dict['delta_loo_df'], 
                palette=model_colors)
    ax.set(ylabel=r'$\Delta$' + 'LOO'.upper() + '\n(vs. no history)', 
           xlabel='', xticklabels=['$z$', '$v_{bias}$', 'both'])

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=40, ha='right')
    fig.tight_layout()
    sns.despine(trim=False)
    fig.savefig(os.path.join(output_dir, "model_comp_preprint.pdf"))



def print_summary_statistics(stats_dict):
    """
    Print comprehensive summary statistics.
    
    Parameters:
    -----------
    stats_dict : dict
        Dictionary containing summary statistics
    """
    print("\n" + "="*60)
    print("INDIVIDUAL MOUSE MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nTotal number of mice analyzed: {stats_dict['total_mice']}")
    
    print("\n--- Model Win Frequencies ---")
    wins = stats_dict['wins']
    win_props = stats_dict['win_proportions']
    for model in sorted(wins.index):
        print(f"{model}: {wins[model]} wins ({win_props[model]*100:.1f}%)")
    
    print("\n--- Model Performance Statistics (ΔLOO vs. No History) ---")
    model_summary = stats_dict['model_summary']
    for _, row in model_summary.iterrows():
        print(f"{row['model']}:")
        print(f"  Mean ΔLOO: {row['mean_delta_loo']:.2f} ± {row['sem_delta_loo']:.2f} (SEM)")
        print(f"  Std Dev: {row['std_delta_loo']:.2f}")
        print(f"  Median: {row['median_delta_loo']:.2f}")
    
    print("\n--- Average Model Rankings ---")
    avg_ranks = stats_dict['avg_ranks']
    for i, (model, rank) in enumerate(avg_ranks.items(), 1):
        print(f"{i}. {model}: {rank:.3f}")
    

def main():
    """
    Main function to run individual mouse model comparison analysis.
    """
    # Define directories
    summaries_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/mouse_analysis/summaries"
    output_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures"
    try:
        assert os.path.exists(summaries_dir)
    except:
        summaries_dir = "/Users/anneurai/Documents/code/2023_choicehistory_HSSM/results/figures/mouse_analysis/summaries"
        output_dir = "/Users/anneurai/Documents/code/2023_choicehistory_HSSM/results/figures"

    print("Individual Mouse Model Comparison Analysis")
    print(f"Loading data from: {summaries_dir}")
    
    try:
        # Load and aggregate individual mouse data
        df = load_individual_comparisons(summaries_dir)
        
        # Calculate summary statistics
        stats_dict = calculate_model_statistics(df)
        
        # Create comprehensive visualization
        create_model_comparison_plot(df, stats_dict, output_dir)
        
        # Print summary statistics
        #print_summary_statistics(stats_dict)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print("  - individual_mouse_model_comparison.png")
        print("  - individual_mouse_model_comparison.pdf")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()