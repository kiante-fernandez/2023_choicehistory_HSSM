"""
Figure 3: Scatter plots of repetition behavior vs posterior estimates for previous response effects.
Individual mouse analysis version using individual mouse parameter summaries.

This script creates scatter plots showing the relationship between individual mice's repetition behavior 
and their estimated previous response effects on drift rate (v) and starting point bias (z) parameters.
Each data point represents one individual mouse rather than hierarchical participant estimates.

Generates two main outputs:
1. Main scatter plot grid (2x4): Shows all DDM model variants
2. Correlation comparison plot (1x3): Focuses on the combined model (ddmd) with statistical 
   comparison between z and v parameter correlations, including:
   - Left panel: z parameter correlation scatter plot
   - Middle panel: v parameter correlation scatter plot  
   - Right panel: Permutation test histogram showing null distribution of correlation differences
   
The permutation test randomly reassigns z and v parameters within each mouse across 10,000 
iterations to test if the correlation difference is statistically significant.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import glob
import re

# Configuration
INDIVIDUAL_SUMMARIES_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/mouse_analysis/summaries'
DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv'
OUTPUT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'

def find_individual_mouse_files():
    """
    Find all individual mouse summary files for DDM models.
    
    Returns:
    --------
    dict: Dictionary mapping model variants to lists of (mouse_id, file_path) tuples
    """
    model_variants = {
        'ddma': 'nohist',
        'ddmb': 'prevresp_z', 
        'ddmc': 'prevresp_v',
        'ddmd': 'prevresp_zv'
    }
    
    mouse_files = {variant: [] for variant in model_variants.keys()}
    
    for model_code, variant_name in model_variants.items():
        pattern = f'*_{model_code}_summary.csv'
        search_path = os.path.join(INDIVIDUAL_SUMMARIES_DIR, pattern)
        matching_files = glob.glob(search_path)
        
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            # Extract mouse ID from filename like 'CSH_ZAD_029_ddma_summary.csv'
            mouse_id = filename.replace(f'_{model_code}_summary.csv', '')
            mouse_files[model_code].append((mouse_id, file_path))
        
        print(f"Found {len(mouse_files[model_code])} files for {variant_name} ({model_code})")
    
    return mouse_files

def load_and_preprocess_data(file_path):
    """Load and preprocess behavioral data for individual mice."""
    print(f"Loading behavioral data from {file_path}")
    mouse_data = pd.read_csv(file_path)
    
    # Recode prevresp from {-1, 1} to {0, 1} to match response coding
    mouse_data['prevresp_recoded'] = mouse_data['prevresp'].map({-1.0: 0.0, 1.0: 1.0})
    
    # Create repetition variable (1 if current response equals previous response)
    mouse_data['repeat'] = np.where(mouse_data['response'] == mouse_data['prevresp_recoded'], 1, 0)
    mouse_data['stimrepeat'] = np.where(mouse_data.groupby(['subj_idx'])['signed_contrast'].shift(1) == mouse_data['signed_contrast'], 1, 0)
    
    print(f"Recoded prevresp and calculated repeat variable")

    # Clean data
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    print(f"Loaded data for {mouse_data['subj_idx'].nunique()} individual mice")
    return mouse_data

def calculate_individual_repeat_rates(data):
    """Calculate repetition rates by individual mouse."""
    repeat_rates = data.groupby(['subj_idx'])[['repeat', 'stimrepeat']].mean().reset_index()
    print(f"Calculated repeat rates for {len(repeat_rates)} individual mice")
    return repeat_rates

def load_individual_mouse_summary(file_path):
    """Load parameter estimates from individual mouse CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded individual mouse summary from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def extract_individual_mouse_parameter(df, param_name):
    """
    Extract specific parameter estimate from individual mouse summary.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Individual mouse summary DataFrame
    param_name : str
        Parameter name to extract (e.g., 'z_prevresp_cat[prev_right]', 'v_prevresp_cat[prev_right]')
    
    Returns:
    --------
    float or np.nan: Parameter mean estimate
    """
    if df is None or df.empty:
        return np.nan
    
    # Look for exact parameter name match in the index (first column, unnamed)
    param_rows = df[df.iloc[:, 0] == param_name]
    
    if param_rows.empty:
        return np.nan
    
    # Return the mean estimate
    return param_rows['mean'].iloc[0]

def process_individual_mouse_data(mouse_files, repeat_rates, model_code, model_name):
    """
    Process all individual mouse data for a specific model variant.
    
    Parameters:
    -----------
    mouse_files : list
        List of (mouse_id, file_path) tuples for this model
    repeat_rates : pandas.DataFrame
        Repeat rates by mouse
    model_code : str
        Model code (ddma, ddmb, ddmc, ddmd)
    model_name : str
        Human-readable model name
        
    Returns:
    --------
    dict: Dictionary with 'z_data' and 'v_data' DataFrames
    """
    z_estimates = []
    v_estimates = []
    
    for mouse_id, file_path in mouse_files:
        df = load_individual_mouse_summary(file_path)
        
        if df is None:
            continue
            
        # Extract z parameter (starting point bias)
        z_param = extract_individual_mouse_parameter(df, 'z_prevresp_cat[prev_right]')
        
        # Extract v parameter (drift rate bias)
        v_param = extract_individual_mouse_parameter(df, 'v_prevresp_cat[prev_right]')
        
        if not np.isnan(z_param):
            z_estimates.append({
                'mouse_id': mouse_id,
                'param_mean': z_param
            })
            
        if not np.isnan(v_param):
            v_estimates.append({
                'mouse_id': mouse_id,
                'param_mean': v_param
            })
    
    # Convert to DataFrames
    z_df = pd.DataFrame(z_estimates) if z_estimates else pd.DataFrame()
    v_df = pd.DataFrame(v_estimates) if v_estimates else pd.DataFrame()
    
    # Merge with repeat rates using mouse_id (subj_idx)
    z_data = pd.DataFrame()
    v_data = pd.DataFrame()
    
    if not z_df.empty:
        z_data = pd.merge(z_df, repeat_rates, left_on='mouse_id', right_on='subj_idx', how='inner')
        print(f"Merged z data for {model_name}: {len(z_data)} mice")
    
    if not v_df.empty:
        v_data = pd.merge(v_df, repeat_rates, left_on='mouse_id', right_on='subj_idx', how='inner')
        print(f"Merged v data for {model_name}: {len(v_data)} mice")
    
    return {
        'model_name': model_name,
        'z_data': z_data,
        'v_data': v_data
    }

def corrfunc(x, y, ax=None, color='k'):
    """Plot correlation with regression line and statistics."""
    if ax is None:
        ax = plt.gca()
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        return np.nan, np.nan
    
    # Calculate correlation and regression
    r, p = stats.spearmanr(x_clean, y_clean)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # Create scatter plot
    scatter = ax.scatter(x_clean, y_clean, alpha=0.7, s=80, 
                        color=color, edgecolors='white', linewidth=1.5)
    
    # Add regression line
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=3, alpha=0.8)
    
    # Calculate proper confidence interval for regression line
    def calculate_ci(x_vals, y_vals, x_pred, confidence=0.95):
        """Calculate confidence interval for regression line."""
        n = len(x_vals)
        if n < 3:
            return np.zeros_like(x_pred), np.zeros_like(x_pred)
        
        # Calculate residuals and standard error
        y_pred = slope * x_vals + intercept
        residuals = y_vals - y_pred
        mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
        
        # Standard error of prediction
        x_mean = np.mean(x_vals)
        sxx = np.sum((x_vals - x_mean)**2)
        
        # Standard error for each prediction point
        se_pred = np.sqrt(mse * (1/n + (x_pred - x_mean)**2 / sxx))
        
        # t-value for confidence interval
        from scipy import stats as sp_stats
        t_val = sp_stats.t.ppf((1 + confidence) / 2, n - 2)
        
        # Confidence interval
        y_pred_line = slope * x_pred + intercept
        ci_lower = y_pred_line - t_val * se_pred
        ci_upper = y_pred_line + t_val * se_pred
        
        return ci_lower, ci_upper
    
    # Add confidence interval
    if len(x_clean) >= 3:
        ci_lower, ci_upper = calculate_ci(x_clean, y_clean, x_line)
        ax.fill_between(x_line, ci_lower, ci_upper, 
                       color=color, alpha=0.2, label='95% CI')
    
    # Add significance stars
    stars = ''
    if p < 0.05: stars = '*'
    if p < 0.01: stars = '**' 
    if p < 0.001: stars = '***'
    
    # Add correlation text with better formatting
    ax.text(0.05, 0.95, f'r = {r:.3f}{stars}\np = {p:.3f}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=color, alpha=0.9, linewidth=2))
    
    return r, p

def calculate_permutation_test(data_z, data_v, x_col='param_mean', y_col='repeat', n_permutations=10000):
    """
    Perform permutation test to assess significance of correlation difference.
    
    Parameters:
    -----------
    data_z : pd.DataFrame
        Data for z parameter correlations
    data_v : pd.DataFrame  
        Data for v parameter correlations
    x_col : str
        Column name for x variable (parameter values)
    y_col : str
        Column name for y variable (repeat rates)
    n_permutations : int
        Number of permutations for the test
        
    Returns:
    --------
    dict: Statistics including correlations, observed difference, null distribution, and p-value
    """
    from scipy.stats import spearmanr
    import numpy as np
    
    # Remove NaN values for both datasets
    z_clean = data_z.dropna(subset=[x_col, y_col])
    v_clean = data_v.dropna(subset=[x_col, y_col])
    
    if len(z_clean) < 3 or len(v_clean) < 3:
        return {
            'r_z': np.nan, 'p_z': np.nan,
            'r_v': np.nan, 'p_v': np.nan,
            'r_diff_obs': np.nan, 'null_distribution': np.array([]),
            'p_value_one_tail': np.nan, 'p_value_two_tail': np.nan,
            'n_z': len(z_clean), 'n_v': len(v_clean)
        }
    
    # Calculate observed correlations
    r_z, p_z = spearmanr(z_clean[x_col], z_clean[y_col])
    r_v, p_v = spearmanr(v_clean[x_col], v_clean[y_col])
    
    # Calculate observed correlation difference
    r_diff_obs = r_z - r_v
    
    # Combine all data for permutation
    # Need to ensure we can match parameter values with repeat rates by mouse
    # Create combined dataset with parameter type labels
    z_combined = z_clean[[x_col, y_col, 'mouse_id']].copy()
    z_combined['param_type'] = 'z'
    z_combined = z_combined.rename(columns={x_col: 'param_value'})
    
    v_combined = v_clean[[x_col, y_col, 'mouse_id']].copy()
    v_combined['param_type'] = 'v'
    v_combined = v_combined.rename(columns={x_col: 'param_value'})
    
    # Only use mice that have both z and v parameters
    common_mice = set(z_combined['mouse_id']) & set(v_combined['mouse_id'])
    z_for_perm = z_combined[z_combined['mouse_id'].isin(common_mice)]
    v_for_perm = v_combined[v_combined['mouse_id'].isin(common_mice)]
    
    if len(z_for_perm) < 3 or len(v_for_perm) < 3:
        print(f"Warning: Only {len(common_mice)} mice have both z and v parameters")
        return {
            'r_z': r_z, 'p_z': p_z,
            'r_v': r_v, 'p_v': p_v,
            'r_diff_obs': r_diff_obs, 'null_distribution': np.array([]),
            'p_value_one_tail': np.nan, 'p_value_two_tail': np.nan,
            'n_z': len(z_clean), 'n_v': len(v_clean)
        }
    
    # Perform permutation test
    np.random.seed(2025)  # For reproducibility
    null_diffs = []
    
    print(f"Running permutation test with {n_permutations} permutations...")
    
    for i in range(n_permutations):
        # For each mouse, randomly assign their z and v parameters to groups
        # This preserves the within-mouse pairing while testing if parameter type matters
        perm_z_data = []
        perm_v_data = []
        
        for mouse in common_mice:
            mouse_z = z_for_perm[z_for_perm['mouse_id'] == mouse].iloc[0]
            mouse_v = v_for_perm[v_for_perm['mouse_id'] == mouse].iloc[0]
            
            # Randomly assign which parameter goes to which group
            if np.random.random() < 0.5:
                # Keep original assignment
                perm_z_data.append([mouse_z['param_value'], mouse_z[y_col]])
                perm_v_data.append([mouse_v['param_value'], mouse_v[y_col]])
            else:
                # Swap assignment
                perm_z_data.append([mouse_v['param_value'], mouse_z[y_col]])  # v param with z's repeat rate
                perm_v_data.append([mouse_z['param_value'], mouse_v[y_col]])  # z param with v's repeat rate
        
        # Calculate correlations for permuted data
        if len(perm_z_data) >= 3 and len(perm_v_data) >= 3:
            perm_z_array = np.array(perm_z_data)
            perm_v_array = np.array(perm_v_data)
            
            try:
                r_z_perm, _ = spearmanr(perm_z_array[:, 0], perm_z_array[:, 1])
                r_v_perm, _ = spearmanr(perm_v_array[:, 0], perm_v_array[:, 1])
                null_diffs.append(r_z_perm - r_v_perm)
            except:
                continue
    
    null_distribution = np.array(null_diffs)
    
    if len(null_distribution) == 0:
        return {
            'r_z': r_z, 'p_z': p_z,
            'r_v': r_v, 'p_v': p_v,
            'r_diff_obs': r_diff_obs, 'null_distribution': null_distribution,
            'p_value_one_tail': np.nan, 'p_value_two_tail': np.nan,
            'n_z': len(z_clean), 'n_v': len(v_clean)
        }
    
    # Calculate p-values
    # One-tailed: proportion of null distribution >= observed difference
    p_one_tail = 1 - np.mean(null_distribution >= r_diff_obs)
    
    # Two-tailed: proportion of null distribution with absolute value >= observed absolute difference
    p_two_tail = np.mean(np.abs(null_distribution) >= np.abs(r_diff_obs))
    
    print(f"Permutation test completed: {len(null_distribution)} valid permutations")
    
    return {
        'r_z': r_z, 'p_z': p_z,
        'r_v': r_v, 'p_v': p_v,
        'r_diff_obs': r_diff_obs, 'null_distribution': null_distribution,
        'p_value_one_tail': p_one_tail, 'p_value_two_tail': p_two_tail,
        'n_z': len(z_clean), 'n_v': len(v_clean), 'n_common': len(common_mice)
    }

def create_ddmd_correlation_comparison_plot(model_data_list, output_dir):
    """
    Create correlation comparison plot specifically for the ddmd model (both z and v effects).
    
    Parameters:
    -----------
    model_data_list : list of dict
        List of dictionaries containing model data
    output_dir : str
        Directory to save the plot
    """
    # Find ddmd model data
    ddmd_data = None
    for model_data in model_data_list:
        if model_data is not None and 'z,v' in model_data['model_name']:
            ddmd_data = model_data
            break
    
    if ddmd_data is None:
        print("No ddmd model data found for correlation comparison")
        return
    
    z_data = ddmd_data.get('z_data', pd.DataFrame())
    v_data = ddmd_data.get('v_data', pd.DataFrame())
    
    if z_data.empty or v_data.empty:
        print("Insufficient data for ddmd correlation comparison")
        return
    
    # Calculate permutation test statistics
    perm_stats = calculate_permutation_test(z_data, v_data)
    
    # Set publication-ready style
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
    
    # Create figure with side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color scheme
    z_color = '#28A745'  # Green for z
    v_color = '#2E86AB'  # Blue for v
    
    # Left panel: Z parameter correlation
    ax_z = axes[0]
    r_z, p_z = corrfunc(z_data['param_mean'], z_data['repeat'], ax=ax_z, color=z_color)
    ax_z.set_title('Starting Point Bias (z)\nvs Repetition Behavior', 
                   fontsize=14, fontweight='bold', pad=15)
    ax_z.set_xlabel('z Previous Response Effect', fontsize=13, fontweight='bold')
    ax_z.set_ylabel('P(repeat choice)', fontsize=13, fontweight='bold')
    
    # Right panel: V parameter correlation  
    ax_v = axes[1]
    r_v, p_v = corrfunc(v_data['param_mean'], v_data['repeat'], ax=ax_v, color=v_color)
    ax_v.set_title('Drift Rate Bias (v)\nvs Repetition Behavior', 
                   fontsize=14, fontweight='bold', pad=15)
    ax_v.set_xlabel('v Previous Response Effect', fontsize=13, fontweight='bold')
    ax_v.set_ylabel('P(repeat choice)', fontsize=13, fontweight='bold')
    
    # Right panel: Permutation test histogram
    ax_hist = axes[2]
    
    # Check if we have valid permutation results
    if len(perm_stats['null_distribution']) > 0:
        null_dist = perm_stats['null_distribution']
        r_diff_obs = perm_stats['r_diff_obs']
        p_two_tail = perm_stats['p_value_two_tail']
        
        # Create histogram of null distribution
        n_bins = 50
        counts, bins, patches = ax_hist.hist(null_dist, bins=n_bins, alpha=0.7, 
                                           color='lightgray', edgecolor='black', 
                                           linewidth=0.5, density=True)
        
        # Add vertical line for observed difference
        ax_hist.axvline(r_diff_obs, color='red', linewidth=3, linestyle='-', 
                       label=f'Observed difference: {r_diff_obs:.3f}')
        
        # Shade the tail area for p-value visualization (one-tailed test)
        p_one_tail = perm_stats['p_value_one_tail']
        if not np.isnan(p_one_tail):
            # Shade only the tail in the direction of the observed difference
            if r_diff_obs > 0:
                # Shade positive tail (observed difference is positive)
                for i, (count, bin_left, bin_right) in enumerate(zip(counts, bins[:-1], bins[1:])):
                    if bin_left >= r_diff_obs:
                        patches[i].set_facecolor('red')
                        patches[i].set_alpha(0.3)
            else:
                # Shade negative tail (observed difference is negative)
                for i, (count, bin_left, bin_right) in enumerate(zip(counts, bins[:-1], bins[1:])):
                    if bin_right <= r_diff_obs:
                        patches[i].set_facecolor('red')
                        patches[i].set_alpha(0.3)
        
        # Add statistical annotations
        stats_text = f"Permutation Test\n"
        stats_text += f"Observed diff: {r_diff_obs:.3f}\n"
        if not np.isnan(p_one_tail):
            stats_text += f"p-value: {p_one_tail:.3f}"
            if p_one_tail < 0.05:
                stats_text += "*"
            if p_one_tail < 0.01:
                stats_text += "*"
            if p_one_tail < 0.001:
                stats_text += "*"
        stats_text += f"\n{len(null_dist)} permutations"
        
        ax_hist.text(0.98, 0.98, stats_text, transform=ax_hist.transAxes, 
                    fontsize=11, fontweight='bold', ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='red', alpha=0.9, linewidth=2))
        
        # Set axis labels and title
        ax_hist.set_xlabel('Δ (ρz - ρv)', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax_hist.set_title('Permutation Test\nNull Distribution', fontsize=14, fontweight='bold', pad=15)
        
        # Add reference line at 0
        ax_hist.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
    else:
        # If no valid permutation results, show message
        ax_hist.text(0.5, 0.5, 'Insufficient data\nfor permutation test', 
                    transform=ax_hist.transAxes, ha='center', va='center', 
                    fontsize=14, style='italic', color='gray')
        ax_hist.set_title('Permutation Test\nNull Distribution', fontsize=14, fontweight='bold', pad=15)
        ax_hist.set_xlabel('Correlation Δ', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('Density', fontsize=12, fontweight='bold')
    
    # Style all axes consistently
    for ax in axes:
        # Add reference line at 0.5 for repeat rate plots (but not histogram)
        if ax != ax_hist:
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_ylim(0.47, 0.63)
        
        # Remove grid and style spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Enhance tick formatting
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
        # Add subtle background
        ax.set_facecolor('#white')
    
    # Add overall title
    fig.suptitle('DDM (z,v): ΔCorrelation ', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add sample size information
    if 'n_common' in perm_stats:
        fig.text(0.02, 0.02, f'Sample: {perm_stats["n_z"]} mice (z), {perm_stats["n_v"]} mice (v), {perm_stats["n_common"]} common', 
                 fontsize=10, style='italic', alpha=0.7)
    else:
        fig.text(0.02, 0.02, f'Sample: {perm_stats["n_z"]} mice (z), {perm_stats["n_v"]} mice (v)', 
                 fontsize=10, style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save plot
    output_path = os.path.join(output_dir, 'figure3_individual_mouse_ddmd_correlation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved ddmd correlation comparison plot to {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DDMD MODEL PERMUTATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Starting Point Bias (z) correlation: r = {perm_stats['r_z']:.3f}, p = {perm_stats['p_z']:.3f}")
    print(f"Drift Rate Bias (v) correlation: r = {perm_stats['r_v']:.3f}, p = {perm_stats['p_v']:.3f}")
    print(f"Observed correlation difference (z - v): {perm_stats['r_diff_obs']:.3f}")
    
    if len(perm_stats['null_distribution']) > 0:
        print(f"Permutation test results (one-tailed):")
        print(f"  - Number of permutations: {len(perm_stats['null_distribution'])}")
        print(f"  - One-tailed p-value: {perm_stats['p_value_one_tail']:.3f}")
        print(f"  - Two-tailed p-value: {perm_stats['p_value_two_tail']:.3f}")
        
        if perm_stats['p_value_one_tail'] < 0.05:
            print("*** Significant difference between correlations (one-tailed, p < 0.05) ***")
        else:
            print("No significant difference between correlations (one-tailed test)")
            
        # Additional descriptive statistics about null distribution
        null_mean = np.mean(perm_stats['null_distribution'])
        null_std = np.std(perm_stats['null_distribution'])
        print(f"  - Null distribution mean: {null_mean:.3f}")
        print(f"  - Null distribution std: {null_std:.3f}")
        
    else:
        print("Permutation test could not be completed (insufficient data)")
    
    if 'n_common' in perm_stats:
        print(f"Sample sizes: z = {perm_stats['n_z']} mice, v = {perm_stats['n_v']} mice, common = {perm_stats['n_common']} mice")
    else:
        print(f"Sample sizes: z = {perm_stats['n_z']} mice, v = {perm_stats['n_v']} mice")
    
    # Show plot
    plt.show()

def create_individual_mouse_scatter_plot(model_data_list, output_path):
    """
    Create publication-ready scatter plot for individual mouse DDM analysis.
    
    Parameters:
    -----------
    model_data_list : list of dict
        List of dictionaries containing model data with keys: 
        'model_name', 'z_data', 'v_data'
    output_path : str
        Path to save the plot
    """
    # Set publication-ready style
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
    
    # Set up the plot - 2 rows (z and v parameters), 4 columns (model variants)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    
    # Color scheme
    z_color = '#28A745'  # Green for z
    v_color = '#2E86AB'  # Blue for v
    
    model_codes = ['ddma', 'ddmb', 'ddmc', 'ddmd']
    model_names = ['No History', 'Previous Response → z', 'Previous Response → v', 'Previous Response → z,v']
    
    for col, (model_data, model_code, model_name) in enumerate(zip(model_data_list, model_codes, model_names)):
        if model_data is None:
            continue
            
        z_data = model_data.get('z_data', pd.DataFrame())
        v_data = model_data.get('v_data', pd.DataFrame())
        
        # Z parameter plot (top row)
        ax_z = axes[0, col]
        if not z_data.empty:
            r_z, p_z = corrfunc(z_data['param_mean'], z_data['repeat'], 
                               ax=ax_z, color=z_color)
        else:
            ax_z.text(0.5, 0.5, 'No z parameter\nin this model', 
                     transform=ax_z.transAxes, ha='center', va='center', 
                     fontsize=14, style='italic', color='gray')
            r_z, p_z = np.nan, np.nan
        
        ax_z.set_title(f'DDM {model_name}', 
                      fontsize=14, fontweight='bold', pad=15)
        if col == 0:
            ax_z.set_ylabel('P(repeat choice)\nStarting Point Bias (z)', 
                           fontsize=13, fontweight='bold')
        
        # V parameter plot (bottom row)
        ax_v = axes[1, col]
        if not v_data.empty:
            r_v, p_v = corrfunc(v_data['param_mean'], v_data['repeat'],
                               ax=ax_v, color=v_color)
        else:
            ax_v.text(0.5, 0.5, 'No v parameter\nin this model', 
                     transform=ax_v.transAxes, ha='center', va='center', 
                     fontsize=14, style='italic', color='gray')
            r_v, p_v = np.nan, np.nan
        
        ax_v.set_xlabel('Previous Response Effect', fontsize=13, fontweight='bold')
        if col == 0:
            ax_v.set_ylabel('P(repeat choice)\nDrift Rate Bias (v)', 
                           fontsize=13, fontweight='bold')
    
    # Style all axes
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            # Add reference line at 0.5 (no bias)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
            
            # Let x-axis scale freely based on data, set y-axis for repeat rate range
            ax.set_ylim(0.47, 0.63)  # Adjusted for repeat rate range
            
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
    
    # Add overall title
    fig.suptitle('Individual Mouse DDM Analysis: Previous Response Effects vs Repetition Behavior', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved individual mouse DDM plot to {output_path}")
    
    # Show plot
    plt.show()

def main():
    """Main execution function."""
    print("Starting Individual Mouse Figure 3 analysis: Previous response effects vs repetition behavior")
    print("Processing individual mouse DDM model summaries...")
    
    # Find individual mouse files
    mouse_files = find_individual_mouse_files()
    
    if not any(mouse_files.values()):
        print("Error: No individual mouse summary files found!")
        return
    
    # Load behavioral data
    behavioral_data = load_and_preprocess_data(DATA_PATH)
    repeat_rates = calculate_individual_repeat_rates(behavioral_data)
    
    # Process each DDM model variant
    model_data_list = []
    model_mapping = {
        'ddma': 'No History',
        'ddmb': 'Previous Response → z', 
        'ddmc': 'Previous Response → v',
        'ddmd': 'Previous Response → z,v'
    }
    
    for model_code, model_name in model_mapping.items():
        if mouse_files[model_code]:
            print(f"\nProcessing {model_name} ({model_code})...")
            model_data = process_individual_mouse_data(
                mouse_files[model_code], repeat_rates, model_code, model_name
            )
            model_data_list.append(model_data)
        else:
            print(f"No files found for {model_name} ({model_code})")
            model_data_list.append(None)
    
    # Create output path
    output_path = os.path.join(OUTPUT_DIR, 'figure3_individual_mouse_ddm_scatter.png')
    
    # Create scatter plot
    create_individual_mouse_scatter_plot(model_data_list, output_path)
    
    # Create correlation comparison plot for ddmd model
    create_ddmd_correlation_comparison_plot(model_data_list, OUTPUT_DIR)
    
    print(f"\nIndividual mouse analysis complete!")
    print(f"Main plot saved to: {output_path}")
    print(f"Correlation comparison plot saved to: {os.path.join(OUTPUT_DIR, 'figure3_individual_mouse_ddmd_correlation_comparison.png')}")

if __name__ == "__main__":
    main()