#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from pathlib import Path
import re
#%%
# Define paths and constants
SUMMARY_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/model_results/summaries'
PLOT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'
DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv'

# Create plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """Load and preprocess mouse data."""
    print(f"Loading data from {file_path}")
    mouse_data = pd.read_csv(file_path)
    
    # Mark repetitions - whether current response equals previous response
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    
    # Recode response to be -1 and 1 rather than 0 and 1
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['prevresp'] = mouse_data['prevresp'].replace({0: -1, 1: 1})
    
    # Clean and filter data
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    # Add stimulus repetition
    mouse_data['stimrepeat'] = np.where(
        mouse_data.groupby(['subj_idx'])['signed_contrast'].shift(1) == mouse_data['signed_contrast'], 
        1, 0
    )
    
    return mouse_data

def get_repeat_rates(mouse_data):
    """Calculate repeat rates by subject."""
    return mouse_data.groupby(['subj_idx'])[['stimrepeat', 'repeat']].mean().reset_index()

def get_model_summary_files(directory, model_type='ddmd'):
    """Get all summary files for a specific model type."""
    pattern = os.path.join(directory, f"*_{model_type}_summary.csv")
    return sorted([f for f in os.listdir(directory) if f.endswith(f'_{model_type}_summary.csv')])

def extract_full_mouse_id(filename):
    """Extract the complete mouse ID from the filename, removing model type suffixes."""
    # First, remove model suffixes if present
    clean_name = filename.replace('_ddmd_summary.csv', '').replace('_angled_summary.csv', '')
    clean_name = clean_name.replace('_ddmd', '').replace('_angled', '')
    
    # Handle different naming patterns
    if clean_name.startswith('CSH_ZAD_'):
        return '_'.join(clean_name.split('_')[:3])  # CSH_ZAD_001
    elif clean_name.startswith('MFD_'):
        return '_'.join(clean_name.split('_')[:2])  # MFD_06
    elif clean_name.startswith('NR_'):
        return '_'.join(clean_name.split('_')[:2])  # NR_0017
    elif clean_name.startswith('SWC_'):
        return '_'.join(clean_name.split('_')[:2])  # SWC_023
    elif clean_name.startswith('NYU-'):
        # NYU has dashes instead of underscores
        match = re.match(r'(NYU-\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('UCLA'):
        # Extract UCLA ID
        match = re.match(r'(UCLA\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('KS'):
        # Extract KS ID
        match = re.match(r'(KS\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ZFM-'):
        # Extract ZFM ID
        match = re.match(r'(ZFM-\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ZM_'):
        # Extract ZM ID
        match = re.match(r'(ZM_\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ibl_witten_'):
        # Extract ibl_witten ID
        match = re.match(r'(ibl_witten_\d+)', clean_name)
        if match:
            return match.group(1)
    
    # Return cleaned name
    return clean_name

def extract_model_params(file_path):
    """Extract model parameters from a summary file."""
    filename = os.path.basename(file_path)
    mouse_id = extract_full_mouse_id(filename)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Look for the column that contains parameter names
    param_col = None
    for col in df.columns:
        if df[col].astype(str).str.contains('prev').any():
            param_col = col
            break
    
    if param_col is None:
        print(f"  Could not find parameter column in {filename}")
        return {'filename': filename, 'mouse_id': mouse_id, 'v_effect': None, 'z_effect': None}
    
    # Extract drift rate (v) parameters related to previous choice
    v_prev_right = None
    v_prev_left = None
    z_prev_right = None 
    z_prev_left = None
    
    # Find parameters with an explicit search
    for idx, row in df.iterrows():
        param_name = str(row[param_col])
        
        # For v parameters
        if 'v_prevresp_cat[prev_right]' in param_name:
            v_prev_right = row['mean']
        elif 'v_prevresp_cat[prev_left]' in param_name:
            v_prev_left = row['mean']
            
        # For z parameters
        if 'z_prevresp_cat[prev_right]' in param_name:
            z_prev_right = row['mean']
        elif 'z_prevresp_cat[prev_left]' in param_name:
            z_prev_left = row['mean']
    
    # Calculate effects
    v_effect = None
    if v_prev_right is not None and v_prev_left is not None:
        v_effect = v_prev_right - v_prev_left
    elif v_prev_right is not None:
        v_effect = v_prev_right
    elif v_prev_left is not None:
        v_effect = -v_prev_left
        
    z_effect = None
    if z_prev_right is not None and z_prev_left is not None:
        z_effect = z_prev_right - z_prev_left
    elif z_prev_right is not None:
        z_effect = z_prev_right
    elif z_prev_left is not None:
        z_effect = -z_prev_left
    
    return {
        'filename': filename,
        'mouse_id': mouse_id,
        'v_effect': v_effect,
        'z_effect': z_effect
    }

def corrfunc(x, y, ax=None, color='k'):
    """Plot correlation coefficient and p-value on the given axis with confidence interval."""
    if ax is None:
        ax = plt.gca()
    
    # Calculate correlation and p-value
    r, p = sp.stats.spearmanr(x, y, nan_policy='omit')
    
    # Format and display the correlation text
    stars = ''
    if p < 0.05: 
        stars = '*'
    if p < 0.01: 
        stars = '**'
    if p < 0.001: 
        stars = '***'
    
    text = f'ρ = {r:.2f}{stars}'
    ax.annotate(text, xy=(.1, .9), xycoords=ax.transAxes, color=color, fontsize=12)
    
    # Add a regression line with confidence interval
    sns.regplot(x=x, y=y, ax=ax, color=color, scatter=True, 
                scatter_kws={'alpha': 0.6, 's': 50},
                line_kws={'linewidth': 2}, 
                ci=95)
    
    return r, p

def dependent_corr(r12, r13, r23, n, twotailed=True, conf_level=0.95, method='steiger'):
    """Calculate the statistical significance of difference between dependent correlations."""
    # Steiger's Z test
    diff = r12 - r13
    determin = 1 - r12*r12 - r13*r13 - r23*r23 + 2*r12*r13*r23
    av = (r12 + r13)/2
    cube = (1 - r23) * (1 - r23) * (1 - r23)
    
    if method == 'steiger':
        # Formula from Steiger (1980)
        t = diff * np.sqrt((n-1) * (1 + r23) / (2 * ((n-1)/(n-3)) * determin + av * av * cube))
    else:
        # Williams' modification
        t = (r12 - r13) * np.sqrt((n-1) * (1 + r23) / (2 * determin * (1 - r23)))
    
    if twotailed:
        p = 2 * (1 - sp.stats.t.cdf(abs(t), n-3))
    else:
        p = 1 - sp.stats.t.cdf(t, n-3)
    
    return t, p

# def create_scatter_plot(z_df, v_df, plot_path, model_type):    
    """Create scatter plot comparing z and v history effects with repetition behavior."""
    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(12, 6))
    
    # Colors
    pal = sns.color_palette("Paired")
    pal2 = pal[2:4] + pal[0:2] + pal[8:10]
    
    # Plot correlation for z
    r_z, p_z = corrfunc(x=z_df.z_effect, y=z_df.repeat, ax=ax[0], color=pal2[1])
    ax[0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[0].set(xlabel='History effect in starting point (z)', ylabel='P(repeat choice)')
    
    # Label each point with mouse ID
    for idx, row in z_df.iterrows():
        # Create shortened label for display
        label = row['mouse_id']
        if '_' in label:
            parts = label.split('_')
            if len(parts) > 2:
                label = f"{parts[0]}_{parts[-1]}"  # e.g., CSH_001
        elif '-' in label:
            label = label  # Keep NYU-xx as is
            
        ax[0].annotate(label, 
                      (row['z_effect'], row['repeat']),
                      xytext=(5, 0), 
                      textcoords='offset points',
                      fontsize=8)
    
    # Plot correlation for v
    r_v, p_v = corrfunc(x=v_df.v_effect, y=v_df.repeat, ax=ax[1], color=pal2[3])
    ax[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[1].set(xlabel='History effect in drift rate (v)', ylabel='P(repeat choice)')
    
    # Label each point with mouse ID
    for idx, row in v_df.iterrows():
        # Create shortened label for display
        label = row['mouse_id']
        if '_' in label:
            parts = label.split('_')
            if len(parts) > 2:
                label = f"{parts[0]}_{parts[-1]}"  # e.g., CSH_001
        elif '-' in label:
            label = label  # Keep NYU-xx as is
            
        ax[1].annotate(label, 
                      (row['v_effect'], row['repeat']),
                      xytext=(5, 0), 
                      textcoords='offset points',
                      fontsize=8)
    
    # Calculate difference in correlation
    common_mice = set(z_df['mouse_id']).intersection(set(v_df['mouse_id']))
    if len(common_mice) >= 5:
        z_common = z_df[z_df['mouse_id'].isin(common_mice)]
        v_common = v_df[v_df['mouse_id'].isin(common_mice)]
        
        # Ensure same order
        z_common = z_common.sort_values('mouse_id')
        v_common = v_common.sort_values('mouse_id')
        
        r_zv = sp.stats.spearmanr(z_common.z_effect, v_common.v_effect, nan_policy='omit')[0]
        
        tstat, pval = dependent_corr(r_z, r_v, r_zv, len(common_mice), twotailed=True)
        deltarho = r_z - r_v
        
        if pval < 0.0001:
            fig.suptitle(f'{model_type}: Δρ = {deltarho:.3f}, p < 0.0001', fontsize=12)
        else:
            fig.suptitle(f'{model_type}: Δρ = {deltarho:.3f}, p = {pval:.4f}', fontsize=12)
    
    # Add sample size to the plot
    fig.text(0.5, 0.01, f'n = {len(z_df)} mice', ha='center', fontsize=10)
    
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    
    return fig

def create_scatter_plot(z_df, v_df, plot_path, model_type, show_labels=True):
    """Create scatter plot comparing z and v history effects with repetition behavior."""
    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(10, 5))
    
    # Colors - matching the style from the second code
    pal = sns.color_palette("Paired")
    pal2 = pal[2:4] + pal[0:2] + pal[8:10]
    
    # Plot correlation for z
    r_z, p_z = corrfunc(x=z_df.z_effect, y=z_df.repeat, ax=ax[0], color=pal2[1])
    ax[0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[0].set(xlabel='History effect in starting point (z)', ylabel='P(repeat choice)')
    
    # Add mouse ID labels if requested
    if show_labels:
        for idx, row in z_df.iterrows():
            # Create shortened label for display
            label = row['mouse_id']
            if '_' in label:
                parts = label.split('_')
                if len(parts) > 2:
                    label = f"{parts[0]}_{parts[-1]}"  # e.g., CSH_001
            elif '-' in label:
                label = label  # Keep NYU-xx as is
                
            ax[0].annotate(label, 
                          (row['z_effect'], row['repeat']),
                          xytext=(5, 0), 
                          textcoords='offset points',
                          fontsize=8)
    
    # Plot correlation for v
    r_v, p_v = corrfunc(x=v_df.v_effect, y=v_df.repeat, ax=ax[1], color=pal2[3])
    ax[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[1].set(xlabel='History effect in drift rate (v)', ylabel='P(repeat choice)')
    
    # Add mouse ID labels if requested
    if show_labels:
        for idx, row in v_df.iterrows():
            # Create shortened label for display
            label = row['mouse_id']
            if '_' in label:
                parts = label.split('_')
                if len(parts) > 2:
                    label = f"{parts[0]}_{parts[-1]}"  # e.g., CSH_001
            elif '-' in label:
                label = label  # Keep NYU-xx as is
                
            ax[1].annotate(label, 
                          (row['v_effect'], row['repeat']),
                          xytext=(5, 0), 
                          textcoords='offset points',
                          fontsize=8)
    
    # Calculate difference in correlation
    common_mice = set(z_df['mouse_id']).intersection(set(v_df['mouse_id']))
    if len(common_mice) >= 5:
        z_common = z_df[z_df['mouse_id'].isin(common_mice)]
        v_common = v_df[v_df['mouse_id'].isin(common_mice)]
        
        # Ensure same order
        z_common = z_common.sort_values('mouse_id')
        v_common = v_common.sort_values('mouse_id')
        
        r_zv = sp.stats.spearmanr(z_common.z_effect, v_common.v_effect, nan_policy='omit')[0]
        
        tstat, pval = dependent_corr(r_z, r_v, r_zv, len(common_mice), twotailed=True)
        deltarho = r_z - r_v
        
        if pval < 0.0001:
            fig.suptitle(f'{model_type}: Δρ = {deltarho:.3f}, p < 0.0001', fontsize=14)
        else:
            fig.suptitle(f'{model_type}: Δρ = {deltarho:.3f}, p = {pval:.4f}', fontsize=14)
    
    # Add sample size to the plot
    fig.text(0.5, -0.02, f'n = {len(z_df)} mice', ha='center', fontsize=12)
    
    # Use arviz style if available
    try:
        az.style.use("arviz-doc")
    except:
        pass
    
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    
    return fig

#%%
def main():
    # Load and preprocess data
    print("Loading and preprocessing mouse data...")
    mouse_data = load_and_preprocess_data(DATA_PATH)
    
    # Calculate repeat rates by subject
    repeat_rates = get_repeat_rates(mouse_data)
    print(f"Calculated repeat rates for {len(repeat_rates)} mice")
    
    # Get the list of expected mice
    
    # Get summary files
    print("\nFinding model summary files...")
    ddmd_files = get_model_summary_files(SUMMARY_DIR, model_type='ddmd')
    print(f"Found {len(ddmd_files)} ddmd model summary files")
    
    angled_files = get_model_summary_files(SUMMARY_DIR, model_type='angled')
    print(f"Found {len(angled_files)} angled model summary files")
    
    # Process model files
    print("\nProcessing model files...")
    
    # Process DDM-d files
    ddmd_params = []
    for filename in ddmd_files:
        file_path = os.path.join(SUMMARY_DIR, filename)
        params = extract_model_params(file_path)
        if params['v_effect'] is not None or params['z_effect'] is not None:
            ddmd_params.append(params)
            print(f"Extracted DDM-d parameters for mouse: {params['mouse_id']}")
    
    # Process Angle-d files
    angled_params = []
    for filename in angled_files:
        file_path = os.path.join(SUMMARY_DIR, filename)
        params = extract_model_params(file_path)
        if params['v_effect'] is not None or params['z_effect'] is not None:
            angled_params.append(params)
            print(f"Extracted Angle-d parameters for mouse: {params['mouse_id']}")
    
    # Create DataFrames
    ddmd_df = pd.DataFrame(ddmd_params)
    angled_df = pd.DataFrame(angled_params)
    
    # Print detailed diagnostics
    print("\nModel mouse IDs extracted:")
    print(f"DDM-d mice: {sorted(ddmd_df['mouse_id'].unique())}")
    print(f"Angle-d mice: {sorted(angled_df['mouse_id'].unique())}")
    
    print("\nBehavioral data mouse IDs (first 20):")
    print(sorted(repeat_rates['subj_idx'].unique())[:20])
    
    # Merge with repeat rates
    print("\nMerging with repeat rates...")
    ddmd_merged = pd.merge(ddmd_df, repeat_rates, left_on='mouse_id', right_on='subj_idx', how='inner')
    angled_merged = pd.merge(angled_df, repeat_rates, left_on='mouse_id', right_on='subj_idx', how='inner')
    
    print(f"After merge, DDM-d dataframe has {len(ddmd_merged)} rows")
    print(f"After merge, Angle-d dataframe has {len(angled_merged)} rows")
    
    if len(ddmd_merged) > 0:
        # Check the number of unique values in 'repeat' column
        unique_repeats = ddmd_merged['repeat'].nunique()
        print(f"Number of unique P(repeat) values in DDM-d data: {unique_repeats}")
        
        # Check if any mice have the same repeat value
        repeat_counts = ddmd_merged['repeat'].value_counts()
        if any(repeat_counts > 1):
            print("Warning: Some mice have identical P(repeat) values:")
            for val, count in repeat_counts.items():
                if count > 1:
                    mice = ddmd_merged[ddmd_merged['repeat'] == val]['mouse_id'].tolist()
                    print(f"  P(repeat) = {val:.3f}: {mice}")
    
    # Create scatter plots
    if len(ddmd_merged) > 0:
        print("\nCreating scatter plot for DDM-d model...")
        z_df_ddmd = ddmd_merged.dropna(subset=['z_effect', 'repeat'])
        v_df_ddmd = ddmd_merged.dropna(subset=['v_effect', 'repeat'])
        
        print(f"DDM-d data points: {len(z_df_ddmd)} for z-effect, {len(v_df_ddmd)} for v-effect")
        
        if len(z_df_ddmd) > 0 and len(v_df_ddmd) > 0:
            plot_path = os.path.join(PLOT_DIR, 'ddmd_history_effects.png')
            create_scatter_plot(z_df_ddmd, v_df_ddmd, plot_path, "DDM Model", show_labels=True)
            print(f"Saved DDM-d plot to {plot_path}")
    
    if len(angled_merged) > 0:
        print("\nCreating scatter plot for Angle-d model...")
        z_df_angled = angled_merged.dropna(subset=['z_effect', 'repeat'])
        v_df_angled = angled_merged.dropna(subset=['v_effect', 'repeat'])
        
        print(f"Angle-d data points: {len(z_df_angled)} for z-effect, {len(v_df_angled)} for v-effect")
        
        if len(z_df_angled) > 0 and len(v_df_angled) > 0:
            plot_path = os.path.join(PLOT_DIR, 'angled_history_effects.png')
            create_scatter_plot(z_df_angled, v_df_angled, plot_path, "Angle Model", show_labels=True)
            print(f"Saved Angle-d plot to {plot_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    
    if len(ddmd_merged) > 0:
        print("\nDDM-d Model Summary:")
        print(f"Z-effect correlation with repeat: {sp.stats.spearmanr(ddmd_merged.z_effect.dropna(), ddmd_merged.repeat, nan_policy='omit')}")
        print(f"V-effect correlation with repeat: {sp.stats.spearmanr(ddmd_merged.v_effect.dropna(), ddmd_merged.repeat, nan_policy='omit')}")
    
    if len(angled_merged) > 0:
        print("\nAngle-d Model Summary:")
        print(f"Z-effect correlation with repeat: {sp.stats.spearmanr(angled_merged.z_effect.dropna(), angled_merged.repeat, nan_policy='omit')}")
        print(f"V-effect correlation with repeat: {sp.stats.spearmanr(angled_merged.v_effect.dropna(), angled_merged.repeat, nan_policy='omit')}")

if __name__ == "__main__":
    main()
# %%
