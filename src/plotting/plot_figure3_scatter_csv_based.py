"""
Figure 3: Scatter plots of repetition behavior vs posterior estimates for previous response effects.
CSV-based version using parameter summaries instead of model traces.

This script creates scatter plots showing the relationship between subjects' repetition behavior 
and their estimated previous response effects on drift rate (v) and starting point bias (z) parameters.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path

# Configuration
ANALYSIS_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/src/analysis'
MODEL_SUMMARIES_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/model_summaries'
DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250310.csv'
OUTPUT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'

def find_latest_model_files():
    """
    Automatically find the most recent CSV files for each model type.
    
    Returns:
    --------
    tuple: (ddm_model_files, angle_model_files) dictionaries with model names and file paths
    """
    import glob
    import re
    from datetime import datetime
    
    model_types = ['ddm', 'angle']
    model_variants = ['nohist', 'prevresp_v', 'prevresp_z', 'prevresp_zv']
    
    ddm_files = {}
    angle_files = {}
    
    for model_type in model_types:
        for variant in model_variants:
            pattern = f'{model_type}_{variant}_*_results_combined.csv'
            search_path = os.path.join(ANALYSIS_DIR, pattern)
            matching_files = glob.glob(search_path)
            
            if matching_files:
                # Extract timestamps and find the most recent
                latest_file = None
                latest_timestamp = None
                
                for file_path in matching_files:
                    filename = os.path.basename(file_path)
                    # Extract timestamp from filename like 'ddm_nohist_20250819_115333_results_combined.csv'
                    match = re.search(r'(\d{8}_\d{6})', filename)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                                latest_file = file_path
                        except ValueError:
                            continue
                
                if latest_file:
                    model_key = f'{model_type}_{variant}'
                    filename = os.path.basename(latest_file)
                    
                    if model_type == 'ddm':
                        ddm_files[model_key] = filename
                    else:
                        angle_files[model_key] = filename
                    
                    print(f"Found latest {model_key}: {filename}")
    
    return ddm_files, angle_files

def copy_latest_files_to_summaries(ddm_files, angle_files):
    """Copy the latest model files to the model_summaries directory."""
    os.makedirs(MODEL_SUMMARIES_DIR, exist_ok=True)
    
    all_files = {**ddm_files, **angle_files}
    
    for model_key, filename in all_files.items():
        source_path = os.path.join(ANALYSIS_DIR, filename)
        dest_path = os.path.join(MODEL_SUMMARIES_DIR, filename)
        
        if os.path.exists(source_path):
            # Only copy if the file doesn't exist or source is newer
            if not os.path.exists(dest_path) or os.path.getmtime(source_path) > os.path.getmtime(dest_path):
                import shutil
                shutil.copy2(source_path, dest_path)
                print(f"Copied {filename} to model_summaries")
            else:
                print(f"File {filename} already up to date in model_summaries")
        else:
            print(f"Warning: Source file not found: {source_path}")
    
    return all_files

def load_and_preprocess_data(file_path):
    """Load and preprocess behavioral data."""
    print(f"Loading behavioral data from {file_path}")
    mouse_data = pd.read_csv(file_path)
    
    # Create repetition variable (1 if current response equals previous response)
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    mouse_data['stimrepeat'] = np.where(mouse_data.groupby(['subj_idx'])['signed_contrast'].shift(1) == mouse_data['signed_contrast'], 1, 0)

    # Clean data
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    print(f"Loaded data for {mouse_data['subj_idx'].nunique()} participants")
    return mouse_data

def calculate_repeat_rates(data):
    """Calculate repetition rates by participant."""
    repeat_rates = data.groupby(['subj_idx'])[['repeat', 'stimrepeat']].mean().reset_index()
    # Create a sequential participant_id for merging with model parameters
    repeat_rates['participant_id'] = range(len(repeat_rates))
    print(f"Calculated repeat rates for {len(repeat_rates)} participants")
    return repeat_rates

def load_csv_results(file_path):
    """Load parameter estimates from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def extract_csv_parameters(df, param_pattern):
    """
    Extract participant-specific parameter estimates from CSV results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results DataFrame from CSV
    param_pattern : str
        Pattern to match parameter names (e.g., 'z_prevresp|participant_id', 'v_prevresp|participant_id')
    
    Returns:
    --------
    pandas.DataFrame with participant_id and parameter mean estimates
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Look for parameters matching the pattern
    if param_pattern == 'z_prevresp|participant_id':
        # Look for z parameters with previous response effect - format: z_prevresp_cat|participant_id[N, prev_right]
        param_rows = df[df['index'].str.contains('z_prevresp_cat\|participant_id\[\d+, prev_right\]', na=False, regex=True)]
    elif param_pattern == 'v_prevresp|participant_id':
        # Look for v parameters with previous response effect - format: v_prevresp_cat|participant_id[N, prev_right]
        param_rows = df[df['index'].str.contains('v_prevresp_cat\|participant_id\[\d+, prev_right\]', na=False, regex=True)]
    else:
        print(f"Unknown parameter pattern: {param_pattern}")
        return pd.DataFrame()
    
    if param_rows.empty:
        print(f"No parameters found matching pattern: {param_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(param_rows)} parameter estimates for pattern: {param_pattern}")
    
    # Extract participant IDs from parameter names
    participant_estimates = []
    for _, row in param_rows.iterrows():
        param_name = row['index']
        param_mean = row['mean']
        
        # Extract participant ID number from parameter name like 'z_prevresp_cat|participant_id[2, prev_right]'
        import re
        match = re.search(r'participant_id\[(\d+),', param_name)
        if match:
            participant_id = int(match.group(1))
            participant_estimates.append({
                'original_participant_id': participant_id,
                'param_mean': param_mean
            })
    
    if not participant_estimates:
        print(f"No valid participant estimates found for pattern: {param_pattern}")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by original participant_id
    df_estimates = pd.DataFrame(participant_estimates).sort_values('original_participant_id').reset_index(drop=True)
    
    # Create sequential participant_id for merging (0-indexed)
    df_estimates['participant_id'] = range(len(df_estimates))
    
    print(f"Extracted {len(df_estimates)} parameter estimates")
    return df_estimates

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

def create_model_type_scatter_plot(model_data_list, model_type, output_path):
    """
    Create publication-ready scatter plot for a specific model type (DDM or Angle).
    
    Parameters:
    -----------
    model_data_list : list of dict
        List of dictionaries containing model data with keys: 
        'model_name', 'z_data', 'v_data'
    model_type : str
        Model type ('DDM' or 'Angle')
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
    
    model_names = ['nohist', 'prevresp_v', 'prevresp_z', 'prevresp_zv']
    
    for col, (model_data, model_name) in enumerate(zip(model_data_list, model_names)):
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
        
        ax_z.set_title(f'{model_type} {model_name.replace("_", " ").title()}', 
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
            
            # Set consistent limits
            ax.set_xlim(-2, 2)
            ax.set_ylim(0.50, 0.65)
            
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
            
            # Add subtle background
            ax.set_facecolor('#fafafa')
    
    # Add overall title
    fig.suptitle(f'{model_type} Models: Previous Response Effects vs Repetition Behavior', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved {model_type} plot to {output_path}")
    
    # Show plot
    plt.show()

def process_model_type(model_files, model_type, repeat_rates):
    """Process all models of a specific type (DDM or Angle)."""
    print(f"\n{'='*60}")
    print(f"Processing {model_type} models")
    print(f"{'='*60}")
    
    model_data_list = []
    model_names = ['nohist', 'prevresp_v', 'prevresp_z', 'prevresp_zv']
    
    for variant in model_names:
        model_key = f'{model_type.lower()}_{variant}'
        filename = model_files.get(model_key)
        
        if filename is None:
            print(f"No file found for {model_key}")
            model_data_list.append(None)
            continue
            
        print(f"\nProcessing {model_key}...")
        
        # Load CSV results
        file_path = os.path.join(MODEL_SUMMARIES_DIR, filename)
        df = load_csv_results(file_path)
        
        if df is None:
            print(f"Skipping {model_key} due to loading error")
            model_data_list.append(None)
            continue
        
        # Extract z parameter estimates (starting point bias)
        z_estimates = extract_csv_parameters(df, 'z_prevresp|participant_id')
        
        # Extract v parameter estimates (drift rate bias)  
        v_estimates = extract_csv_parameters(df, 'v_prevresp|participant_id')
        
        # Merge with repeat rates
        z_data = pd.DataFrame()
        v_data = pd.DataFrame()
        
        if not z_estimates.empty:
            z_data = pd.merge(z_estimates, repeat_rates, on='participant_id', how='inner')
            print(f"Merged z data: {len(z_data)} participants")
        
        if not v_estimates.empty:
            v_data = pd.merge(v_estimates, repeat_rates, on='participant_id', how='inner')
            print(f"Merged v data: {len(v_data)} participants")
        
        model_data_list.append({
            'model_name': model_key,
            'z_data': z_data,
            'v_data': v_data
        })
    
    # Create output path
    output_path = os.path.join(OUTPUT_DIR, f'figure3_{model_type.lower()}_models_csv_scatter.png')
    
    # Create scatter plot
    create_model_type_scatter_plot(model_data_list, model_type, output_path)
    
    return model_data_list

def main():
    """Main execution function."""
    print("Starting Figure 3 analysis: CSV-based Previous response effects vs repetition behavior")
    print("Automatically detecting most recent model files...")
    
    # Find the latest model files
    ddm_files, angle_files = find_latest_model_files()
    
    if not ddm_files and not angle_files:
        print("Error: No model files found in the analysis directory!")
        return
    
    # Copy latest files to model_summaries directory
    copy_latest_files_to_summaries(ddm_files, angle_files)
    
    # Load behavioral data
    behavioral_data = load_and_preprocess_data(DATA_PATH)
    repeat_rates = calculate_repeat_rates(behavioral_data)
    
    # Process DDM models
    if ddm_files:
        print(f"\nFound {len(ddm_files)} DDM model files")
        ddm_results = process_model_type(ddm_files, 'DDM', repeat_rates)
    else:
        print("No DDM model files found")
    
    # Process Angle models
    if angle_files:
        print(f"\nFound {len(angle_files)} Angle model files")
        angle_results = process_model_type(angle_files, 'Angle', repeat_rates)
    else:
        print("No Angle model files found")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()