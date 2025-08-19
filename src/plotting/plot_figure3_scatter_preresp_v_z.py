"""
Figure 3: Scatter plots of repetition behavior vs posterior estimates for previous response effects.

This script creates scatter plots showing the relationship between subjects' repetition behavior 
and their estimated previous response effects on drift rate (v) and starting point bias (z) parameters.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import arviz as az
from pathlib import Path

# Configuration #TODO add angle models one we have some fits. For now use old fits for the DDM for testing
MODEL_PATHS = {
    'ddm_nohist': '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/models/ddm_nohist_model.nc',
    'ddm_prevresp_v': '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/models/ddm_prevresp_v_model.nc',
    'ddm_prevresp_z': '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/models/ddm_prevresp_z_model.nc',
    'ddm_prevresp_zv': '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/models/ddm_prevresp_zv_model.nc'
}

DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250310.csv'
OUTPUT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'

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

def load_model_traces(model_path):
    """Load model traces from NetCDF file."""
    try:
        traces = az.from_netcdf(model_path)
        print(f"Successfully loaded traces from {model_path}")
        return traces
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

def extract_parameter_estimates(traces, param_pattern):
    """
    Extract participant-specific parameter estimates from traces.
    
    Parameters:
    -----------
    traces : arviz.InferenceData
        Model traces
    param_pattern : str
        Pattern to match parameter names (e.g., 'z_prevresp|participant_id_offset')
    
    Returns:
    --------
    pandas.DataFrame with participant_id and parameter mean estimates
    """
    if traces is None:
        return pd.DataFrame()
    
    # Look for parameters matching the pattern
    param_vars = [var for var in traces.posterior.data_vars if param_pattern in var]
    
    if not param_vars:
        print(f"No parameters found matching pattern: {param_pattern}")
        return pd.DataFrame()
    
    # Use the first matching parameter (should be the offset parameter)
    param_var = param_vars[0]
    print(f"Extracting estimates for parameter: {param_var}")
    
    # Get posterior samples and calculate means
    param_samples = traces.posterior[param_var]
    param_means = param_samples.mean(dim=['chain', 'draw'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': range(len(param_means)),
        'param_mean': param_means.values
    })
    
    print(f"Extracted {len(df)} parameter estimates")
    return df

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

def create_scatter_plot(z_data, v_data, model_name, output_path):
    """
    Create publication-ready scatter plot for z and v parameter effects vs repetition behavior.
    
    Parameters:
    -----------
    z_data : pandas.DataFrame
        Data with z parameter estimates and repeat rates
    v_data : pandas.DataFrame  
        Data with v parameter estimates and repeat rates
    model_name : str
        Name of the model for plot title
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
    
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Color scheme matching previous plots
    colors = ['#28A745', '#2E86AB']  # Green for z, Blue for v
    
    # Z parameter plot (left)
    if not z_data.empty:
        r_z, p_z = corrfunc(z_data['param_mean'], z_data['repeat'], 
                           ax=axes[0], color=colors[0])
        axes[0].set_xlabel('Previous Response Effect on z\n(Starting Point Bias)', 
                          fontsize=13, fontweight='bold')
        axes[0].set_ylabel('P(repeat choice)', fontsize=13, fontweight='bold')
        axes[0].set_title('Starting Point Bias Effect', fontsize=14, fontweight='bold', pad=20)
    else:
        axes[0].text(0.5, 0.5, 'No z parameter\nin this model', 
                    transform=axes[0].transAxes, ha='center', va='center', 
                    fontsize=16, style='italic', color='gray')
        axes[0].set_xlabel('Previous Response Effect on z\n(Starting Point Bias)', 
                          fontsize=13, fontweight='bold')
        axes[0].set_ylabel('P(repeat choice)', fontsize=13, fontweight='bold')
        axes[0].set_title('Starting Point Bias Effect', fontsize=14, fontweight='bold', pad=20)
        r_z, p_z = np.nan, np.nan
    
    # V parameter plot (right)
    if not v_data.empty:
        r_v, p_v = corrfunc(v_data['param_mean'], v_data['repeat'],
                           ax=axes[1], color=colors[1])
        axes[1].set_xlabel('Previous Response Effect on v\n(Drift Rate Bias)', 
                          fontsize=13, fontweight='bold')
        axes[1].set_title('Drift Rate Bias Effect', fontsize=14, fontweight='bold', pad=20)
    else:
        axes[1].text(0.5, 0.5, 'No v parameter\nin this model', 
                    transform=axes[1].transAxes, ha='center', va='center', 
                    fontsize=16, style='italic', color='gray')
        axes[1].set_xlabel('Previous Response Effect on v\n(Drift Rate Bias)', 
                          fontsize=13, fontweight='bold')
        axes[1].set_title('Drift Rate Bias Effect', fontsize=14, fontweight='bold', pad=20)
        r_v, p_v = np.nan, np.nan
    
    # Style both axes
    for i, ax in enumerate(axes):
        # Add reference line at 0.5 (no bias)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(1.8, 0.502, 'No bias', ha='right', va='bottom', 
                fontsize=10, style='italic', color='gray')
        
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
    
    # Add difference in correlations at the top (no title interference)
    if not np.isnan(r_z) and not np.isnan(r_v):
        diff = abs(r_z - r_v)
        fig.text(0.5, 0.92, f'|Δr| = {diff:.3f}', ha='center', va='top', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                         edgecolor='navy', alpha=0.8, linewidth=2))
    
    # Remove overall title to avoid interference with correlation difference
    # fig.suptitle removed
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Leave space for correlation difference
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")
    
    # Show plot
    plt.show()
    
    return r_z, p_z, r_v, p_v

def process_model(model_name, model_path, repeat_rates):
    """Process a single model and create scatter plot."""
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")
    
    # Load model traces
    traces = load_model_traces(model_path)
    if traces is None:
        print(f"Skipping {model_name} due to loading error")
        return
    
    # Extract z parameter estimates (starting point bias)
    z_estimates = extract_parameter_estimates(traces, 'z_prevresp|participant_id_offset')
    
    # Extract v parameter estimates (drift rate bias)  
    v_estimates = extract_parameter_estimates(traces, 'v_prevresp|participant_id_offset')
    
    # Merge with repeat rates
    z_data = pd.DataFrame()
    v_data = pd.DataFrame()
    
    if not z_estimates.empty:
        z_data = pd.merge(z_estimates, repeat_rates, on='participant_id', how='inner')
        print(f"Merged z data: {len(z_data)} participants")
    
    if not v_estimates.empty:
        v_data = pd.merge(v_estimates, repeat_rates, on='participant_id', how='inner')
        print(f"Merged v data: {len(v_data)} participants")
    
    # Create output path
    output_path = os.path.join(OUTPUT_DIR, f'figure3_{model_name}_scatter.png')
    
    # Create scatter plot
    r_z, p_z, r_v, p_v = create_scatter_plot(z_data, v_data, model_name, output_path)
    
    # Print summary
    print(f"\nResults for {model_name}:")
    if not np.isnan(r_z):
        print(f"  Z parameter correlation: r = {r_z:.3f}, p = {p_z:.3f}")
    if not np.isnan(r_v):
        print(f"  V parameter correlation: r = {r_v:.3f}, p = {p_v:.3f}")

def main():
    """Main execution function."""
    print("Starting Figure 3 analysis: Previous response effects vs repetition behavior")
    
    # Load behavioral data
    behavioral_data = load_and_preprocess_data(DATA_PATH)
    repeat_rates = calculate_repeat_rates(behavioral_data)
    
    # Process each model
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            process_model(model_name, model_path, repeat_rates)
        else:
            print(f"Model file not found: {model_path}")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
