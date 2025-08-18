# hssm_reproduce_choice_history_ddm.py - a reproduction of models from
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
# Choice history biases subsequent evidence accumulation. eLife
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/11/17      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2024/01/08      Kianté Fernandez<kiantefernan@gmail.com>   added util functions for workflow
# 2024/07/03      Kianté Fernandez<kiantefernan@gmail.com>   added util functions for workflow

# %% load lib
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import hssm
import seaborn as sns
import matplotlib.pyplot as plt
import re
#import jax
from ..utils.utils_hssm_modelspec import make_model 
from ..utils.utils_hssm import run_model
from scipy import stats

#%matplotlib inline

#os.environ["JAX_PLATFORMS"] = "cpu"
# %% load data
#mouse data
# Set up file paths
script_dir = os.path.dirname(os.path.realpath(__file__))

#%% Data loading and preprocessing function
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the mouse data with comprehensive cleaning and feature creation
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing mouse data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed mouse data
    """
    # Load the data
    mouse_data = pd.read_csv(file_path)
    
    # Create participant ID from subject index
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    
    # Limit to first 350 trials per session per subject
    mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(
        lambda group: group.head(350)
    ).reset_index(drop=True)
    # mouse_data_limited = mouse_data
    # Drop rows with missing values in key columns
    mouse_data_limited = mouse_data_limited.dropna(subset=[
        'movement_onset', 'rt', 'prevresp', 'signed_contrast', 'response'
    ])
    
    # Filter based on movement onset and reaction time
    mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] < 5) & 
                                           (mouse_data_limited['rt'] < 5)]
    mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] > 0.08) & 
                                           (mouse_data_limited['rt'] > 0.08)]
    
    # Check for mice with less than 200 trials and exclude them
    trial_counts = mouse_data_limited.groupby('participant_id').size()
    mice_with_enough_trials = trial_counts[trial_counts >= 200].index
    print(f"\nMice with at least 200 trials: {len(mice_with_enough_trials)}")
    print(f"Mice excluded for having less than 200 trials: {len(trial_counts[trial_counts < 200])}")
    
    # Filter to keep only mice with at least 200 trials
    mouse_data_limited = mouse_data_limited[mouse_data_limited['participant_id'].isin(mice_with_enough_trials)]
    
    # Add repeat variable (1 if current response matches previous response)
    mouse_data_limited['repeat'] = np.where(mouse_data_limited.response == mouse_data_limited.prevresp, 1, 0)

    # Recode response to be -1 and 1 rather than 0 and 1
    mouse_data_limited['response'] = mouse_data_limited['response'].replace({0: -1, 1: 1})
    mouse_data_limited['prevresp'] = mouse_data_limited['prevresp'].replace({0: -1, 1: 1})
    
    # Create categorical variables for contrast and previous response
    contrast_values = sorted(mouse_data_limited['signed_contrast'].unique())
    contrast_mapping = {value: f'c_{value}' for value in contrast_values}
    mouse_data_limited['contrast_category'] = mouse_data_limited['signed_contrast'].map(contrast_mapping)
    mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    
    # Convert categorical columns to category dtype
    mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp_cat'].astype('category')
    mouse_data_limited['contrast_category'] = mouse_data_limited['contrast_category'].astype('category')
        
    # Round RT for consistency
    mouse_data_limited['rt'] = mouse_data_limited['rt'].round(6)
    
    print(f"Final dataset: {mouse_data_limited['participant_id'].nunique()} mice and {len(mouse_data_limited)} trials")
    
    return mouse_data_limited

#%% Conditional bias function computation
def compute_conditional_bias_function(data, n_quantiles=5):
    """
    Compute conditional bias function with quantiles
    Returns both summary statistics and subject-level data
    """
    # Create lists to store data for each quantile
    quantile_means = []
    quantile_sems = []
    rt_means = []
    
    # Store subject-level data for plotting individual points
    subject_data = []
    
    # Process each RT quantile separately
    for q in range(n_quantiles):
        # For each subject, get their data for this quantile
        subject_means = []
        subject_rt_means = []
        
        for subj_id in data['participant_id'].unique():
            subj_data = data[data['participant_id'] == subj_id].copy()
            
            # Get quantile boundaries for this subject's RT
            quantile_edges = np.percentile(subj_data['rt'], 
                                          np.linspace(0, 100, n_quantiles+1))
            
            # Select data in this quantile
            if q < n_quantiles-1:
                q_data = subj_data[(subj_data['rt'] >= quantile_edges[q]) & 
                                  (subj_data['rt'] < quantile_edges[q+1])]
            else:
                q_data = subj_data[subj_data['rt'] >= quantile_edges[q]]
            
            if len(q_data) > 0:
                repeat_mean = q_data['repeat'].mean()
                rt_mean = q_data['rt'].mean()
                
                subject_means.append(repeat_mean)
                subject_rt_means.append(rt_mean)
                
                # Store individual subject data for later plotting
                subject_data.append({
                    'subject_id': subj_id,
                    'rt_quantile': q,
                    'repeat_mean': repeat_mean,
                    'rt_mean': rt_mean
                })
        
        # Calculate mean and SEM across subjects for this quantile
        quantile_means.append(np.mean(subject_means))
        quantile_sems.append(stats.sem(subject_means))
        rt_means.append(np.mean(subject_rt_means))
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'rt_quantile': range(n_quantiles),
        'repeat_mean': quantile_means,
        'repeat_sem': quantile_sems,
        'rt_mean': rt_means
    })
    subject_df = pd.DataFrame(subject_data)
    
    return summary, subject_df

#%% Professional plotting function with jitter
def plot_conditional_bias_function(summary, subject_df, n_quantiles=5):
    """
    Create conditional bias function plot with jittered data points
    """
    # Set figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual subject data points with jitter
    for q in range(n_quantiles):
        q_data = subject_df[subject_df['rt_quantile'] == q]
        
        # Add horizontal jitter to x-coordinates
        jitter_amount = 0.15  # Adjust this value to control jitter width
        x_jittered = np.array([q] * len(q_data)) + np.random.uniform(-jitter_amount, jitter_amount, size=len(q_data))
        
        ax.scatter(
            x_jittered, 
            q_data['repeat_mean'], 
            color='gray', 
            alpha=0.2,  # Slightly increased alpha for better visibility
            s=25,       # Slightly larger points
            zorder=1
        )
    
    # Plot main line with error bars - thicker and more prominent
    ax.errorbar(
        range(n_quantiles), 
        summary['repeat_mean'], 
        yerr=summary['repeat_sem'],
        fmt='-o', 
        color='black', 
        ecolor='black', 
        capsize=6, 
        linewidth=2.5, 
        markersize=9,
        zorder=3
    )
    
    # Add reference line at 0.5 (no bias)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, zorder=2)
    
    # Set labels with better font
    ax.set_xlabel('Response time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Choice bias (fraction)', fontsize=16, fontweight='bold')
    
    # Set axis limits
    ax.set_ylim(0.48, 0.61)
    ax.set_xlim(-0.5, n_quantiles-0.5)
    
    # Set custom x-tick labels to show actual RT values
    ax.set_xticks(range(n_quantiles))
    ax.set_xticklabels([f"{rt:.2f}" for rt in summary['rt_mean']], fontsize=13)
    
    # Set y-ticks and make them more readable
    ax.set_yticks(np.arange(0.50, 0.61, 0.04))
    ax.tick_params(axis='y', labelsize=12)
    
    # Remove top and right spines
    sns.despine(ax=ax)
    
    ax.grid(False)

    plt.tight_layout()
    
    return fig, ax

#%% Load data and process
MOUSE_DATA_PATH = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
mouse_data = load_and_preprocess_data(MOUSE_DATA_PATH)
print(f"Analysis includes {mouse_data['participant_id'].nunique()} mice and {len(mouse_data)} trials")

#%% Compute statistics
summary, subject_df = compute_conditional_bias_function(mouse_data, n_quantiles=5)
print("\nSummary statistics for each RT quantile:")
print(summary)

#%% Generate and save plot
fig, ax = plot_conditional_bias_function(summary, subject_df, n_quantiles=5)

# Save the figure
PLOT_DIRECTORY = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures')
os.makedirs(PLOT_DIRECTORY, exist_ok=True)
plot_path = os.path.join(PLOT_DIRECTORY, 'conditional_bias_function_jittered.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{plot_path}'")

# Show the plot
plt.show()

mouse_data_limited = mouse_data.copy()  # Use the preprocessed data for modeling
# %% specify HSSM models
# Define models
model_names = [
    # "ddm_nohist",
    # "ddm_prevresp_v",
    # "ddm_prevresp_z",
    "ddm_prevresp_zv"
]

# model_names = [
#     "full_ddm_nohist"
# ]
#testing
# ddm_models = {name: make_model(dataset, name) for name in model_names}

# ddm_models['ddm_nohist'].restore_traces("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_nohist_model.nc")
# ddm_models['ddm_prevresp_v'].restore_traces("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_prevresp_v_model.nc")
# ddm_models['ddm_prevresp_z'].restore_traces("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_prevresp_z_model.nc")
# ddm_models['ddm_prevresp_zv'].restore_traces("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_prevresp_zv_model.nc")

# %% run parameter estimation
# Parameters for sampling
sampling_params = {
    "chains": 4,
    "cores": 4,
    "draws": 2000,
    "tune": 2000,
}

# # Sample from the posterior for each model
model_run_results = {name: run_model(mouse_data_limited, name, script_dir, **sampling_params) for name in model_names}
#exit()

