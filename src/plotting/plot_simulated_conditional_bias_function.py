"""
Conditional bias function plot for simulated DDM datasets.

This script creates a publication-ready conditional bias function plot showing 
how different DDM model variants predict choice repetition patterns across 
reaction time quantiles. Based on Urai et al. 2019 methodology but applied 
to HSSM simulation results.

Features:
- Loads all available simulated datasets from results/simulated_datasets/
- Computes conditional bias function for each DDM model (ddma, ddmb, ddmc, ddmd)
- Creates RT quantiles and computes conditional bias function per model
- Left panel: Model predictions as different colored lines
- Right panel: Fast vs Slow RT comparison with different shapes/colors per model
- Publication-ready styling and saves PNG + PDF formats

Model predictions from HSSM DDM simulations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import utils_plot as tools
## INITIALIZE A FEW THINGS
tools.seaborn_style()

# grab the utils that are already defined
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils_plot as tools
tools.seaborn_style()

# Configuration
SIMULATED_DATA_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/simulated_datasets'
OUTPUT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'

try:
    assert os.path.exists(SIMULATED_DATA_DIR)
except:
    SIMULATED_DATA_DIR = '/Users/uraiae/Documents/code/2023_choicehistory_HSSM/results/simulated_datasets'
    OUTPUT_DIR = "/Users/uraiae/Documents/code/2023_choicehistory_HSSM/results/figures"


def load_all_simulated_data(sample_size_per_subject=1000, 
                            max_subjects=62):
    """
    Load and combine all available simulated datasets.
    
    Parameters:
    -----------
    sample_size_per_subject : int
        Number of trials to sample per subject to make processing manageable
    max_subjects : int
        Maximum number of subjects to process (for testing purposes)
    
    Returns:
    --------
    pd.DataFrame
        Combined simulated data with all models and subjects
    """
    print(f"Loading simulated data from {SIMULATED_DATA_DIR}")
    
    # Find all simulation files
    sim_files = glob.glob(os.path.join(SIMULATED_DATA_DIR, "*/*_all_models_simulated.csv"))
    print(f"Found {len(sim_files)} simulation files")
    
    all_data = []
    subjects_processed = 0
    
    for sim_file in sim_files[:max_subjects]:  # Limit for testing
        try:
            # Extract subject name from path
            subject_name = os.path.basename(os.path.dirname(sim_file))
            
            # Load simulation data - sample to make it manageable
            print(f"  Loading {subject_name}...")
            sim_data = pd.read_csv(sim_file)
            
            if len(sim_data) > sample_size_per_subject:
                # Simple random sampling to make processing faster
                sim_data_sampled = sim_data.sample(n=sample_size_per_subject, random_state=42)
                print(f"    Sampled {len(sim_data_sampled)} from {len(sim_data)} trials")
            else:
                sim_data_sampled = sim_data
            
            if len(sim_data_sampled) > 0:
                sim_data_sampled['subject_name'] = subject_name
                all_data.append(sim_data_sampled)
                subjects_processed += 1
                print(f"    Using {len(sim_data_sampled)} trials for {subject_name}")
            
        except Exception as e:
            print(f"  Failed to load {sim_file}: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No simulation data could be loaded!")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Create unique participant IDs for each model-subject combination
    combined_data['participant_id'] = (
        combined_data['subject_name'] + '_' + combined_data['model']
    )
    
    # Compute repeat variable for simulated data: compare simulated response to previous observed response
    # Note: response_sim and prevresp are both in {-1, 1} coding, so we can compare directly
    combined_data['repeat_sim'] = np.where(
        combined_data['response_sim'] == combined_data['prevresp'], 1, 0
    )
    
    # Clean data - remove NaN values and negative RTs
    combined_data = combined_data.dropna(subset=['rt_sim', 'response_sim', 'prevresp'])
    combined_data = combined_data[combined_data['rt_sim'] >= 0]
    
    print(f"\nLoaded simulation data:")
    print(f"  Total subjects: {subjects_processed}")
    print(f"  Total models: {combined_data['model'].nunique()}")
    print(f"  Models: {sorted(combined_data['model'].unique())}")
    print(f"  Total simulated trials: {len(combined_data):,}")
    
    # Summary by model
    for model in sorted(combined_data['model'].unique()):
        model_data = combined_data[combined_data['model'] == model]
        mean_repeat = model_data['repeat_sim'].mean()
        print(f"  {model}: {len(model_data):,} trials, repeat rate: {mean_repeat:.3f}")
    
    return combined_data


def load_observational_data(subject_names):
    """
    Load observational data from the main IBL dataset, filtered to subjects with simulation data.
    
    Parameters:
    -----------
    subject_names : list
        List of subject names that have simulation data
        
    Returns:
    --------
    pd.DataFrame
        IBL observational data for subjects with simulations
    """
    print(f"\nLoading observational data from main IBL dataset for {len(subject_names)} subjects...")
    
    # Define IBL dataset path (same as original script)
    IBL_DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv'
    
    try:
        assert os.path.exists(IBL_DATA_PATH)
    except:
        IBL_DATA_PATH = '/Users/uraiae/Documents/code/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv'
    
    print(f"  Loading IBL data from: {IBL_DATA_PATH}")
    
    try:
        # Load the main IBL dataset
        ibl_data = pd.read_csv(IBL_DATA_PATH)
        print(f"  Loaded full IBL dataset: {len(ibl_data):,} trials from {ibl_data['subj_idx'].nunique()} subjects")
        
        # Filter to only subjects that have simulation data
        obs_data = ibl_data[ibl_data['subj_idx'].isin(subject_names)].copy()
        
        if len(obs_data) == 0:
            print("  No matching subjects found in IBL dataset!")
            return pd.DataFrame()
        
        # Apply same preprocessing as original script
        # Recode prevresp from {-1, 1} to {0, 1} to match response coding
        obs_data['prevresp_recoded'] = obs_data['prevresp'].map({-1.0: 0.0, 1.0: 1.0})
        
        # Create repetition variable (1 if current response equals previous response)
        obs_data['repeat_obs'] = np.where(obs_data['response'] == obs_data['prevresp_recoded'], 1, 0)
        
        # Create participant IDs for grouping (same format as original)
        obs_data['participant_id'] = obs_data['subj_idx'] + '_observed'
        obs_data['subject_name'] = obs_data['subj_idx']
        
        # Clean data - remove NaN values and negative RTs
        obs_data = obs_data.dropna(subset=['rt', 'response', 'prevresp'])
        obs_data = obs_data[obs_data['rt'] >= 0]
        
        print(f"  Filtered to subjects with simulations: {obs_data['subj_idx'].nunique()} subjects")
        print(f"  Total observational trials: {len(obs_data):,}")
        print(f"  Mean observed repeat rate: {obs_data['repeat_obs'].mean():.3f}")
        
        return obs_data
        
    except Exception as e:
        print(f"  Failed to load IBL dataset: {e}")
        return pd.DataFrame()


def compute_conditional_bias_function_simulated(data, n_quantiles=5):
    """
    Compute conditional bias function for simulated data by model.
    
    For each participant-model combination:
    1. Divide trials into RT quantiles 
    2. Calculate fraction of choices in direction of individual's history bias per quantile
    3. Aggregate by model across participants (mean ± SEM)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with columns: participant_id, subject_name, model, rt_sim, repeat_sim
    n_quantiles : int
        Number of RT quantiles (default=5)
        
    Returns:
    --------
    tuple: (summary_df, subject_df)
        summary_df: Summary statistics by model across participants
        subject_df: Individual subject-model data for plotting
    """
    subject_data = []
    
    # Process each participant-model combination separately
    for participant_id in data['participant_id'].unique():
        subj_data = data[data['participant_id'] == participant_id].copy()
        
        if len(subj_data) < n_quantiles:
            continue
            
        # Extract subject and model info
        subject_name = subj_data['subject_name'].iloc[0]
        model = subj_data['model'].iloc[0]
        
        # Determine individual's overall bias direction (repeat vs alternate)
        overall_repeat_rate = subj_data['repeat_sim'].mean()
        
        # Create RT quantiles for this participant
        try:
            subj_data['rt_quantile'] = pd.qcut(
                subj_data['rt_sim'], q=n_quantiles, labels=False, duplicates='drop'
            )
        except ValueError:
            # If RT values are too similar, use rank-based quantiles
            subj_data['rt_quantile'] = pd.qcut(
                subj_data['rt_sim'].rank(method='first'), q=n_quantiles, labels=False
            )
        
        # For each quantile, calculate raw repeat rate (P(repeat))
        for q in range(n_quantiles):
            q_data = subj_data[subj_data['rt_quantile'] == q]
            
            if len(q_data) > 0:
                # Use raw repeat rate for direct comparison between models and data
                choice_bias = q_data['repeat_sim'].mean()
                
                rt_mean = q_data['rt_sim'].mean()
                
                subject_data.append({
                    'participant_id': participant_id,
                    'subject_name': subject_name,
                    'model': model,
                    'rt_quantile': q,
                    'choice_bias': choice_bias,
                    'repeat_rate': q_data['repeat_sim'].mean(),
                    'rt_mean': rt_mean,
                    'n_trials': len(q_data),
                    'overall_bias': 'repeat' if overall_repeat_rate > 0.5 else 'alternate',
                    'overall_repeat_rate': overall_repeat_rate
                })
    
    subject_df = pd.DataFrame(subject_data)
    
    if len(subject_df) == 0:
        raise ValueError("No conditional bias data could be computed!")
    
    # Compute summary statistics by model and quantile
    summary_stats = []
    
    for model in subject_df['model'].unique():
        model_data = subject_df[subject_df['model'] == model]
        
        for q in range(n_quantiles):
            q_data = model_data[model_data['rt_quantile'] == q]
            
            if len(q_data) > 0:
                summary_stats.append({
                    'model': model,
                    'rt_quantile': q,
                    'choice_bias_mean': q_data['choice_bias'].mean(),
                    'choice_bias_sem': stats.sem(q_data['choice_bias']),
                    'rt_mean': q_data['rt_mean'].mean(),
                    'rt_sem': stats.sem(q_data['rt_mean']),
                    'n_subjects': len(q_data)
                })
    
    summary = pd.DataFrame(summary_stats)
    
    print(f"\nComputed conditional bias function:")
    print(f"  Subject-model-quantile combinations: {len(subject_df)}")
    print(f"  Models with data: {sorted(summary['model'].unique())}")
    
    for model in sorted(summary['model'].unique()):
        model_summary = summary[summary['model'] == model]
        bias_range = f"{model_summary['choice_bias_mean'].min():.3f} to {model_summary['choice_bias_mean'].max():.3f}"
        n_subjects = model_summary['n_subjects'].iloc[0] if len(model_summary) > 0 else 0
        print(f"  {model}: n={n_subjects} subjects, bias range: {bias_range}")
    
    return summary, subject_df


def compute_conditional_bias_function_observed(data, n_quantiles=5):
    """
    Compute conditional bias function for observational data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Observational data with columns: participant_id, subject_name, rt, repeat_obs
    n_quantiles : int
        Number of RT quantiles (default=5)
        
    Returns:
    --------
    tuple: (summary_df, subject_df)
        summary_df: Summary statistics across participants
        subject_df: Individual subject data for plotting
    """
    if len(data) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    subject_data = []
    
    # Process each subject separately
    for participant_id in data['participant_id'].unique():
        subj_data = data[data['participant_id'] == participant_id].copy()
        
        if len(subj_data) < n_quantiles:
            continue
            
        # Extract subject info
        subject_name = subj_data['subject_name'].iloc[0]
        
        # Determine individual's overall bias direction (repeat vs alternate)
        overall_repeat_rate = subj_data['repeat_obs'].mean()
        
        # Create RT quantiles for this participant
        try:
            subj_data['rt_quantile'] = pd.qcut(
                subj_data['rt'], q=n_quantiles, labels=False, duplicates='drop'
            )
        except ValueError:
            # If RT values are too similar, use rank-based quantiles
            subj_data['rt_quantile'] = pd.qcut(
                subj_data['rt'].rank(method='first'), q=n_quantiles, labels=False
            )
        
        # For each quantile, calculate raw repeat rate (P(repeat))
        for q in range(n_quantiles):
            q_data = subj_data[subj_data['rt_quantile'] == q]
            
            if len(q_data) > 0:
                # Use raw repeat rate for direct comparison between models and data
                choice_bias = q_data['repeat_obs'].mean()
                
                rt_mean = q_data['rt'].mean()
                
                subject_data.append({
                    'participant_id': participant_id,
                    'subject_name': subject_name,
                    'model': 'observed',
                    'rt_quantile': q,
                    'choice_bias': choice_bias,
                    'repeat_rate': q_data['repeat_obs'].mean(),
                    'rt_mean': rt_mean,
                    'n_trials': len(q_data),
                    'overall_bias': 'repeat' if overall_repeat_rate > 0.5 else 'alternate',
                    'overall_repeat_rate': overall_repeat_rate
                })
    
    subject_df = pd.DataFrame(subject_data)
    
    if len(subject_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute summary statistics
    summary_stats = []
    
    for q in range(n_quantiles):
        q_data = subject_df[subject_df['rt_quantile'] == q]
        
        if len(q_data) > 0:
            summary_stats.append({
                'model': 'observed',
                'rt_quantile': q,
                'choice_bias_mean': q_data['choice_bias'].mean(),
                'choice_bias_sem': stats.sem(q_data['choice_bias']),
                'rt_mean': q_data['rt_mean'].mean(),
                'rt_sem': stats.sem(q_data['rt_mean']),
                'n_subjects': len(q_data)
            })
    
    summary = pd.DataFrame(summary_stats)
    
    print(f"\nComputed conditional bias function for observed data:")
    print(f"  Subject-quantile combinations: {len(subject_df)}")
    if len(summary) > 0:
        bias_range = f"{summary['choice_bias_mean'].min():.3f} to {summary['choice_bias_mean'].max():.3f}"
        n_subjects = summary['n_subjects'].iloc[0] if len(summary) > 0 else 0
        print(f"  Observed: n={n_subjects} subjects, bias range: {bias_range}")
    
    return summary, subject_df


def get_model_label(model):
    """
    Convert model codes to proper legend labels with subscripts.
    """
    model_labels = {
        'ddma': 'No history',
        'ddmb': r'v$_{\mathrm{bias}}$',
        'ddmc': 'z',
        'ddmd': 'Both'
    }
    return model_labels.get(model, model.upper())

#%%

def plot_simulated_conditional_bias_function(summary, subject_df, obs_summary=None, obs_subject_df=None, n_quantiles=5):
    """
    Create two-panel plot showing simulated model predictions and observational data.
    
    Left panel: conditional bias function by model (different colors) + observed data (black)
    Right panel: Fast vs Slow RT comparison by model (different shapes/colors) + observed data (black)
    
    Parameters:
    -----------
    summary : pd.DataFrame
        Summary statistics by model
    subject_df : pd.DataFrame
        Individual subject-model data
    obs_summary : pd.DataFrame, optional
        Summary statistics for observed data
    obs_subject_df : pd.DataFrame, optional
        Individual subject data for observed data
    n_quantiles : int
        Number of RT quantiles
        
    Returns:
    --------
    tuple: (fig, (ax1, ax2))
        Matplotlib figure and axes objects
    """
    # Get model colors from utils
    model_colors, _ = tools.get_colors()
    
    # Define markers for right panel
    model_markers = {
        'ddma': 'o',      # circle
        'ddmb': 's',      # square  
        'ddmc': '^',      # triangle up
        'ddmd': 'D'       # diamond
    }
    
    # Create two-panel figure with extra width for external legends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # ===== LEFT PANEL: Conditional bias function by model =====
    
    models_to_plot = sorted(summary['model'].unique())
    
    # Plot simulation models
    for model in models_to_plot:
        model_summary = summary[summary['model'] == model]
        model_subjects = subject_df[subject_df['model'] == model]
        color = model_colors.get(model, 'black')
        
        if len(model_summary) > 0:
            # # Plot individual subject lines (faint)
            # for subject in model_subjects['participant_id'].unique():
            #     subject_data = model_subjects[
            #         model_subjects['participant_id'] == subject
            #     ].sort_values('rt_quantile')
                
            #     if len(subject_data) == n_quantiles:  # Only if subject has all quantiles
            #         ax1.plot(subject_data['rt_quantile'], subject_data['choice_bias'], 
            #                 color=color, alpha=0.01, linewidth=0.5, zorder=1)
            
            # Plot model mean with error bars

            sns.lineplot(ax=ax1, data=model_summary,
                         x='rt_quantile', y='choice_bias_mean',
                         color=color)
            # ax1.errorbar(
            #     model_summary['rt_quantile'], 
            #     model_summary['choice_bias_mean'], 
            #     yerr=model_summary['choice_bias_sem'],
            #     fmt='-0', 
            #     color=color, 
            #     ecolor=color, 
            #     capsize=0, 
            #     linewidth=3, 
            #     markersize=0,
            #     zorder=3,
            #     label=get_model_label(model)
            # )
    
    # Plot observed data if available
    if obs_summary is not None and len(obs_summary) > 0:
        # # Plot individual observed subject lines (faint)
        # if obs_subject_df is not None:
        #     for subject in obs_subject_df['participant_id'].unique():
        #         subject_data = obs_subject_df[
        #             obs_subject_df['participant_id'] == subject
        #         ].sort_values('rt_quantile')
                
        #         if len(subject_data) == n_quantiles:  # Only if subject has all quantiles
        #             ax1.plot(subject_data['rt_quantile'], subject_data['choice_bias'], 
        #                     color='black', alpha=0.05, linewidth=0.5, zorder=2)
        
        # Plot observed mean with error bars
        # sns.lineplot(ax=ax1, data=obs_summary,
        #                  x='rt_quantile', y='choice_bias_mean',
        #                  color='black', err_style='bars', 
        #                  errorbar='choice_bias_sem',
        #                  markers=True)
        ax1.errorbar(
            obs_summary['rt_quantile'], 
            obs_summary['choice_bias_mean'], 
            yerr=obs_summary['choice_bias_sem'],
            fmt='-o', 
            color='black', 
            # ecolor='black', 
            # capsize=2, 
            # linewidth=4, 
            # markersize=8,
            # markerfacecolor='white',
            # zorder=4,
            # label='OBSERVED'
        )
    
    # Customize left panel
    ax1.set_xlabel('RT (quantiles)')
    ax1.set_ylabel('P(repeat)')
    # ax1.set_title('A. Model Predictions: Conditional Bias', fontweight='bold', pad=15)
    
    # Set axis limits and ticks
    ax1.set_xlim(-0.5, n_quantiles-0.5)
    ax1.set_xticks(range(n_quantiles))
    ax1.set_xticklabels([f'Q{i+1}' for i in range(n_quantiles)])
    ax1.set_xticklabels('')
    
    # Legend will be added at the figure level later
    
    # Add reference line at 0.5 (no bias)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # ===== RIGHT PANEL: Fast vs Slow by model =====
    
    if subject_df is not None and len(subject_df) > 0:
        # Extract Fast (Q1) and Slow (Q5) data for each model
        fast_slow_data = []
        
        for model in models_to_plot:
            model_subjects = subject_df[subject_df['model'] == model]
            
            for subject in model_subjects['participant_id'].unique():
                subject_data = model_subjects[model_subjects['participant_id'] == subject]
                
                # Get Q1 (Fast) and Q5 (Slow) - using 0-based indexing
                fast_data = subject_data[subject_data['rt_quantile'] == 0]  # Q1
                slow_data = subject_data[subject_data['rt_quantile'] == n_quantiles-1]  # Q5
                
                if len(fast_data) > 0:
                    fast_slow_data.append({
                        'participant_id': subject,
                        'model': model,
                        'condition': 'Fast',
                        'choice_bias': fast_data['choice_bias'].iloc[0]
                    })
                
                if len(slow_data) > 0:
                    fast_slow_data.append({
                        'participant_id': subject,
                        'model': model, 
                        'condition': 'Slow',
                        'choice_bias': slow_data['choice_bias'].iloc[0]
                    })
        
        fast_slow_df = pd.DataFrame(fast_slow_data)
        
        if len(fast_slow_df) > 0:
            # Plot by model with offset positions - increased spacing
            model_positions = {'ddma': -0.4, 'ddmb': -0.2, 'ddmc': 0.2, 'ddmd': 0.4}
            
            for model in models_to_plot:
                model_data = fast_slow_df[fast_slow_df['model'] == model]
                color = model_colors.get(model, 'black')
                marker = model_markers.get(model, 'o')
                offset = model_positions.get(model, 0)
                
                if len(model_data) > 0:
                    # Plot Fast and Slow conditions
                    for i, condition in enumerate(['Fast', 'Slow']):
                        condition_data = model_data[model_data['condition'] == condition]
                        x_pos = i + offset
                        
                        if len(condition_data) > 0:
                            # Individual points (jittered)
                            jitter = np.random.normal(0, 0.02, len(condition_data))
                            ax2.scatter(x_pos + jitter, condition_data['choice_bias'], 
                                      color=color, alpha=0.20, s=25, marker=marker, 
                                      zorder=3, label=get_model_label(model) if condition == 'Fast' else "")
                    
                    # Connect paired Fast-Slow points for each subject
                    fast_data = model_data[model_data['condition'] == 'Fast']
                    slow_data = model_data[model_data['condition'] == 'Slow']
                    
                    # Merge to get paired data
                    paired_data = pd.merge(fast_data, slow_data, on='participant_id', suffixes=('_fast', '_slow'))
                    
                    for _, row in paired_data.iterrows():
                        ax2.plot([0 + offset, 1 + offset], 
                               [row['choice_bias_fast'], row['choice_bias_slow']], 
                               color=color, alpha=0.05, linewidth=0.5, zorder=1)
    
    # Add box plots for each model and condition
    if len(fast_slow_df) > 0:
        # Box plot styling
        box_width = 0.12
        
        # Create box plots for each model in both Fast and Slow conditions
        for model in models_to_plot:
            model_data = fast_slow_df[fast_slow_df['model'] == model]
            color = model_colors.get(model, 'black')
            offset = model_positions.get(model, 0)
            
            if len(model_data) > 0:
                # Fast condition box plot
                fast_model_data = model_data[model_data['condition'] == 'Fast']['choice_bias'].values
                if len(fast_model_data) > 0:
                    ax2.boxplot([fast_model_data], positions=[0 + offset], widths=box_width,
                               patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=color, alpha=0.6, linewidth=1),
                               whiskerprops=dict(color=color, linewidth=1),
                               capprops=dict(color=color, linewidth=1),
                               medianprops=dict(color='black', linewidth=1.5))
                
                # Slow condition box plot
                slow_model_data = model_data[model_data['condition'] == 'Slow']['choice_bias'].values
                if len(slow_model_data) > 0:
                    ax2.boxplot([slow_model_data], positions=[1 + offset], widths=box_width,
                               patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=color, alpha=0.6, linewidth=1),
                               whiskerprops=dict(color=color, linewidth=1),
                               capprops=dict(color=color, linewidth=1),
                               medianprops=dict(color='black', linewidth=1.5))

    # Add observed data to right panel if available
    if obs_subject_df is not None and len(obs_subject_df) > 0:
        # Extract Fast (Q1) and Slow (Q5) data for observed data
        obs_fast_slow_data = []
        
        for subject in obs_subject_df['participant_id'].unique():
            subject_data = obs_subject_df[obs_subject_df['participant_id'] == subject]
            
            # Get Q1 (Fast) and Q5 (Slow) - using 0-based indexing
            fast_data = subject_data[subject_data['rt_quantile'] == 0]  # Q1
            slow_data = subject_data[subject_data['rt_quantile'] == n_quantiles-1]  # Q5
            
            if len(fast_data) > 0:
                obs_fast_slow_data.append({
                    'participant_id': subject,
                    'condition': 'Fast',
                    'choice_bias': fast_data['choice_bias'].iloc[0]
                })
            
            if len(slow_data) > 0:
                obs_fast_slow_data.append({
                    'participant_id': subject,
                    'condition': 'Slow',
                    'choice_bias': slow_data['choice_bias'].iloc[0]
                })
        
        obs_fast_slow_df = pd.DataFrame(obs_fast_slow_data)
        
        if len(obs_fast_slow_df) > 0:
            # Plot observed data with no offset (centered)
            for i, condition in enumerate(['Fast', 'Slow']):
                condition_data = obs_fast_slow_df[obs_fast_slow_df['condition'] == condition]
                x_pos = i
                
                if len(condition_data) > 0:
                    # Individual points (jittered)
                    jitter = np.random.normal(0, 0.02, len(condition_data))
                    ax2.scatter(x_pos + jitter, condition_data['choice_bias'], 
                              color='black', alpha=0.20, s=40, marker='o', 
                              zorder=5, label='OBSERVED' if condition == 'Fast' else "")
            
            # Connect paired Fast-Slow points for observed data
            fast_data = obs_fast_slow_df[obs_fast_slow_df['condition'] == 'Fast']
            slow_data = obs_fast_slow_df[obs_fast_slow_df['condition'] == 'Slow']
            
            # Merge to get paired data
            paired_obs_data = pd.merge(fast_data, slow_data, on='participant_id', suffixes=('_fast', '_slow'))
            
            for _, row in paired_obs_data.iterrows():
                ax2.plot([0, 1], 
                       [row['choice_bias_fast'], row['choice_bias_slow']], 
                       color='black', alpha=0.05, linewidth=1, zorder=4)
            
            # Add box plots for observed data
            fast_obs_data = obs_fast_slow_df[obs_fast_slow_df['condition'] == 'Fast']['choice_bias'].values
            slow_obs_data = obs_fast_slow_df[obs_fast_slow_df['condition'] == 'Slow']['choice_bias'].values
            
            if len(fast_obs_data) > 0:
                ax2.boxplot([fast_obs_data], positions=[0], widths=0.15,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='black', alpha=0.6, linewidth=1.5),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           medianprops=dict(color='white', linewidth=2))
            
            if len(slow_obs_data) > 0:
                ax2.boxplot([slow_obs_data], positions=[1], widths=0.15,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='black', alpha=0.6, linewidth=1.5),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           medianprops=dict(color='white', linewidth=2))
    
    # Customize right panel
    ax2.set_xlabel('RT')
    ax2.set_ylabel('P(repeat)')
    # ax2.set_title('B. Fast vs Slow RT by Model', fontweight='bold', pad=15)

    # Set axis properties
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Fast', 'Slow'])
    
    # Legend will be added at the figure level later
    
    # Add reference line
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # # Apply styling to both panels
    # for ax in [ax1, ax2]:
    #     # Remove grid lines
    #     ax.grid(False)
        
    #     # Create detached axis style
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_position(('outward', 10))
    #     ax.spines['bottom'].set_position(('outward', 10))
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
        
    #     # Enhance tick formatting
    #     ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        
    #     # Set white background
    #     ax.set_facecolor('white')
    
    plt.tight_layout()
    tools.seaborn_style()
    sns.despine(trim=True)

    # # Create a single legend for the entire figure, positioned to the right
    # # Collect handles and labels from the first axis (they're the same for both)
    # handles, labels = ax1.get_legend_handles_labels()
    
    # # Add the figure-level legend positioned to the right of both plots
    # fig.legend(handles, labels, bbox_to_anchor=(0.98, 0.5), loc='center right', 
    #            frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # # Adjust layout to accommodate external legend
    # plt.subplots_adjust(right=0.82)
    
    return fig, (ax1, ax2)

#%%

def main():

    #%%
    """Main execution function."""
    print("="*70)
    print("SIMULATED DDM CONDITIONAL BIAS FUNCTION ANALYSIS")
    print("="*70)
    
    # Load and preprocess simulated data
    try:
        data = load_all_simulated_data(sample_size_per_subject=100000)
    except Exception as e:
        print(f"Error loading data: {e}")
        #return
    
    # Get list of subjects with simulation data
    subject_names = data['subject_name'].unique()
    
    # Load observational data for the same subjects
    obs_data = load_observational_data(subject_names)
    
    #%% Compute conditional bias function by model
    print("\nComputing conditional bias function by model...")
    try:
        summary, subject_df = compute_conditional_bias_function_simulated(data, n_quantiles=5)
    except Exception as e:
        print(f"Error computing conditional bias function: {e}")
        #return
    
    if len(summary) == 0:
        print("Error: No bias function data computed!")
        #return
    
    # Compute conditional bias function for observed data
    obs_summary = pd.DataFrame()
    obs_subject_df = pd.DataFrame()
    if len(obs_data) > 0:
        try:
            obs_summary, obs_subject_df = compute_conditional_bias_function_observed(obs_data, n_quantiles=5)
        except Exception as e:
            print(f"Error computing observed conditional bias function: {e}")
    
    #%% Create plot
    print("\nCreating plot...")
    try:
        fig, _ = plot_simulated_conditional_bias_function(
            summary, subject_df, 
            obs_summary=obs_summary, obs_subject_df=obs_subject_df,
            n_quantiles=5
        )
        
        # Save plots
        output_base = os.path.join(OUTPUT_DIR, "simulated_datasets_conditional_bias_function")
        
        png_path = f"{output_base}.png"
        pdf_path = f"{output_base}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        
        print(f"\nSaved plots:")
        print(f"  PNG: {png_path}")
        print(f"  PDF: {pdf_path}")
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Total subjects with simulations: {subject_df['subject_name'].nunique()}")
        print(f"Models analyzed: {sorted(summary['model'].unique())}")
        print(f"Total subject-model combinations: {len(subject_df['participant_id'].unique())}")
        
        # Model-specific summaries
        for model in sorted(summary['model'].unique()):
            model_summary = summary[summary['model'] == model]
            model_subjects = subject_df[subject_df['model'] == model]
            n_subjects = model_subjects['subject_name'].nunique()
            bias_range = f"{model_summary['choice_bias_mean'].min():.3f} to {model_summary['choice_bias_mean'].max():.3f}"
            
            print(f"\n{model.upper()}:")
            print(f"  Subjects: {n_subjects}")
            print(f"  Choice bias range: {bias_range}")
            
            # Test for significant deviation from chance across quantiles
            choice_bias_values = model_summary['choice_bias_mean'].values
            if len(choice_bias_values) > 1:
                t_stat, p_val = stats.ttest_1samp(choice_bias_values, 0.5)
                print(f"  One-sample t-test vs chance (0.5): t = {t_stat:.3f}, p = {p_val:.4f}")
        
        # Observed data summary
        if len(obs_summary) > 0:
            obs_subjects = obs_subject_df['subject_name'].nunique()
            bias_range = f"{obs_summary['choice_bias_mean'].min():.3f} to {obs_summary['choice_bias_mean'].max():.3f}"
            
            print(f"\nOBSERVED:")
            print(f"  Subjects: {obs_subjects}")
            print(f"  Choice bias range: {bias_range}")
            
            # Test for significant deviation from chance across quantiles
            choice_bias_values = obs_summary['choice_bias_mean'].values
            if len(choice_bias_values) > 1:
                t_stat, p_val = stats.ttest_1samp(choice_bias_values, 0.5)
                print(f"  One-sample t-test vs chance (0.5): t = {t_stat:.3f}, p = {p_val:.4f}")
        
        print(f"\nAnalysis complete!")
        

    except Exception as e:
        print(f"Error creating plot: {e}")
        #return
    #%%

if __name__ == "__main__":
    main()
# %%
