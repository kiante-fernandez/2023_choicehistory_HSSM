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
# 2025/08/18      Kianté Fernandez<kiantefernan@gmail.com>   refactored model estimation workflow
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/Users/kiante/Documents/2023_choicehistory_HSSM/src')
from utils.utils_hssm_modelspec import make_model
from utils.utils_hssm import run_model
# Set up file paths
script_dir = os.path.dirname(os.path.realpath(__file__))

def apply_rt_exclusions(data, lower_bound=0.1, iqr_multiplier=2):
    """Apply RT exclusions per subject"""
    
    # Apply IQR-based exclusion per subject
    def apply_iqr_exclusion(group):
        Q1 = group['rt'].quantile(0.25)
        Q3 = group['rt'].quantile(0.75)
        IQR = Q3 - Q1
        return group[
            (group['rt'] > (Q1 - iqr_multiplier * IQR)) & 
            (group['rt'] < (Q3 + iqr_multiplier * IQR))
        ]
    
    # Apply exclusions per subject
    data = data.groupby('subj_idx').apply(apply_iqr_exclusion).reset_index(drop=True)
    
    return data

# Data loading and preprocessing function
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
    
    # Filter based on movement onset and reaction time (hard cutoffs)
    mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] < 5) & 
                                           (mouse_data_limited['rt'] < 5)]
    mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] > 0.08) & 
                                           (mouse_data_limited['rt'] > 0.08)]
    
    # Apply RT exclusions using IQR method
    mouse_data_limited = apply_rt_exclusions(mouse_data_limited)
    
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
    # signed contrast solutions from meeting
    # Clip signed contrast to [-25, 25] and then divide by 100
    # This preserves the original contrast values while scaling them appropriately
    mouse_data_limited['squeezed_signed_contrast'] = mouse_data_limited['signed_contrast'].clip(upper=25, lower=-25)
    mouse_data_limited['scaled_signed_contrast'] = mouse_data_limited['squeezed_signed_contrast'] / 100
    
    # Use the scaled (not z-scored) contrast to maintain interpretable contrast levels
    mouse_data_limited['signed_contrast'] = mouse_data_limited['scaled_signed_contrast']

    # Round RT for consistency
    mouse_data_limited['rt'] = mouse_data_limited['rt'].round(6)
    
    print(f"Final dataset: {mouse_data_limited['participant_id'].nunique()} mice and {len(mouse_data_limited)} trials")
    
    return mouse_data_limited

#%% Load data and process
MOUSE_DATA_PATH = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250310.csv')
mouse_data = load_and_preprocess_data(MOUSE_DATA_PATH)
print(f"Analysis includes {mouse_data['participant_id'].nunique()} mice and {len(mouse_data)} trials")

mouse_data_limited = mouse_data.copy()

# %% specify HSSM models
model_names = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv",
    "angle_nohist",
    "angle_prevresp_v",
    "angle_prevresp_z",
    "angle_prevresp_zv",
]

ddm_models = {name: make_model(mouse_data, name) for name in model_names}

# %% run parameter estimation
# %% Run Variational Inference (VI)
print("\n" + "="*60)
print("RUNNING VARIATIONAL INFERENCE")
print("="*60)

# VI configuration parameters
vi_config = {
    "vi_niter": 100000,  # Increased for better convergence
    "vi_method": "fullrank_advi",  # Recommended method
    "vi_optimizer": "adamax",  # Recommended optimizer
    "vi_learning_rate": 0.0001,  # Recommended learning rate for adamax
}

print(f"VI Configuration:")
for key, value in vi_config.items():
    print(f"  {key}: {value}")
print()

# Run VI for each model
vi_results = {}
for i, name in enumerate(model_names, 1):
    print(f"Running VI for model {i}/{len(model_names)}: {name}")
    vi_results[name] = run_model(
        mouse_data_limited, 
        name, 
        script_dir, 
        sampling_method="vi",
        **vi_config
    )
    print(f"Completed VI for {name}\n")

print("All VI runs completed!")

# For MCMC sampling, use full dataset (VI already completed above)
# np.random.seed(2025)  # For reproducibility
# all_subjects = mouse_data_limited['participant_id'].unique()
# selected_subjects = np.random.choice(all_subjects, size=10, replace=False)
# print(f"Selected subjects for contrast specification testing: {selected_subjects}")

# Use full dataset for MCMC sampling
mouse_data_subset = mouse_data_limited.copy()
print(f"Full dataset includes {len(mouse_data_subset)} trials from {mouse_data_subset['participant_id'].nunique()} subjects")

# Parameters for sampling
sampling_params = {
    "chains": 4,
    "cores": 4,
    "draws": 300,
    "tune": 1000
}

# Sample from the posterior for each model
model_run_results = {name: run_model(mouse_data_limited, name, script_dir, **sampling_params) for name in model_names}
