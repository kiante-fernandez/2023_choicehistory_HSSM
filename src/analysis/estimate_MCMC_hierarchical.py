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
# 2025/08/20      Kianté Fernandez<kiantefernan@gmail.com>   MCMC-only version with custom initial values
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append('/Users/kiante/Documents/2023_choicehistory_HSSM/src')
from utils.utils_hssm_modelspec import make_model
from utils.utils_hssm import run_model
from utils.hierarchical_initvals import get_single_initvals
# Set up file paths
script_dir = os.path.dirname(os.path.realpath(__file__))

def apply_rt_exclusions(data, iqr_multiplier=2):
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
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the mouse data with comprehensive cleaning and feature creation.
    Now includes standardization of continuous predictors.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing mouse data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned, preprocessed, and scaled mouse data
    """
    MAX_MOVEMENT_ONSET = 2.0
    MAX_RT = 2.0
    MIN_MOVEMENT_ONSET = 0.08
    MIN_RT = 0.08
    # Load the data
    mouse_data = pd.read_csv(file_path)
    
    # Create participant ID from subject index
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    
    #mouse_data_limited = mouse_data
    # Note: Other cleaning steps (RT exclusions, etc.) are performed elsewhere
    mouse_data_limited = mouse_data[
        (mouse_data['movement_onset'] < MAX_MOVEMENT_ONSET) & 
        (mouse_data['rt'] < MAX_RT) &
        (mouse_data['movement_onset'] > MIN_MOVEMENT_ONSET) & 
        (mouse_data['rt'] > MIN_RT)
    ].copy()

    # Check for mice with less than 200 trials and exclude them
    trial_counts = mouse_data_limited.groupby('participant_id').size()
    mice_with_enough_trials = trial_counts[trial_counts >= 200].index
    print(f"\nMice with at least 200 trials: {len(mice_with_enough_trials)}")
    print(f"Mice excluded for having less than 200 trials: {len(trial_counts[trial_counts < 200])}")
    
    # Filter to keep only mice with at least 200 trials
    mouse_data_limited = mouse_data_limited[mouse_data_limited['participant_id'].isin(mice_with_enough_trials)]
        
    # Create categorical variables for contrast and previous response
    contrast_values = sorted(mouse_data_limited['signed_contrast'].unique())
    contrast_mapping = {value: f'c_{value}' for value in contrast_values}
    mouse_data_limited['contrast_category'] = mouse_data_limited['signed_contrast'].map(contrast_mapping)
    mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    
    # Convert categorical columns to category dtype
    mouse_data_limited['contrast_category'] = mouse_data_limited['contrast_category'].astype('category')

    # Recode response to be -1 and 1 rather than 0 and 1 (HSSM requirement)
    mouse_data_limited['response'] = mouse_data_limited['response'].replace({0: -1, 1: 1})
    
    # Use the pre-computed signed_contrast_squeezed column
    mouse_data_limited['signed_contrast'] = mouse_data_limited['signed_contrast_squeezed']

    # Create categorical variable for previous response
    mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp_cat'].astype('category')
    
    # Round RT for consistency
    mouse_data_limited['rt'] = mouse_data_limited['rt'].round(6)

    print("\nStandardizing continuous predictors...")
    cols_to_standardize = ['signed_contrast']
    
    scaler = StandardScaler()
    mouse_data_limited[cols_to_standardize] = scaler.fit_transform(mouse_data_limited[cols_to_standardize])
    print("Standardization complete.")
    
    print(f"\nFinal dataset: {mouse_data_limited['participant_id'].nunique()} mice and {len(mouse_data_limited)} trials")
    
    return mouse_data_limited

#%% Load data and process
MOUSE_DATA_PATH = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv')
mouse_data = load_and_preprocess_data(MOUSE_DATA_PATH)
print(f"Analysis includes {mouse_data['participant_id'].nunique()} mice and {len(mouse_data)} trials")

mouse_data_limited = mouse_data.copy()

# %% specify HSSM models
model_names = [
    "ddm_nohist",
    # "ddm_prevresp_v",
    # "ddm_prevresp_z",
    # "ddm_prevresp_zv",
    # "angle_nohist",
    # "angle_prevresp_v",
    # "angle_prevresp_z",
    # "angle_prevresp_zv",
]

#ddm_models = {name: make_model(mouse_data, name, parameterization= "noncentered") for name in model_names}
#ddm_models["angle_prevresp_zv"].initvals

#ddm_models = {name: make_model(mouse_data, name, parameterization= "centered") for name in model_names}
#ddm_models["angle_prevresp_zv"].initvals
#%% 
np.random.seed(2025)  # For reproducibility
all_subjects = mouse_data_limited['participant_id'].unique()
selected_subjects = np.random.choice(all_subjects, size=5, replace=False)
print(f"Selected subjects for contrast specification testing: {selected_subjects}")

# Use full dataset for MCMC sampling
mouse_data_subset = mouse_data_limited.copy()
# use the subset
mouse_data_subset = mouse_data_subset[mouse_data_subset['participant_id'].isin(selected_subjects)]

print(f"Full dataset includes {len(mouse_data_subset)} trials from {mouse_data_subset['participant_id'].nunique()} subjects")

# %% MCMC Sampling with Custom Initial Values
print("\n" + "="*60)
print("RUNNING MCMC SAMPLING WITH CUSTOM INITIAL VALUES")
print("="*60)

# Parameters for sampling
sampling_params = {
    'sampler': 'nuts_numpyro',
    "chains": 3,
    "cores": 3,
    "draws": 200,
    "tune": 500
}

# Option to use custom initial values
use_custom_initvals = True

ddm_models = {name: make_model(mouse_data, name, parameterization= "noncentered") for name in model_names}
print(ddm_models["ddm_nohist"].initvals)

if use_custom_initvals:
    print("Using custom hierarchical initial values based on fitted results...")
    
    # Sample from the posterior for each model with custom initial values
    model_run_results = {}
    for name in model_names:
        print(f"Getting custom initial values for {name}...")
        
        # Get single set of initial values (HSSM will handle multiple chains internally)
        n_subjects = mouse_data_limited['participant_id'].nunique()
        custom_initvals = get_single_initvals(
            model_name=name, 
            parameterization='noncentered',  # Match the default in make_model
            jitter_scale=0.001,
            n_subjects=n_subjects
        )
        print(custom_initvals)
        print(f"Running MCMC for {name} with custom initial values...")
        model_run_results[name] = run_model(
            mouse_data_limited, 
            name, 
            script_dir, 
            initvals=custom_initvals,
            **sampling_params
        )
        print(f"Completed MCMC for {name}\n")
else:
    print("Using default HSSM initial values...")
    # Sample from the posterior for each model without custom initial values
    model_run_results = {name: run_model(mouse_data_limited, name, script_dir, **sampling_params) for name in model_names}

print("\n" + "="*60)
print("ALL MCMC SAMPLING COMPLETED!")
print("="*60)

# %%