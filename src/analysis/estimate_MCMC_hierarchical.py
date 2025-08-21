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
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv",
    "angle_nohist",
    "angle_prevresp_v",
    "angle_prevresp_z",
    "angle_prevresp_zv",
]

#ddm_models = {name: make_model(mouse_data, name, parameterization= "noncentered") for name in model_names}
#ddm_models["angle_prevresp_zv"].initvals

#ddm_models = {name: make_model(mouse_data, name, parameterization= "centered") for name in model_names}
#ddm_models["angle_prevresp_zv"].initvals
#%% 
# np.random.seed(2025)  # For reproducibility
# all_subjects = mouse_data_limited['participant_id'].unique()
# selected_subjects = np.random.choice(all_subjects, size=5, replace=False)
# print(f"Selected subjects for contrast specification testing: {selected_subjects}")

# # Use full dataset for MCMC sampling
# mouse_data_subset = mouse_data_limited.copy()
# # use the subset
# mouse_data_subset = mouse_data_subset[mouse_data_subset['participant_id'].isin(selected_subjects)]

# print(f"Full dataset includes {len(mouse_data_subset)} trials from {mouse_data_subset['participant_id'].nunique()} subjects")

# %% MCMC Sampling with Custom Initial Values
print("\n" + "="*60)
print("RUNNING MCMC SAMPLING WITH CUSTOM INITIAL VALUES")
print("="*60)

# Parameters for sampling
sampling_params = {
    'sampler': 'nuts_numpyro',
    "chains": 4,
    "cores": 4,
    "draws": 200,
    "tune": 50
}

# Option to use custom initial values
use_custom_initvals = True

#ddm_models = {name: make_model(mouse_data_limited, name, parameterization= "noncentered") for name in model_names}
#print(ddm_models["angle_prevresp_zv"].initvals)

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
            parameterization='centered',  # Match the default in make_model
            jitter_scale=0,
            n_subjects=n_subjects
        )
        #print(custom_initvals)
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

# """
# angle_theta_rt_exploration.py

# Simulate the angle-boundary DDM using ssms and inspect how the RT distribution
# changes as a function of the 'theta' parameter.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# import ssms
# from ssms.basic_simulators.simulator import simulator

# # -----------------------
# # Simulation parameters
# # -----------------------
# n_samples = 5000            # number of trials per theta
# rng_seed = 42               # reproducible-ish (ssms may or may not expose seed)
# v = 0.5                     # drift
# a = 1.2                     # boundary separation
# z = 0.5                     # starting point (proportion of a)
# t0 = 0.2                    # non-decision time
# theta_values = np.linspace(-1.2, 1.4, 10)  # values of theta to explore

# # Optional: pick a narrower theta range if you prefer
# # theta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# # -----------------------
# # Helper functions
# # -----------------------
# def run_angle_sim(theta, n_samples=n_samples, v=v, a=a, z=z, t=t0):
#     """
#     Run ssms simulator for the angle model with a single theta value.
#     Returns the rts and choices arrays from the simulator output.
#     """
#     params = {"v": v, "a": a, "z": z, "t": t, "theta": float(theta)}
#     out = simulator(model="angle", theta=params, n_samples=n_samples)
#     rts = out["rts"]
#     choices = out["choices"]
#     return np.asarray(rts), np.asarray(choices), out.get("metadata", {})

# def compute_summary_stats(rts):
#     return {
#         "N": len(rts),
#         "mean": float(np.mean(rts)),
#         "median": float(np.median(rts)),
#         "std": float(np.std(rts)),
#         "25%": float(np.percentile(rts, 25)),
#         "75%": float(np.percentile(rts, 75)),
#     }

# # -----------------------
# # Run simulations
# # -----------------------
# all_rts = {}
# all_choices = {}
# metadata_store = {}

# print(f"Running simulations for theta values: {theta_values}")

# for th in theta_values:
#     rts, choices, meta = run_angle_sim(th)
#     # filter or remove impossible RTs if present (defensive)
#     rts = rts[np.isfinite(rts)]
#     rts = rts[rts > 0]  # drop non-positive RTs if any
#     all_rts[th] = rts
#     all_choices[th] = choices
#     metadata_store[th] = meta
#     stats = compute_summary_stats(rts)
#     print(f"theta={th:.3f}: N={stats['N']}, mean={stats['mean']:.3f}, median={stats['median']:.3f}, std={stats['std']:.3f}")


# %matplotlib inline
# # -----------------------
# # Plotting: KDE overlays + histograms
# # -----------------------
# plt.figure(figsize=(10, 6))
# xmin = min(rts.min() for rts in all_rts.values())
# xmax = max(rts.max() for rts in all_rts.values())
# xs = np.linspace(xmin, xmax, 1000)

# # Use a consistent, semi-transparent color palette
# colors = plt.cm.viridis(np.linspace(0, 1, len(theta_values)))

# for idx, th in enumerate(theta_values):
#     rts = all_rts[th]
#     if len(rts) < 10:
#         continue
#     # KDE
#     try:
#         kde = gaussian_kde(rts)
#         density = kde(xs)
#     except Exception:
#         # fallback: simple histogram-based density smoothing
#         hist, bin_edges = np.histogram(rts, bins=80, density=True)
#         # linear interp to xs
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         density = np.interp(xs, bin_centers, hist)
#     plt.plot(xs, density, label=f"theta={th:.3f}", color=colors[idx], linewidth=1.6)

# plt.xlabel("Reaction time (s)")
# plt.ylabel("Density")
# plt.title("RT density for different theta values (angle DDM)")
# plt.xlim(0, xmax)
# plt.legend(title="theta", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

# # -----------------------
# # Optional: stacked histograms (same bins) for visual comparison
# # -----------------------
# plt.figure(figsize=(10, 6))
# bins = np.linspace(0, xmax, 80)
# for idx, th in enumerate(theta_values):
#     rts = all_rts[th]
#     plt.hist(rts, bins=bins, alpha=0.25, label=f"theta={th:.3f}", density=True, color=colors[idx])
# plt.xlabel("Reaction time (s)")
# plt.ylabel("Density")
# plt.title("Overlaid histograms of RT by theta")
# plt.xlim(0, xmax)
# plt.legend(title="theta", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.show()
