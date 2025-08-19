# hssm_mice_choice_history_ddm.py - a reproduction of models from
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
# Choice history biases subsequent evidence accumulation. eLife
# using The International Brain Laboratory data
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2024/10/06      Kianté Fernandez<kiantefernan@gmail.com>   updated workflow from hssm_reproduce for mice
# 2024/10/18      Kianté Fernandez<kiantefernan@gmail.com>   added model comparisons

import os
import shutil
from tokenize import group
import numpy as np
import pandas as pd
from ..utils.utils_hssm_modelspec import make_model 
from ..utils.utils_hssm import run_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jax
import pytensor  # Graph-based tensor library

import numpyro
import jax

from hssm import load_data
# Set the number of CPU devices you want to use (at the start of your script)
numpyro.set_host_device_count(4)

# Verify the device count
print(f"Available devices: {jax.local_device_count()}")

def load_and_preprocess_mouse_data(file_path):
    mouse_data = pd.read_csv(file_path)
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    
    # Clean data
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    return mouse_data

def apply_rt_exclusions(data, lower_bound=0.1, iqr_multiplier=2):
    """Apply RT exclusions per subject:
    1. Remove RTs below lower bound
    2. Apply IQR-based exclusion"""
    
    # First apply the lower bound
    data = data[data['rt'] > lower_bound]
    
    # Apply IQR-based exclusion per subject
    def apply_iqr_exclusion(group, iqr_multiplier=2):
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

def print_dataset_stats(dataset, show_excluded=True, min_trials=30):
    """Print RT statistics before and after exclusions"""
    
    if show_excluded:
        print("\nBEFORE EXCLUSIONS:")
        print(f"Number of trials: {len(dataset)}")
        print(f"Number of unique subjects: {dataset['subj_idx'].nunique()}")
        print(f"Overall RT range: {dataset['rt'].min():.3f} to {dataset['rt'].max():.3f} seconds")
    
    def limit_trials(group):
        return group.head(350)  # Take only first 350 trials

    # Apply the function to each session group
    mouse_data_limited = dataset.groupby(['subj_idx', 'session']).apply(limit_trials).reset_index(drop=True)
    
    # Apply exclusions
    clean_dataset = apply_rt_exclusions(mouse_data_limited)

    # Check for subjects with fewer than min_trials trials
    trials_per_subject = clean_dataset.groupby('subj_idx').size()
    subjects_to_exclude = trials_per_subject[trials_per_subject < min_trials].index
    
    if len(subjects_to_exclude) > 0:
        print(f"\nExcluding {len(subjects_to_exclude)} subjects with fewer than {min_trials} trials:")
        for subj in subjects_to_exclude:
            print(f"Subject {subj}: {trials_per_subject[subj]} trials")
        
        # Remove subjects with too few trials
        clean_dataset = clean_dataset[~clean_dataset['subj_idx'].isin(subjects_to_exclude)]
    
    print("\nAFTER ALL EXCLUSIONS (RT and minimum trials):")
    print(f"Number of trials: {len(clean_dataset)}")
    print(f"Number of unique subjects: {clean_dataset['subj_idx'].nunique()}")
    print(f"Overall RT range: {clean_dataset['rt'].min():.3f} to {clean_dataset['rt'].max():.3f} seconds")

    # Calculate detailed RT statistics per mouse after exclusions
    rt_stats = clean_dataset.groupby('subj_idx').agg({
        'rt': ['count', 'mean', 'std', 'min', 'max', 
               lambda x: x.quantile(0.25),  # Q1
               lambda x: x.quantile(0.50),  # median
               lambda x: x.quantile(0.75)]  # Q3
    })
    
    # Rename columns for clarity
    rt_stats.columns = ['n_trials', 'mean_rt', 'std_rt', 'min_rt', 'max_rt', 'q1_rt', 'median_rt', 'q3_rt']
    
    # Calculate IQR
    rt_stats['iqr_rt'] = rt_stats['q3_rt'] - rt_stats['q1_rt']
    
    # Calculate percentage of trials retained per mouse
    if show_excluded:
        original_counts = dataset.groupby('subj_idx')['rt'].count()
        rt_stats['original_trials'] = original_counts
        rt_stats['percent_retained'] = (rt_stats['n_trials'] / rt_stats['original_trials'] * 100).round(1)
    
    # Round all RT values to 3 decimal places
    rt_stats = rt_stats.round(3)
    
    print("\nRT Statistics per mouse after exclusions:")
    print(rt_stats)
    
    print("\nSummary of per-mouse RT statistics after exclusions:")
    summary_stats = rt_stats.agg(['mean', 'min', 'max']).round(3)
    print(summary_stats)
        
    return clean_dataset

def move_models_to_folder(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        if filename.endswith("_model.nc"):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            if os.path.exists(dest_file):
                print(f"Model {filename} already exists in destination. Skipping.")
            else:
                shutil.move(src_file, dest_file)
                print(f"Moved {filename} to {dest_dir}")

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
    results_dir = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'results')
    models_dir = os.path.join(results_dir, 'models')

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Move any existing models from src to models folder
    move_models_to_folder(script_dir, models_dir)

    # Load and preprocess mouse data
    mouse_data_path = os.path.join(data_file_path, 'ibl_trainingChoiceWorld_clean_20241003.csv')
    mouse_data = load_and_preprocess_mouse_data(mouse_data_path)

    columns_to_use = ['subj_idx', 'participant_id', 'rt', 'response', 'signed_contrast', 'prevresp', 'eid', 'session']
    dataset = mouse_data[columns_to_use]

    # Print dataset stats
    dataset = print_dataset_stats(dataset)
    # Filter dataset to only include specified participant IDs
    # participants_to_use = [
    #     5, 11, 12, 18, 21, 22, 25, 29, 30, 31, 36, 38, 46, 47, 52,
    #     54, 57, 58, 60, 62, 65, 67, 68
    # ]
    # dataset = dataset[dataset['participant_id'].isin(participants_to_use)]
    # cav_data = load_data("cavanagh_theta")
    # cav_data['signed_contrast'] = cav_data['theta']
    # cav_data['prevresp'] = cav_data['conf']
    # dataset = cav_data
    
    # Define models to estimate
    model_names = [
        "ddm_nohist", "ddm_prevresp_v", "ddm_prevresp_z", "ddm_prevresp_zv"]
        # "angle_nohist", "angle_prevresp_v", "angle_prevresp_z", "angle_prevresp_zv"]
        # "weibull_nohist", "weibull_prevresp_v", "weibull_prevresp_z", "weibull_prevresp_zv"]

    # model_names = ["ddm_catnohist"]
    # Parameters for sampling
    sampling_params = {"chains": 4, "cores": 4, "draws": 2000, "tune": 2000}
    
    # Run models
    for name in model_names:
        model_file = os.path.join(models_dir, f"{name}_model.nc")
        if os.path.exists(model_file):
            print(f"Model {name} already exists. Skipping.")
        else:
            print(f"Running model: {name}")
            model = run_model(dataset, name, script_dir, **sampling_params)
#            model = run_model(dataset, name, script_dir, sampling_method="vi", **sampling_params)

            print(f"Model {name} completed and saved.")

    # Model comparison
#    combine_model_comparison_csvs(script_dir, model_names)
#    plot_model_comparison(script_dir)

if __name__ == "__main__":
    main()
    
    