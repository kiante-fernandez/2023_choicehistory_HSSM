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
import numpy as np
import pandas as pd
from utils_hssm_modelspec import make_model 
from utils_hssm import run_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jax
import pytensor  # Graph-based tensor library

pytensor.config.floatX = "float32"
jax.config.update("jax_enable_x64", False)

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

def print_dataset_stats(dataset):
    print(f"\nDataset summary:")
    print(f"Number of trials: {len(dataset)}")
    print(f"Number of unique subjects: {dataset['subj_idx'].nunique()}")
    print(f"RT range: {dataset['rt'].min():.3f} to {dataset['rt'].max():.3f} seconds")

    print("\nUnique subject IDs (subj_idx) in final dataset:")
    print(dataset['subj_idx'].unique())

    print("\nNumber of trials per subject in final dataset:")
    print(dataset['subj_idx'].value_counts())

    print("\nMapping between subj_idx and participant_id:")
    print(dataset[['subj_idx', 'participant_id']].drop_duplicates().sort_values('participant_id'))

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

def combine_model_comparison_csvs(script_dir, model_names):
    input_files = [f"{name}_model_comparison.csv" for name in model_names]
    
    def process_csv(file_path, model_name):
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        df['Metric'] = df['Metric'].str.upper()
        return df

    combined_df = pd.concat([process_csv(os.path.join(script_dir, file), model) 
                             for file, model in zip(input_files, model_names)])
    
    result_df = combined_df.pivot(index='Model', columns='Metric', values='Value').reset_index()
    result_df.to_csv(os.path.join(script_dir, 'model_comparison.csv'), index=False)
    print("Combined CSV file 'model_comparison.csv' has been created.")
    
def plot_model_comparison(script_dir):
    # Set the style
    plt.style.use("seaborn-v0_8")
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Color palette
    colors = sns.color_palette("colorblind", n_colors=4)
    comparison_data = pd.read_csv(os.path.join(script_dir, 'model_comparison.csv'))
    # Melt the dataframe to long format
    comparison_data = comparison_data.melt(id_vars='Model', var_name='Metric', value_name='Value')
    # Filter the data for the reference model 'ddm_nohist'
    reference_data = comparison_data[comparison_data['Model'] == 'ddm_nohist']
    # Merge the reference data with the full dataset to calculate the differences
    merged_data = comparison_data.merge(reference_data, on='Metric', suffixes=('', '_ref'))
    # Calculate the difference in value for each metric compared to the reference model
    merged_data['Value_diff'] = merged_data['Value'] - merged_data['Value_ref']
    # Filter out the reference model from the comparison
    comparison_data = merged_data[merged_data['Model'] != merged_data['Model_ref']]

    # Function to calculate confidence intervals
    def calculate_ci(data):
        mean = np.mean(data)
        se = stats.sem(data)
        ci = stats.t.interval(confidence=0.95, df=len(data)-1, loc=mean, scale=se)
        return pd.Series({'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]})

    # Calculate confidence intervals
    ci_data = comparison_data.groupby(['Model', 'Metric'])['Value_diff'].apply(calculate_ci).reset_index()
    ci_data = ci_data.pivot(index=['Model', 'Metric'], columns='level_2', values='Value_diff').reset_index()

    # Plot
    metrics = comparison_data['Metric'].unique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))
    fig.suptitle('Model Comparison: Difference from ddm_nohist', fontsize=16, fontweight='bold', y=1.05)

    # Store bar containers for legend
    bar_containers = []

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = ci_data[ci_data['Metric'] == metric]
        
        bars = ax.bar(metric_data['Model'], metric_data['mean'], yerr=[metric_data['mean'] - metric_data['ci_lower'], 
                      metric_data['ci_upper'] - metric_data['mean']], 
                      capsize=5, color=colors, alpha=0.8)
        
        bar_containers.append(bars)
        
        ax.set_title(f'{metric}', fontsize=14)
        ax.set_ylabel('Difference in Value' if i == 0 else '')
        ax.set_xlabel('Model')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # Remove top and right spines
        sns.despine(ax=ax)
        
        # Adjust y-axis limits to add some padding
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout()

    # Add a legend
    fig.legend(bar_containers[0], ci_data['Model'].unique(), loc='lower center', ncol=len(ci_data['Model'].unique()), 
               bbox_to_anchor=(0.5, -0.1))

    # Save the figure
    plt.savefig(os.path.join(script_dir, 'model_comparison_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
    results_dir = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'results')
    models_dir = os.path.join(results_dir, 'models')

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Move any existing models from src to models folder
#    move_models_to_folder(script_dir, models_dir)

    # Load and preprocess mouse data
    mouse_data_path = os.path.join(data_file_path, 'ibl_trainingChoiceWorld_clean_20241003.csv')
    mouse_data = load_and_preprocess_mouse_data(mouse_data_path)
    columns_to_use = ['subj_idx', 'participant_id', 'rt', 'response', 'signed_contrast', 'prevresp', 'eid']
    dataset = mouse_data[columns_to_use]
    
    # Print dataset stats
    print_dataset_stats(dataset)
    
    # Define models to estimate
    model_names = [
        "ddm_nohist", "ddm_prevresp_v", "ddm_prevresp_z", "ddm_prevresp_zv",
        "angle_nohist", "angle_prevresp_v", "angle_prevresp_z", "angle_prevresp_zv",
        "weibull_nohist", "weibull_prevresp_v", "weibull_prevresp_z", "weibull_prevresp_zv"]

    # Parameters for sampling
    sampling_params = {"chains": 4, "cores": 4, "draws": 2000, "tune": 2000}
    
    # Run models
    for name in model_names:
        model_file = os.path.join(models_dir, f"{name}_model.nc")
        if os.path.exists(model_file):
            print(f"Model {name} already exists. Skipping.")
        else:
            print(f"Running model: {name}")
#            model = run_model(dataset, name, script_dir, **sampling_params)
            model = run_model(dataset, name, script_dir, sampling_method="vi", **sampling_params)

            print(f"Model {name} completed and saved.")

    # Model comparison
    combine_model_comparison_csvs(script_dir, model_names)
#    plot_model_comparison(script_dir)

if __name__ == "__main__":
    main()
    
    