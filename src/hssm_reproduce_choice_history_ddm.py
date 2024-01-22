# hssm_reproduce_choice_history_ddm.py - a reproduction of models from
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
# Choice history biases subsequent evidence accumulation. eLife
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/17/11      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2024/08/01      Kianté Fernandez<kiantefernan@gmail.com>   added util functions for workflow

# %%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import hssm
import seaborn as sns
import matplotlib.pyplot as plt
import re

from ssms.basic_simulators.simulator import simulator
hssm.set_floatX("float32")

from hssm_modelspec import make_model # specifically for hssm models
#from utils_hssm import run_model, saveInferenceData
from utils_hssm import run_model, dic, aggregate_model_comparisons
#import utils_hssm
#from utils_hssm import saveInferenceData

# %% load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})
def get_prep(data):
    data['prevresp_temp'] = data['prevresp'].map({-1: 0, 1: 1})
    grouped_data = data.groupby('subj_idx')['prevresp_temp'].apply(lambda x: x.value_counts(normalize=True))
    return grouped_data

prep = pd.DataFrame(get_prep(elife_data))
#prep = prep[prep.index.get_level_values(1) == 1]

# Define the number of subjects you want to sample
# num_subjects = 5
# # Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
# subsample = elife_data[elife_data['subj_idx'].isin(elife_data.groupby('subj_idx').size().sample(num_subjects).index)]

# %%
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

ddm_models = {name: make_model(elife_data, name) for name in model_names}

# %% parameter estimation
# Parameters for sampling
sampling_params = {
    "sampler": "nuts_numpyro",
    "chains": 4,
    "cores": 12,
    "draws": 3000,
    "tune": 3000,
    "idata_kwargs": dict(log_likelihood=True)  # return log likelihood
}

# Sample from the posterior for each model
model_run_results = {name: run_model(elife_data, name, script_dir, **sampling_params) for name in model_names}
# %%
# Base directory for model files and figures
model_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/src/"
figure_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/"

# Function to plot and save the model
def plot_and_save_trace_plot(model_file_path, plot_file_path):
    model_data = az.from_netcdf(model_file_path)
    az.style.use("arviz-doc")
    az.plot_trace(model_data)
    plt.savefig(plot_file_path)

# Loop through each model and apply the function
for model_name in model_names:
    # Construct the file paths
    model_file_path = os.path.join(model_dir, model_name + "_model.nc")
    plot_file_path = os.path.join(figure_dir, model_name + "_plot.pdf")

    plot_and_save_trace_plot(model_file_path, plot_file_path)

# %%model comparisons
#%matplotlib inline

aggregate_model_comparisons("/Users/kiante/Documents/2023_choicehistory_HSSM/data")
# load data
comparison_data = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/src/aggregated_model_comparisons.csv")

# Filter the data for the reference model 'ddm_nohist_stimcat'
reference_data = comparison_data[comparison_data['Model'] == 'ddm_nohist_stimcat']
# Merge the reference data with the full dataset to calculate the differences
merged_data = comparison_data.merge(reference_data, on='Metric', suffixes=('', '_ref'))
# Calculate the difference in value for each metric compared to the reference model
merged_data['Value_diff'] =  merged_data['Value'] - merged_data['Value_ref']
# Filter out the reference model from the comparison
comparison_data = merged_data[merged_data['Model'] != merged_data['Model_ref']]

# plot
def annotate_bars(ax, precision="{:.2f}"):
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = precision.format(p.get_height())
        ax.text(_x, _y, value, ha="center") 

metrics = comparison_data['Metric'].unique()       
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)))

for i, metric in enumerate(metrics):
    ax = axes[i]
    #sorted_data = comparison_data[comparison_data['Metric'] == metric].sort_values('Value_diff')
    sns.barplot(x='Model', y='Value_diff', data=comparison_data[comparison_data['Metric'] == metric], ax=ax, palette="deep")
    ax.set_title(f'Difference in {metric} Compared to ddm_nohist_stimcat', fontsize=14, fontweight='bold')
    ax.set_ylabel('Difference in Value',fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    sns.despine(ax=ax) 
    annotate_bars(ax, "{:.2f}")
    plt.gca().tick_params(bottom=True, left=True)
    plt.gca().tick_params(labelbottom=True, labelleft=True)

plt.tight_layout()
plt.show()

# %% 
res_v = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/data/ddm_prevresp_v_results_combined.csv")
res_z = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/data/ddm_prevresp_z_results_combined.csv")

pattern = r'v_prevresp\|subj_idx\[\d+\]'
v_1_subj_idx_specific_rows = res_v[res_v['index'].str.contains(pattern, na=False, regex=True)]
v_1_subj_idx_df = v_1_subj_idx_specific_rows[['index', 'mean']]
v_1_subj_idx_df = v_1_subj_idx_specific_rows[['index', 'mean']].rename(columns={'mean': 'v_mean'})

pattern = r'z_prevresp\|subj_idx\[\d+\]'
z_1_subj_idx_specific_rows = res_z[res_z['index'].str.contains(pattern, na=False, regex=True)]
z_1_subj_idx_df = z_1_subj_idx_specific_rows[['index', 'mean']]
z_1_subj_idx_df = z_1_subj_idx_specific_rows[['index', 'mean']].rename(columns={'mean': 'z_mean'})

# prep = pd.DataFrame(prep).iloc[v_1_subj_idx_specific_rows.iloc[:, 0],:]

# Merging the two dataframes on 'subj_idx'
v_1_subj_idx_df_reset = v_1_subj_idx_df.reset_index(drop=True)
z_1_subj_idx_df_reset = z_1_subj_idx_df.reset_index(drop=True)

prep_reset = prep.reset_index(drop=True)
merged_df = pd.concat([v_1_subj_idx_df_reset,z_1_subj_idx_df_reset, prep_reset], axis=1)
# Renaming columns for clarity
merged_df.rename(columns={'prevresp_temp': 'prevresp'}, inplace=True)

# %% plot it
# %matplotlib inline

# Set the seaborn style to white which has no grid by default and a clear background
sns.set_style("white")
# Create a scatter plot with grey, unfilled circles
sns.scatterplot(
    data=merged_df, 
    x="v_mean", 
    y="prevresp", 
    edgecolor='gray', 
    facecolors='none', 
    s=50,  # size of the markers
    linewidth=1.5  # edge line width
)
# Add thin lines at specific values
plt.axhline(y=0.5, color='black', linestyle='-', linewidth=.5)  # thin horizontal line at y=0.5
plt.axvline(x=0, color='black', linestyle='-', linewidth=.5)  # thin vertical line at x=0
# customization
plt.title('Visual motion 2AFC (FD)', fontsize=14, fontweight='bold')
plt.xlabel('History shift in v_{bias}', fontsize=12, fontweight='bold')
plt.ylabel('P(repeat)', fontsize=12, fontweight='bold')
# Remove the grid
plt.grid(False)
sns.despine()
# Set ticks on the bottom and left axes
plt.gca().tick_params(bottom=True, left=True)
# Set the tick labels to be visible (they are sometimes turned off by default in seaborn's white style)
plt.gca().tick_params(labelbottom=True, labelleft=True)
plt.show()
sns.scatterplot(
    data=merged_df, 
    x="z_mean", 
    y="prevresp", 
    edgecolor='gray', 
    facecolors='none', 
    s=50,  # size of the markers
    linewidth=1.5  # edge line width
)
# Add thin lines at specific values
plt.axhline(y=0.5, color='black', linestyle='-', linewidth=.5)  # thin horizontal line at y=0.5
plt.axvline(x=0, color='black', linestyle='-', linewidth=.5)  # thin vertical line at x=0
# customization
plt.title('Visual motion 2AFC (FD)', fontsize=14, fontweight='bold')
plt.xlabel('History shift in z', fontsize=12, fontweight='bold')
plt.ylabel('P(repeat)', fontsize=12, fontweight='bold')
# Remove the grid
plt.grid(False)
sns.despine()
# Set ticks on the bottom and left axes
plt.gca().tick_params(bottom=True, left=True)
# Set the tick labels to be visible (they are sometimes turned off by default in seaborn's white style)
plt.gca().tick_params(labelbottom=True, labelleft=True)
plt.show()
# %% plot conditional bias function
def conditional_history_plot(df, quantiles):
    import seaborn as sns

    # Binning response times
    df['rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    
    # Compute the normalized value counts for prevresp_temp within each group
    normalized_counts = df.groupby(['subj_idx','rt_bin'])['prevresp_temp'].value_counts(normalize=True)
    normalized_counts = normalized_counts.rename('fraction').reset_index()
    
    # Filter rows where prevresp_temp is 1
    filtered_counts = normalized_counts[normalized_counts['prevresp_temp'] == 1]
    
    # Calculate the mean of each group
    mean_values = df.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    
    # Merge the mean values with the filtered normalized counts
    merged_data = pd.merge(mean_values, filtered_counts, on=['rt_bin', 'subj_idx'], how='left')
    
    # Plotting
    fig, ax = plt.subplots()
    sns.pointplot(x="rt_bin", y="fraction", data=merged_data, join=True, capsize=0.2)
    
    # Set plot details
    ax.set_title('Conditional bias')
    ax.set_xlabel('RT Bin')
    ax.set_ylabel('Choice Bias (fraction)')
    # ax.set_xlim(0, quantiles - 1)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=.5) 
    ax.set_ylim(.45, .60)
    sns.despine(trim=True)
    plt.tight_layout()
    
    return fig

# Define the quantiles and mean_response for the plot
quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
# Generate the plot
fig = conditional_history_plot(elife_data, quantiles)
plt.show()

#%%
# def reattach(filename, model):
#     import arviz as az
#     #load the inferenceData object
#     temp_inferd = az.from_netcdf(filename)
#     #reattch to the model
#     return model

# test_inferd = az.from_netcdf("/Users/kiante/Documents/2023_choicehistory_HSSM/data/ddm_nohist_stimcat_model.nc")
# #reattch to the corresponding HSSM object
# ddm_models["ddm_nohist_stimcat"].trace = test_inferd
# #hssm.HSSM.sample_posterior_predictive(idata=test,data=subsample)
# # %%
# #note once model is rerun you can sample
# ddm_models["ddm_nohist_stimcat"].sample_posterior_predictive(test_inferd)