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

#%matplotlib inline
from ssms.basic_simulators.simulator import simulator
hssm.set_floatX("float32")

from hssm_modelspec import make_model # specifically for hssm models
#from utils_hssm import run_model, saveInferenceData
from utils_hssm import run_model, dic, aggregate_model_comparisons, reattach
#import utils_hssm
#from utils_hssm import saveInferenceData

# %% load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})

# # add some more history measures
elife_data['stimrepeat'] = np.where(elife_data.stimulus == elife_data.prevstim, 1, 0)
elife_data['repeat'] = np.where(elife_data.response == elife_data.prevresp, 1, 0)

excluded_participants = [11, 19, 20, 22, 26, 27, 28] #whose subject level fits did not converge
elife_data = elife_data[~elife_data['participant_id'].isin(excluded_participants)]

def get_prep(data):
     # grouped_data = data.groupby(['subj_idx'])['stimrepeat','repeat'].apply(lambda x: x.value_counts(normalize=True))
    grouped_data = data.groupby(['subj_idx'])[['stimrepeat','repeat']].mean().reset_index()
    return grouped_data


prep = pd.DataFrame(get_prep(elife_data))
#prep = prep[prep.index.get_level_values(1) == 1]

# Define the number of subjects you want to sample
num_subjects = 5
# # Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
subsample = elife_data[elife_data['subj_idx'].isin(elife_data.groupby('subj_idx').size().sample(num_subjects).index)]

# %%
# Define models
model_names = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv"
]

# model_names = [
#     "full_ddm_nohist"
# ]

#ddm_models = {name: make_model(elife_data, name) for name in model_names}

# %% parameter estimation
# Parameters for sampling
sampling_params = {
    "sampler": "nuts_numpyro",
    "chains": 10,
    "cores": 10,
    "draws": 2000,
    "tune": 1500,
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

aggregate_model_comparisons("/Users/kiante/Documents/2023_choicehistory_HSSM/src")
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
#plt.savefig(os.path.join(figure_dir, "model_comparisons_v2.png"), dpi=300)

# %% 
res_v = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_prevresp_v_results_combined.csv")
res_z = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_prevresp_z_results_combined.csv")

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
merged_df.rename(columns={'repeat': 'repetition'}, inplace=True)

# %% plot it

# Set the seaborn style to white which has no grid by default and a clear background
sns.set_style("white")
# Create a scatter plot with grey, unfilled circles
sns.scatterplot(
    data=merged_df, 
    x="v_mean", 
    y="repetition", 
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
# plt.savefig(os.path.join(figure_dir, "fig4A_vbias_v2.png"), dpi=300)

sns.scatterplot(
    data=merged_df, 
    x="z_mean", 
    y="repetition", 
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
# plt.savefig(os.path.join(figure_dir, "fig4A_z_v2.png"), dpi=300)

# %% plot conditional bias function
df = elife_data
def conditional_history_plot(df, quantiles):
    import seaborn as sns

    # Binning response times
    df['rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    
    rep_data = df.groupby(['rt_bin','subj_idx'])[['repeat']].mean().reset_index()
    rep_data['repetition'] = rep_data.repeat

    # Calculate the mean of each group
    # mean_values = df.groupby(['rt_bin', 'subj_idx']).mean().reset_index()
    
    # Merge the mean values with the filtered normalized counts
    # merged_data = pd.merge(mean_values, rep_data, on=['rt_bin', 'subj_idx'], how='left')
    
    # Plotting
    fig, ax = plt.subplots()
    sns.pointplot(x="rt_bin", y="repetition", data=rep_data, join=True, capsize=0.2)
    
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

#%% Posterior predictives on psychometric/chronometric functions

# reattch to the corresponding HSSM object
no_hist = reattach("/Users/kiante/Documents/2023_choicehistory_HSSM/data/ddm_nohist_model.nc", "ddm_nohist", elife_data)
no_hist.traces
no_hist.sample_posterior_predictive(data=elife_data, n_samples = 10, include_group_specific = True)
# #note once model is rerun you can sample
hssm.hssm.plotting.plot_posterior_predictive(model= no_hist)
hssm.hssm.plotting.plot_quantile_probability(model= no_hist, cond="stimulus")

plotting_df = hssm.hssm.plotting.posterior_predictive._get_plotting_df(idata=no_hist.traces, data=elife_data,extra_dims=["subj_idx","prevresp"])

def get_prep(data):
    data['prevresp_temp'] = data['prevresp'].map({-1: 0, 1: 1})
    grouped_data = data.groupby(['observed','subj_idx'])['prevresp_temp'].apply(lambda x: x.value_counts(normalize=True))
    return grouped_data

get_prep(plotting_df)

# %% ================================= #
# QP plots 
# ================================= #
fig = hssm.plotting.quantile_probability(ddm_models['ddm_nohist'], cond="signed_contrast")
fig.set_ylim(0, 3);
# %%

#quantile plots 
#> just with single condition correct vs error...
#> Alex would like to see a PR that looks that the qp 
#> he would also like a test (?) that compares to the plots made in R

# df_subj = elife_data.groupby(['subj_idx'])[['stimrepeat','repeat']].mean().reset_index()
# df_subj['repetition'] = df_subj.repeat
# df_subj_melt = df_subj.melt(id_vars=['subj_idx', 'repetition'])
# df_subj_melt['x'] = np.where(df_subj_melt.variable == 'repeat', 1, 0) + df_subj_melt.repetition

# plt.close('all')
# fig, ax = plt.subplots(figsize=(6,8))

# sns.lineplot(data=df_subj_melt, x='x', y='value', units='subj_idx', hue='repetition',
#               estimator=None, legend=False, palette='PuOr', hue_norm=(0.4,0.6),#palette='ch:s=.25,rot=-.25',
#               ax=ax, dashes=False)
# ax.set(yticks=[0.4, 0.5, 0.6], xticks=[0.5, 1.5],
#           ylabel='Repetition probability', xlabel='')
# ax.set_xticklabels(['stimuli', 'choices'], rotation=-45)
# sns.despine(trim=True)
# plt.tight_layout()
#TODO see code Anne sent.
# https://github.com/anne-urai/2022_Urai_choicehistory_MEG/blob/main/behavior_plots.py#L28
# df_subj = df.groupby(['subj_idx'])['stimrepeat', 'repeat', 'group'].mean().reset_index()
# df_subj['repetition'] = df_subj.repeat
# df_subj_melt = df_subj.melt(id_vars=['subj_idx', 'repetition', 'group'])
# df_subj_melt['x'] = np.where(df_subj_melt.variable == 'repeat', 1, 0) + df_subj_melt.repetition