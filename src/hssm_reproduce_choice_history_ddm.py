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

# %%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import hssm
import seaborn as sns
import matplotlib.pyplot as plt
import re

from src.utils_hssm_modelspec import make_model # specifically for hssm models
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

elife_data['participant_id'] = elife_data['subj_idx']
excluded_participants = [11, 19, 20, 22, 26, 27, 28] #whose subject level fits did not converge

elife_data = elife_data[~elife_data['participant_id'].isin(excluded_participants)]

def get_prep(data):
    grouped_data = data.groupby(['subj_idx'])[['stimrepeat','repeat']].mean().reset_index()
    return grouped_data

prep = pd.DataFrame(get_prep(elife_data))

# Define the number of subjects you want to sample
# num_subjects = 5
# # # Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
# subsample = elife_data[elife_data['subj_idx'].isin(elife_data.groupby('subj_idx').size().sample(num_subjects).index)]

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

ddm_models = {name: make_model(elife_data, name) for name in model_names}

# %% parameter estimation
# Parameters for sampling
sampling_params = {
    "sampler": "nuts_numpyro",
    "chains": 6,
    "cores": 6,
    "draws": 3000,
    "tune": 3000,
    "idata_kwargs": dict(log_likelihood=True)  # return log likelihood
}

# Sample from the posterior for each model
model_run_results = {name: run_model(elife_data, name, script_dir, **sampling_params) for name in model_names}
# %% here we would use Z's plotting code
# Base directory for model files and figures
results_dir = os.path.join(script_dir, 'results', 'figures')

# %% plot it
# %% plot conditional bias function
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

#compute choice prevresp on the simulated data from PPC

# %% # Quantile probability plots 
fig = hssm.plotting.quantile_probability(ddm_models['ddm_nohist'], cond="signed_contrast")
fig.set_ylim(0, 3);
