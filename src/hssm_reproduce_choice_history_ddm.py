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

from utils_hssm_modelspec import make_model 
from utils_hssm import run_model, reattach

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
# %% specify HSSM models
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
# %% run parameter estimation
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
# %% plotting here we would use Z's plotting code
# Base directory for model files and figures
results_dir = os.path.join(script_dir, 'results', 'figures')





# %% plot other things not yet included in Z's code: conditional bias function

## Conditional bia function 
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

## Quantile probability plot 
fig = hssm.plotting.quantile_probability(ddm_models['ddm_nohist'], cond="signed_contrast")
fig.set_ylim(0, 3);
plt.show()

#%% Posterior predictives checks on models

# reattch to the corresponding HSSM object
prevresp_zv = reattach(os.path.join(data_file_path, 'ddm_prevresp_zv.nc'), "ddm_prevresp_zv", elife_data)
prevresp_zv.traces
prevresp_zv.sample_posterior_predictive(data=elife_data, n_samples = 10, include_group_specific = True)
# #note once model is rerun you can sample
hssm.hssm.plotting.plot_posterior_predictive(model= prevresp_zv)

plotting_df = hssm.hssm.plotting.posterior_predictive._get_plotting_df(idata=prevresp_zv.traces, data=elife_data,extra_dims=["subj_idx","prevresp"])

#compute choice prevresp on the simulated data from PPC
