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

from ssms.basic_simulators.simulator import simulator
hssm.set_floatX("float32")

from hssm_modelspec import make_model # specifically for hssm models
#from utils_hssm import run_model, saveInferenceData
from utils_hssm import run_model, dic
#import utils_hssm
#from utils_hssm import saveInferenceData

# %% load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})

# Define the number of subjects you want to sample
num_subjects = 5
# Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
subsample = elife_data[elife_data['subj_idx'].isin(elife_data.groupby('subj_idx').size().sample(num_subjects).index)]

# %%
# Define models
# model_names = [
#     "ddm_nohist_stimcat", 
#     "ddm_prevresp_v", 
#     "ddm_prevresp_z", 
#     "ddm_stimcat_prevresp_zv", 
#     "ddm_prevresp_zv"
# ]
model_names = [
    "ddm_nohist_stimcat", 
]

#ddm_models = {name: make_model(subsample, name) for name in model_names}

# %% parameter estimation
# Parameters for sampling
sampling_params = {
    "sampler": "nuts_numpyro",
    "chains": 4,
    "cores": 4,
    "draws": 100,
    "tune": 0,
    "idata_kwargs": dict(log_likelihood=True)  # return log likelihood
}

# Sample from the posterior for each model
model_run_results = {name: run_model(subsample, name, script_dir, **sampling_params) for name in model_names}
# %%
# az.summary(model_res1)
# az.plot_trace(model_res1);
