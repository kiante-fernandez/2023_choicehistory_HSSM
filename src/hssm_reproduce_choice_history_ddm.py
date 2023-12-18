# hssm_reproduce_choice_history_ddm.py - a reproduction of models from
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
# Choice history biases subsequent evidence accumulation. eLife
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2022/17/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
#
# %%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import hssm

hssm.set_floatX("float32")

from hssm_modelspec import make_model # specifically for hssm models

# %% load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))
elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})

# Define the number of subjects you want to sample
num_subjects = 5
# Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
subsample = elife_data[elife_data['subj_idx'].isin(elife_data.groupby('subj_idx').size().sample(num_subjects).index)]

# %%
# Define models
model_names = [
    "ddm_nohist_stimcat", 
    "ddm_nohist_stimcat_dummycode", 
    "ddm_nohist_stimcat_reducedrankcode", 
    "ddm_prevresp_v", 
    "ddm_prevresp_z", 
    "ddm_stimcat_prevresp_zv", 
    "ddm_prevresp_zv"
]

ddm_models = {name: make_model(subsample, name) for name in model_names}

# %%
# Parameters for sampling
sampling_params = {
    "chains": 1,
    "cores": 1,
    "draws": 100,
    "tune": 100
}
# Sample from the posterior for each model
model_results = {name: model.sample(**sampling_params) for name, model in models.items()}
# model_res1 = nohist_stimcat.sample(chains=1, cores=1, draws=100, tune=100)

# %%
# az.summary(model_res1)
# az.plot_trace(model_res1);
