# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2022/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
#

import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from ssms.basic_simulators.simulator import simulator
from hssm.distribution_utils import (
    make_distribution,  # A general function for making Distribution classes
    make_distribution_from_onnx,  # Makes Distribution classes from onnx files
    make_distribution_from_blackbox,  # Makes Distribution classes from callables
)
from hssm.likelihoods import logp_ddm_sdv, DDM
from hssm.utils import download_hf

# Setting float precision in pytensor
pytensor.config.floatX = "float32"

#load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
data = pd.read_csv(os.path.join(data_file_path, 'ibl_trainingchoiceworld_clean.csv'))

# Data preprocessing
#data = data.dropna()
data['rt'] = data.rt_raw
data['subject'] = data.subj_idx
data['response'] = data['response'].replace({0: -1, 1: 1})
data['subject'] = pd.factorize(data['subj_idx'])[0]

n_cats = data['signed_contrast'].nunique()
n_subjects = data['subject'].nunique()

dummy_data = pd.get_dummies(data['signed_contrast'], prefix='signed_contrast')
dataset = pd.concat([data[['rt', 'response', 'subject']], dummy_data], axis=1)
dataset = dataset.apply(pd.to_numeric, errors='coerce')

#See what happens when you build a model directly from PyMC (seem to work okay)

if __name__ == '__main__':
    with pm.Model() as ddm_pymc:
        v = pm.Uniform("v", lower=-10.0, upper=10.0)
        a = pm.HalfNormal("a", sigma=2.0)
        z = pm.Uniform("z", lower=0.01, upper=0.99)
        t = pm.Uniform("t", lower=0.0, upper=0.6, initval=0.1)

        ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt', 'response']])

        nohist_stimcat = pm.sample(chains=1,draws=100,tune=100)
             
    az.summary(nohist_stimcat)
    az.plot_trace(nohist_stimcat);

# draft for the hierarchial DDM w/ factor coded contrast (needs more testing)
# TODO get code ready to run on cluster so you can iterate faster

# Hierarchical DDM model with factor-coded contrast (needs more testing)
if __name__ == '__main__':
    with pm.Model() as ddm_pymc:
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10, shape=n_cats)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=10, shape=n_cats)
        
        # Subject-specific slopes
        slopes = pm.Normal("slopes", mu=mu_beta, sigma=sigma_beta, shape=(n_subjects, n_cats))

        subject = dataset['subject'].values
        diff = dataset.filter(like="signed_contrast_").values
        v = pm.math.sum(slopes[subject] * diff, axis=1)
        v = pm.math.clip(v, -10.0, 10.0)

        a = pm.HalfNormal("a", sigma=2.0)
        z = pm.Uniform("z", lower=0.01, upper=0.99)
        t = pm.Uniform("t", lower=0.0, upper=0.6, initval=0.1)
        
        # Likelihood
        ddm = DDM("ddm", v=v[subject], a=a, z=z, t=t, observed=dataset[['rt', 'response']])
        nohist_stimcat = pm.sample(chains=1, cores=1, draws=100, tune=100)

    az.summary(nohist_stimcat)
    az.plot_trace(nohist_stimcat);

# if __name__ == '__main__':
#     with pm.Model() as ddm_pymc:
#         #prior on drift betas
#         # mu_alpha = pm.Normal("mu_alpha", mu = 0, sigma = 10)
#         # sigma_alpha = pm.HalfNormal("sigma_alpha", mu = 0, sigma = 10)

#         mu_beta = pm.Normal("mu_beta", mu = 0, sigma = 10, shape = n_cats)
#         sigma_beta = pm.HalfNormal("sigma_beta", mu = 0, sigma = 10, shape = n_cats)
        
#         #subject specific
#         #drift_intercept = pm.Normal("intercepts", mu = mu_alpha, sigma = sigma_alpha, shape = n_subjects)
#         drift_slope = pm.Normal("slopes", mu = mu_beta, sigma = sigma_beta, shape = (n_subjects, n_cats))

#         # v = drift_intercept[subject] + drift_slope[subject] * difficulty[subject]
#         subject =  dataset['subject'].values
#         diff = dataset.filter(like="signed_contrast_").values
#         v = 0 + pm.math.sum(drift_slope[subject] * diff, axis = 1)

#         a = pm.HalfNormal("a", sigma=2.0)
#         z = pm.Uniform("z", lower=0.01, upper=0.99)
#         t = pm.Uniform("t", lower=0.0, upper=0.6, initval=0.1)
        
#         #likelihood
#         ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset.values)

#         #nohist_stimcat = pm.sample(chains=2,draws=500,tune=500)
             
#     az.summary(nohist_stimcat)
#     az.plot_trace(nohist_stimcat);
