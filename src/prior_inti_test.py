# %%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from ssms.basic_simulators.simulator import simulator
import glob

from hssm.distribution_utils import (
    make_distribution,  # A general function for making Distribution classes
)

# pm.Distributions that represents the top-level distribution for
# DDM models (the Wiener First-Passage Time distribution)
from hssm.likelihoods import logp_ddm_sdv, DDM
from hssm.utils import download_hf
from hssm import set_floatX
from hssm import load_data
from utils_hssm import  dic, aggregate_model_comparisons

#%matplotlib inline
set_floatX("float32")

# %% load data
script_dir = os.path.dirname(os.path.realpath(__file__))

data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
# mouse_data = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', 'ibl_trainingchoiceworld_clean.csv')
# mouse_data = pd.read_csv(mouse_data)
# mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
# mouse_data['rt'] = mouse_data['rt'].round(5)

elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})
# elife_data['response'] = elife_data['response'].astype(float)

elife_data['rt'] = elife_data['rt'].round(5)
# # add some more history measures
elife_data['stimrepeat'] = np.where(elife_data.stimulus == elife_data.prevstim, 1, 0)
elife_data['repeat'] = np.where(elife_data.response == elife_data.prevresp, 1, 0)

elife_data['participant_id'] = elife_data['subj_idx']

participants = elife_data['participant_id'].unique()
# num_subjects = 20
# #first 10 are fine. 
# # selected_participants = participants[:num_subjects]
# selected_participants = participants[26:32]
# subsample = elife_data[elife_data['participant_id'].isin(selected_participants)]
# subsample = subsample[['rt','response','signed_contrast','prevresp','participant_id','subj_idx']]

# %%
dataset = elife_data 
excluded_participants = [11, 19, 20, 22, 26, 27, 28] #we want diff combos
dataset = dataset[~dataset['participant_id'].isin(excluded_participants)]
# dataset = mouse_data 

n_subjects = len(dataset['participant_id'].unique())
participant_id, unique_participants = pd.factorize(dataset['subj_idx'])
signed_contrast = dataset['signed_contrast'].values
# %%

with pm.Model() as no_hist_pymc:
    sigma_intercept_v = pm.Weibull("v_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_signed_contrast_v = pm.Weibull("v_signed_contrast|participant_id_sigma", alpha = 1, beta = 0.2)
    
    sigma_intercept_z = pm.Weibull("z_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_a = pm.Weibull("a_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_t = pm.Weibull("t_1|participant_id_sigma", alpha = 1.5, beta = 0.2)

    # Hierarchical
    v_signed_contrast_group = pm.Normal("v_signed_contrast", mu=0, sigma=2)
    v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=3)
    gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10)
    z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
    a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=1)
    # t_Intercept_group = pm.Gamma("t_Intercept",mu=0.4, sigma=2)
    t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2)


    v_intercept_prior = pm.Normal("v_1|participant_id_offset", mu=0, sigma=sigma_intercept_v, shape=n_subjects)
    v_signed_contrast_prior = pm.Normal("v_signed_contrast|participant_id_offset", mu=0, sigma=sigma_signed_contrast_v, shape=n_subjects)
    z_intercept_prior = pm.Normal("z_1|participant_id_offset", mu=0, sigma=sigma_intercept_z, shape=n_subjects)
    a_intercept_prior = pm.Normal("a_1|participant_id_offset", mu=0, sigma=sigma_intercept_a, shape=n_subjects)
    t_intercept_prior = pm.Normal("t_1|participant_id_offset", mu=0, sigma=sigma_intercept_t, shape=n_subjects)

    # Linear combinations
    v = (v_Intercept_group + v_intercept_prior[participant_id]) + ((v_signed_contrast_group + v_signed_contrast_prior[participant_id])*signed_contrast)
    z = z_Intercept_group + z_intercept_prior[participant_id]
    a = a_Intercept_group + a_intercept_prior[participant_id]
    t = t_Intercept_group + t_intercept_prior[participant_id]

    # Drift Diffusion Model as the likelihood function
    ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    # Sampling
    no_hist_pymc_trace = pm.sample(init="adapt_diag",
                               cores=4,
                               chains=4,
                               draws=1000,
                               tune=1000,
                               discard_tuned_samples = True,
                               idata_kwargs=dict(log_likelihood=True))
    # no_hist_pymc_trace = pm.sample(nuts_sampler="numpyro", 
    #                            cores=8,
    #                            chains=8,
    #                            draws=8000,
    #                            tune=8000,
    #                            idata_kwargs=dict(log_likelihood=False))


print("saving summary stats")

results =  az.summary(no_hist_pymc_trace).reset_index()  # point estimate for each parameter and subject
results.to_csv('m1_test_results_v.csv')
# %% HSSM version
import hssm 

param_v = hssm.Param(
    "v",
    formula="v ~ 1 + signed_contrast + (1 + signed_contrast|participant_id)",
    link="identity",
    prior={
        "Intercept": hssm.Prior("Normal", mu=0.0, sigma=3),
        "1|participant_id": hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
        ),
        "signed_contrast": hssm.Prior("Normal", mu=0.0, sigma=2),
        "signed_contrast|participant_id": hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("Weibull",alpha = 1, beta = 0.2), 
        ),
    },
)

param_z = hssm.Param(
    "z",
    formula="z ~ 1 + (1|participant_id)",
    link="identity",
    prior={
        "Intercept": hssm.Prior("Gamma", mu=10, sigma=10, bounds=(0.01,.99)),
        "1|participant_id": hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
        )
    },
    bounds= (0.01,.99)
)

param_a = hssm.Param(
    "a",
    formula="a ~ 1 + (1|participant_id)",
    link="identity",
    prior={
        "Intercept": hssm.Prior("Gamma", mu=1.5, sigma=1),
        "1|participant_id": hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
        )
    },
    bounds=(0.01, np.inf)
)

param_t = hssm.Param(
    "t",
    formula="t ~ 1 + (1|participant_id)",
    link="identity",
    prior={
        "Intercept": hssm.Prior("Weibull", alpha=1, beta=0.2),
        "1|participant_id": hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("Weibull", alpha = 1.5, beta = 0.3), 
        )
    },
    bounds=(0.100, np.inf)
)

model_specs = [param_v, param_z, param_a, param_t]

hssm_model = hssm.HSSM(data = dataset,
                       model= "ddm",
                       include=model_specs,
                       loglik_kind="analytical")

# hssm_model.model.plot_priors()

hssm_model.sample(sampler="nuts_numpyro",
                  cores=4,
                  chains=4,
                  draws=3000,
                  tune=3000,
                  idata_kwargs=dict(log_likelihood=True)
    
)

# %%
prevresp = dataset['prevresp'].values

with pm.Model() as prevresp_v_pymc:
    sigma_intercept_v = pm.Weibull("v_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_signed_contrast_v = pm.Weibull("v_signed_contrast|participant_id_sigma", alpha = 1, beta = 0.2)
    sigma_prevresp_v = pm.Weibull("v_prevresp|participant_id_sigma", alpha = 1, beta = 0.2)
    
    sigma_intercept_z = pm.Weibull("z_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_a = pm.Weibull("a_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_t = pm.Weibull("t_1|participant_id_sigma", alpha = 1.5, beta = 0.3)

    # Hierarchical
    v_signed_contrast_group = pm.Normal("v_signed_contrast", mu=0, sigma=2)
    v_prevresp_group = pm.Normal("v_prevresp", mu=0, sigma=2)

    v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=3)
    gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10)
    z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
    a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=0.75)
    t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2)

    v_intercept_prior = pm.Normal("v_1|participant_id_offset", mu=0, sigma=sigma_intercept_v, shape=n_subjects)
    v_signed_contrast_prior = pm.Normal("v_signed_contrast|participant_id_offset", mu=0, sigma=sigma_signed_contrast_v, shape=n_subjects)
    v_prevresp_prior = pm.Normal("v_prevresp|participant_id_offset", mu=0, sigma=sigma_prevresp_v, shape=n_subjects)

    z_intercept_prior = pm.Normal("z_1|participant_id_offset", mu=0, sigma=sigma_intercept_z, shape=n_subjects)
    a_intercept_prior = pm.Normal("a_1|participant_id_offset", mu=0, sigma=sigma_intercept_a, shape=n_subjects)
    t_intercept_prior = pm.Normal("t_1|participant_id_offset", mu=0, sigma=sigma_intercept_t, shape=n_subjects)

    # Linear combinations
    v = (v_Intercept_group + v_intercept_prior[participant_id]) + ((v_signed_contrast_group + v_signed_contrast_prior[participant_id])*signed_contrast) + ((v_prevresp_group + v_prevresp_prior[participant_id])*prevresp)
    z = z_Intercept_group + z_intercept_prior[participant_id]
    a = a_Intercept_group + a_intercept_prior[participant_id]
    t = t_Intercept_group + t_intercept_prior[participant_id]
    
    # Drift Diffusion Model as the likelihood function
    ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    # Sampling
    prevresp_v_pymc_trace = pm.sample(init="adapt_diag",
                               cores=4,
                               chains=4,
                               draws=1000,
                               tune=1000,
                               idata_kwargs=dict(log_likelihood=True))
    
results =  az.summary(prevresp_v_pymc_trace).reset_index()  # point estimate for each parameter and subject
results.to_csv('m2_test_results_combined.csv')
#%%

with pm.Model() as prevresp_z_pymc:
    sigma_intercept_v = pm.Weibull("v_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_signed_contrast_v = pm.Weibull("v_signed_contrast|participant_id_sigma", alpha = 1, beta = 0.2)
    
    sigma_intercept_z = pm.Weibull("z_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_prevresp_z = pm.Weibull("z_prevresp|participant_id_sigma", alpha = 1, beta = 0.2)

    sigma_intercept_a = pm.Weibull("a_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_t = pm.Weibull("t_1|participant_id_sigma", alpha = 1.5, beta = 0.3)

    # Hierarchical
    v_signed_contrast_group = pm.Normal("v_signed_contrast", mu=0, sigma=2)

    v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=3)
    gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10)
    z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
    z_prevresp_group = pm.Normal("z_prevresp", mu=0, sigma=2)
    
    a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=0.75)
    t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2)

    v_intercept_prior = pm.Normal("v_1|participant_id_offset", mu=0, sigma=sigma_intercept_v, shape=n_subjects)
    v_signed_contrast_prior = pm.Normal("v_signed_contrast|participant_id_offset", mu=0, sigma=sigma_signed_contrast_v, shape=n_subjects)

    z_intercept_prior = pm.Normal("z_1|participant_id_offset", mu=0, sigma=sigma_intercept_z, shape=n_subjects)
    z_prevresp_prior = pm.Normal("z_prevresp|participant_id_offset", mu=0, sigma=sigma_prevresp_z, shape=n_subjects)

    a_intercept_prior = pm.Normal("a_1|participant_id_offset", mu=0, sigma=sigma_intercept_a, shape=n_subjects)
    t_intercept_prior = pm.Normal("t_1|participant_id_offset", mu=0, sigma=sigma_intercept_t, shape=n_subjects)

    # Linear combinations
    v = (v_Intercept_group + v_intercept_prior[participant_id]) + ((v_signed_contrast_group + v_signed_contrast_prior[participant_id])*signed_contrast)
    z = z_Intercept_group + z_intercept_prior[participant_id] + ((z_prevresp_group + z_prevresp_prior[participant_id])*prevresp)
    a = a_Intercept_group + a_intercept_prior[participant_id]
    t = t_Intercept_group + t_intercept_prior[participant_id]
    
    # Drift Diffusion Model as the likelihood function
    ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    # Sampling
    prevresp_z_pymc_trace = pm.sample(init="adapt_diag",
                               cores=4,
                               chains=4,
                               draws=1000,
                               tune=1000,
                               idata_kwargs=dict(log_likelihood=True))
    
results =  az.summary(prevresp_z_pymc_trace).reset_index()  # point estimate for each parameter and subject
results.to_csv('m3_test_results_combined.csv')

#%%

with pm.Model() as prevresp_zv_pymc:
    sigma_intercept_v = pm.Weibull("v_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_signed_contrast_v = pm.Weibull("v_signed_contrast|participant_id_sigma", alpha = 1, beta = 0.2)
    sigma_prevresp_v = pm.Weibull("v_prevresp|participant_id_sigma", alpha = 1, beta = 0.2)
    
    sigma_intercept_z = pm.Weibull("z_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_prevresp_z = pm.Weibull("z_prevresp|participant_id_sigma", alpha = 1, beta = 0.2)

    sigma_intercept_a = pm.Weibull("a_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
    sigma_intercept_t = pm.Weibull("t_1|participant_id_sigma", alpha = 1.5, beta = 0.3)

    # Hierarchical
    v_signed_contrast_group = pm.Normal("v_signed_contrast", mu=0, sigma=2)
    v_prevresp_group = pm.Normal("v_prevresp", mu=0, sigma=2)
    v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=3)
    
    gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10)
    z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
    z_prevresp_group = pm.Normal("z_prevresp", mu=0, sigma=2)

    a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=0.75)
    t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2)

    v_intercept_prior = pm.Normal("v_1|participant_id_offset", mu=0, sigma=sigma_intercept_v, shape=n_subjects)
    v_signed_contrast_prior = pm.Normal("v_signed_contrast|participant_id_offset", mu=0, sigma=sigma_signed_contrast_v, shape=n_subjects)
    v_prevresp_prior = pm.Normal("v_prevresp|participant_id_offset", mu=0, sigma=sigma_prevresp_v, shape=n_subjects)

    z_intercept_prior = pm.Normal("z_1|participant_id_offset", mu=0, sigma=sigma_intercept_z, shape=n_subjects)
    z_prevresp_prior = pm.Normal("z_prevresp|participant_id_offset", mu=0, sigma=sigma_prevresp_z, shape=n_subjects)

    a_intercept_prior = pm.Normal("a_1|participant_id_offset", mu=0, sigma=sigma_intercept_a, shape=n_subjects)
    t_intercept_prior = pm.Normal("t_1|participant_id_offset", mu=0, sigma=sigma_intercept_t, shape=n_subjects)

    # Linear combinations
    v = (v_Intercept_group + v_intercept_prior[participant_id]) + ((v_signed_contrast_group + v_signed_contrast_prior[participant_id])*signed_contrast) + ((v_prevresp_group + v_prevresp_prior[participant_id])*prevresp)
    z = z_Intercept_group + z_intercept_prior[participant_id] + ((z_prevresp_group + z_prevresp_prior[participant_id])*prevresp)
    a = a_Intercept_group + a_intercept_prior[participant_id]
    t = t_Intercept_group + t_intercept_prior[participant_id]
    
    # Drift Diffusion Model as the likelihood function
    ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    # Sampling
    prevresp_zv_pymc_trace = pm.sample(init="adapt_diag",
                               cores=4,
                               chains=4,
                               draws=1000,
                               tune=1000,
                               idata_kwargs=dict(log_likelihood=True))
    
results =  az.summary(prevresp_zv_pymc_trace).reset_index()  # point estimate for each parameter and subject
#results.to_csv('m4_test_results_combined.csv')
results.to_csv('m3_test_results_combined.csv')
prevresp_zv_pymc_trace.to_netcdf("prevresp_zv_pymc_traces.nc")
#%%
models = [no_hist_pymc_trace, prevresp_v_pymc_trace, prevresp_z_pymc_trace, prevresp_zv_pymc_trace]  # replace with your actual model traces

results = []

for i, model_trace in enumerate(models):
    dics = dic(model_trace)['ddm'].values.item()
    waic = az.waic(model_trace).elpd_waic
    loo = az.loo(model_trace).elpd_loo
    results.append({'Model': f'Model{i+1}','DIC':dics, 'WAIC': waic, 'LOO': loo})

df2 = pd.DataFrame(results)
df2.to_csv('model_comparison.csv', index=False)

df2.to_csv('model_comparison_run.csv', index=False)

#%%
#Do not need to run! RN 
# dataset =  elife_data
# excluded_participants = [11, 19, 20, 22, 26, 27, 28]
# dataset = dataset[~dataset['participant_id'].isin(excluded_participants)]

# # selected_participants = participants[:num_subjects]
# selected_participants = participants[11, 19, 20]
# subsample = elife_data[elife_data['participant_id'].isin(selected_participants)]
# subsample = subsample[['rt','response','signed_contrast','prevresp','participant_id','subj_idx']]

# n_subjects = len(dataset['participant_id'].unique())
# signed_contrast = dataset['signed_contrast'].values
# participant_id, unique_participants = pd.factorize(dataset['subj_idx'])

# with pm.Model() as no_hist_pymc:
#     # sigma_intercept_v = pm.Weibull("v_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
#     # sigma_signed_contrast_v = pm.Weibull("v_signed_contrast|participant_id_sigma", alpha = 1.5, beta = 0.3)
    
#     # sigma_intercept_z = pm.Weibull("z_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
#     # sigma_intercept_a = pm.Weibull("a_1|participant_id_sigma", alpha = 1.5, beta = 0.3)
#     # sigma_intercept_t = pm.Weibull("t_1|participant_id_sigma", alpha = 1.5, beta = 0.3)

#     # Hierarchical
#     v_signed_contrast_group = pm.Normal("v_signed_contrast", mu=0, sigma=2, shape = n_subjects)
#     v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=1,shape = n_subjects)
#     gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10,shape = n_subjects)
#     z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
#     a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=0.5, shape = n_subjects)
#     # t_Intercept_group = pm.Gamma("t_Intercept",mu=0.4, sigma=2)
#     t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2, shape = n_subjects)


#     # v_intercept_prior = pm.Normal("v_1|participant_id_offset", mu=0, sigma=1, shape=n_subjects)
#     # v_signed_contrast_prior = pm.Normal("v_signed_contrast|participant_id_offset", mu=0, sigma=1, shape=n_subjects)
#     # z_intercept_prior = pm.Normal("z_1|participant_id_offset", mu=0, sigma=1, shape=n_subjects)
#     # a_intercept_prior = pm.Normal("a_1|participant_id_offset", mu=0, sigma=1, shape=n_subjects)
#     # t_intercept_prior = pm.Normal("t_1|participant_id_offset", mu=0, sigma=1, shape=n_subjects)

#     # Linear combinations
#     v = v_Intercept_group[participant_id] + v_signed_contrast_group[participant_id] * signed_contrast
#     z = z_Intercept_group[participant_id]
#     a = a_Intercept_group[participant_id]
#     t = t_Intercept_group[participant_id] 

#     # Drift Diffusion Model as the likelihood function
#     ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

#     # Sampling
#     no_hist_pymc_trace = pm.sample(init="adapt_diag",
#                                cores=4,
#                                chains=4,
#                                draws=1000,
#                                tune=1000,
#                                discard_tuned_samples = True,
#                                idata_kwargs=dict(log_likelihood=True))
    
# %% singel suibject tests
# from matplotlib import pyplot as plt
# az.plot_trace(no_hist_pymc_trace)
# plt.tight_layout()
# results =  az.summary(no_hist_pymc_trace).reset_index()  # point estimate for each parameter and subject

# %%
# n_coherence = len(dataset['signed_contrast'].unique())
# signed_contrast_id, unique_contrast = pd.factorize(dataset['signed_contrast'])

# with pm.Model() as no_hist_stim_pymc:

#     v_Intercept_group = pm.Normal("v_Intercept",mu=0, sigma=1,shape = n_coherence)
#     gamma_dist_z = pm.Gamma.dist(mu= 10, sigma= 10)
#     z_Intercept_group =  pm.Truncated("z_Intercept",gamma_dist_z, lower= 0, upper= 1)
#     a_Intercept_group = pm.Gamma("a_Intercept",mu=1.5, sigma=0.5)
#     t_Intercept_group = pm.Weibull("t_Intercept",alpha=1, beta=0.2)

#     # Linear combinations
#     v = v_Intercept_group[signed_contrast_id]
#     z = z_Intercept_group
#     a = a_Intercept_group
#     t = t_Intercept_group

#     # Drift Diffusion Model as the likelihood function
#     ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

#     # Sampling
#     no_hist_stim_pymc_trace = pm.sample(init="adapt_diag",
#                                cores=4,
#                                chains=4,
#                                draws=1000,
#                                tune=1000,
#                                idata_kwargs=dict(log_likelihood=False))
    
# %%
from matplotlib import pyplot as plt

az.plot_trace(no_hist_stim_pymc_trace)
plt.tight_layout()
plt.savefig("file_name.png", dpi=300)
az.plot_trace(no_hist_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_v_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_z_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_zv_pymc_trace)
plt.tight_layout()
# results =  az.summary(no_hist_stim_pymc_trace).reset_index()  # point estimate for each parameter and subject

# %% quick model compare.
import seaborn as sns
%matplotlib inline

model_names = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv"
]

# load data
comparison_data = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/src/model_comparison.csv")
comparison_data = comparison_data.melt(id_vars='Model', value_vars=['DIC','WAIC', 'LOO'], var_name='Metric', value_name='Value')
# Create a dictionary mapping the old model names to the new ones
model_name_mapping = {f'Model{i+1}': name for i, name in enumerate(model_names)}
# Replace the model names
comparison_data['Model'] = comparison_data['Model'].replace(model_name_mapping)

# Filter the data for the reference model 'ddm_nohist_stimcat'
reference_data = comparison_data[comparison_data['Model'] == 'ddm_nohist']
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
    ax.set_title(f'Difference in {metric} Compared to ddm_nohist', fontsize=14, fontweight='bold')
    ax.set_ylabel('Difference in Value',fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    sns.despine(ax=ax) 
    annotate_bars(ax, "{:.2f}")
    plt.gca().tick_params(bottom=True, left=True)
    plt.gca().tick_params(labelbottom=True, labelleft=True)

plt.tight_layout()
plt.show()

# %% Get unique participant IDs
import hssm 

participants = elife_data['participant_id'].unique()

# Initialize a dictionary to store summaries
summaries = {}

# Loop over each participant
for participant in participants:
    # Subset data for the current participant
    subj_dataset = elife_data[elife_data['participant_id'] == participant]

    # Define model
    sub_hssm_model = hssm.HSSM(data=subj_dataset,
                               model="ddm",
                            #    model="full_ddm",
                               include=[
        {
            "name": "v",
            "formula":"v ~ 1 + signed_contrast + prevresp"
        },
        {
            "name": "z",
            "formula":"z ~ 1 + prevresp",
        }
    ],
                                # loglik_kind="blackbox",
                                # sv = 0, sz = 0, st = 0)      
                            #    loglik_kind="analytical")
                               loglik_kind="approx_differentiable")

    # Estimate model
    subj_sample_res = sub_hssm_model.sample(
        sampler="nuts_numpyro",
        # sampler="mcmc",
        cores=4,
        chains=4,
        draws=1000,
        tune=1000,
        # step = pm.Slice(model=sub_hssm_model.pymc_model)
    )
    # Create summary and store it in the dictionary
    summaries[participant] = az.summary(subj_sample_res)

for participants, dataframe in summaries.items():
    # filename = f'HSSM_M4_fit_{participants}_MCMC.csv'
    filename = f'HSSM_M4_fit_{participants}_nuts_numpyro_approx_differentiable.csv'
    # filename = f'HSSM_M4_fit_{participants}_MCMC_analytical.csv'
    # filename = f'HSSM_M4_fit_{participants}_MCMC_SLICE.csv'

    dataframe.to_csv(filename, index=True)
    print(f'Saved data for participant {participants} as {filename}')

# %%

def load_files(base_path):
    dataframes_mcmc = {}
    dataframes_slice = {}
    dataframes_nuts_analytical = {}
    dataframes_approx_differentiable = {}

    # Find all MCMC and SLICE files
    mcmc_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC.csv'))
    slice_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC_SLICE.csv'))
    analytical_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC_analytical.csv'))
    approx_differentiable_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_nuts_numpyro_approx_differentiable.csv'))

    # Load SLICE files and add participant_id, retain index
    for file in slice_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_slice = pd.read_csv(file, index_col=0)
        df_slice['participant_id'] = subject_number
        df_slice['sampler'] = "slice"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_slice[subject_number] = df_slice.reset_index().rename(columns={'index': 'parameter'})

    # Load MCMC files and add participant_id, set parameter based on corresponding SLICE file
    for file in mcmc_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_mcmc = pd.read_csv(file)
        df_mcmc['participant_id'] = subject_number
        df_mcmc['sampler'] = "nuts"
        # Assign parameter column from corresponding SLICE DataFrame
        if subject_number in dataframes_slice:
            df_mcmc['parameter'] = dataframes_slice[subject_number]['parameter'].values
        dataframes_mcmc[subject_number] = df_mcmc

    for file in analytical_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_ana = pd.read_csv(file, index_col=0)
        df_ana['participant_id'] = subject_number
        df_ana['sampler'] = "nuts_analytical"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_nuts_analytical[subject_number] = df_ana.reset_index().rename(columns={'index': 'parameter'})
        
    for file in approx_differentiable_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_LAN = pd.read_csv(file, index_col=0)
        df_LAN['participant_id'] = subject_number
        df_LAN['sampler'] = "numpyro_approx_differentiable"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_approx_differentiable[subject_number] = df_LAN.reset_index().rename(columns={'index': 'parameter'})
      
      
    return dataframes_mcmc, dataframes_slice, dataframes_nuts_analytical, dataframes_approx_differentiable

base_path = script_dir 

dataframes_mcmc, dataframes_slice, dataframes_nuts_analytical, dataframes_approx_differentiable = load_files(base_path)

# Concatenate all the DataFrames into a single DataFrame
all_data = pd.concat([pd.concat(dataframes_slice.values(), ignore_index=True),
                      pd.concat(dataframes_mcmc.values(), ignore_index=True),
                      pd.concat(dataframes_nuts_analytical.values(), ignore_index=True),
                      pd.concat(dataframes_approx_differentiable.values(), ignore_index=True)], ignore_index=True)

required_parameters = ['z_Intercept', 'z_prevresp', 'a', 't', 'v_prevresp', 'v_Intercept', 'v_signed_contrast']
filtered_data = all_data[all_data['parameter'].isin(required_parameters)]

filtered_data.to_csv('individual_fits_combined_data.csv', index=False)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

average_r_hat = filtered_data.groupby(['sampler', 'parameter'])['r_hat'].mean().reset_index()
average_estimate = filtered_data.groupby(['sampler', 'parameter'])['mean'].mean().reset_index()

# Define the color palette
palette = sns.color_palette("viridis", as_cmap=False)

# Adjust overall aesthetics for publication quality
sns.set(style='ticks', context='talk', palette=palette)

# Plotting Average r_hat for each sampler by parameter
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='parameter', y='r_hat', hue='sampler', data=average_r_hat, palette=palette)
plt.title('Average $\\hat{R}$ by Parameter and Sampler')
plt.xticks(rotation=45)
plt.xlabel('Parameter')
plt.ylabel('Average $\\hat{R}$')
plt.legend(title='Sampler')
sns.despine()  # Remove the top and right spines
plt.tight_layout()
plt.savefig('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/Average_R_hat_by_Parameter_and_Sampler.png')
plt.show()

# Plotting Mean estimate for each sampler by parameter
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='parameter', y='mean', hue='sampler', data=average_estimate, palette=palette)
plt.title('Mean Estimates by Parameter and Sampler')
plt.xticks(rotation=45)
plt.xlabel('Parameter')
plt.ylabel('Estimate')
plt.legend(title='Sampler')
sns.despine()  # Remove the top and right spines
plt.tight_layout()
plt.savefig('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/Mean_Estimates_by_Parameter_and_Sampler.png')
plt.show()

# %%
pivot_data = filtered_data.pivot_table(index=['participant_id', 'parameter'], columns='sampler', values='mean').reset_index()

# Drop rows with any missing data to ensure a clean comparison
pivot_data.dropna(inplace=True)

# Set up the facet grid for plotting with scales specific to each plot for better visibility
g = sns.FacetGrid(pivot_data, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x='slice', y='nuts_analytical')

# Draw identity lines and set axes limits specific to each plot for clarity
for ax, (_, subdata) in zip(g.axes.flat, pivot_data.groupby('parameter')):
    min_val = min(subdata['slice'].min(), subdata['nuts_analytical'].min())
    max_val = max(subdata['slice'].max(), subdata['nuts_analytical'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

# Set titles and labels
g.set_axis_labels('Slice Mean Estimate', 'Nuts Mean Estimate')
g.set_titles(col_template="{col_name}")
g.add_legend()
plt.show()

# %%
# Load the dataset
data_path = '/Users/kiante/Downloads/visual_motion_2afc_fd_hddmfits.csv'
data = pd.read_csv(data_path)
# Select columns that include the subject number and columns with 'regressdczprevresplag1' in their names
selected_data = data.filter(regex='subjnr|regressdczprevresplag1|t__stimcodingnohist')
#t__regressdclag1
# Rename the columns as specified previously
selected_data = selected_data.rename(columns={
    'subjnr': 'participant_id',
    'a__regressdczprevresplag1': 'a',
    't__stimcodingnohist': 't',
    'v_Intercept__regressdczprevresplag1': 'v_Intercept',
    'v_stimulus__regressdczprevresplag1': 'v_signed_contrast',
    'v_prevresp__regressdczprevresplag1': 'v_prevresp',
    'z_Intercept__regressdczprevresplag1': 'z_Intercept',
    'z_prevresp__regressdczprevresplag1': 'z_prevresp'
})
 # 't__regressdczprevresplag1': 't',
#t__regressdclag1
selected_data['sampler'] = 'old_slice'

melted_df = selected_data.melt(id_vars=["participant_id", "sampler"], var_name="parameter", value_name="old_slice")

melted_df['participant_id'] = melted_df['participant_id'].astype(int)
pivot_data['participant_id'] = pivot_data['participant_id'].astype(int)

combined_df = pd.merge(pivot_data, melted_df, on=["participant_id", "parameter"], how="outer")

# %% plot identiy lines 
# Define palettes for each FacetGrid
palette_deep = sns.color_palette('deep', n_colors=len(combined_df['parameter'].unique()))
palette_pastel = sns.color_palette('pastel', n_colors=len(combined_df['parameter'].unique()))
palette_set2 = sns.color_palette('Set2', n_colors=len(combined_df['parameter'].unique()))

# First FacetGrid: Comparing 'slice' and 'nuts_analytical'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_deep)
g.map_dataframe(sns.scatterplot, x='slice', y='nuts_analytical')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['slice'].min(), subdata['nuts_analytical'].min())
#     max_val = max(subdata['slice'].max(), subdata['nuts_analytical'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('Slice Mean Estimate', 'Nuts Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/Slice_vs_Nuts_Analytical.png')
plt.show()

# Second FacetGrid: Comparing 'slice' and 'old_slice'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_pastel)
g.map_dataframe(sns.scatterplot, x='slice', y='old_slice')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['slice'].min(), subdata['old_slice'].min())
#     max_val = max(subdata['slice'].max(), subdata['old_slice'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('New Slice Mean Estimate', 'Old Slice Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/New_vs_Old_Slice.png')
plt.show()

# Third FacetGrid: Comparing 'nuts_analytical' and 'old_slice'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_set2)
g.map_dataframe(sns.scatterplot, x='nuts_analytical', y='old_slice')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['nuts_analytical'].min(), subdata['old_slice'].min())
#     max_val = max(subdata['nuts_analytical'].max(), subdata['old_slice'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('Nuts Analytical Mean Estimate', 'Old Slice Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig('/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures/Nuts_Analytical_vs_Old_Slice.png')
plt.show()

# %% investiate really low estimates
data_path = '/Users/kiante/Downloads/visual_motion_2afc_fd_hddmfits.csv'
data = pd.read_csv(data_path)
# Select columns that include the subject number and columns with 'regressdczprevresplag1' in their names
# selected_data_ndt = data.filter(regex='subjnr|t_')
# selected_data_t = data.filter(regex='prevresp')
selected_data_t = data.filter(regex='^t_')

# Define number of columns per row
cols_per_row = 4  # Adjust based on how many plots you want per row
num_rows = (len(selected_data_t.columns) + cols_per_row - 1) // cols_per_row  # Calculate the required number of rows

# Plotting
plt.figure(figsize=(20, 4 * num_rows))  # Adjust figure size based on number of rows
for i, column in enumerate(selected_data_t.columns):
    plt.subplot(num_rows, cols_per_row, i + 1)  # Create a subplot for each column
    sns.histplot(selected_data_t[column], kde=False)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# %%
