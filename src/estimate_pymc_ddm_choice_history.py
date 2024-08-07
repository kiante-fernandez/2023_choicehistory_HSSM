# pymc_ddm_choice_history.py - simple DDMs choice history test M1-M4 with pymc model specifications
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/17/11      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2024/08/01      Kianté Fernandez<kiantefernan@gmail.com>   added util functions for workflow
# 2024/08/01      Kianté Fernandez<kiantefernan@gmail.com>   refactored


# %%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from ssms.basic_simulators.simulator import simulator
import glob
import seaborn as sns
from matplotlib import pyplot as plt

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

mouse_data = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', 'ibl_trainingChoiceWorld_20240715.csv')
mouse_data = pd.read_csv(mouse_data)
mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
mouse_data['rt'] = mouse_data['rt'].round(6)
mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0]
mouse_data['participant_id'] = mouse_data['participant_id'] + 1
nan_rt_count = mouse_data['rt'].isna().sum()
print(f"Number of NaN rt values: {nan_rt_count}")
negative_rt_count = (mouse_data['rt'] < 0).sum()
print(f"Number of negative rt values: {negative_rt_count}")
#remove the NA cols and negatives
mouse_data = mouse_data.dropna(subset=['rt'])
mouse_data = mouse_data[mouse_data['rt'] >= 0]
mouse_data = mouse_data.dropna(subset=['response'])
mouse_data = mouse_data.dropna(subset=['prevresp'])

# Define the number of subjects you want to sample
# num_subjects = 3
# Group the data by subject, sample the desired number of subjects, and get all rows for these subjects
# subsample = mouse_data[mouse_data['participant_id'].isin(mouse_data.groupby('participant_id').size().sample(num_subjects).index)]
# %% create subset of data and objects for pymc
# dataset = elife_data 
# excluded_participants = [11, 19, 20, 22, 26, 27, 28] #we want diff combos
# dataset = dataset[~dataset['participant_id'].isin(excluded_participants)]

dataset = mouse_data 

# excluded_participants = [5, 7, 9, 10, 14,16, 23, 38, 41, 45]
# dataset = dataset[~dataset['participant_id'].isin(excluded_participants)]

n_subjects = len(dataset['participant_id'].unique())
participant_id, unique_participants = pd.factorize(dataset['subj_idx'])
signed_contrast = dataset['signed_contrast'].values
prevresp = dataset['prevresp'].values

#%% check individual subjects
import hssm
import os
import bambi as bmb

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
output_folder = "subject_plots"
ensure_dir(output_folder)

unique_subjects = dataset['participant_id'].unique()

# Loop through each subject
for subject_id in unique_subjects:
    print(f"Processing subject {subject_id}")
    
    # Filter data for the current subject
    subject_data = dataset[dataset['participant_id'] == subject_id]
    
    # Create a new model instance for the current subject
    subject_model = hssm.HSSM(
        data=subject_data,
        p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.50},
        lapse=bmb.Prior("Uniform", lower=0.0, upper=30.0),
        include=[
        {
            "name": "v",
            "formula":"v ~ 1 + signed_contrast + prevresp"
        },
        {
            "name": "z",
            "formula":"z ~ 1 + prevresp",
        }
    ]
    )
    subject_model.sample(
        step = pm.NUTS(
            model=subject_model.pymc_model,
            target_accept=0.96,
            max_treedepth=12,
            adapt_step_size=True),
        cores=4,
        chains=4,
        draws=6000,
        tune=6000,
        idata_kwargs=dict(log_likelihood=False)
    )
    results =  subject_model.summary().reset_index()
    sub_res_filename = os.path.join(output_folder, f"res_subject_{subject_id}_samples_summary.csv")
    results.to_csv(sub_res_filename)
    # Create and save the trace plot
    plt.figure(figsize=(12, 8))
    subject_model.plot_trace()
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_folder, f"subject_{subject_id}_trace_plot.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Saved plot for subject {subject_id}")

# %% model one. no history effects
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
    # no_hist_pymc_trace = pm.sample(init="adapt_diag",
    #                            cores=4,
    #                            chains=4,
    #                            draws=1000,
    #                            tune=1000,
    #                            discard_tuned_samples = True,
    #                            idata_kwargs=dict(log_likelihood=True))
    step = pm.NUTS(
        target_accept=0.99, #.999
        # step_scale=0.001,
        # max_treedepth=20,
        adapt_step_size=True
    )
    no_hist_pymc_trace = pm.sample(
        nuts_sampler="numpyro",
        step=step,
        cores=4,
        chains=4,
        draws=5000,
        tune=5000,
        idata_kwargs=dict(log_likelihood=False)
    )
    
print("saving summary stats")
results =  az.summary(no_hist_pymc_trace).reset_index()  # point estimate for each parameter and subject
results.to_csv('m1_trainingChoiceWorld_results.csv')
# %% model 2 choice history on drift
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
results.to_csv('m2_trainingChoiceWorld_results.csv')
# %% model 3 choice history on starting point
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
results.to_csv('m3_trainingChoiceWorld_results.csv')
#%% model 4 choice history on drift and starting point
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
results.to_csv('m4_trainingChoiceWorld_results.csv')
prevresp_zv_pymc_trace.to_netcdf("prevresp_zv_pymc_traces.nc")

# %% plot traces across each model
az.plot_trace(no_hist_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_v_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_z_pymc_trace)
plt.tight_layout()
az.plot_trace(prevresp_zv_pymc_trace)
plt.tight_layout()
#%% run model comparisons across dic waic and loo
models = [no_hist_pymc_trace, prevresp_v_pymc_trace, prevresp_z_pymc_trace, prevresp_zv_pymc_trace]  # replace with your actual model traces
compare_results = []

for i, model_trace in enumerate(models):
    dics = dic(model_trace)['ddm'].values.item()
    waic = az.waic(model_trace).elpd_waic
    loo = az.loo(model_trace).elpd_loo
    compare_results.append({'Model': f'Model{i+1}','DIC':dics, 'WAIC': waic, 'LOO': loo})

df2 = pd.DataFrame(compare_results)
df2.to_csv('model_comparison.csv', index=False)

# %% quick model compare plot
model_names = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv"
]
## find compare data saved previously
comparison_data = pd.read_csv(os.path.join(script_dir, 'model_comparison.csv'))
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
