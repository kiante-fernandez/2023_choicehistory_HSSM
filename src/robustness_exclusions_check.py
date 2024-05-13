# %%
import pymc as pm
import arviz as az
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
# DDM models (the Wiener First-Passage Time distribution)
from hssm.likelihoods import DDM
from matplotlib import pyplot as plt
import signal
import itertools
from hssm import set_floatX

set_floatX("float32")

#%matplotlib inline
# %%
def run_model_with_timeout(model_context, trace_filename, results_filename, timeout_seconds):
    import signal
    
    def handler(signum, frame):
        raise TimeoutError("Model run exceeded the allowed time limit.")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)  # Set the alarm
    
    try:
        with model_context:
            trace = pm.sample(init="adapt_diag", cores=4, chains=4, draws=1000, tune=1000, idata_kwargs={'log_likelihood': True})
            results = az.summary(trace).reset_index()
            results.to_csv(results_filename)
            trace.to_netcdf(trace_filename)
    except TimeoutError:
        print(f"Timeout occurred while processing {trace_filename}")
    finally:
        signal.alarm(0)  # Disable the alarm
        
def run_all_models(dataset, n_subjects, participant_id, signed_contrast, prevresp, subset_id, timeout_hours):
    # Convert hours to seconds for the timeout
    timeout_seconds = timeout_hours * 3600
    
    model_contexts = {
        "no_hist": pm.Model(),
        "prevresp_v": pm.Model(),
        "prevresp_z": pm.Model(),
        "prevresp_zv": pm.Model()
    }

    with model_contexts["no_hist"]:
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

        ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)
                
    with model_contexts["prevresp_v"]:
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
    
        ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    with model_contexts["prevresp_z"]:
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
    
        ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)
    
    
    with model_contexts["prevresp_zv"]:
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
    
        ddm = DDM("ddm", v=v, a=a, z=z, t=t, observed=dataset[['rt','response']].values)

    model_args = []
    for i, (name, context) in enumerate(model_contexts.items(), start=1):
        trace_filename = f"{name}_traces{subset_id}.nc"
        results_filename = f"m{i}_results{subset_id}.csv"
        model_args.append((context, trace_filename, results_filename, timeout_seconds))

    for args in model_args:
        run_model_with_timeout(*args)
        
        
#%%    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', 'visual_motion_2afc_fd.csv')
    elife_data = pd.read_csv(data_file_path)

    # Data preprocessing
    elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']
    elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})
    elife_data['rt'] = elife_data['rt'].round(5)
    elife_data['stimrepeat'] = np.where(elife_data['stimulus'] == elife_data['prevstim'], 1, 0)
    elife_data['repeat'] = np.where(elife_data['response'] == elife_data['prevresp'], 1, 0)
    elife_data['participant_id'] = elife_data['subj_idx']

    timeout_hours = 8
    all_participants = list(range(1, 31))  # Assuming 30 participants total
    exclude_candidates = [11, 19, 20, 22, 26, 27, 28]

    exclude_combinations = []
    for r in range(1, len(exclude_candidates)+1):
        exclude_combinations.extend(itertools.combinations(exclude_candidates, r))
    
    exclude_combinations.reverse()
    
    for exclude in exclude_combinations:
        included_participants = set(all_participants) - set(exclude)
        dataset = elife_data[elife_data['participant_id'].isin(included_participants)]
        n_subjects = len(dataset['participant_id'].unique())
        participant_id, unique_participants = pd.factorize(dataset['participant_id'])
        signed_contrast = dataset['signed_contrast'].values
        prevresp = dataset['prevresp'].values
        
        if exclude:
            subset_id = '_excluded_' + '_'.join(map(str, exclude))
        else:
            subset_id = '_all'
        print(exclude)
        # Call to run all models with unique filenames for each subset and model
        run_all_models(dataset, n_subjects, participant_id, signed_contrast, prevresp, subset_id, timeout_hours=timeout_hours)
