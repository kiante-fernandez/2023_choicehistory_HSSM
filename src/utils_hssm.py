import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob, time
import datetime

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pymc.variational.callbacks import CheckParametersConvergence

# more handy imports
import hssm
from utils_plot import results_long2wide_hddmnn, corrfunc
import pymc as pm

# ============================================ #
# define some functions
# ============================================ #

# Get around a problem with saving regression outputs in Python 3
# TODO look at the new saving protocol following the InferenceData with netCDF4
# see: https://python.arviz.org/en/stable/getting_started/XarrayforArviZ.html

        
def saveInferenceData(model, fname):
    model.to_netcdf(fname)

import arviz as az

def dic(inference_data):
    """
    Calculate the Deviance Information Criterion (DIC) for a given model.

    Parameters:
    inference_data (arviz.InferenceData): An ArviZ InferenceData object containing the posterior samples

    Returns:
    float: The computed DIC value.
    """

    # Extract log likelihood from the inference data
    log_likelihood = inference_data.log_likelihood

    # Calculate the point-wise deviance
    D_bar = -2 * np.mean(log_likelihood)

    # Calculate the effective number of parameters
    p_D = 2 * (D_bar + 2 * np.mean(log_likelihood) - np.mean(log_likelihood))

    # Calculate DIC
    dic = D_bar + p_D

    return dic

# Assuming `HSSM` is the class of your HSSM objects and `inference_data` is the InferenceData attribute
hssm.HSSM.saveInferenceData = saveInferenceData

def run_model(data, modelname, mypath, trace_id=0, sampling_method ="mcmc", **kwargs):

    from utils_hssm_modelspec import make_model # specifically for HSSM models

    print('HSSM version: ', hssm.__version__)

    sampling_params = {
        "sampler": kwargs.get('sampler', 'nuts_numpyro'),
        "chains": kwargs.get('chains', 4),
        "cores": kwargs.get('cores', 4),
        "draws": kwargs.get('draws', 1000),
        "tune": kwargs.get('tune', 1000),
        "idata_kwargs": kwargs.get('idata_kwargs', dict(log_likelihood=True))  # return log likelihood
    }
    m = make_model(data, modelname)
    time.sleep(trace_id) # to avoid different jobs trying to make the same folder

    # make a new folder if it doesn't exist yet
    if not os.path.exists(mypath):
        os.makedirs(mypath)
        print('creating directory %s' % mypath)

    print("begin estimation")

    # Sample from the model
    if sampling_method == "mcmc":
        # Run sampling
        # inference_data = m.sample(**sampling_params)
        inference_data = m.sample() #hard coded for debug. Fine for most uses now. 
        
        print('saving model itself')

        # Save the InferenceData object
        inference_data.to_netcdf(os.path.join(mypath, f'{modelname}_model.nc'))
    
        print("save model comparison indices")
        df = dict()
        import arviz as az
        df = dict()
        df['dic'] = dic(inference_data)['rt,response'].values.item()
        df['waic'] = az.waic(inference_data).elpd_waic
        df['loo'] = az.loo(inference_data).elpd_loo
        df2 = pd.DataFrame(list(df.items()), columns=['Metric', 'Value'])
        df2.to_csv(os.path.join(mypath, f'{modelname}_model_comparison.csv'))

        # save useful output
        print("saving summary stats")
        results =  az.summary(inference_data).reset_index()  # point estimate for each parameter and subject
        results.to_csv(os.path.join(mypath, f'{modelname}_results_combined.csv')) 
        
    elif sampling_method == "vi":
        import arviz as az
        # Run VI sampling
        vi_approx = m.vi(niter=60000,method="fullrank_advi", callbacks=[CheckParametersConvergence(diff='absolute')])
        vi_samples = m.vi_approx.sample(draws=1000)

        #az_data = az.from_pymc3(vi_samples)
        
        # Save the VI samples
        #m.to_netcdf(os.path.join(mypath, f'{modelname}_model.nc'))
        plt.plot(m.vi_approx.hist)
        hist_file = os.path.join(mypath, f'{modelname}_vi_loss_hist.png')
        plt.savefig(hist_file)
        plt.close()
        # Extract and save summary statistics
        print("saving summary stats")
        results = az.summary(m.vi_idata).reset_index()
        results.to_csv(os.path.join(mypath, f'{modelname}_results_combined.csv')) 
        
        idatavi = m.vi_idata
        
        with m.pymc_model:
            pm.compute_log_likelihood(idatavi)
        
        print("save model comparison indices")
        df = dict()
#        df['dic'] = dic(inference_data)['rt,response'].values.item()
        df['waic'] =  az.waic(idatavi).elpd_waic
        df['loo'] = az.loo(idatavi).elpd_loo
        df2 = pd.DataFrame(list(df.items()), columns=['Metric', 'Value'])
        df2.to_csv(os.path.join(mypath, f'{modelname}_model_comparison.csv'))
        
    else:
        raise ValueError("Unsupported sampling method. Use 'mcmc' or 'vi'.")
    
    return m

def aggregate_model_comparisons(directory):
    # Initialize an empty DataFrame to store all data
    aggregated_data = pd.DataFrame()

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        # Check if 'model_comparison' is in the file name
        if 'model_comparison' in file:
            # Extract model name from filename (assuming the model name is the part of the filename before '_model_comparison')
            model_name = file.split('_model_comparison')[0]

            # Construct full file path
            file_path = os.path.join(directory, file)
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Add a new column with the model name
            data['Model'] = model_name

            # Append the data to the aggregated DataFrame
            aggregated_data = pd.concat([aggregated_data, data], ignore_index=True)
            #aggregated_data = aggregated_data.append(data, ignore_index=True)

    # Reorder columns if necessary
    aggregated_data = aggregated_data[['Model', 'Metric', 'Value']]

    # Save the aggregated data to a new CSV file
    aggregated_data.to_csv('aggregated_model_comparisons.csv', index=False)

    return 'File saved as aggregated_model_comparisons.csv'

# hssm new verions has something that does this now
# def reattach(filename, model, data):
#     import arviz as az
#     from utils_hssm_modelspec import make_model
#     #load the inferenceData object
#     inferd = az.from_netcdf(filename)
#     #reattch to the model
#     m = make_model(data,  model)
#     m._inference_obj = inferd
#     return m



