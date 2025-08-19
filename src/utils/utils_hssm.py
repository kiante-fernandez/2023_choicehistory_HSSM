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
from .utils_plot import results_long2wide_hddmnn, corrfunc
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

def plot_traces_func(inference_data, modelname, mypath, timestamp):
    """
    Generate trace plots for HSSM model parameters.
    
    Parameters:
    -----------
    inference_data : arviz.InferenceData
        The inference data object containing posterior samples
    modelname : str
        Name of the model for file naming
    mypath : str
        Path to save the trace plots
    timestamp : str
        Timestamp string for unique file naming
    """
    print("Generating trace plots...")
    
    try:
        # Create trace plots with massive height for maximum readability
        fig = az.plot_trace(inference_data, figsize=(24, 48))
        
        # Adjust layout with massive vertical spacing
        plt.tight_layout(pad=8.0)  # Much more padding between subplots
        plt.subplots_adjust(hspace=0.8, wspace=0.3)  # Massive vertical spacing
        
        # Save as both PNG and PDF
        trace_png = os.path.join(mypath, f'{modelname}_{timestamp}_traces.png')
        trace_pdf = os.path.join(mypath, f'{modelname}_{timestamp}_traces.pdf')
        
        plt.savefig(trace_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(trace_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Trace plots saved: {trace_png} and {trace_pdf}")
        
    except Exception as e:
        print(f"Error generating trace plots: {e}")
        print("Continuing without trace plots...")
        # Close any open figures to prevent memory issues
        plt.close('all')


def run_model(data, modelname, mypath, trace_id=0, sampling_method="mcmc", plot_traces=True, **kwargs):
    """
    Run HSSM model with trace plotting and unique timestamped file naming.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for model fitting
    modelname : str
        Name of the model to run
    mypath : str
        Base path - models will be saved in organized subdirectories
    trace_id : int, default=0
        Sleep time to avoid concurrent folder creation conflicts
    sampling_method : str, default="mcmc"
        Sampling method: "mcmc" or "vi"
    plot_traces : bool, default=True
        Whether to generate and save trace plots
    **kwargs : dict
        Additional sampling parameters
        
    Returns:
    --------
    hssm.HSSM
        The fitted HSSM model object
    """
    from .utils_hssm_modelspec import make_model # specifically for HSSM models

    # Generate timestamp for unique file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Starting model run at {timestamp}')
    print('HSSM version: ', hssm.__version__)
    
    # Create organized directory structure - find the project root
    # Navigate from mypath to the project root (2023_choicehistory_HSSM)
    current_path = os.path.abspath(mypath)
    while not os.path.basename(current_path) == '2023_choicehistory_HSSM' and current_path != '/':
        current_path = os.path.dirname(current_path)
    
    if current_path == '/':
        # Fallback if we can't find the project root
        project_root = '/Users/kiante/Documents/2023_choicehistory_HSSM'
    else:
        project_root = current_path
    
    results_base = os.path.join(project_root, 'results', 'figures')
    models_dir = os.path.join(results_base, 'models')
    comparisons_dir = os.path.join(results_base, 'model_comparisons')
    traces_dir = os.path.join(results_base, 'traces')
    
    print(f'Project root: {project_root}')
    print(f'Results base: {results_base}')
    
    # Create directories if they don't exist
    for directory in [models_dir, comparisons_dir, traces_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'Created directory {directory}')

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
        inference_data = m.sample(**sampling_params)
        # inference_data = m.sample(init="advi+adapt_diag",
        #                           chains=4, cores=4, draws= 1000, tune= 1000) #hard coded for debug. Fine for most uses now. 
        # inference_data = m.sample(chains=1, cores=1, sampler="nuts_numpyro") 
        #inference_data = m.sample()

        print('saving model itself')

        # Save the InferenceData object with timestamp
        model_file = os.path.join(models_dir, f'{modelname}_{timestamp}_model.nc')
        inference_data.to_netcdf(model_file)
        print(f'Model saved: {model_file}')
        
        # Generate trace plots if requested
        if plot_traces:
            plot_traces_func(inference_data, modelname, traces_dir, timestamp)
    
        print("save model comparison indices")
        df = dict()
        import arviz as az
        
        try:
            # Calculate DIC more robustly
            dic_result = dic(inference_data)
            if isinstance(dic_result, dict):
                df['dic'] = dic_result.get('rt,response', 'N/A')
            else:
                df['dic'] = float(dic_result) if dic_result is not None else 'N/A'
        except Exception as e:
            print(f"Warning: Could not calculate DIC: {e}")
            df['dic'] = 'N/A'
        
        try:
            df['waic'] = az.waic(inference_data).elpd_waic
            df['loo'] = az.loo(inference_data).elpd_loo
        except Exception as e:
            print(f"Warning: Could not calculate WAIC/LOO: {e}")
            df['waic'] = 'N/A'
            df['loo'] = 'N/A'
        
        df2 = pd.DataFrame(list(df.items()), columns=['Metric', 'Value'])
        comparison_file = os.path.join(comparisons_dir, f'{modelname}_{timestamp}_model_comparison.csv')
        df2.to_csv(comparison_file)
        print(f'Model comparison saved: {comparison_file}')

        # save useful output
        print("saving summary stats")
        results =  az.summary(inference_data).reset_index()  # point estimate for each parameter and subject
        results_file = os.path.join(mypath, f'{modelname}_{timestamp}_results_combined.csv')
        results.to_csv(results_file)
        print(f'Results summary saved: {results_file}') 
        
    elif sampling_method == "vi":
        import arviz as az
        # Run VI sampling
        vi_approx = m.vi(niter=60000,method="fullrank_advi", callbacks=[CheckParametersConvergence(diff='absolute')])
        vi_samples = m.vi_approx.sample(draws=1000)

        #az_data = az.from_pymc3(vi_samples)
        
        # Save the VI samples
        #m.to_netcdf(os.path.join(mypath, f'{modelname}_model.nc'))
        plt.plot(m.vi_approx.hist)
        hist_file = os.path.join(traces_dir, f'{modelname}_{timestamp}_vi_loss_hist.png')
        plt.savefig(hist_file)
        plt.close()
        print(f'VI loss history saved: {hist_file}')
        # Extract and save summary statistics
        print("saving summary stats")
        results = az.summary(m.vi_idata).reset_index()
        results_file = os.path.join(mypath, f'{modelname}_{timestamp}_results_combined.csv')
        results.to_csv(results_file)
        print(f'VI results summary saved: {results_file}') 
        
        idatavi = m.vi_idata
        
        with m.pymc_model:
            pm.compute_log_likelihood(idatavi)
        
        print("save model comparison indices")
        df = dict()
        
        try:
            df['waic'] =  az.waic(idatavi).elpd_waic
            df['loo'] = az.loo(idatavi).elpd_loo
        except Exception as e:
            print(f"Warning: Could not calculate WAIC/LOO for VI: {e}")
            df['waic'] = 'N/A'
            df['loo'] = 'N/A'
        
        df2 = pd.DataFrame(list(df.items()), columns=['Metric', 'Value'])
        comparison_file = os.path.join(comparisons_dir, f'{modelname}_{timestamp}_model_comparison.csv')
        df2.to_csv(comparison_file)
        print(f'VI model comparison saved: {comparison_file}')
        
    else:
        raise ValueError("Unsupported sampling method. Use 'mcmc' or 'vi'.")
    
    return m


