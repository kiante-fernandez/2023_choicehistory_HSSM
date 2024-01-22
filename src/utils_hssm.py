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

# more handy imports
import hssm
from utils_plot import results_long2wide_hddmnn, corrfunc

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

def run_model(data, modelname, mypath, trace_id=0, **kwargs):

    from hssm_modelspec import make_model # specifically for HSSM models

    print('HSSM version: ', hssm.__version__)

    sampling_params = {
        "sampler": kwargs.get('sampler', 'nuts_numpyro'),
        "chains": kwargs.get('chains', 4),
        "cores": kwargs.get('cores', 4),
        "draws": kwargs.get('draws', 1000),
        "tune": kwargs.get('tune', 1000),
        "idata_kwargs": kwargs.get('idata_kwargs', dict(log_likelihood=False))  # return log likelihood
    }
    m = make_model(data, modelname)
    time.sleep(trace_id) # to avoid different jobs trying to make the same folder

    # make a new folder if it doesn't exist yet
    if not os.path.exists(mypath):
        os.makedirs(mypath)
        print('creating directory %s' % mypath)

    print("begin sampling") # this is the core of the fitting

    # Sample from the model
    inference_data = m.sample(**sampling_params)

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

    # print("saving traces")
    # # get the names for all nodes that are available here
    # group_traces = inference_data.posterior.to_dataframe()
    # group_traces.to_csv(os.path.join(mypath, f'{modelname}_group_traces.csv'))    
    
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

# TODO write something for simple plotting
# def plot_model(m, savepath):

#     # MAKE SOME PLOTS
#     # 'Note: The posterior pair plot does not support regression models at this point! Aborting...'
#     # hddm.plotting.plot_posterior_pair(m, samples=50,
#     #                                   save=True, save_path=savepath)

#     # quick overview of the parameters
#     hddm.plotting.plot_caterpillar(hddm_model = m,
#                                    save=True, path=savepath,
#                                    drop_sd = True,
#                                    keep_key=list(m.get_group_nodes().reset_index()['index']),
#                                    columns=5)
        
#     # more classical posterior predictive on the RT distributions
#     # the likelihood-based posterior predictives (_plot_func_posterior_pdf_node_nn) are not yet implemented for HDDMnn regression models, use simulated ones instead
#     hddm.plotting.plot_posterior_predictive(model = m,
#                                             save=True, path=savepath,
#                                             # columns = 4, #groupby = ['subj_idx'],
#                                             value_range = np.arange(-2, 2, 0.01),
#                                             plot_func = hddm.plotting._plot_func_posterior_node_from_sim,
#                                             figsize=(15,24),
#                                             parameter_recovery_mode = False,
#                                             **{'alpha': 0.01,
#                                             'add_legend':False,
#                                             'ylim': 3,
#                                             'bin_size': 0.05,
#                                             'add_posterior_mean_rts': True,
#                                             'add_posterior_uncertainty_rts': False,
#                                             #'samples': 30,
#                                             'data_color':'darkblue',
#                                             'posterior_mean_color':'firebrick',
#                                             # 'legend_fontsize': 7,
#                                             'subplots_adjust': {'top': 0.9, 'hspace': 1, 'wspace': 0.3}})
    

#     # across-subject parameter correlation plot
#     results = results_long2wide_hddmnn(m.gen_stats().reset_index())  # point estimate for each parameter and subject
#     g = sns.PairGrid(results, vars=list(set(results.columns) - set(['subj_idx'])))
#     g.map_diag(sns.histplot)
#     g.map_lower(corrfunc)
#     g.map_upper(sns.kdeplot)
#     g.savefig(os.path.join(savepath, 'pairgrid_corrplot.png'))



