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
def savePatch(self, fname):
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(self, f)
        
hssm.HSSM.savePatch = savePatch

def run_model(data, modelname, mypath, n_samples=1000, trace_id=0):

    from hssm_modelspec import make_model # specifically for HDDMnn models

    print('HSSM version: ', hssm.__version__)

    # get the model
    m = make_model(data, modelname)
    time.sleep(trace_id) # to avoid different jobs trying to make the same folder

    # make a new folder if it doesn't exist yet
    if not os.path.exists(mypath):
        os.makedirs(mypath)
        print('creating directory %s' % mypath)

    print("begin sampling") # this is the core of the fitting

    # If you are running multiple models simultaneously, make sure to name the .db files differently, otherwise they might all 
    # write to the same location and when HDDM tries to save the model by consulting the .db file it'll just get a jumble.
    m.sample(n_samples, burn = np.max([n_samples/10, 100]),
             dbname= os.path.join(mypath, 'traces.db'), 
             db='pickle')

    print('saving model itself')
    m.savePatch(os.path.join(mypath, 'model.mdl')) # use the patched code
    
    print("save model comparison indices")
    df = dict()
    df['dic'] = [m.dic]
    df['aic'] = [aic(m)]
    df['bic'] = [bic(m)]
    df2 = pd.DataFrame(df)
    df2.to_csv(os.path.join(mypath, 'model_comparison.csv'))

    # save useful output
    print("saving summary stats")
    # results = m.gen_stats().reset_index()  # point estimate for each parameter and subject
    results =  az.summary(m).reset_index()  # point estimate for each parameter and subject
    results.to_csv(os.path.join(mypath, 'results_combined.csv'))

    print("saving traces")
    # get the names for all nodes that are available here
    group_traces = m.get_group_traces()
    group_traces.to_csv(os.path.join(mypath, 'group_traces.csv'))

    return m


def plot_model(m, savepath):

    # MAKE SOME PLOTS
    # 'Note: The posterior pair plot does not support regression models at this point! Aborting...'
    # hddm.plotting.plot_posterior_pair(m, samples=50,
    #                                   save=True, save_path=savepath)

    # quick overview of the parameters
    hddm.plotting.plot_caterpillar(hddm_model = m,
                                   save=True, path=savepath,
                                   drop_sd = True,
                                   keep_key=list(m.get_group_nodes().reset_index()['index']),
                                   columns=5)
        
    # more classical posterior predictive on the RT distributions
    # the likelihood-based posterior predictives (_plot_func_posterior_pdf_node_nn) are not yet implemented for HDDMnn regression models, use simulated ones instead
    hddm.plotting.plot_posterior_predictive(model = m,
                                            save=True, path=savepath,
                                            # columns = 4, #groupby = ['subj_idx'],
                                            value_range = np.arange(-2, 2, 0.01),
                                            plot_func = hddm.plotting._plot_func_posterior_node_from_sim,
                                            figsize=(15,24),
                                            parameter_recovery_mode = False,
                                            **{'alpha': 0.01,
                                            'add_legend':False,
                                            'ylim': 3,
                                            'bin_size': 0.05,
                                            'add_posterior_mean_rts': True,
                                            'add_posterior_uncertainty_rts': False,
                                            #'samples': 30,
                                            'data_color':'darkblue',
                                            'posterior_mean_color':'firebrick',
                                            # 'legend_fontsize': 7,
                                            'subplots_adjust': {'top': 0.9, 'hspace': 1, 'wspace': 0.3}})
    

    # across-subject parameter correlation plot
    results = results_long2wide_hddmnn(m.gen_stats().reset_index())  # point estimate for each parameter and subject
    g = sns.PairGrid(results, vars=list(set(results.columns) - set(['subj_idx'])))
    g.map_diag(sns.histplot)
    g.map_lower(corrfunc)
    g.map_upper(sns.kdeplot)
    g.savefig(os.path.join(savepath, 'pairgrid_corrplot.png'))



