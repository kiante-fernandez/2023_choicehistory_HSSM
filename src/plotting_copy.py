
#load nc
#%%
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import hssm
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pymc as pm
import statsmodels.api as sm
import statsmodels
import scipy as sp
from utils_plot import corrfunc, dependent_corr,rz_ci,rho_rxy_rxz
#import corrstats

#%matplotlib inline
from ssms.basic_simulators.simulator import simulator
hssm.set_floatX("float32")

from hssm_modelspec import make_model # specifically for hssm models
from utils_hssm import run_model, dic, aggregate_model_comparisons, reattach,filter_group_level_params

# %% load data
# Define the directory containing the .nc files
#dir_path = os.path.dirname(os.path.realpath(__file__)) #gets the current path

directory = r'c:\Users\Usuario\Desktop\Zeynep\2023_choicehistory_HSSM\results\models'
plot_directory = r'c:\Users\Usuario\Desktop\Zeynep\2023_choicehistory_HSSM\results\figures'

# Define elife_data (assuming it's already defined)
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})

# # add some more history measures
elife_data['stimrepeat'] = np.where(elife_data.stimulus == elife_data.prevstim, 1, 0)
elife_data['repeat'] = np.where(elife_data.response == elife_data.prevresp, 1, 0)
def get_prep(data):
     # grouped_data = data.groupby(['subj_idx'])['stimrepeat','repeat'].apply(lambda x: x.value_counts(normalize=True))
    grouped_data = data.groupby(['subj_idx'])[['stimrepeat','repeat']].mean().reset_index()
    return grouped_data


prep = pd.DataFrame(get_prep(elife_data))


#%%
# List of subjects to exclude
excluded_participants = [11, 19, 20, 22, 26, 27, 28]

# Filtering out the excluded subjects
prep = prep[~prep['subj_idx'].isin(excluded_participants)]

for file_name in os.listdir(directory):
    if file_name.endswith('.nc'):
        # Extract model name from the file name
        model_name = file_name.split('_model')[0]

        # Construct file path for the current .nc file
        file_path = os.path.join(directory, file_name)

        model = az.from_netcdf(file_path)
        model_type = None

        if any(char in model_name for char in ['v', 'z']):
            model_type = ''.join(char for char in ['v', 'z'] if char in model_name)

        print(f"The model type is: {model_type}" if model_type else "The model does not contain 'v' or 'z'")
        
        results_file_path = os.path.join(directory, f"{model_name}_results_combined.csv")

        # Read the CSV file
        results = pd.read_csv(results_file_path)
        model_type = model_name.split("_")  # Extracting the model type from the model name
        
        pattern_v = r'v_{}\|participant_id_offset\[\d+\]'.format(model_type[0])
        pattern_z = r'z_{}\|participant_id_offset\[\d+\]'.format(model_type[0])

        # 'v' model_type part
        subj_idx_specific_rows_v = results[results['index'].str.contains(pattern_v, na=False, regex=True)]
        subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']]
        new_column_name_v = 'v_mean'
        subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']].rename(columns={'mean': new_column_name_v})
        subj_idx_df_reset_v = subj_idx_df_v.reset_index(drop=True)
        prep_reset_v = prep.reset_index(drop=True)
        merged_df_v = pd.concat([subj_idx_df_reset_v, prep_reset_v], axis=1)
        #merged_df_v.rename(columns={'repeat': 'repetition'}, inplace=True)

        # 'z' model_type part
        subj_idx_specific_rows_z = results[results['index'].str.contains(pattern_z, na=False, regex=True)]
        subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']]
        new_column_name_z = 'z_mean'
        subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']].rename(columns={'mean': new_column_name_z})
        subj_idx_df_reset_z = subj_idx_df_z.reset_index(drop=True)
        prep_reset_z = prep.reset_index(drop=True)
        merged_df_z = pd.concat([subj_idx_df_reset_z, prep_reset_z], axis=1)
        #merged_df_z.rename(columns={'repeat': 'repetition'}, inplace=True)

        # Plotting
        fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(12, 6))

        # 'v' model_type subplot
        corrfunc(x=merged_df_z.z_mean, y=merged_df_z.repeat, ax=ax[0], color='blue')
        ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')

        # 'z' model_type subplot
        corrfunc(x=merged_df_v.v_mean, y=merged_df_v.repeat, ax=ax[1], color='green')
        ax[1].set(xlabel='History shift in drift bias', ylabel='P(repeat)')

        # Performing Steiger's test
        tstat, pval = dependent_corr(
            sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0],
            sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0],
            sp.stats.spearmanr(merged_df_z.z_mean, merged_df_v.v_mean, nan_policy='omit')[0],
            len(merged_df_z),
            twotailed=True, conf_level=0.95, method='steiger'
        )
        deltarho = sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0] - \
                sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0]

        if pval < 0.0001:
            fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho), fontsize=10)
        else:
            fig.suptitle(r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)

        sns.despine(trim=True)
        plt.tight_layout()
        #plt.title('Correlation plots for %s'%model_name, fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
        fig.savefig(os.path.join(plot_directory, 'scatterplot_trainingchoiceworld_%s_prevchoice_zv.png'%model_type))
        plt.close()


        # Posterior Plot #
        plot_file_name = f"{model_name}_posteriorplot.png"
        plot_file_path = os.path.join(plot_directory, plot_file_name)
        #plot_file_path = os.path.join(file_path, plot_directory + "_posteriorplot.pdf")
        az.style.use("arviz-doc")
        suffix_to_filter = "participant_id_offset"
        filtered_vars = [var for var in model.posterior.data_vars if filter_group_level_params(var, suffix_to_filter)]
        #az.plot_posterior(model, kind="scatter", var_names=filtered_vars)
        az.plot_posterior(model,var_names=["~participant_id_offset"],filter_vars="like")
        #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
        plt.tight_layout()
        plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
        plt.savefig(plot_file_path,dpi=300)
        plt.close()

        # Trace Plot #
        plot_file_name = f"{model_name}_traceplot.png"
        plot_file_path = os.path.join(plot_directory, plot_file_name)
        az.style.use("arviz-doc")
        az.plot_trace(model,var_names=["~participant_id_offset"],filter_vars="like")
        #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
        plt.tight_layout()
        plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
        plt.savefig(plot_file_path,dpi=300)
        plt.close()
        
        # Forest Plot #
        plot_file_name = f"{model_name}_forestplot.png"
        plot_file_path = os.path.join(plot_directory, plot_file_name)
        az.style.use("arviz-doc")
        #az.plot_forest(model, var_names=["~z"]) 
        az.plot_forest(model,var_names=["~participant_id_offset"],filter_vars="like")
        #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
        plt.tight_layout()
        plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
        plt.savefig(plot_file_path,dpi=300,bbox_inches='tight',pad_inches=0.1)
        plt.close()

        # Pair Plot #
        plot_file_name = f"{model_name}_pair_plot.png"
        plot_file_path = os.path.join(plot_directory, plot_file_name)
        #reattach(file_path, model, elife_data)
        az.style.use("arviz-doc")
        az.plot_pair(model,var_names=["~participant_id_offset"],filter_vars="like",kind="kde")
        #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
        plt.tight_layout()
        plt.figtext(0.5, 0.95, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=60, va='top')
        plt.savefig(plot_file_path,dpi=300)
        plt.close()

        # Posterior Pred Plot #
        #plot_file_name = f"{model_name}_ppc_plot.png"
        #plot_file_path = os.path.join(plot_directory, plot_file_name)
        #reattach(file_path, model, elife_data)
        #az.style.use("arviz-doc")
        #pm.sample_posterior_predictive(model) #### before saving nc file
        #az.plot_ppc(model)
        #plt.tight_layout()
        #plt.savefig(plot_file_path,dpi=300)
        #plt.close()

        

# %%
