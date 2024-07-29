
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
import glob

#%matplotlib inline
from ssms.basic_simulators.simulator import simulator
hssm.set_floatX("float32")

from utils_hssm_modelspec import make_model # specifically for hssm models
from utils_hssm import run_model, dic, aggregate_model_comparisons, reattach
#ask about the function below, do not have? 

#filter_group_level_params
# %% load data
#dir_path = os.path.dirname(os.path.realpath(__file__)) #gets the current path

# Map the model types for cvs files
model_mapping = {
    'no_hist': 'm1',
    'prevresp': 'm2',
    'prevresp_z': 'm3',
    'prevresp_zv': 'm4'
}

# Directory for the nc files and for saving the plots
#directory = r'c:\Users\Usuario\Desktop\Zeynep\2023_choicehistory_HSSM\results\models'
#plot_directory = r'c:\Users\Usuario\Desktop\Zeynep\2023_choicehistory_HSSM\results\figures'



# Load the data (elife)
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']

elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})

# Add some more history measures
elife_data['stimrepeat'] = np.where(elife_data.stimulus == elife_data.prevstim, 1, 0)
elife_data['repeat'] = np.where(elife_data.response == elife_data.prevresp, 1, 0)
def get_prep(data):
    # grouped_data = data.groupby(['subj_idx'])['stimrepeat','repeat'].apply(lambda x: x.value_counts(normalize=True))
    grouped_data = data.groupby(['subj_idx'])[['stimrepeat','repeat']].mean().reset_index()
    return grouped_data


prep = pd.DataFrame(get_prep(elife_data))

#%% zeys code
# ## Plotting loop ##
# for file_name in os.listdir(directory):
#     if file_name.endswith('.nc'):
#         # Extract model name from the file name
#         model_name = file_name.split('_traces')[0]
#         # List of subjects to exclude
#         excluded_participants = [int(num.replace(".nc", "")) for num in file_name.split("_excluded")[1].split("_") if num]

#         # Filtering out the excluded subjects
#         prep = prep[~prep['subj_idx'].isin(excluded_participants)].reset_index()
#         elife_data_excluded = elife_data[~elife_data['subj_idx'].isin(excluded_participants)].reset_index()
#         # Remove or reset the index ! sample_subj_idx as a name !

#         # Construct file path for the current .nc file
#         file_path = os.path.join(directory, file_name)
        
#         # Load the model
#         model = az.from_netcdf(file_path)
#         model_type = None

#         # Extract the model type informations
#         if any(char in model_name for char in ['v', 'z']):
#             model_type = ''.join(char for char in ['v', 'z'] if char in model_name)

#         print(f"The model type is: {model_type}" if model_type else "The model does not contain 'v' or 'z'")

#         # Prepare for loading the csv result file
#         model_value = model_mapping.get(model_name, model_name)
#         results_file_path = os.path.join(directory, f"{model_value}_test_results_combined.csv")

#         # Read the CSV file
#         results = pd.read_csv(results_file_path)
#         ## depends on the model name and order
#         model_type = model_name.split("_")  # Extracting the model type from the model name
        
#         ## may need to change subj_idx depend on the model

#         # Creates the regular expression pattern to extract subject predictions
#         pattern_v = r'v_{}\|participant_id_offset\[\d+\]'.format(model_type[0]) 
#         pattern_z = r'z_{}\|participant_id_offset\[\d+\]'.format(model_type[0])
#         pal = sns.color_palette("Paired")
#         pal2 = pal[2:4] + pal[0:2] + pal[8:10]
#         # 'v' model_type part
#         subj_idx_specific_rows_v = results[results['index'].str.contains(pattern_v, na=False, regex=True)]
#         subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']]
#         new_column_name_v = 'v_mean'
#         subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']].rename(columns={'mean': new_column_name_v})
#         subj_idx_df_reset_v = subj_idx_df_v.reset_index(drop=True)
#         prep_reset_v = prep.reset_index(drop=True)
#         merged_df_v = pd.concat([subj_idx_df_reset_v, prep_reset_v], axis=1)
#         #merged_df_v.rename(columns={'repeat': 'repetition'}, inplace=True)

#         # 'z' model_type part
#         subj_idx_specific_rows_z = results[results['index'].str.contains(pattern_z, na=False, regex=True)]
#         subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']]
#         new_column_name_z = 'z_mean'
#         subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']].rename(columns={'mean': new_column_name_z})
#         subj_idx_df_reset_z = subj_idx_df_z.reset_index(drop=True)
#         prep_reset_z = prep.reset_index(drop=True)
#         merged_df_z = pd.concat([subj_idx_df_reset_z, prep_reset_z], axis=1)
#         #merged_df_z.rename(columns={'repeat': 'repetition'}, inplace=True)

#         # Plotting
#         fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(6,3))

#         # 'v' model_type subplot
#         corrfunc(x=merged_df_z.z_mean, y=merged_df_z.repeat, ax=ax[0], color=pal2[1])
#         ax[0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
#         ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')

#         # 'z' model_type subplot
#         ax[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
#         corrfunc(x=merged_df_v.v_mean, y=merged_df_v.repeat, ax=ax[1], color=pal2[3])
#         ax[1].set(xlabel='History shift in v', ylabel='P(repeat)')
        
#         # Performing Steiger's test
#         tstat, pval = dependent_corr(
#             sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0],
#             sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0],
#             sp.stats.spearmanr(merged_df_z.z_mean, merged_df_v.v_mean, nan_policy='omit')[0],
#             len(merged_df_z),
#             twotailed=True, conf_level=0.95, method='steiger'
#         )
#         deltarho = sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0] - \
#                 sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0]

#         if pval < 0.0001:
#             fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho), fontsize=10)
#         else:
#             fig.suptitle(r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)

#         sns.despine(trim=True)
#         plt.tight_layout()
#         #plt.title('Correlation plots for %s'%model_name, fontsize=14, fontweight='bold')
#         #plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
#         fig.savefig(os.path.join(plot_directory, 'scatterplot_trainingchoiceworld_%s_prevchoice_zv.png'%model_type))
#         plt.close()


#         # Posterior Plot #
#         plot_file_name = f"{model_name}_posteriorplot.png"
#         plot_file_path = os.path.join(plot_directory, plot_file_name)
#         #plot_file_path = os.path.join(file_path, plot_directory + "_posteriorplot.pdf")
#         az.style.use("arviz-doc")
#         suffix_to_filter = "participant_id_offset"
#         filtered_vars = [var for var in model.posterior.data_vars if filter_group_level_params(var, suffix_to_filter)]
#         #az.plot_posterior(model, kind="scatter", var_names=filtered_vars)
#         az.plot_posterior(model,var_names=["~participant_id_offset"],filter_vars="like")
#         #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
#         plt.tight_layout()
#         plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
#         plt.savefig(plot_file_path,dpi=300)
#         plt.close()

#         # Trace Plot #
#         plot_file_name = f"{model_name}_traceplot.png"
#         plot_file_path = os.path.join(plot_directory, plot_file_name)
#         az.style.use("arviz-doc")
#         az.plot_trace(model,var_names=["~participant_id_offset"],filter_vars="like")
#         #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
#         plt.tight_layout()
#         plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
#         plt.savefig(plot_file_path,dpi=300)
#         plt.close()
        
#         # Forest Plot #
#         plot_file_name = f"{model_name}_forestplot.png"
#         plot_file_path = os.path.join(plot_directory, plot_file_name)
#         az.style.use("arviz-doc")
#         #az.plot_forest(model, var_names=["~z"]) 
#         az.plot_forest(model,var_names=["~participant_id_offset"],filter_vars="like")
#         #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
#         plt.tight_layout()
#         plt.figtext(0.5, 0.01, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=8, va='bottom')
#         plt.savefig(plot_file_path,dpi=300,bbox_inches='tight',pad_inches=0.1)
#         plt.close()

#         # Pair Plot #
#         plot_file_name = f"{model_name}_pair_plot.png"
#         plot_file_path = os.path.join(plot_directory, plot_file_name)
#         #reattach(file_path, model, elife_data)
#         az.style.use("arviz-doc")
#         az.plot_pair(model,var_names=["~participant_id_offset"],filter_vars="like",kind="kde")
#         #plt.figure(figsize=(10, 6))  # Adjust dimensions as needed
#         plt.tight_layout()
#         plt.figtext(0.5, 0.95, "Excluded participants = %s"%excluded_participants, ha='center', fontsize=60, va='top')
#         plt.savefig(plot_file_path,dpi=300)
#         plt.close()

#         # Posterior Pred Plot #
#         #plot_file_name = f"{model_name}_ppc_plot.png"
#         #plot_file_path = os.path.join(plot_directory, plot_file_name)
#         #reattach(file_path, model, elife_data_excluded)
#         #az.style.use("arviz-doc")
#         #pm.sample_posterior_predictive(model) #### before saving nc file
#         #az.plot_ppc(model)
#         #plt.tight_layout()
#         #plt.savefig(plot_file_path,dpi=300)
#         #plt.close()

# %%
#%matplotlib inline

def load_and_preprocess_data(file_path):
    mouse_data = pd.read_csv(file_path)
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    mouse_data['stimrepeat'] = np.where(mouse_data['signed_contrast'].abs() == mouse_data['prevcontrast'], 1, 0)
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    return mouse_data

def get_prep(data):
    return data.groupby(['participant_id'])[['stimrepeat','repeat']].mean().reset_index()

def extract_prevresp_values(file_path):
    results = pd.read_csv(file_path)
    v_prevresp = results.loc[results['index'] == 'v_prevresp', 'mean'].values[0]
    z_prevresp = results.loc[results['index'] == 'z_prevresp', 'mean'].values[0]
    return v_prevresp, z_prevresp

def process_summary_files(summary_dir):
    summary_files = glob.glob(os.path.join(summary_dir, "res_subject_*_samples_summary.csv"))
    prevresp_values = {}
    for file_path in summary_files:
        participant_id = int(os.path.basename(file_path).split('_')[2])
        v_prevresp, z_prevresp = extract_prevresp_values(file_path)
        prevresp_values[participant_id] = {'v_prevresp': v_prevresp, 'z_prevresp': z_prevresp}
    return pd.DataFrame.from_dict(prevresp_values, orient='index').reset_index().rename(columns={'index': 'participant_id'})

def corrfunc(x, y, ax, color):
    r, p = sp.stats.spearmanr(x, y, nan_policy='omit')
    ax.scatter(x, y, color=color, alpha=0.5)
    ax.annotate(f"r = {r:.3f}\np = {p:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=8, ha='left', va='top')

def plot_results(merged_data, fig_file_path=None):
    pal = sns.color_palette("Paired")
    pal2 = pal[2:4] + pal[0:2] + pal[8:10]

    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(6,3))

    corrfunc(x=merged_data.z_prevresp, y=merged_data.repeat, ax=ax[0], color=pal2[1])
    ax[0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')

    ax[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    corrfunc(x=merged_data.v_prevresp, y=merged_data.repeat, ax=ax[1], color=pal2[3])
    ax[1].set(xlabel='History shift in v', ylabel='P(repeat)')

    tstat, pval = dependent_corr(
        sp.stats.spearmanr(merged_data.z_prevresp, merged_data.repeat, nan_policy='omit')[0],
        sp.stats.spearmanr(merged_data.v_prevresp, merged_data.repeat, nan_policy='omit')[0],
        sp.stats.spearmanr(merged_data.z_prevresp, merged_data.v_prevresp, nan_policy='omit')[0],
        len(merged_data),
        twotailed=True, conf_level=0.95, method='steiger'
    )
    deltarho = sp.stats.spearmanr(merged_data.z_prevresp, merged_data.repeat, nan_policy='omit')[0] - \
            sp.stats.spearmanr(merged_data.v_prevresp, merged_data.repeat, nan_policy='omit')[0]

    fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho) if pval < 0.0001 else r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)

    sns.despine(trim=True)
    plt.tight_layout()
    
    # Save the figure if a file path is provided
    if fig_file_path:
        fig_name = 'history_shifts_correlation.png'
        fig_path = os.path.join(fig_file_path, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fig_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
    mouse_data_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', 'ibl_trainingChoiceWorld_20240715.csv')
    summary_dir = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/estimation_mice_batch_1_20240717/est_laspse_M4_mice"

    mouse_data = load_and_preprocess_data(mouse_data_path)
    prep = get_prep(mouse_data)
    prevresp_df = process_summary_files(summary_dir)
    merged_data = pd.merge(prep, prevresp_df, on='participant_id', how='left')

    print("Data statistics:")
    print(f"Number of NaN rt values: {mouse_data['rt'].isna().sum()}")
    print(f"Number of negative rt values: {(mouse_data['rt'] < 0).sum()}")
    print("\nMerged data head:")
    print(merged_data.head())

    plot_results(merged_data, fig_file_path)
    
if __name__ == "__main__":
    main()
