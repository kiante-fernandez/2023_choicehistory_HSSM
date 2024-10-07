import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
import scipy as sp
import hssm
from utils_hssm_modelspec import make_model
from utils_plot import corrfunc, dependent_corr

MODEL_NAMES = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv"
]
DIRECTORY = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/models'
PLOT_DIRECTORY = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures"
MOUSE_DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20240908.csv'

def load_and_preprocess_data(file_path):
    mouse_data = pd.read_csv(file_path)
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    selected_subjects = ['CSHL052', 'CSHL059', 'CSHL060', 'CSH_ZAD_019', 'KS046', 'PL037', 'SWC_058', 'UCLA036', 'UCLA048', 'ZFM-01936']
    mouse_data = mouse_data[mouse_data['subj_idx'].isin(selected_subjects)]
    
    mouse_data['stimrepeat'] = np.where(mouse_data.signed_contrast == mouse_data.signed_contrast.shift(1), 1, 0)
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    
    return mouse_data

def get_prep(data):
    return data.groupby(['subj_idx'])[['stimrepeat', 'repeat']].mean().reset_index()

def create_scatter_plot(merged_df_z, merged_df_v, plot_path):
    fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(6,3))
    pal = sns.color_palette("Paired")
    pal2 = pal[2:4] + pal[0:2] + pal[8:10]

    corrfunc(x=merged_df_z.z_mean, y=merged_df_z.repeat, ax=ax[0], color=pal2[1])
    ax[0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')

    corrfunc(x=merged_df_v.v_mean, y=merged_df_v.repeat, ax=ax[1], color=pal2[3])
    ax[1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1)
    ax[1].set(xlabel='History shift in v', ylabel='P(repeat)')

    tstat, pval = dependent_corr(
        sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0],
        sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0],
        sp.stats.spearmanr(merged_df_z.z_mean, merged_df_v.v_mean, nan_policy='omit')[0],
        len(merged_df_z),
        twotailed=True, conf_level=0.95, method='steiger'
    )
    deltarho = sp.stats.spearmanr(merged_df_z.z_mean, merged_df_z.repeat, nan_policy='omit')[0] - \
            sp.stats.spearmanr(merged_df_v.v_mean, merged_df_v.repeat, nan_policy='omit')[0]

    fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho) if pval < 0.0001 else r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)

    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close()

def process_model(model, prep, dataset_name, model_name):
    results = model.summary()

    pattern_v = r'v_prevresp\|participant_id_offset\[\d+\]'
    pattern_z = r'z_prevresp\|participant_id_offset\[\d+\]'

    subj_idx_specific_rows_v = results[results.index.str.contains(pattern_v, na=False, regex=True)].reset_index()
    subj_idx_specific_rows_z = results[results.index.str.contains(pattern_z, na=False, regex=True)].reset_index()

    subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']].rename(columns={'mean': 'v_mean'}).reset_index(drop=True)
    subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']].rename(columns={'mean': 'z_mean'}).reset_index(drop=True)

    merged_df_v = pd.concat([subj_idx_df_v, prep.reset_index(drop=True)], axis=1)
    merged_df_z = pd.concat([subj_idx_df_z, prep.reset_index(drop=True)], axis=1)

    # Create scatter plot
    plot_path = os.path.join(PLOT_DIRECTORY, f'scatterplot_{dataset_name}_{model_name}_prevchoice_zv.png')
    create_scatter_plot(merged_df_z, merged_df_v, plot_path)

    # Create other plots
    az.style.use("arviz-doc")

    for plot_type in ['posterior', 'trace', 'forest', 'pair']:
        plot_file_name = f"{dataset_name}_{model_name}_{plot_type}plot.png"
        plot_file_path = os.path.join(PLOT_DIRECTORY, plot_file_name)
        
        if plot_type == 'posterior':
            az.plot_posterior(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        elif plot_type == 'trace':
            az.plot_trace(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        elif plot_type == 'forest':
            az.plot_forest(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        elif plot_type == 'pair':
            az.plot_pair(model.traces, var_names=["~participant_id_offset"], filter_vars="like", kind="kde", marginals=True)
        
        plt.tight_layout()
        plt.savefig(plot_file_path, dpi=300)
        plt.close()

    # Posterior Predictive Check
    plot_file_name = f"{dataset_name}_{model_name}_posterior_predictive_check.png"
    plot_file_path = os.path.join(PLOT_DIRECTORY, plot_file_name)
    hssm.plotting.plot_posterior_predictive(model, col="participant_id", col_wrap=4, range=(-3,3))
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

def main():
    # Load and preprocess data
    mouse_data = load_and_preprocess_data(MOUSE_DATA_PATH)
    dataset_name = os.path.splitext(os.path.basename(MOUSE_DATA_PATH))[0]
    prep = get_prep(mouse_data)

    # Select columns
    columns_to_use = ['subj_idx', 'participant_id', 'rt', 'response', 'signed_contrast', 'prevresp', 'eid']
    dataset = mouse_data[columns_to_use]

    # Create models
    ddm_models = {name: make_model(dataset, name) for name in MODEL_NAMES}

    # Restore traces for each model
    for model_name in MODEL_NAMES:
        file_path = os.path.join(DIRECTORY, f"{model_name}_model.nc")
        ddm_models[model_name].restore_traces(file_path)

    # Process each model
    for model_name, model in ddm_models.items():
        process_model(model, prep, dataset_name, model_name)

if __name__ == "__main__":
    main()