import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
import scipy as sp
import hssm
from ..utils.utils_hssm_modelspec import make_model
from ..utils.utils_plot import corrfunc, dependent_corr

MODEL_NAMES = [
    "ddm_nohist",
    "ddm_prevresp_v",
    "ddm_prevresp_z",
    "ddm_prevresp_zv"
]

DIRECTORY = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/models'
PLOT_DIRECTORY = "/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures"
MOUSE_DATA_PATH = '/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv'

def load_and_preprocess_data(file_path):
    mouse_data = pd.read_csv(file_path)
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    # selected_subjects = ['CSHL052', 'CSHL059', 'CSHL060', 'CSH_ZAD_019', 'KS046', 'PL037', 'SWC_058', 'UCLA036', 'UCLA048', 'ZFM-01936']
    # mouse_data = mouse_data[mouse_data['subj_idx'].isin(selected_subjects)]
    
    mouse_data['stimrepeat'] = np.where(mouse_data.signed_contrast == mouse_data.signed_contrast.shift(1), 1, 0)
    
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


def extract_posterior_predictions(model, n_draws=None):
    """
    Extract posterior predictive samples from HSSM/DDM model output.
    
    Parameters:
    -----------
    model : HSSM model
        The fitted model
    n_draws : int, optional
        Number of posterior draws to use per chain. If None, uses all draws.
    
    Returns:
    --------
    DataFrame with columns: observation, chain, draw, rt, repeat
    """
    # Debug prints
    print(f"\nModel info:")
    print(f"Model type: {type(model)}")
    print(f"Available variables in posterior_predictive: {list(model.traces.posterior_predictive.data_vars)}")
    print(f"Sample values from predictions:")
    # Get the posterior predictive samples
    traces = model.traces.posterior_predictive['rt,response']
    
    # Convert to numpy array for faster processing
    array_data = traces.values
    
    print(f"Shape of predictions: {array_data.shape}")
    print(f"First few values: {array_data[0,0,0:5]}")
    # Get dimensions
    n_chains = array_data.shape[0]
    total_draws = array_data.shape[1]
    n_obs = array_data.shape[2]
    
    # Select draws if specified
    if n_draws is not None:
        draws_per_chain = min(n_draws, total_draws)
        draw_indices = np.random.choice(total_draws, draws_per_chain, replace=False)
        array_data = array_data[:, draw_indices]
    else:
        draws_per_chain = total_draws
    
    # Create DataFrame with all predictions
    draws_list = []
    for chain in range(n_chains):
        for draw in range(draws_per_chain):
            draw_data = array_data[chain, draw]
            df = pd.DataFrame({
                'observation': range(n_obs),
                'chain': chain,
                'draw': draw,
                'rt': draw_data[:, 0],
                'repeat': draw_data[:, 1]
            })
            draws_list.append(df)
    
    return pd.concat(draws_list, ignore_index=True)

def prepare_for_conditional_plot(posterior_df, original_data, max_draws=100):
    """
    Prepare posterior predictive samples for conditional history plotting.
    
    Parameters:
    -----------
    posterior_df : DataFrame
        Output from extract_posterior_predictions
    original_data : DataFrame
        Original data with rt and participant_id columns
    max_draws : int, optional
        Maximum number of draws to use per observation
    
    Returns:
    --------
    tuple: (pred_df, data_df) containing processed prediction and data DataFrames
    """
    # Sample draws if needed
    if max_draws is not None:
        draws_per_obs = len(posterior_df) // len(original_data)
        if draws_per_obs > max_draws:
            sampled_indices = []
            for obs in posterior_df['observation'].unique():
                obs_indices = posterior_df[posterior_df['observation'] == obs].index
                sampled_indices.extend(np.random.choice(obs_indices, max_draws, replace=False))
            posterior_df = posterior_df.loc[sampled_indices]
    
    # Get participant IDs
    mapping_df = pd.DataFrame({
        'observation': range(len(original_data)),
        'participant_id': original_data['participant_id'].values
    })
    
    # Merge predictions with participant info and transform coding
    pred_df = posterior_df.merge(mapping_df, on='observation')
    pred_df['repeat'] = (pred_df['repeat'] + 1) / 2  # Transform from [-1,1] to [0,1]
    
    # Prepare data
    data_df = original_data.copy()
    
    return pred_df, data_df

def plot_conditional_history(pred_df, data_df, n_quantiles=5):
    """
    Create conditional history plot comparing model predictions to data.
    
    Parameters:
    -----------
    pred_df : DataFrame
        Processed model predictions
    data_df : DataFrame
        Original data
    n_quantiles : int
        Number of RT quantiles for binning
    
    Returns:
    --------
    matplotlib figure
    """
    # Create RT quantiles
    pred_df['rt_quantile'] = pd.qcut(pred_df['rt'], q=n_quantiles, labels=False)
    data_df['rt_quantile'] = pd.qcut(data_df['rt'], q=n_quantiles, labels=False)
    
    # Compute subject-level means
    grouped_pred = pred_df.groupby(['participant_id', 'rt_quantile'])['repeat'].mean().reset_index()
    grouped_data = data_df.groupby(['participant_id', 'rt_quantile'])['repeat'].mean().reset_index()
    
    # Compute summary statistics
    summary_pred = grouped_pred.groupby('rt_quantile').agg({
        'repeat': ['mean', 'sem']
    }).reset_index()
    summary_pred.columns = ['rt_quantile', 'mean_repeat', 'sem_repeat']
    
    summary_data = grouped_data.groupby('rt_quantile').agg({
        'repeat': ['mean', 'sem']
    }).reset_index()
    summary_data.columns = ['rt_quantile', 'mean_repeat', 'sem_repeat']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot individual subject lines
    for subject in grouped_data['participant_id'].unique():
        # Data lines (light gray)
        subject_data = grouped_data[grouped_data['participant_id'] == subject]
        ax.plot(subject_data['rt_quantile'], subject_data['repeat'], 
                color='gray', alpha=0.1, linewidth=0.5, zorder=1)
        
        # Model lines (light blue)
        subject_pred = grouped_pred[grouped_pred['participant_id'] == subject]
        ax.plot(subject_pred['rt_quantile'], subject_pred['repeat'], 
                color='blue', alpha=0.2, linewidth=1.0, zorder=2)
    
    # Plot mean with error bars
    ax.errorbar(summary_data['rt_quantile'], summary_data['mean_repeat'],
                yerr=summary_data['sem_repeat'], 
                fmt='-o', color='black', ecolor='black', 
                capsize=5, linewidth=2, markersize=8,
                label='Data', zorder=3)
    
    ax.errorbar(summary_pred['rt_quantile'], summary_pred['mean_repeat'],
                yerr=summary_pred['sem_repeat'], 
                fmt='-o', color='blue', ecolor='blue', 
                capsize=5, linewidth=3, markersize=10,
                label='Model', zorder=4)
    
    # Customize plot
    ax.set_xlabel('RT Quantile')
    ax.set_ylabel('P(repeat)')
    ax.set_title('Conditional Bias Function')
    ax.set_xticks(range(n_quantiles))
    ax.set_xticklabels([f'Q{i+1}' for i in range(n_quantiles)])
    ax.set_ylim(0.45, 0.60)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend()
    
    sns.despine(trim=True)
    plt.tight_layout()
    
    return fig

def plot_model_comparison(model, original_data, n_draws=5, n_quantiles=5):
    """
    Convenience function to create model comparison plot from raw model output.
    
    Parameters:
    -----------
    model : HSSM/DDM model instance
        The fitted model
    original_data : DataFrame
        Original data
    n_draws : int
        Number of draws to use per chain
    n_quantiles : int
        Number of RT quantiles for binning
    
    Returns:
    --------
    matplotlib figure
    """
    # Extract predictions
    predictions = extract_posterior_predictions(model, n_draws=n_draws)
    
    # Prepare data
    pred_df, data_df = prepare_for_conditional_plot(predictions, original_data)
    
    # Create plot
    return plot_conditional_history(pred_df, data_df, n_quantiles=n_quantiles)
        
def process_model(model, prep, dataset_name, model_name, mouse_data):
    """Process model and create all plots."""
    print(f"\nProcessing model: {model_name}")
    
    print("  Computing model summary...")
    results = model.summary()

    print("  Extracting participant-specific parameters...")
    pattern_v = r'v_prevresp\|participant_id_offset\[\d+\]'
    pattern_z = r'z_prevresp\|participant_id_offset\[\d+\]'

    subj_idx_specific_rows_v = results[results.index.str.contains(pattern_v, na=False, regex=True)].reset_index()
    subj_idx_specific_rows_z = results[results.index.str.contains(pattern_z, na=False, regex=True)].reset_index()

    subj_idx_df_v = subj_idx_specific_rows_v[['index', 'mean']].rename(columns={'mean': 'v_mean'}).reset_index(drop=True)
    subj_idx_df_z = subj_idx_specific_rows_z[['index', 'mean']].rename(columns={'mean': 'z_mean'}).reset_index(drop=True)

    merged_df_v = pd.concat([subj_idx_df_v, prep.reset_index(drop=True)], axis=1)
    merged_df_z = pd.concat([subj_idx_df_z, prep.reset_index(drop=True)], axis=1)

    print("  Creating plots...")
    print("    - Scatter plot...")
    plot_path = os.path.join(PLOT_DIRECTORY, f'scatterplot_{dataset_name}_{model_name}_prevchoice_zv.png')
    create_scatter_plot(merged_df_z, merged_df_v, plot_path)

    az.style.use("arviz-doc")
    # for plot_type in ['posterior', 'trace', 'forest', 'pair']:
    for plot_type in ['posterior', 'trace']:
        print(f"    - {plot_type} plot...")
        plot_file_name = f"{dataset_name}_{model_name}_{plot_type}plot.png"
        plot_file_path = os.path.join(PLOT_DIRECTORY, plot_file_name)
        
        if plot_type == 'posterior':
            az.plot_posterior(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        elif plot_type == 'trace':
            az.plot_trace(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        # elif plot_type == 'forest':
        #     az.plot_forest(model.traces, var_names=["~participant_id_offset"], filter_vars="like")
        # elif plot_type == 'pair':
        #     az.plot_pair(model.traces, var_names=["~participant_id_offset"], filter_vars="like", kind="kde", marginals=True)
        
        plt.tight_layout()
        plt.savefig(plot_file_path, dpi=300)
        plt.close()

    # print("    - Posterior predictive check...")
    # plot_file_name = f"{dataset_name}_{model_name}_posterior_predictive_check.png"
    # plot_file_path = os.path.join(PLOT_DIRECTORY, plot_file_name)
    # hssm.plotting.plot_posterior_predictive(model, col="participant_id", col_wrap=4, range=(-3,3))
    # plt.tight_layout()
    # plt.savefig(plot_file_path, dpi=300)
    # plt.close()
    
    # print("    - Conditional bias plot...")
    # print("      Extracting posterior predictions...")
    # plot_file_name = f"{dataset_name}_{model_name}_conditional_bias.png"
    # plot_file_path = os.path.join(PLOT_DIRECTORY, plot_file_name)
    # model.sample_posterior_predictive()
    # predictions = extract_posterior_predictions(model, n_draws=1000)
    # print("      Preparing data for plotting...")
    # pred_df, data_df = prepare_for_conditional_plot(predictions, mouse_data, max_draws=100)
    # fig = plot_conditional_history(pred_df, data_df, n_quantiles=5)
    # plt.tight_layout()
    # plt.savefig(plot_file_path, dpi=300)
    # plt.close()
    
    print(f"Completed processing for {model_name}\n")

def main():
    print("\nStarting analysis pipeline...")
    
    print("\nLoading and preprocessing data...")
    mouse_data = load_and_preprocess_data(MOUSE_DATA_PATH)
    dataset_name = os.path.splitext(os.path.basename(MOUSE_DATA_PATH))[0]
    print(f"Loaded {len(mouse_data)} trials from {mouse_data['participant_id'].nunique()} participants")
    
    prep = get_prep(mouse_data)
    columns_to_use = ['subj_idx', 'participant_id', 'rt', 'response', 'signed_contrast', 'prevresp', 'eid']
    dataset = mouse_data[columns_to_use]

    print("\nInitializing models...")
    ddm_models = {name: make_model(dataset, name) for name in MODEL_NAMES}

    print("\nRestoring model traces...")
    for model_name in MODEL_NAMES:
        print(f"  Loading {model_name}...")
        file_path = os.path.join(DIRECTORY, f"{model_name}_model.nc")
        ddm_models[model_name].restore_traces(file_path)

    print("\nProcessing models and generating plots...")
    for i, (model_name, model) in enumerate(ddm_models.items(), 1):
        print(f"\nProcessing model {i}/{len(MODEL_NAMES)}: {model_name}")
        process_model(model, prep, dataset_name, model_name, mouse_data)

    print("\nAnalysis complete! All plots have been saved to:", PLOT_DIRECTORY)

if __name__ == "__main__":
    main()