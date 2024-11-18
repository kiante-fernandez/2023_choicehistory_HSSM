# estimate_vi_mouse_choice_history.py - history test M4 with pymc model specifications
# with parameter optimization throught variational inference
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2024/11/17      Kianté Fernandez<kiantefernan@gmail.com>   coded version single subject

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import bambi as bmb
import hssm
from matplotlib import gridspec
from pymc.blocking import DictToArrayBijection, RaveledVars
import xarray as xr

# Set floating point precision
hssm.set_floatX("float32")

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def tracker_to_idata(tracker, model):
    """
    Convert a tracker object into an InferenceData object.
    
    Args:
        tracker: The tracker object containing optimization history
        model: The PyMC model
    
    Returns:
        az.InferenceData: The converted inference data
    """
    tracker_groups = list(tracker.whatchdict.keys())
    stacked_results = {
        tracker_group: {
            key: np.stack([d[key] for d in tracker[tracker_group]])
            for key in tracker[tracker_group][0]
        }
        for tracker_group in tracker_groups
    }

    var_to_dims = {
        var.name: ("vi_step", *(model.named_vars_to_dims.get(var.name, ())))
        for var in model.continuous_value_vars
    }
    datasets = {
        key: xr.Dataset(
            {
                var: (var_to_dims[var], stacked_results[key][var])
                for var in stacked_results[key].keys()
            }
        )
        for key in tracker_groups
    }

    with warnings.catch_warnings(action="ignore"):
        return az.InferenceData(**datasets)

def untransform_params(idata, model):
    """
    Transform parameters back to their original scale.
    
    Args:
        idata: InferenceData object containing parameter values
        model: The PyMC model
    
    Returns:
        xr.Dataset: Dataset with untransformed parameters
    """
    suffixes = ["_interval__", "_log__"]

    def remove_suffixes(word, suffixes):
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    free_rv_names = [rv_.name for rv_ in model.free_RVs]
    transformed_vars = list(idata.mean.data_vars.keys())
    collect_untransformed_vars = []
    collect_untransformed_xarray_datasets = []

    for var_ in transformed_vars:
        var_untrans = remove_suffixes(var_, suffixes=suffixes)
        if var_untrans in free_rv_names:
            rv = model.free_RVs[free_rv_names.index(var_untrans)]
            if model.rvs_to_transforms[rv] is not None:
                untransformed_var = (
                    model.rvs_to_transforms[rv]
                    .backward(idata.mean[var_].values, *rv.owner.inputs)
                    .eval()
                )
                collect_untransformed_vars.append(var_)
                collect_untransformed_xarray_datasets.append(
                    xr.Dataset(
                        data_vars={var_untrans: (("vi_step"), untransformed_var)}
                    )
                )

    return xr.merge([idata.mean] + collect_untransformed_xarray_datasets).drop_vars(
        collect_untransformed_vars
    )

def plot_vi_traces(idata):
    """
    Plot parameter history of the optimization algorithm.
    
    Args:
        idata: InferenceData object containing optimization history
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if not isinstance(idata, az.InferenceData):
        raise ValueError("idata must be an InferenceData object")
    if "loss" not in idata.groups():
        raise ValueError("InferenceData object must contain a 'loss' group")
    if "mean_untransformed" not in idata.groups():
        print("Using transformed variables because 'mean_untransformed' group not found")
        data_vars = list(idata["mean"].data_vars.keys())
    else:
        data_vars = list(idata["mean_untransformed"].data_vars.keys())

    fig = plt.figure(figsize=(8, 1.5 * len(data_vars)))
    gs = gridspec.GridSpec(
        len(data_vars) // 2 + 2
        if (len(data_vars) % 2) == 0
        else (len(data_vars) // 2) + 3,
        2,
    )

    for i, var_ in enumerate(data_vars):
        ax_tmp = fig.add_subplot(gs[i // 2, i % 2])
        idata["mean_untransformed"][var_].plot(ax=ax_tmp)
        ax_tmp.set_title(var_)

    last_ax = fig.add_subplot(gs[-2:, :])
    idata["loss"].loss.plot(ax=last_ax)
    gs.tight_layout(fig)
    return fig

def load_and_prepare_data(file_path):
    """
    Load and prepare mouse data for analysis.
    
    Args:
        file_path: Path to the mouse data CSV file
    
    Returns:
        pd.DataFrame: Prepared dataset
    """
    mouse_data = pd.read_csv(file_path)
    
    # Data preparation steps
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    
    # Clean data
    print(f"Number of NaN rt values: {mouse_data['rt'].isna().sum()}")
    print(f"Number of negative rt values: {(mouse_data['rt'] < 0).sum()}")
    
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    return mouse_data

def create_subject_model(subject_data):
    """Create HSSM model for a subject."""
    return hssm.HSSM(
        data=subject_data,
        model="angle",
        loglik_kind="approx_differentiable",
        p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.50},
        lapse=bmb.Prior("Uniform", lower=0.0, upper=30.0),
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                    "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
                    "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
                },
                "formula": "v ~ signed_contrast + prevresp",
                "link": "identity"
            },
            {
                "name": "z",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                    "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
                },
                "formula": "z ~ 1 + prevresp"
            },
        ]
    )

def fit_subject_model(subject_model):
    """Fit the HSSM model for a subject."""
    with subject_model.pymc_model:
        advi = pm.FullRankADVI()
    
    # Setup initial point
    start = subject_model.pymc_model.initial_point()
    vars_dict = {var.name: var for var in subject_model.pymc_model.continuous_value_vars}
    x0 = DictToArrayBijection.map(
        {var_name: value for var_name, value in start.items() if var_name in vars_dict}
    )
    
    # Setup tracker
    tracker = pm.callbacks.Tracker(
        mean=lambda: DictToArrayBijection.rmap(
            RaveledVars(advi.approx.mean.eval(), x0.point_map_info), start
        ),
        std=lambda: DictToArrayBijection.rmap(
            RaveledVars(advi.approx.std.eval(), x0.point_map_info), start
        ),
    )
    
    # Fit model
    approx = advi.fit(n=30000, callbacks=[tracker])
    vi_posterior_samples = approx.sample(1000)
    #drop p_outlier
    #vi_posterior_samples.posterior = vi_posterior_samples.posterior.drop_vars("p_outlier")
    
    # Process results
    result = tracker_to_idata(tracker, subject_model.pymc_model)
    result.add_groups({
        "mean_untransformed": untransform_params(result.copy(), subject_model.pymc_model)
    })
    result.add_groups({
        "loss": xr.Dataset(data_vars={"loss": ("vi_step", np.array(approx.hist))})
    })
    
    return result, vi_posterior_samples

def save_subject_results(subject_id, result, vi_posterior_samples, output_folder):
    """
    Save analysis results for a subject, filtering out trailing indices.
    Only saves core parameters: v_Intercept, v_prevresp, v_signed_contrast, and non-indexed parameters.
    """
    # Get summary statistics
    res_summary = az.summary(vi_posterior_samples.posterior).reset_index()
    
    # Filter out parameters with trailing indices and keep only core parameters
    mask = (~res_summary['index'].str.match(r'.*\[\d+\]$')) | (res_summary['index'].isin(['v_Intercept', 'v_prevresp', 'v_signed_contrast']))
    filtered_summary = res_summary[mask]
    
    # Save filtered summary statistics
    summary_filename = os.path.join(output_folder, f"res_subject_{subject_id}_angle_FullRankADVI_summary.csv")
    filtered_summary.to_csv(summary_filename, index=False)
    
    # Save convergence plot
    fig = plot_vi_traces(result)
    plot_filename = os.path.join(output_folder, f"subject_{subject_id}_convergence_plot.png")
    plt.savefig(plot_filename)
    plt.close()

def main():
    """Main execution function."""
    # Setup paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    mouse_data_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 
                                  'data', 'ibl_trainingChoiceWorld_clean_20241003.csv')
    output_folder = "subject_plots"
    ensure_dir(output_folder)
    
    # Load and prepare data
    dataset = load_and_prepare_data(mouse_data_path)
    
    # Select relevant columns
    columns_to_use = ['participant_id', 'rt', 'response', 'signed_contrast', 'prevresp']
    dataset = dataset[columns_to_use]
    
    # Process each subject
    for subject_id in dataset['participant_id'].unique():
        print(f"Processing subject {subject_id}")
        
        # Filter data for current subject
        subject_data = dataset[dataset['participant_id'] == subject_id]
        
        # Create and configure model
        subject_model = create_subject_model(subject_data)
        
        # Fit model and process results
        result, vi_posterior_samples = fit_subject_model(subject_model)
        
        # Save results
        save_subject_results(subject_id, result, vi_posterior_samples, output_folder)

if __name__ == "__main__":
    main()