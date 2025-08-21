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

def generate_ppc_plots(model, modelname, ppc_group_dir, ppc_subject_dir, timestamp, 
                      n_samples=10, subject_level=True):
    """
    Generate comprehensive posterior predictive check plots.
    
    Parameters:
    -----------
    model : hssm.HSSM
        The fitted HSSM model object
    modelname : str
        Name of the model for file naming
    ppc_group_dir : str
        Directory to save group-level PPC plots
    ppc_subject_dir : str
        Directory to save subject-level PPC plots
    timestamp : str
        Timestamp string for unique file naming
    n_samples : int, default=10
        Number of samples to use for PPC generation
    subject_level : bool, default=True
        Whether to generate subject-level plots
    """
    print(f"Generating PPC plots with {n_samples} samples...")
    
    # Group-level PPC plots
    try:
        print("  Generating group-level PPC plot...")
        
        # Generate the plot
        fig = model.plot_posterior_predictive(n_samples=n_samples, range=(-3, 3))
        
        # Enhance the plot styling
        plt.suptitle(f"Posterior Predictive Check - {modelname.replace('_', ' ').title()}", 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save as both PNG and PDF
        group_png = os.path.join(ppc_group_dir, f'{modelname}_{timestamp}_group_ppc.png')
        group_pdf = os.path.join(ppc_group_dir, f'{modelname}_{timestamp}_group_ppc.pdf')
        
        plt.savefig(group_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(group_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"    ✓ Group-level PPC saved: {group_png} and {group_pdf}")
        
    except Exception as e:
        print(f"    ✗ Failed to generate group-level PPC plot: {e}")
        plt.close('all')  # Ensure no hanging figures
    
    # Subject-level PPC plots
    if subject_level:
        try:
            print("  Generating subject-level PPC plots...")
            
            # Generate subject-level plot
            fig = model.plot_posterior_predictive(
                n_samples=n_samples, 
                col="participant_id", 
                col_wrap=3,
                range=(-3, 3)
            )
            
            # Enhance the plot styling
            if hasattr(fig, 'figure'):
                fig.figure.suptitle(f"Posterior Predictive Check by Participant - {modelname.replace('_', ' ').title()}", 
                                   fontsize=16, y=1.02)
            
            # Save as both PNG and PDF
            subject_png = os.path.join(ppc_subject_dir, f'{modelname}_{timestamp}_subject_ppc.png')
            subject_pdf = os.path.join(ppc_subject_dir, f'{modelname}_{timestamp}_subject_ppc.pdf')
            
            if hasattr(fig, 'savefig'):
                fig.savefig(subject_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                fig.savefig(subject_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
            else:
                plt.savefig(subject_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.savefig(subject_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            plt.close()
            
            print(f"    ✓ Subject-level PPC saved: {subject_png} and {subject_pdf}")
            
        except Exception as e:
            print(f"    ✗ Failed to generate subject-level PPC plot: {e}")
            print("    Continuing without subject-level plots...")
            plt.close('all')  # Ensure no hanging figures


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


def run_model(data, modelname, mypath, trace_id=0, sampling_method="mcmc", plot_traces=True, 
             plot_ppc=True, ppc_n_samples=10, ppc_subject_level=True,
             vi_niter=100000, vi_method="fullrank_advi", vi_optimizer="adamax", 
             vi_learning_rate=0.01, vi_scheduler=None, scheduler_params=None,
             vi_grad_clip=None, vi_convergence_tolerance=None, vi_convergence_every=None,
             vi_min_iterations=None, 
             # Pathfinder parameters
             pathfinder_num_paths=8, pathfinder_num_draws=2000, pathfinder_num_single_draws=2000,
             pathfinder_max_lbfgs_iters=2000, **kwargs):
    """
    Run HSSM model with trace plotting, PPC generation, and unique timestamped file naming.
    
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
        Sampling method: "mcmc", "vi", or "pathfinder"
    plot_traces : bool, default=True
        Whether to generate and save trace plots
    plot_ppc : bool, default=True
        Whether to generate posterior predictive check plots
    ppc_n_samples : int, default=10
        Number of samples to use for PPC generation
    ppc_subject_level : bool, default=True
        Whether to generate subject-level PPC plots in addition to group-level plots
    vi_niter : int, default=100000
        Number of iterations for VI (when sampling_method="vi")
    vi_method : str, default="fullrank_advi"
        VI method: "advi" or "fullrank_advi" (recommended)
    vi_optimizer : str, default="adamax"
        Optimizer for VI: "adamax" (recommended), "adam", "adagrad", "sgd"
    vi_learning_rate : float, default=0.01
        Learning rate for VI optimizer (0.01 recommended for adamax)
    vi_scheduler : str, optional
        Learning rate scheduler type: "plateau", "step", "exponential", or None
    scheduler_params : dict, optional
        Parameters for the learning rate scheduler
    vi_grad_clip : float, optional
        Gradient clipping constraint to prevent exploding gradients (PyMC total_grad_norm_constraint)
        Typical values: 1.0-5.0, or None to disable clipping
    vi_convergence_tolerance : float, optional
        Convergence tolerance for CheckParametersConvergence callback
        Higher values (e.g., 0.01) are more lenient for noisy datasets. Default: 0.001
    vi_convergence_every : int, optional
        How often to check convergence (iterations). Higher values check less frequently.
        Default: 100
    vi_min_iterations : int, optional
        Minimum iterations before convergence checking starts. Ensures models run at least
        this many iterations before early stopping is allowed. Default: None
    pathfinder_num_paths : int, default=4
        Number of parallel Pathfinder chains to run (when sampling_method="pathfinder")
    pathfinder_num_draws : int, default=1000
        Number of posterior samples to draw per path (when sampling_method="pathfinder")
    pathfinder_num_single_draws : int, default=1000
        Number of samples from single best path (when sampling_method="pathfinder")
    pathfinder_max_lbfgs_iters : int, default=1000
        Maximum L-BFGS iterations for optimization (when sampling_method="pathfinder")
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
    PPC_dir = os.path.join(results_base, 'PPC')

    print(f'Project root: {project_root}')
    print(f'Results base: {results_base}')
    
    # Create directories if they don't exist
    for directory in [models_dir, comparisons_dir, traces_dir, PPC_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'Created directory {directory}')
    
    # Create subdirectories for PPC plots
    ppc_group_dir = os.path.join(PPC_dir, 'group_level')
    ppc_subject_dir = os.path.join(PPC_dir, 'subject_level')
    if plot_ppc:
        for ppc_dir in [ppc_group_dir, ppc_subject_dir]:
            if not os.path.exists(ppc_dir):
                os.makedirs(ppc_dir)
                print(f'Created PPC directory {ppc_dir}')

    sampling_params = {
        "sampler": kwargs.get('sampler', 'nuts_numpyro'),
        "chains": kwargs.get('chains', 4),
        "cores": kwargs.get('cores', 4),
        "draws": kwargs.get('draws', 1000),
        "tune": kwargs.get('tune', 1000),
        "idata_kwargs": kwargs.get('idata_kwargs', dict(log_likelihood=True))  # return log likelihood
    }
    
    # Add initvals if provided
    if 'initvals' in kwargs:
        sampling_params['initvals'] = kwargs['initvals']
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
        m.save_model(model_name = f'{modelname}_{timestamp}_model')
        print(f'Model saved: {model_file}')
        
        # Generate comprehensive PPC plots if requested
        if plot_ppc:
            generate_ppc_plots(
                model=m, 
                modelname=modelname, 
                ppc_group_dir=ppc_group_dir, 
                ppc_subject_dir=ppc_subject_dir, 
                timestamp=timestamp,
                n_samples=ppc_n_samples, 
                subject_level=ppc_subject_level
            )
        
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

        # save useful output - only main parameters, not deterministic variables
        print("saving summary stats")
        import arviz as az
        
        # Filter to only include the main model parameters (exclude deterministic trial-wise variables)
        if hasattr(inference_data, 'posterior'):
            # Get list of variables, filtering out trial-wise deterministic variables
            var_names = []
            for var_name in inference_data.posterior.data_vars:
                # Include only parameters that are not trial-wise (exclude __obs__ dimension)
                if '__obs__' not in inference_data.posterior[var_name].dims:
                    var_names.append(var_name)
            
            print(f"Including {len(var_names)} main parameters (excluding {len(inference_data.posterior.data_vars) - len(var_names)} trial-wise variables)")
            
            # Generate summary only for main parameters
            if var_names:
                results = az.summary(inference_data, var_names=var_names).reset_index()
            else:
                print("Warning: No main parameters found, using all variables")
                results = az.summary(inference_data).reset_index()
        else:
            results = az.summary(inference_data).reset_index()
            
        results_file = os.path.join(mypath, f'{modelname}_{timestamp}_results_combined.csv')
        results.to_csv(results_file)
        print(f'Results summary saved: {results_file} ({len(results)} parameters)') 
        
    elif sampling_method == "vi":
        import arviz as az
        # Run VI sampling
        # Configure VI optimizer based on HSSM best practices
        import pymc as pm
        import pytensor
        from .utils_hssm_schedulers import create_scheduler
        
        # Create shared learning rate variable for scheduler support
        learning_rate_shared = pytensor.shared(vi_learning_rate, name='learning_rate')
        
        # Set up optimizer with configurable parameters using shared learning rate
        if vi_optimizer.lower() == "adamax":
            optimizer = pm.adamax(learning_rate=learning_rate_shared)
        elif vi_optimizer.lower() == "adam":
            optimizer = pm.adam(learning_rate=learning_rate_shared)
        elif vi_optimizer.lower() == "adagrad":
            optimizer = pm.adagrad(learning_rate=learning_rate_shared)
        elif vi_optimizer.lower() == "sgd":
            optimizer = pm.sgd(learning_rate=learning_rate_shared)
        else:
            print(f"Warning: Optimizer '{vi_optimizer}' not recognized, using adamax (recommended)")
            optimizer = pm.adamax(learning_rate=learning_rate_shared)
        
        # Set up callbacks with configurable convergence checking
        convergence_tolerance = vi_convergence_tolerance if vi_convergence_tolerance is not None else 0.001
        convergence_every = vi_convergence_every if vi_convergence_every is not None else 100
        min_iterations = vi_min_iterations if vi_min_iterations is not None else 0
        
        # Create a custom convergence checker that respects minimum iterations
        class MinIterationCheckParametersConvergence(CheckParametersConvergence):
            def __init__(self, min_iter, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.min_iter = min_iter
                
            def __call__(self, approx, loss_hist, i):
                # Only check convergence after minimum iterations
                if i >= self.min_iter:
                    super().__call__(approx, loss_hist, i)
        
        callbacks = [MinIterationCheckParametersConvergence(
            min_iterations,
            diff='absolute',
            tolerance=convergence_tolerance,
            every=convergence_every
        )]
        
        print(f"Convergence check: tolerance={convergence_tolerance}, every={convergence_every} iterations, min_iter={min_iterations}")
        
        # Add learning rate scheduler if specified
        scheduler_callback = None
        if vi_scheduler is not None:
            if scheduler_params is None:
                scheduler_params = {}
            
            try:
                scheduler_callback = create_scheduler(vi_scheduler, learning_rate_shared, **scheduler_params)
                callbacks.append(scheduler_callback)
                print(f"Using {vi_scheduler} learning rate scheduler with params: {scheduler_params}")
            except Exception as e:
                print(f"Warning: Could not create scheduler '{vi_scheduler}': {e}")
                print("Continuing without scheduler...")
        
        # Set up VI kwargs with optional gradient clipping
        vi_kwargs = {
            "niter": vi_niter,
            "method": vi_method,
            "obj_optimizer": optimizer,
            "callbacks": callbacks
        }
        
        # Add initial values if provided
        if 'initvals' in kwargs:
            vi_kwargs["start"] = kwargs['initvals']
            print(f"Using custom initial values for VI with {len(kwargs['initvals'])} parameters")
        
        # Add gradient clipping if specified
        if vi_grad_clip is not None:
            vi_kwargs["total_grad_norm_constraint"] = vi_grad_clip
            grad_clip_info = f"grad_clip: {vi_grad_clip}"
        else:
            grad_clip_info = "no grad_clip"
        
        scheduler_info = f"scheduler: {vi_scheduler}" if vi_scheduler else "no scheduler"
        print(f"Running VI with {vi_niter} iterations, method: {vi_method}, optimizer: {vi_optimizer}, lr: {vi_learning_rate}, {scheduler_info}, {grad_clip_info}")
        
        # Run VI with configurable parameters and convergence monitoring
        m.vi(ignore_mcmc_start_point_defaults = True, **vi_kwargs)
        
        # Sample from VI approximation
        m.vi_approx.sample(draws=1000)

        #az_data = az.from_pymc3(vi_samples)
        
        # Save the VI loss history plot for convergence diagnostics
        if scheduler_callback is not None and hasattr(scheduler_callback, 'get_lr_history'):
            # Create dual-axis plot with loss and learning rate
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot loss history
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(m.vi_approx.hist, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Plot learning rate history
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Learning Rate', color=color)
            lr_history = scheduler_callback.get_lr_history()
            if len(lr_history) > 0:
                ax2.plot(lr_history, color=color, alpha=0.7)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_yscale('log')  # Log scale for learning rate
            
            plt.title(f'VI Loss & Learning Rate - {modelname}\n'
                     f'Final Loss: {m.vi_approx.hist[-1]:.2f} | '
                     f'Final LR: {learning_rate_shared.get_value():.2e} | '
                     f'Optimizer: {vi_optimizer} | Scheduler: {vi_scheduler}')
            
            fig.tight_layout()
            hist_file = os.path.join(traces_dir, f'{modelname}_{timestamp}_vi_loss_lr_hist.png')
            plt.savefig(hist_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'VI loss & LR history saved: {hist_file}')
            print(f'Final VI loss: {m.vi_approx.hist[-1]:.2f}')
            print(f'Final learning rate: {learning_rate_shared.get_value():.2e}')
            
        else:
            # Standard single-axis loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(m.vi_approx.hist)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            final_lr = learning_rate_shared.get_value()
            plt.title(f'VI Loss History - {modelname}\n'
                     f'Final Loss: {m.vi_approx.hist[-1]:.2f} | '
                     f'LR: {final_lr:.2e} | Optimizer: {vi_optimizer}')
            plt.grid(True, alpha=0.3)
            hist_file = os.path.join(traces_dir, f'{modelname}_{timestamp}_vi_loss_hist.png')
            plt.savefig(hist_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'VI loss history saved: {hist_file}')
            print(f'Final VI loss: {m.vi_approx.hist[-1]:.2f}')
        # Extract and save summary statistics - only main parameters, not deterministic variables
        print("saving summary stats")
        import arviz as az
        
        # Filter to only include the main model parameters (exclude deterministic trial-wise variables)
        if hasattr(m.vi_idata, 'posterior'):
            # Get list of variables, filtering out trial-wise deterministic variables
            var_names = []
            for var_name in m.vi_idata.posterior.data_vars:
                # Include only parameters that are not trial-wise (exclude __obs__ dimension)
                if '__obs__' not in m.vi_idata.posterior[var_name].dims:
                    var_names.append(var_name)
            
            print(f"Including {len(var_names)} main parameters (excluding {len(m.vi_idata.posterior.data_vars) - len(var_names)} trial-wise variables)")
            
            # Generate summary only for main parameters
            if var_names:
                results = az.summary(m.vi_idata, var_names=var_names).reset_index()
            else:
                print("Warning: No main parameters found, using all variables")
                results = az.summary(m.vi_idata).reset_index()
        else:
            results = az.summary(m.vi_idata).reset_index()
            
        results_file = os.path.join(mypath, f'{modelname}_{timestamp}_results_combined.csv')
        results.to_csv(results_file)
        print(f'VI results summary saved: {results_file} ({len(results)} parameters)') 
        
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
        
    elif sampling_method == "pathfinder":
        print("Running Pathfinder VI...")
        start_time = time.time()
        
        # Import pathfinder functionality
        try:
            import pymc_extras as pmx
        except ImportError:
            raise ImportError(
                "pymc-extras is required for Pathfinder VI. "
                "Install it with: pip install git+https://github.com/pymc-devs/pymc-extras"
            )
        
        # Set up Pathfinder arguments
        pathfinder_args = {
            "num_paths": pathfinder_num_paths,
            "num_draws": pathfinder_num_draws,
            "num_draws_per_path": pathfinder_num_single_draws,  # Correct parameter name
            "maxiter": pathfinder_max_lbfgs_iters,
            # "jitter": pathfinder_jitter,
            # "concurrent": "process"
        }
        
        # Add any additional pathfinder kwargs (but exclude non-pathfinder parameters and duplicates)
        excluded_keys = {'draws', 'chains', 'cores', 'tune', 'sampler', 'max_lbfgs_iters'}  # MCMC-specific parameters and duplicates
        pathfinder_kwargs_keys = [k for k in kwargs.keys() if k.startswith('pathfinder_')]
        for key in pathfinder_kwargs_keys:
            clean_key = key.replace('pathfinder_', '')
            if clean_key not in excluded_keys and clean_key not in pathfinder_args:
                pathfinder_args[clean_key] = kwargs[key]
        
        print(f"Pathfinder settings: {pathfinder_args}")
        
        # Run Pathfinder VI with error handling for broadcasting issues
        with m.pymc_model:
            try:
                inference_data = pmx.fit(method="pathfinder", **pathfinder_args)
            except (ValueError, RuntimeError) as e:
                if any(phrase in str(e) for phrase in ["Runtime broadcasting not allowed", "broadcasting error", "distinct dimension length"]):
                    print("WARNING: Encountered broadcasting error during trace conversion.")
                    print("This is a known issue with Pathfinder on complex hierarchical models.")
                    print("Attempting fallback with reduced num_draws_per_path...")
                    
                    # Try with reduced draws per path to avoid broadcasting issues
                    fallback_args = pathfinder_args.copy()
                    fallback_args["num_draws_per_path"] = min(250, fallback_args.get("num_draws_per_path", 1000))
                    fallback_args["num_draws"] = min(500, fallback_args.get("num_draws", 2000))
                    fallback_args["num_paths"] = min(4, fallback_args.get("num_paths", 8))
                    
                    print(f"Fallback Pathfinder settings: {fallback_args}")
                    
                    try:
                        inference_data = pmx.fit(method="pathfinder", **fallback_args)
                        print("Fallback successful!")
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
                        print("Attempting conservative Pathfinder configuration...")
                        
                        # Very conservative configuration as last resort
                        conservative_args = {
                            "num_paths": 4,
                            "num_draws": 200,
                            "num_draws_per_path": 100,
                            "maxiter": 300
                        }
                        
                        print(f"Conservative Pathfinder settings: {conservative_args}")
                        try:
                            inference_data = pmx.fit(method="pathfinder", **conservative_args)
                            print("Conservative configuration successful!")
                        except Exception as e3:
                            print(f"Conservative configuration also failed: {e3}")
                            print("Pathfinder VI appears to be incompatible with this model complexity.")
                            print("Consider using MCMC (nuts) or standard VI (advi) instead.")
                            # Return None to indicate failure
                            return None
                else:
                    raise e
        
        # Check if inference was successful
        if inference_data is None:
            print("Pathfinder VI failed - returning None")
            return None
            
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Pathfinder VI completed in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
        # Compute log likelihood for model comparison
        print("Computing log likelihood...")
        try:
            import pymc as pm
            with m.pymc_model:
                pm.compute_log_likelihood(inference_data)
        except Exception as e:
            print(f"Warning: Could not compute log likelihood: {e}")
        
        # Save the model
        print('Saving Pathfinder model...')
        model_file = os.path.join(models_dir, f'{modelname}_pathfinder_{timestamp}_model.nc')
        inference_data.to_netcdf(model_file)
        print(f'Pathfinder model saved: {model_file}')
        
        # Generate PPC plots if requested
        if plot_ppc:
            print("Generating PPC plots...")
            try:
                # Set inference object for plotting
                m._inference_obj = inference_data
                generate_ppc_plots(
                    model=m,
                    modelname=f"{modelname}_pathfinder",
                    ppc_group_dir=ppc_group_dir,
                    ppc_subject_dir=ppc_subject_dir,
                    timestamp=timestamp,
                    n_samples=ppc_n_samples,
                    subject_level=ppc_subject_level
                )
            except Exception as e:
                print(f"Warning: Could not generate PPC plots: {e}")
        
        # Generate trace plots if requested
        if plot_traces:
            print("Generating trace plots...")
            try:
                plot_traces_func(inference_data, f"{modelname}_pathfinder", traces_dir, timestamp)
            except Exception as e:
                print(f"Warning: Could not generate trace plots: {e}")
        
        # Save model comparison metrics
        print("Save model comparison indices")
        df = {}
        
        if 'log_likelihood' in inference_data.groups():
            try:
                import arviz as az
                df['waic'] = az.waic(inference_data).elpd_waic
                df['loo'] = az.loo(inference_data).elpd_loo
            except Exception as e:
                print(f"Warning: Could not calculate WAIC/LOO: {e}")
                df['waic'] = 'N/A'
                df['loo'] = 'N/A'
        else:
            df['waic'] = 'N/A'
            df['loo'] = 'N/A'
        
        # Add Pathfinder-specific metrics
        df['method'] = 'pathfinder'
        df['num_paths'] = pathfinder_num_paths
        df['num_draws'] = pathfinder_num_draws
        df['runtime_seconds'] = runtime
        
        df2 = pd.DataFrame(list(df.items()), columns=['Metric', 'Value'])
        comparison_file = os.path.join(comparisons_dir, f'{modelname}_pathfinder_{timestamp}_model_comparison.csv')
        df2.to_csv(comparison_file)
        print(f'Pathfinder model comparison saved: {comparison_file}')
        
        # Save summary statistics - only main parameters, not deterministic variables
        print("Saving summary stats")
        import arviz as az
        
        # Filter to only include the main model parameters (exclude deterministic trial-wise variables)
        if hasattr(inference_data, 'posterior'):
            # Get list of variables, filtering out trial-wise deterministic variables
            var_names = []
            for var_name in inference_data.posterior.data_vars:
                # Include only parameters that are not trial-wise (exclude __obs__ dimension)
                if '__obs__' not in inference_data.posterior[var_name].dims:
                    var_names.append(var_name)
            
            print(f"Including {len(var_names)} main parameters (excluding {len(inference_data.posterior.data_vars) - len(var_names)} trial-wise variables)")
            
            # Generate summary only for main parameters
            if var_names:
                results = az.summary(inference_data, var_names=var_names).reset_index()
            else:
                print("Warning: No main parameters found, using all variables")
                results = az.summary(inference_data).reset_index()
        else:
            results = az.summary(inference_data).reset_index()
            
        results_file = os.path.join(mypath, f'{modelname}_pathfinder_{timestamp}_results_combined.csv')
        results.to_csv(results_file)
        print(f'Pathfinder results summary saved: {results_file} ({len(results)} parameters)')
        
        # Store inference data in model object for compatibility
        m._inference_obj = inference_data
        
    else:
        raise ValueError("Unsupported sampling method. Use 'mcmc', 'vi', or 'pathfinder'.")
    
    return m


