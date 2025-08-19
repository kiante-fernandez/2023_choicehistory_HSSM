"""
Mouse Choice History Analysis using Hierarchical Sequential Sampling Models (HSSM)

This script analyzes mouse choice behavior data using various DDM and Angle models
to understand how previous responses influence current decisions.

Author: Kianté Fernandez
Date: 2025-08-19
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import logging
from pathlib import Path
import json
import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import hssm
import bambi as bmb

# Import utility functions from project
try:
    from ..utils.utils_hssm import run_model
    from ..utils.utils_hssm_modelspec import make_model
except ImportError:
    # Handle case when running as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.utils_hssm import run_model
    from utils.utils_hssm_modelspec import make_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_PATH = Path('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
MAX_TRIALS_PER_SESSION = 350
MIN_TRIALS_FOR_ANALYSIS = 100
MAX_RT = 5.0
MIN_RT = 0.08
MAX_MOVEMENT_ONSET = 5.0
MIN_MOVEMENT_ONSET = 0.08

# Model configuration - All 8 model variants
MODELS_TO_FIT = ['ddma', 'ddmb', 'ddmc', 'ddmd', 'anglea', 'angleb', 'anglec', 'angled']
SAMPLING_CONFIG = {
    'draws': 200,
    'tune': 1000,
    'chains': 3,
    'cores': 3,
    'target_accept': 0.95
}

# Mice to exclude from analysis
EXCLUDED_MICE = [
    'CSHL059', 'CSHL060', 'CSHL_015', 'CSH_ZAD_017', 'DY_018', 
    'KS043', 'KS044', 'KS045', 'KS046', 'KS086', 'KS091', 'MFD_07', 
    'NR_0020', 'NYU-47', 'PL015', 'PL016', 'PL037', 'PL050', 'SWC_022', 
    'SWC_038', 'SWC_058', 'UCLA033', 'UCLA048', 'ZFM-01577', 'ZM_1898',
    'PL024', 'PL031', 'SWC_021', 'ZFM-01935', 'ZFM-04308', 'CSHL045', 
    'CSHL052', 'CSH_ZAD_022', 'CSH_ZAD_024', 'DY_008', 'DY_020', 'NR_0027',
    'CSHL049', 'CSHL053', 'CSHL047', 'KS017', 'KS094', 'NR_0019', 'ZM_2245', 
    'ibl_witten_27', 'DY_014', 'KS084', 'NYU-11', 'NYU-37', 'SWC_061', 
    'UCLA011', 'UCLA014', 'ZFM-02369', 'ibl_witten_14', 'ibl_witten_16',
    'CSHL054', 'SWC_053', 'SWC_054', 'UCLA017', 'ZFM-01592', 'UCLA017', 
    'ibl_witten_13', 'SWC_066'
]

# Output directory configuration
PROJECT_ROOT = Path('/Users/kiante/Documents/2023_choicehistory_HSSM')
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "mouse_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)
(OUTPUT_DIR / "summaries").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "traces").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "posterior_predictive").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "model_comparison").mkdir(exist_ok=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

def load_and_filter_data() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load mouse data and filter to included mice.
    
    Returns:
        Tuple of (filtered_data, list_of_included_mice)
    """
    logger.info(f"Loading data from {DATA_PATH}")
    mouse_data = pd.read_csv(DATA_PATH)
    
    # Limit to first N trials per session
    mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(
        lambda group: group.head(MAX_TRIALS_PER_SESSION)
    ).reset_index(drop=True)
    
    # Filter data to exclude the specified mice
    all_mouse_ids = mouse_data_limited['subj_idx'].unique()
    included_mice = [m for m in all_mouse_ids if m not in EXCLUDED_MICE]
    
    logger.info(f"Found {len(all_mouse_ids)} mice total")
    logger.info(f"Excluding {len(EXCLUDED_MICE)} mice")
    logger.info(f"Including {len(included_mice)} mice in analysis")
    
    return mouse_data_limited, included_mice


def preprocess_mouse_data(mouse_data: pd.DataFrame, mouse_id: str) -> Optional[pd.DataFrame]:
    """
    Preprocess data for a single mouse.
    
    Args:
        mouse_data: Full dataset
        mouse_id: ID of the mouse to process
        
    Returns:
        Preprocessed data for the mouse, or None if insufficient data
    """
    logger.info(f"Processing mouse {mouse_id}")
    
    # Get data for this mouse
    mouse_subset = mouse_data[mouse_data['subj_idx'] == mouse_id].copy()
    
    # Basic preprocessing
    valid_data = mouse_subset.dropna(subset=['movement_onset', 'rt', 'prevresp', 'signed_contrast', 'response'])
    valid_data = valid_data[
        (valid_data['movement_onset'] < MAX_MOVEMENT_ONSET) & 
        (valid_data['rt'] < MAX_RT) &
        (valid_data['movement_onset'] > MIN_MOVEMENT_ONSET) & 
        (valid_data['rt'] > MIN_RT)
    ]
    
    # Recode response to be -1 and 1 rather than 0 and 1
    valid_data['response'] = valid_data['response'].replace({0: -1, 1: 1})
    valid_data['prevresp'] = valid_data['prevresp'].replace({0: -1, 1: 1})
    
    # Create scaled signed contrast (following hierarchical script pattern)
    # Clip and divide by 100 to preserve original contrast values while scaling appropriately
    valid_data['squeezed_signed_contrast'] = valid_data['signed_contrast'].clip(upper=25, lower=-25)
    valid_data['scaled_signed_contrast'] = valid_data['squeezed_signed_contrast'] / 100
    valid_data['signed_contrast'] = valid_data['scaled_signed_contrast']
    
    # Create categorical variable for previous response
    valid_data['prevresp_cat'] = valid_data['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    valid_data['prevresp_cat'] = valid_data['prevresp_cat'].astype('category')
    
    logger.info(f"Preprocessed {len(valid_data)} trials for mouse {mouse_id}")
    logger.info(f"Scaled contrast range: [{valid_data['signed_contrast'].min():.3f}, {valid_data['signed_contrast'].max():.3f}]")
    
    # Skip if not enough data
    if len(valid_data) < MIN_TRIALS_FOR_ANALYSIS:
        logger.warning(f"Skipping mouse {mouse_id} - not enough valid data ({len(valid_data)} trials)")
        return None
        
    logger.info(f"Successfully preprocessed {len(valid_data)} trials for mouse {mouse_id}")
    return valid_data

# =============================================================================
# MODEL SPECIFICATION AND FITTING FUNCTIONS
# =============================================================================

def create_mouse_specific_models(valid_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create HSSM models for mouse-specific analysis.
    
    Creates all 8 model variants (DDM and Angle × a,b,c,d) with proper priors
    based on the utility function specifications, adapted for single-mouse analysis.
    
    Args:
        valid_data: Preprocessed data for the mouse
        
    Returns:
        Dictionary of model name -> HSSM model objects
    """
    models = {}
    
    # Define priors based on modelspec file (adapted for single-mouse)
    v_priors = {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
        "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
        "prevresp_cat": {"name": "Normal", "mu": 0.0, "sigma": 1.0}
    }
    
    z_priors = {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.3},
        "prevresp_cat": {"name": "Normal", "mu": 0.0, "sigma": 0.5}
    }
    
    theta_prior = {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.5}
    }
    
    # Base parameters (same for all models) - no link for simple parameters
    base_specs = [
        {"name": "t", "prior": {"name": "Normal", "mu": -1.9, "sigma": 0.3}},
        {"name": "a", "prior": {"name": "Normal", "mu": -0.1, "sigma": 0.3}}
    ]
    
    # Standard lapse and outlier priors
    lapse = bmb.Prior("Uniform", lower=0.0, upper=5.0)
    
    try:
        for model_name in MODELS_TO_FIT:
            if model_name not in ['ddma', 'ddmb', 'ddmc', 'ddmd', 'anglea', 'angleb', 'anglec', 'angled']:
                logger.warning(f"Unknown model name: {model_name}")
                continue
                
            base_model = "ddm" if model_name.startswith("ddm") else "angle"
            variant = model_name[-1]  # Get the variant (a, b, c, d)
            
            # Build include specifications based on variant
            include_specs = []
            
            # Add v specification based on variant
            if variant == 'a':  # Only contrast affecting drift rate
                include_specs.append({
                    "name": "v", 
                    "formula": "v ~ 1 + signed_contrast", 
                    "link": "identity",
                    "prior": {k: v for k, v in v_priors.items() if k in ["Intercept", "signed_contrast"]}
                })
                include_specs.append({
                    "name": "z", 
                    "formula": "z ~ 1", 
                    "link": "logit",
                    "prior": {"Intercept": z_priors["Intercept"]}
                })
            elif variant == 'b':  # Contrast + previous response affecting drift rate
                include_specs.append({
                    "name": "v", 
                    "formula": "v ~ 1 + prevresp_cat + signed_contrast", 
                    "link": "identity",
                    "prior": v_priors
                })
                include_specs.append({
                    "name": "z", 
                    "formula": "z ~ 1", 
                    "link": "logit",
                    "prior": {"Intercept": z_priors["Intercept"]}
                })
            elif variant == 'c':  # Contrast affecting drift + previous response affecting starting point
                include_specs.append({
                    "name": "v", 
                    "formula": "v ~ 1 + signed_contrast", 
                    "link": "identity",
                    "prior": {k: v for k, v in v_priors.items() if k in ["Intercept", "signed_contrast"]}
                })
                include_specs.append({
                    "name": "z", 
                    "formula": "z ~ 1 + prevresp_cat", 
                    "link": "logit",
                    "prior": z_priors
                })
            elif variant == 'd':  # Both contrast + previous response affecting drift AND starting point
                include_specs.append({
                    "name": "v", 
                    "formula": "v ~ 1 + prevresp_cat + signed_contrast", 
                    "link": "identity",
                    "prior": v_priors
                })
                include_specs.append({
                    "name": "z", 
                    "formula": "z ~ 1 + prevresp_cat", 
                    "link": "logit",
                    "prior": z_priors
                })
            
            # Add base parameters
            include_specs.extend(base_specs)
            
            # Add theta for angle models
            if base_model == "angle":
                include_specs.append({
                    "name": "theta", 
                    "formula": "theta ~ 1", 
                    "link": "identity",
                    "prior": theta_prior
                })
            
            # Set p_outlier based on model type
            p_outlier_alpha = 6
            p_outlier_beta = 24 if base_model == "ddm" else 26
            p_outlier = {"name": "Beta", "alpha": p_outlier_alpha, "beta": p_outlier_beta}
            
            # Create the model
            models[model_name] = hssm.HSSM(
                data=valid_data,
                model=base_model,
                loglik_kind="approx_differentiable",
                include=include_specs,
                lapse=lapse,
                p_outlier=p_outlier
            )
            
            logger.info(f"Created {model_name} model successfully")
            
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return {}
    
    return models


def get_default_init_values(model_name: str = None) -> Dict[str, np.ndarray]:
    """
    Get default initialization values for model parameters.
    
    Args:
        model_name: Name of the model to get init values for
    
    Returns:
        Dictionary of parameter names to initial values
    """
    # Base initialization values that work for all models
    base_init = {
        'a': np.array(1.0),              # boundary separation
        't': np.array(0.1),              # non-decision time
        'p_outlier': np.array(0.05),     # outlier probability
    }
    
    # Only include theta for angle models
    if model_name and model_name.startswith('angle'):
        base_init['theta'] = np.array(0.0)
    
    return base_init

def fit_models_for_mouse(models: Dict[str, Any], mouse_id: str) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Fit all models for a single mouse.
    
    Args:
        models: Dictionary of model name -> HSSM model objects
        mouse_id: ID of the mouse being processed
        
    Returns:
        Tuple of (traces_dict, summaries_dict)
    """
    traces = {}
    summaries = {}
    
    for model_name, model in models.items():
        logger.info(f"Fitting {model_name} for mouse {mouse_id}")
        try:
            start_time = time.time()
            # Sample without custom initvals to avoid parameter name mismatches
            traces[model_name] = model.sample(
                **SAMPLING_CONFIG
            )
            
            end_time = time.time()
            summaries[model_name] = az.summary(traces[model_name])
            
            # Save the summary
            summary_file = OUTPUT_DIR / "summaries" / f"{mouse_id}_{model_name}_summary.csv"
            summaries[model_name].to_csv(summary_file)
            
            # Generate and save trace plot
            plt.figure(figsize=(12, 8))
            az.plot_trace(traces[model_name])
            plt.tight_layout()
            trace_file = OUTPUT_DIR / "plots" / "traces" / f"{mouse_id}_{model_name}_trace.png"
            plt.savefig(trace_file)
            plt.close()
            
            # Generate and save posterior predictive plot
            plt.figure(figsize=(10, 6))
            ax = hssm.plotting.plot_posterior_predictive(model, range=(-2, 2))
            sns.despine()
            ax.set_ylabel("")
            plt.title(f"{model_name} Posterior Predictive Plot for {mouse_id}")
            plt.tight_layout()
            pp_file = OUTPUT_DIR / "plots" / "posterior_predictive" / f"{mouse_id}_{model_name}_pp.png"
            plt.savefig(pp_file)
            plt.close()
            
            logger.info(f"Successfully fit {model_name} for mouse {mouse_id} in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting {model_name} for mouse {mouse_id}: {str(e)}")
            continue
    
    return traces, summaries


def perform_model_comparison(traces: Dict[str, Any], summaries: Dict[str, pd.DataFrame], 
                           mouse_id: str) -> Optional[Dict[str, Any]]:
    """
    Perform model comparison for a single mouse.
    
    Args:
        traces: Dictionary of model traces
        summaries: Dictionary of model summaries
        mouse_id: ID of the mouse
        
    Returns:
        Dictionary with comparison results, or None if comparison failed
    """
    if len(traces) < 2:
        logger.warning(f"Not enough models fit for mouse {mouse_id} to perform comparison")
        return None
        
    try:
        # Compare models
        comparison = az.compare({model_name: trace for model_name, trace in traces.items()})
        
        # Save comparison results
        comparison_file = OUTPUT_DIR / "summaries" / f"{mouse_id}_model_comparison.csv"
        comparison.to_csv(comparison_file)
        
        # Plot model comparison
        plt.figure(figsize=(10, 6))
        az.plot_compare(comparison)
        plt.tight_layout()
        comparison_plot_file = OUTPUT_DIR / "plots" / "model_comparison" / f"{mouse_id}_model_comparison.png"
        plt.savefig(comparison_plot_file)
        plt.close()
        
        # Extract results
        winning_model = comparison.index[0]
        best_model_summary = summaries[winning_model]
        
        mouse_results = {
            'winning_model': winning_model,
            'model_comparison': comparison.to_dict(),
            'best_model_summary': best_model_summary.to_dict()
        }
        
        # Extract parameter estimates
        mouse_results.update(extract_parameter_estimates(best_model_summary))
        
        logger.info(f"Model comparison complete for mouse {mouse_id}. Winning model: {winning_model}")
        return mouse_results
        
    except Exception as e:
        logger.error(f"Error in model comparison for mouse {mouse_id}: {str(e)}")
        return None


def extract_parameter_estimates(summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key parameter estimates from model summary.
    
    Args:
        summary: Model summary DataFrame
        
    Returns:
        Dictionary with extracted parameters
    """
    results = {}
    
    # Extract key parameters
    key_params = {}
    for param in summary.index:
        # Extract all v, z, and theta parameters
        if param.startswith(('v_', 'z_', 'theta_')):
            key_params[param] = {
                'mean': summary.loc[param, 'mean'],
                'sd': summary.loc[param, 'sd'],
                'hdi_3%': summary.loc[param, 'hdi_3%'],
                'hdi_97%': summary.loc[param, 'hdi_97%']
            }
    
    if key_params:
        results['key_params'] = key_params
    
    # Extract previous response parameters specifically
    prev_resp_params = {}
    for param in summary.index:
        if 'prevresp_cat[prev_left]' in param or 'prevresp_cat[prev_right]' in param:
            prev_resp_params[param] = {
                'mean': summary.loc[param, 'mean'],
                'sd': summary.loc[param, 'sd'],
                'hdi_3%': summary.loc[param, 'hdi_3%'],
                'hdi_97%': summary.loc[param, 'hdi_97%']
            }
    
    if prev_resp_params:
        results['prev_resp_params'] = prev_resp_params
    
    # Extract signed contrast parameter
    contrast_params = {}
    for param in summary.index:
        if 'signed_contrast' in param:
            contrast_params[param] = {
                'mean': summary.loc[param, 'mean'],
                'sd': summary.loc[param, 'sd'],
                'hdi_3%': summary.loc[param, 'hdi_3%'],
                'hdi_97%': summary.loc[param, 'hdi_97%']
            }
    
    if contrast_params:
        results['contrast_params'] = contrast_params
    
    return results


def process_mouse(mouse_id: str, mouse_data: pd.DataFrame) -> bool:
    """
    Process a single mouse: preprocess data, fit models, and perform comparison.
    
    Args:
        mouse_id: ID of the mouse to process
        mouse_data: Full dataset
        
    Returns:
        True if processing successful, False otherwise
    """
    valid_data = preprocess_mouse_data(mouse_data, mouse_id)
    if valid_data is None:
        return False
    # Create models (no reference contrast needed with continuous variables)
    models = create_mouse_specific_models(valid_data)
    if not models:
        return False
    # Fit models
    traces, summaries = fit_models_for_mouse(models, mouse_id)
    if not traces:
        return False    
    # Perform model comparison
    mouse_results = perform_model_comparison(traces, summaries, mouse_id)
    if mouse_results:
        return mouse_results
    
    return False

# =============================================================================
# GLOBAL ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def initialize_global_results() -> Dict[str, Any]:
    """
    Initialize the global results container.
    
    Returns:
        Empty global results dictionary
    """
    model_names = MODELS_TO_FIT
    return {
        'mice': {},
        'model_wins': {model: 0 for model in model_names},
        'model_ranks': {model: [] for model in model_names},
        'parameter_estimates': {}
    }


def update_global_results(global_results: Dict[str, Any], mouse_id: str, 
                         mouse_results: Dict[str, Any]) -> None:
    """
    Update global results with results from a single mouse.
    
    Args:
        global_results: Global results dictionary to update
        mouse_id: ID of the mouse
        mouse_results: Results dictionary for this mouse
    """
    global_results['mice'][mouse_id] = mouse_results
    
    # Update model wins and ranks
    winning_model = mouse_results['winning_model']
    global_results['model_wins'][winning_model] += 1
    
    comparison = mouse_results['model_comparison']
    if 'rank' in comparison:
        for model, rank in comparison['rank'].items():
            global_results['model_ranks'][model].append(rank)


def generate_global_summary_plots(global_results: Dict[str, Any]) -> None:
    """
    Generate summary plots across all mice.
    
    Args:
        global_results: Global results dictionary
    """
    # Model winning frequency
    model_win_df = pd.DataFrame(global_results['model_wins'].items(), columns=['model', 'wins'])
    model_win_df['proportion'] = model_win_df['wins'] / model_win_df['wins'].sum() if model_win_df['wins'].sum() > 0 else 0
    model_win_df.to_csv(OUTPUT_DIR / "model_wins.csv")
    
    # Plot model win frequency
    if model_win_df['wins'].sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='proportion', data=model_win_df)
        plt.title("Proportion of Mice Best Fit by Each Model")
        plt.ylabel("Proportion")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / "model_win_frequency.png")
        plt.close()
    
    # Average model ranks
    model_ranks = {}
    for model_name, ranks in global_results['model_ranks'].items():
        if ranks:
            model_ranks[model_name] = np.mean(ranks)
    
    if model_ranks:
        model_rank_df = pd.DataFrame(model_ranks.items(), columns=['model', 'avg_rank'])
        model_rank_df = model_rank_df.sort_values('avg_rank')
        model_rank_df.to_csv(OUTPUT_DIR / "model_ranks.csv")
        
        # Plot model ranks
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='avg_rank', data=model_rank_df)
        plt.title("Average Rank of Each Model Across Mice")
        plt.ylabel("Average Rank (lower is better)")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / "model_ranks.png")
        plt.close()
    
    # Generate parameter summary plots
    generate_parameter_summary_plots(global_results)


def generate_parameter_summary_plots(global_results: Dict[str, Any]) -> None:
    """
    Generate summary plots for parameter estimates across mice.
    
    Args:
        global_results: Global results dictionary
    """
    # Drift rate by contrast across mice
    contrast_drift = {}
    for mouse_id, mouse_data in global_results['mice'].items():
        if 'drift_params' in mouse_data:
            for contrast, params in mouse_data['drift_params'].items():
                if contrast not in contrast_drift:
                    contrast_drift[contrast] = []
                contrast_drift[contrast].append(params['mean'])
    
    # Create dataframe for plotting
    drift_data = []
    for contrast, values in contrast_drift.items():
        for value in values:
            drift_data.append({
                'contrast': contrast,
                'drift_rate': value
            })
    
    if drift_data:
        drift_df = pd.DataFrame(drift_data)
        
        # Plot drift rate by contrast across mice
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='contrast', y='drift_rate', data=drift_df)
        plt.title("Drift Rate by Contrast Across Mice")
        plt.ylabel("Drift Rate (v)")
        plt.xlabel("Contrast")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / "drift_by_contrast.png")
        plt.close()
    
    # Previous response effects across mice
    prev_resp_data = []
    for mouse_id, mouse_data in global_results['mice'].items():
        if 'prev_resp_params' in mouse_data:
            for param, values in mouse_data['prev_resp_params'].items():
                if 'v_prevresp_cat' in param:
                    param_type = 'drift'
                elif 'z_prevresp_cat' in param:
                    param_type = 'starting_point'
                else:
                    continue
                    
                if 'prev_left' in param:
                    direction = 'left'
                elif 'prev_right' in param:
                    direction = 'right'
                else:
                    continue
                    
                prev_resp_data.append({
                    'mouse_id': mouse_id,
                    'parameter_type': param_type,
                    'previous_response': direction,
                    'effect': values['mean']
                })
    
    if prev_resp_data:
        prev_resp_df = pd.DataFrame(prev_resp_data)
        
        # Plot previous response effects
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='parameter_type', y='effect', hue='previous_response', data=prev_resp_df)
        plt.title("Previous Response Effects Across Mice")
        plt.ylabel("Parameter Effect")
        plt.xlabel("Parameter Type")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / "prev_resp_effects.png")
        plt.close()


def save_global_results(global_results: Dict[str, Any]) -> None:
    """
    Save global results to JSON file.
    
    Args:
        global_results: Global results dictionary
    """
    def serialize_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(OUTPUT_DIR / "global_results.json", "w") as f:
        json.dump(global_results, f, default=serialize_numpy, indent=2)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main execution function.
    """
    logger.info("Starting mouse choice history analysis")
    
    # Load and filter data
    mouse_data, included_mice = load_and_filter_data()
    
    # Initialize global results
    global_results = initialize_global_results()
    
    # Process each mouse
    successful_mice = 0
    for mouse_id in included_mice:
        logger.info(f"Processing mouse {mouse_id} ({successful_mice + 1}/{len(included_mice)})")
        
        mouse_results = process_mouse(mouse_id, mouse_data)
        if mouse_results:
            update_global_results(global_results, mouse_id, mouse_results)
            successful_mice += 1
            logger.info(f"Successfully processed mouse {mouse_id}")
        else:
            logger.warning(f"Failed to process mouse {mouse_id}")
    
    logger.info(f"Successfully processed {successful_mice}/{len(included_mice)} mice")
    
    # Generate global summary
    if successful_mice > 0:
        logger.info("Generating global summary plots and saving results")
        generate_global_summary_plots(global_results)
        save_global_results(global_results)
        logger.info("Analysis complete!")
    else:
        logger.error("No mice were successfully processed")


if __name__ == "__main__":
    main()
