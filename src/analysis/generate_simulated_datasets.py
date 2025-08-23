"""
Mouse Choice History Analysis - Simulated Dataset Generation using HSSM

This script generates simulated datasets for each subject using all fitted DDM models.
For each subject, it fits all models (ddma, ddmb, ddmc, ddmd) and generates posterior
predictive samples from each model, creating comprehensive simulated datasets that
preserve all original trial information while adding model-specific simulations.

NOTE: this reruns the estimation for all the animals. So it can take long
TODO: Create version that calls saved models rather than re-estimate
Author: Kianté Fernandez
Date: 2025-08-22
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time

import pandas as pd
import arviz as az
import hssm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_PATH = Path('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_20250819.csv')
MAX_TRIALS_PER_SESSION = 350
MIN_TRIALS_FOR_ANALYSIS = 100
MAX_RT = 5.0
MIN_RT = 0.08
MAX_MOVEMENT_ONSET = 5.0
MIN_MOVEMENT_ONSET = 0.08

# Model configuration - DDM models only
MODELS_TO_FIT = ['ddma', 'ddmb', 'ddmc', 'ddmd']
SAMPLING_CONFIG = {
    'sampler': 'nuts_numpyro',
    'draws': 400,
    'tune': 1200,
    'chains': 4,
    'cores': 4,
    'target_accept': 0.90
}

# Simulation configuration
N_POSTERIOR_DRAWS = 100  # Number of posterior predictive draws per model
EXCLUDED_MICE = ['']

# Output directory configuration
PROJECT_ROOT = Path('/Users/kiante/Documents/2023_choicehistory_HSSM')
OUTPUT_DIR = PROJECT_ROOT / "results" / "simulated_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS (from estimate_individual_mice.py)
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
    
    # Basic preprocessing - filter for valid trials
    valid_data = mouse_subset[
        (mouse_subset['movement_onset'] < MAX_MOVEMENT_ONSET) & 
        (mouse_subset['movement_onset'] > MIN_MOVEMENT_ONSET) 
    ].copy()
    
    # Recode response to be -1 and 1 rather than 0 and 1 (HSSM requirement)
    valid_data['response'] = valid_data['response'].replace({0: -1, 1: 1})
    
    # Use the pre-computed signed_contrast_squeezed column
    valid_data['signed_contrast'] = valid_data['signed_contrast_squeezed']

    # Create categorical variable for previous response
    valid_data['prevresp_cat'] = valid_data['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    valid_data['prevresp_cat'] = valid_data['prevresp_cat'].astype('category')
    
    # Add original trial index for tracking
    valid_data = valid_data.reset_index(drop=True)
    valid_data['original_trial_id'] = valid_data.index
    
    logger.info(f"Preprocessed {len(valid_data)} trials for mouse {mouse_id}")
    
    # Skip if not enough data
    if len(valid_data) < MIN_TRIALS_FOR_ANALYSIS:
        logger.warning(f"Skipping mouse {mouse_id} - not enough valid data ({len(valid_data)} trials)")
        return None
        
    logger.info(f"Successfully preprocessed {len(valid_data)} trials for mouse {mouse_id}")
    return valid_data

# =============================================================================
# MODEL SPECIFICATION AND FITTING FUNCTIONS (from estimate_individual_mice.py)
# =============================================================================

def create_mouse_specific_models(valid_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create HSSM models for mouse-specific analysis.
    
    Args:
        valid_data: Preprocessed data for the mouse
        
    Returns:
        Dictionary of model name -> HSSM model objects
    """
    models = {}
    
    # Define priors based on modelspec file non-centered specification
    v_priors = {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
        "signed_contrast": {"name": "Normal", "mu": 0.5, "sigma": 0.7},
        "prevresp_cat": {"name": "Normal", "mu": -0.5, "sigma": 1.0}
    }
    
    z_priors = {
        "Intercept": {"name": "Beta", "alpha": 2, "beta": 2},
        "prevresp_cat": {"name": "Normal", "mu": 0.0, "sigma": 0.2}
    }
    
    # Base parameters
    base_specs = [
        {"name": "t", "prior": {"name": "Gamma", "alpha": 2, "beta": 6}},
        {"name": "a", "prior": {"name": "Gamma", "alpha": 2, "beta": 2}},
        {"name": "sv", "prior": {"name": "HalfNormal", "sigma": 2}}
    ]
    
    try:
        for model_name in MODELS_TO_FIT:
            if model_name not in ['ddma', 'ddmb', 'ddmc', 'ddmd']:
                logger.warning(f"Unknown model name: {model_name}")
                continue
                
            base_model = "ddm_sdv"
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
                    "link": "identity",
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
                    "link": "identity",
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
                    "link": "identity",
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
                    "link": "identity",
                    "prior": z_priors
                })
            
            # Add base parameters
            include_specs.extend(base_specs)
            
            # Create the model
            models[model_name] = hssm.HSSM(
                data=valid_data,
                model=base_model,
                loglik_kind="analytical",
                include=include_specs
            )
            
            logger.info(f"Created {model_name} model successfully")
            
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return {}
    
    return models


def fit_models_for_mouse(models: Dict[str, Any], mouse_id: str) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Fit all models for a single mouse.
    
    Args:
        models: Dictionary of model name -> HSSM model objects
        mouse_id: ID of the mouse being processed
        
    Returns:
        Tuple of (fitted_models_dict, summaries_dict)
    """
    fitted_models = {}
    summaries = {}
    
    for model_name, model in models.items():
        logger.info(f"Fitting {model_name} for mouse {mouse_id}")
        try:
            start_time = time.time()
            
            # Sample from the model (traces are automatically stored in model.traces)
            model.sample(**SAMPLING_CONFIG)
            
            end_time = time.time()
            summary = az.summary(model.traces)
            
            # Store fitted model and results
            fitted_models[model_name] = model
            summaries[model_name] = summary
            
            logger.info(f"Successfully fit {model_name} for mouse {mouse_id} in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting {model_name} for mouse {mouse_id}: {str(e)}")
            continue
    
    return fitted_models, summaries

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def generate_posterior_predictive_samples(fitted_models: Dict[str, Any], 
                                         original_data: pd.DataFrame,
                                         mouse_id: str,
                                         n_draws: int = N_POSTERIOR_DRAWS) -> Dict[str, pd.DataFrame]:
    """
    Generate posterior predictive samples for all fitted models.
    
    Args:
        fitted_models: Dictionary of fitted HSSM model objects
        original_data: Original preprocessed data for the mouse
        mouse_id: ID of the mouse
        n_draws: Number of posterior predictive draws per model
        
    Returns:
        Dictionary of model_name -> simulated_data_dataframe
    """
    simulated_datasets = {}
    
    for model_name, model in fitted_models.items():
        logger.info(f"Generating posterior predictive samples for {model_name} (mouse {mouse_id})")
        
        try:
            # Check if posterior predictive already exists
            has_existing_predictions = (hasattr(model.traces, 'posterior_predictive') and 
                                      len(model.traces.posterior_predictive.data_vars) > 0)
            
            if has_existing_predictions:
                logger.info(f"  Found existing posterior_predictive samples, using those")
            else:
                logger.info(f"  No existing predictions found, generating new ones...")
                
                # Try different approaches to avoid dimension mismatch (pattern from working code)
                try:
                    # First attempt: use model's original data for consistency
                    logger.info(f"    Attempt 1: Using model's original data with {n_draws} draws")
                    model.sample_posterior_predictive(
                        idata=None,  # Use model's own traces
                        data=model.data,   # Use model's own data explicitly
                        inplace=True,  # Modify traces in place
                        include_group_specific=True,  # Include participant effects
                        kind='response',  # Get actual rt/response predictions
                        draws=n_draws,  # Limit number of draws
                        safe_mode=True  # Prevent memory issues
                    )
                except Exception as e1:
                    logger.info(f"    Attempt 1 failed: {str(e1)}")
                    try:
                        # Second attempt: use smaller number of draws
                        logger.info(f"    Attempt 2: Using fewer draws ({n_draws//2})")
                        model.sample_posterior_predictive(
                            idata=None,
                            data=None,  # Let HSSM figure out data
                            inplace=True,
                            include_group_specific=True,
                            kind='response',
                            draws=n_draws//2,
                            safe_mode=True
                        )
                    except Exception as e2:
                        logger.info(f"    Attempt 2 failed: {str(e2)}")
                        # Third attempt: minimal configuration
                        logger.info(f"    Attempt 3: Minimal configuration")
                        model.sample_posterior_predictive(
                            kind='response',
                            draws=10,
                            safe_mode=True
                        )
            
            # Debug: Print available variables
            logger.info("  Available posterior_predictive variables:")
            if hasattr(model.traces, 'posterior_predictive'):
                logger.info(f"    {list(model.traces.posterior_predictive.data_vars)}")
            else:
                logger.error("    No posterior_predictive group found!")
                continue
            
            # Extract simulated rt and response from the posterior predictive samples
            # HSSM stores them as combined 'rt,response' variable
            if 'rt,response' in model.traces.posterior_predictive.data_vars:
                # Combined format: need to extract rt and response separately
                combined_data = model.traces.posterior_predictive['rt,response'].values  # Shape: (chains, draws, trials, 2)
                simulated_rts = combined_data[:, :, :, 0]  # rt is first dimension
                simulated_responses = combined_data[:, :, :, 1]  # response is second dimension
            else:
                # Separate format (fallback)
                simulated_rts = model.traces.posterior_predictive['rt'].values  # Shape: (chains, draws, trials)
                simulated_responses = model.traces.posterior_predictive['response'].values  # Shape: (chains, draws, trials)
            
            # Flatten across chains and draws to get all simulations
            # Reshape from (chains, draws, trials) to (chains*draws, trials)
            n_trials = simulated_rts.shape[2]
            simulated_rts_flat = simulated_rts.reshape(-1, n_trials)  # (chains*draws, trials)
            simulated_responses_flat = simulated_responses.reshape(-1, n_trials)  # (chains*draws, trials)
            
            # Create list to store all simulated datasets
            model_simulations = []
            
            for draw_idx in range(simulated_rts_flat.shape[0]):
                # Create a copy of original data for this draw
                sim_data = original_data.copy()
                
                # Add simulated values
                sim_data['rt_sim'] = simulated_rts_flat[draw_idx, :]
                sim_data['response_sim'] = simulated_responses_flat[draw_idx, :]
                
                # Add metadata
                sim_data['model'] = model_name
                sim_data['draw_id'] = draw_idx
                sim_data['mouse_id'] = mouse_id
                
                model_simulations.append(sim_data)
            
            # Combine all draws for this model
            simulated_datasets[model_name] = pd.concat(model_simulations, ignore_index=True)
            
            logger.info(f"Generated {len(model_simulations)} simulation draws for {model_name} (mouse {mouse_id})")
            
        except Exception as e:
            logger.error(f"Error generating posterior predictive samples for {model_name} (mouse {mouse_id}): {str(e)}")
            continue
    
    return simulated_datasets


def aggregate_all_model_simulations(simulated_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate simulated datasets from all models into a single dataframe.
    
    Args:
        simulated_datasets: Dictionary of model_name -> simulated_data_dataframe
        
    Returns:
        Combined dataframe with all simulations
    """
    if not simulated_datasets:
        logger.warning("No simulated datasets to aggregate")
        return pd.DataFrame()
    
    # Combine all model simulations
    all_simulations = list(simulated_datasets.values())
    
    combined_data = pd.concat(all_simulations, ignore_index=True)
    
    logger.info(f"Aggregated {len(combined_data)} total simulated trials across {len(simulated_datasets)} models")
    
    return combined_data


def save_simulated_datasets(combined_data: pd.DataFrame, 
                          original_data: pd.DataFrame,
                          summaries: Dict[str, pd.DataFrame],
                          mouse_id: str) -> None:
    """
    Save simulated datasets and metadata to organized directory structure.
    
    Args:
        combined_data: Combined simulated data across all models
        original_data: Original data for reference
        summaries: Model fitting summaries
        mouse_id: ID of the mouse
    """
    # Create subject-specific directory
    subject_dir = OUTPUT_DIR / mouse_id
    subject_dir.mkdir(exist_ok=True)
    
    # Save combined simulated dataset
    sim_file = subject_dir / f"{mouse_id}_all_models_simulated.csv"
    combined_data.to_csv(sim_file, index=False)
    logger.info(f"Saved combined simulated dataset: {sim_file}")
    
    # Save original data for reference
    orig_file = subject_dir / f"{mouse_id}_original_data.csv"
    original_data.to_csv(orig_file, index=False)
    logger.info(f"Saved original data: {orig_file}")
    
    # Save model summaries
    for model_name, summary in summaries.items():
        summary_file = subject_dir / f"{mouse_id}_{model_name}_model_summary.csv"
        summary.to_csv(summary_file)
    
    # Create metadata file
    metadata = {
        'mouse_id': mouse_id,
        'n_original_trials': len(original_data),
        'n_simulated_trials': len(combined_data),
        'models_fitted': list(summaries.keys()),
        'n_posterior_draws': N_POSTERIOR_DRAWS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = subject_dir / f"{mouse_id}_simulation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved simulation metadata: {metadata_file}")

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_mouse_simulations(mouse_id: str, mouse_data: pd.DataFrame) -> bool:
    """
    Process a single mouse: preprocess data, fit models, generate simulations.
    
    Args:
        mouse_id: ID of the mouse to process
        mouse_data: Full dataset
        
    Returns:
        True if processing successful, False otherwise
    """
    logger.info(f"Starting simulation processing for mouse {mouse_id}")
    
    # Preprocess data
    valid_data = preprocess_mouse_data(mouse_data, mouse_id)
    if valid_data is None:
        return False
    
    # Create models
    models = create_mouse_specific_models(valid_data)
    if not models:
        logger.error(f"Failed to create models for mouse {mouse_id}")
        return False
    
    # Fit models
    fitted_models, summaries = fit_models_for_mouse(models, mouse_id)
    if not fitted_models:
        logger.error(f"Failed to fit any models for mouse {mouse_id}")
        return False
    
    # Generate posterior predictive samples
    simulated_datasets = generate_posterior_predictive_samples(fitted_models, valid_data, mouse_id)
    if not simulated_datasets:
        logger.error(f"Failed to generate simulations for mouse {mouse_id}")
        return False
    
    # Aggregate all model simulations
    combined_data = aggregate_all_model_simulations(simulated_datasets)
    if combined_data.empty:
        logger.error(f"Failed to aggregate simulations for mouse {mouse_id}")
        return False
    
    # Save results
    save_simulated_datasets(combined_data, valid_data, summaries, mouse_id)
    
    logger.info(f"Successfully completed simulation processing for mouse {mouse_id}")
    return True


def main() -> None:
    """
    Main execution function.
    """
    logger.info("Starting simulated dataset generation")
    
    # Load and filter data
    mouse_data, included_mice = load_and_filter_data()
    #pick just two
    # included_mice = mouse_data['subj_idx'].unique()[:2]
    # Process each mouse
    successful_mice = 0
    failed_mice = []
    
    for mouse_id in included_mice:
        logger.info(f"Processing mouse {mouse_id} ({successful_mice + 1}/{len(included_mice)})")
        
        try:
            success = process_mouse_simulations(mouse_id, mouse_data)
            if success:
                successful_mice += 1
                logger.info(f"Successfully processed mouse {mouse_id}")
            else:
                failed_mice.append(mouse_id)
                logger.warning(f"Failed to process mouse {mouse_id}")
        except Exception as e:
            logger.error(f"Unexpected error processing mouse {mouse_id}: {str(e)}")
            failed_mice.append(mouse_id)
    
    # Summary
    logger.info(f"Simulation generation complete!")
    logger.info(f"Successfully processed: {successful_mice}/{len(included_mice)} mice")
    if failed_mice:
        logger.info(f"Failed mice: {failed_mice}")
    
    # Save global summary
    global_summary = {
        'total_mice': len(included_mice),
        'successful_mice': successful_mice,
        'failed_mice': failed_mice,
        'models_fitted': MODELS_TO_FIT,
        'n_posterior_draws': N_POSTERIOR_DRAWS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_file = OUTPUT_DIR / "simulation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(global_summary, f, indent=2)
    
    logger.info(f"Global summary saved: {summary_file}")


if __name__ == "__main__":
    main()