# hierarchical_initvals.py - Manual initial values for hierarchical HSSM models
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2025/08/20      Kianté Fernandez<kiantefernan@gmail.com>   Initial version for manual initvals

import numpy as np

# Number of subjects in the dataset
N_SUBJECTS = 62

def get_initvals_for_model(model_name, parameterization='noncentered', n_subjects=None):
    """
    Get manual initial values for hierarchical HSSM models.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'ddm_nohist', 'ddm_prevresp_v', etc.)
    parameterization : str
        Either 'centered' or 'noncentered'
    n_subjects : int, optional
        Number of subjects in the dataset. If None, uses N_SUBJECTS default.
        
    Returns:
    --------
    dict
        Dictionary of initial values for the specified model
    """
    
    if parameterization == 'centered':
        return _get_centered_initvals(model_name, n_subjects)
    elif parameterization == 'noncentered':
        return _get_noncentered_initvals(model_name, n_subjects)
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'")

def _get_noncentered_initvals(model_name, n_subjects=None):
    """Get initial values for non-centered parameterization"""
    
    if n_subjects is None:
        n_subjects = N_SUBJECTS
    
    # Base DDM parameters for NON-CENTERED parameterization (actual HSSM structure)
    # Uses Intercept + _sigma + _offset naming convention
    # Values based on fitted results from ddm_nohist, ddm_prevresp_v, ddm_prevresp_z
    base_initvals = {
        # v parameter (drift rate) - fixed effects and random effects
        'v_Intercept': np.array(-0.02, dtype=np.float32),  # Fixed intercept
        'v_signed_contrast': np.array(0.025, dtype=np.float32),  # Fixed contrast effect
        'v_1|participant_id_sigma': np.array(0.07, dtype=np.float32),  # Random effect variance
        'v_1|participant_id_offset': np.random.normal(0, 0.01, n_subjects).astype(np.float32),  # Subject deviations (small)
        'v_signed_contrast|participant_id_sigma': np.array(0.002, dtype=np.float32),  # Contrast variance
        'v_signed_contrast|participant_id_offset': np.random.normal(0, 0.001, n_subjects).astype(np.float32),  # Contrast deviations
        
        # z parameter (starting point) - should be around 0.5
        'z_Intercept': np.array(0.47, dtype=np.float32),  # Fixed starting point
        'z_1|participant_id_sigma': np.array(0.05, dtype=np.float32),  # Random effect variance
        'z_1|participant_id_offset': np.random.normal(0, 0.01, n_subjects).astype(np.float32),  # Subject deviations
        
        # t parameter (non-decision time) - from fitted: ~0.1
        't_Intercept': np.array(0.1, dtype=np.float32),  # Fixed NDT
        't_1|participant_id_sigma': np.array(0.005, dtype=np.float32),  # Random effect variance
        't_1|participant_id_offset': np.random.normal(0, 0.002, n_subjects).astype(np.float32),  # Subject deviations
        
        # a parameter (boundary separation) - from fitted: ~0.645
        'a_Intercept': np.array(0.645, dtype=np.float32),  # Fixed boundary
        'a_1|participant_id_sigma': np.array(0.014, dtype=np.float32),  # Random effect variance
        'a_1|participant_id_offset': np.random.normal(0, 0.007, n_subjects).astype(np.float32),  # Subject deviations
    }
    
    # Add model-specific parameters based on fitted results (NON-CENTERED)
    if 'prevresp_v' in model_name or 'prevresp_zv' in model_name:
        # prevresp effects use fixed effect + sigma + offset structure
        base_initvals.update({
            'v_prevresp_cat': np.array([-0.003], dtype=np.float32),  # Fixed effect (array for categories)
            'v_prevresp_cat|participant_id_sigma': np.array([0.23], dtype=np.float32),  # Random effect variance
            'v_prevresp_cat|participant_id_offset': np.random.normal(0, 0.01, (n_subjects, 1)).astype(np.float32),  # Subject deviations
        })
    
    if 'prevresp_z' in model_name or 'prevresp_zv' in model_name:
        base_initvals.update({
            'z_prevresp_cat': np.array([0.021], dtype=np.float32),  # Fixed effect (array for categories)
            'z_prevresp_cat|participant_id_sigma': np.array([0.024], dtype=np.float32),  # Random effect variance
            'z_prevresp_cat|participant_id_offset': np.random.normal(0, 0.01, (n_subjects, 1)).astype(np.float32),  # Subject deviations
        })
    
    # Add theta for angle models
    if model_name.startswith('angle'):
        base_initvals.update({
            'theta_Intercept': np.array(0.01, dtype=np.float32),  # Fixed angle parameter
            'theta_1|participant_id_sigma': np.array(0.8, dtype=np.float32),  # Random effect variance
            'theta_1|participant_id_offset': np.random.normal(0, 0.05, n_subjects).astype(np.float32),  # Subject deviations
        })
    
    return base_initvals

def _get_centered_initvals(model_name, n_subjects=None):
    """Get initial values for centered parameterization"""
    
    if n_subjects is None:
        n_subjects = N_SUBJECTS
    
    # Base DDM parameters for centered models - values based on fitted results
    base_initvals = {
        # v parameter (drift rate) - group means and subject-specific values
        'v_1|participant_id_mu': -0.02,  # From fitted: -0.095 to 0.052, start slightly negative
        'v_signed_contrast|participant_id_mu': 0.025,  # Consistent across fitted models: ~0.025
        # v parameter - subject-specific deviations around the group means
        'v_1|participant_id': np.random.normal(-0.02, 0.03, n_subjects),  # Small variation around group mean
        'v_signed_contrast|participant_id': np.random.normal(0.025, 0.001, n_subjects),  # Very small subject variation
        
        # z parameter (starting point) - should be around 0.5 (neutral starting point)
        'z_1|participant_id_mu': 0.47,  # Close to 0.5 as expected for starting point
        'z_1|participant_id': np.random.normal(0.47, 0.02, n_subjects),  # Small variation around group mean
        
        # t parameter (non-decision time) - fitted value ~0.1
        't_1|participant_id_mu': 0.1,  # Consistent fitted value
        't_1|participant_id': np.random.normal(0.1, 0.002, n_subjects),  # Very small variation
        
        # a parameter (boundary separation) - fitted value ~0.645
        'a_1|participant_id_mu': 0.645,  # From fitted results
        'a_1|participant_id': np.random.normal(0.645, 0.007, n_subjects),  # Small variation
    }
    
    # Add model-specific parameters based on fitted results
    if 'prevresp_v' in model_name or 'prevresp_zv' in model_name:
        base_initvals.update({
            # From fitted results: prevresp effects on v have wide subject variation
            'v_prevresp_cat|participant_id_mu': -0.003,  # Close to fitted group mean
            'v_prevresp_cat|participant_id': np.random.normal(-0.003, 0.1, n_subjects),  # Subject effects range ~-0.3 to +0.6
        })
    
    if 'prevresp_z' in model_name or 'prevresp_zv' in model_name:
        base_initvals.update({
            # From fitted results: prevresp effects on z are smaller
            'z_prevresp_cat|participant_id_mu': 0.021,  # From fitted: 0.021
            'z_prevresp_cat|participant_id': np.random.normal(0.021, 0.012, n_subjects),  # Subject effects range ~-0.04 to +0.05
        })
    
    # Add theta for angle models
    if model_name.startswith('angle'):
        base_initvals.update({
            'theta_1|participant_id_mu': -0.09,  # Reasonable middle value for angle parameter
            'theta_1|participant_id': np.random.normal(0.01, 0.015, n_subjects),  # Moderate variation
        })
    
    return base_initvals

def get_jittered_initvals(model_name, parameterization='noncentered', n_chains=4, jitter_scale=0.001, n_subjects=None):
    """
    Get jittered initial values for multiple chains.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    parameterization : str
        Either 'centered' or 'noncentered'
    n_chains : int
        Number of chains to create initial values for
    jitter_scale : float
        Standard deviation for jittering
    n_subjects : int, optional
        Number of subjects in the dataset
        
    Returns:
    --------
    list or dict
        If n_chains > 1: List of initial value dictionaries, one per chain
        If n_chains == 1: Single initial value dictionary
    """
    
    base_initvals = get_initvals_for_model(model_name, parameterization, n_subjects)
    
    if n_chains == 1:
        # Return single dictionary for single chain
        chain_initvals = {}
        for param, value in base_initvals.items():
            if isinstance(value, np.ndarray):
                # Jitter array parameters and maintain float32 dtype
                jittered = value + np.random.normal(0, jitter_scale, value.shape)
                chain_initvals[param] = jittered.astype(np.float32)
            else:
                # Jitter scalar parameters
                chain_initvals[param] = np.float32(value + np.random.normal(0, jitter_scale))
        return chain_initvals
    else:
        # Return list of dictionaries for multiple chains
        jittered_initvals = []
        for chain in range(n_chains):
            chain_initvals = {}
            for param, value in base_initvals.items():
                if isinstance(value, np.ndarray):
                    # Jitter array parameters and maintain float32 dtype
                    jittered = value + np.random.normal(0, jitter_scale, value.shape)
                    chain_initvals[param] = jittered.astype(np.float32)
                else:
                    # Jitter scalar parameters
                    chain_initvals[param] = np.float32(value + np.random.normal(0, jitter_scale))
            jittered_initvals.append(chain_initvals)
        
        return jittered_initvals

def get_single_initvals(model_name, parameterization='noncentered', jitter_scale=0.001, n_subjects=None):
    """
    Get a single set of jittered initial values (convenience function for HSSM).
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    parameterization : str
        Either 'centered' or 'noncentered'
    jitter_scale : float
        Standard deviation for jittering
    n_subjects : int, optional
        Number of subjects in the dataset
        
    Returns:
    --------
    dict
        Single initial value dictionary suitable for HSSM.sample(initvals=...)
    """
    return get_jittered_initvals(model_name, parameterization, n_chains=1, jitter_scale=jitter_scale, n_subjects=n_subjects)

# Convenience function to list all supported models
def get_supported_models():
    """Return list of all supported model names"""
    return [
        "ddm_nohist",
        "ddm_prevresp_v", 
        "ddm_prevresp_z",
        "ddm_prevresp_zv",
        "angle_nohist",
        "angle_prevresp_v",
        "angle_prevresp_z", 
        "angle_prevresp_zv"
    ]

if __name__ == "__main__":
    # Test the functions
    print("Testing hierarchical initial values...")
    
    for model in get_supported_models():
        print(f"\n--- {model} ---")
        initvals = get_initvals_for_model(model, 'noncentered')
        print(f"Parameters: {list(initvals.keys())}")
        
        # Test jittered values
        jittered = get_jittered_initvals(model, 'noncentered', n_chains=3)
        print(f"Generated {len(jittered)} sets of jittered initial values")