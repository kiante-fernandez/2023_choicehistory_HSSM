# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2023/17/05      Kianté Fernandez<kiantefernan@gmail.com>   added make_model draft
# 2024/06/10      Kianté Fernandez<kiantefernan@gmail.com>   updated with models with hierarchical

import pandas as pd
import hssm
import bambi as bmb

# %% make model function
def make_model(data, mname_full):
    """
    Create a Hierarchical Sequential Sampling Model (HSSM) based on the specified model name.

    Parameters:
    data (DataFrame): The data to be used in the model.
    mname_full (str): The full name of the model specifying its configuration.

    Returns:
    HSSM: The constructed HSSM model.
    """
    print('Making HSSM model')

    if mname_full.startswith('full_ddm_'):
        base_model = 'full_ddm'
        mname = mname_full[len('full_ddm_'):]
        spec_loglik_kind = "blackbox"
    else:
        base_model, *mname_parts = mname_full.split('_')
        mname = "_".join(mname_parts)
        spec_loglik_kind = "approx_differentiable"

    print(f'Base model: {base_model}')
    print(f'Model name: {mname}')

    model_specs = {
        'nohist': [
            {"name": "v", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "v ~ signed_contrast + (signed_contrast |participant_id)", 
             "link": "identity"},
            {"name": "z", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "z ~ 1 + (1 |participant_id)"},
            {"name": "t", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "t ~ 1 + (1 |participant_id)"},
            {"name": "a", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "a ~ 1 + (1 |participant_id)"}
        ],
        #add the cat config for the contrast model
        'catnohist': [
            {"name": "v", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "v ~ C(signed_contrast) + (C(signed_contrast) |participant_id)",
            #  "formula": "v ~ 0 + C(signed_contrast) + (0 + C(signed_contrast) |participant_id)", 
 
             "link": "identity"},
            {"name": "z", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "z ~ 1 + (1 |participant_id)"},
            {"name": "t", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "t ~ 1 + (1 |participant_id)"},
            {"name": "a", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "a ~ 1 + (1 |participant_id)"}
        ],
        'prevresp_v': [
            {"name": "v", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "v ~ signed_contrast + prevresp + (signed_contrast + prevresp |participant_id)", 
             "link": "identity"},
            {"name": "z", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "z ~ 1 + (1 |participant_id)"},
            {"name": "t", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "t ~ 1 + (1 |participant_id)"},
            {"name": "a", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "a ~ 1 + (1 |participant_id)"}
        ],
        'prevresp_z': [
            {"name": "v", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "v ~ signed_contrast + (signed_contrast |participant_id)", 
             "link": "identity"},
            {"name": "z", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "z ~ 1 + prevresp + (prevresp |participant_id)"},
            {"name": "t", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "t ~ 1 + (1 |participant_id)"},
            {"name": "a", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "a ~ 1 + (1 |participant_id)"}
        ],
        'prevresp_zv': [
            {"name": "v", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "v ~ signed_contrast + prevresp + (signed_contrast + prevresp |participant_id)", 
             "link": "identity"},
            {"name": "z", 
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1},
                "prevresp": {"name": "Normal", "mu": 0.0, "sigma": 1},
             },
             "formula": "z ~ 1 + prevresp + (prevresp |participant_id)"},
            {"name": "t", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "t ~ 1 + (1 |participant_id)"},
            {"name": "a", 
             "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1}},
             "formula": "a ~ 1 + (1 |participant_id)"}
        ]
    }

    if mname not in model_specs:
        raise ValueError('Model name not recognized!')

    hssm_model = hssm.HSSM(
        data=data, 
        p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.50},
        lapse=bmb.Prior("Uniform", lower=0.0, upper=30.0),
        model="ddm", #hard-coded for now
        # model="angle", #hard-coded for now (need to try others, angle, etc)
        loglik_kind="analytical",
        # loglik_kind="approx_differentiable",
        include=model_specs[mname],
        prior_settings="safe",
        link_settings="log_logit"
    )

    return hssm_model
