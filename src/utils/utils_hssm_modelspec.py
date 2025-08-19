# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2023/17/05      Kianté Fernandez<kiantefernan@gmail.com>   added make_model draft
# 2024/06/10      Kianté Fernandez<kiantefernan@gmail.com>   updated with models with hierarchical
# 2025/08/18      Kianté Fernandez<kiantefernan@gmail.com>   Added both centered and non-centered parameterizations

import pandas as pd
import hssm
import bambi as bmb

# =====================================================================================
# Main Model Selection Function
# =====================================================================================

def make_model(data, mname_full, parameterization='centered'):
    """
    Primary wrapper function to create an HSSM model.

    Selects between a centered or non-centered parameterization based on the
    'parameterization' argument.

    Parameters:
    - data (DataFrame): The data for the model.
    - mname_full (str): The full name of the model configuration.
    - parameterization (str): The desired parameterization.
                               'centered' for the A-centered / 0+ model.
                               'noncentered' for the standard mixed-effects model.

    Returns:
    - HSSM: The constructed HSSM model.
    """
    if parameterization == 'centered':
        print("--- Building A-Centered Model ---")
        return make_model_centered(data, mname_full)
    elif parameterization == 'noncentered':
        print("--- Building Non-Centered Model ---")
        return make_model_noncentered(data, mname_full)
    else:
        raise ValueError("Parameterization must be either 'centered' or 'noncentered'")


# =====================================================================================
# A-Centered Model Specification
# =====================================================================================

def make_model_centered(data, mname_full):
    """
    Creates an HSSM with a CENTERED parameterization ("A-centered" or "0+" model).

    - Uses the formula '~ 0 + (regressors | group)'
    - Sets hyperpriors on the mean (mu) of the group-level distribution.
    - This `mu` represents the fixed effect.
    - `noncentered=False` must be set.
    """
    print('Making HSSM model (A-centered specification)')

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

    nohist_spec = [
        {"name": "v", "formula": "v ~ 0 + (1 + signed_contrast | participant_id)", "link": "identity",
         "prior": {
            "1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.2}},
            "signed_contrast|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.2}}
         }},
        {"name": "z", "formula": "z ~ 0 + (1 | participant_id)", "link": "logit",
         "prior": {"1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.3}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}}},
        {"name": "t", "formula": "t ~ 0 + (1 | participant_id)", "link": "log",
         "prior": {"1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": -1.9, "sigma": 0.3}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.3}}}},
        {"name": "a", "formula": "a ~ 0 + (1 | participant_id)", "link": "log",
         "prior": {"1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": -0.1, "sigma": 0.3}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}}}
    ]

    model_specs = {
        'nohist': nohist_spec,
        'catnohist': [
            {"name": "v", "formula": "v ~ 0 + (1 + C(signed_contrast) | participant_id)", "link": "identity",
             "prior": {
                "1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "C(signed_contrast)|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[1], nohist_spec[2], nohist_spec[3]
        ],
        'prevresp_v': [
            {"name": "v", "formula": "v ~ 0 + (1 + prevresp_cat + signed_contrast | participant_id)", "link": "identity",
             "prior": {
                "1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "prevresp_cat|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "signed_contrast|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[1], nohist_spec[2], nohist_spec[3]
        ],
        'prevresp_z': [
            nohist_spec[0],
            {"name": "z", "formula": "z ~ 0 + (1 + prevresp_cat | participant_id)", "link": "logit",
             "prior": {
                "1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.3}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "prevresp_cat|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[2], nohist_spec[3]
        ]
    }
    model_specs['prevresp_zv'] = [model_specs['prevresp_v'][0], model_specs['prevresp_z'][1], nohist_spec[2], nohist_spec[3]]

    if mname not in model_specs: raise ValueError('Model name not recognized!')
    current_model_spec = model_specs[mname]

    if "angle" in base_model:
        current_model_spec.append({
            "name": "theta", "formula": "theta ~ 0 + (1 | participant_id)", "link": "identity",
            "prior": {"1|participant_id": {"name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5}, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}}
        })

    return hssm.HSSM(
        data=data, model=base_model, loglik_kind=spec_loglik_kind,
        include=current_model_spec, noncentered=False,
        lapse=None,
        p_outlier=None
        # lapse=bmb.Prior("Uniform", lower=0.0, upper=5.0),
        # p_outlier={
        #     "formula": "p_outlier ~ 1 + (1 | participant_id)", "link": "logit",
        #     "prior": {
        #         "Intercept": {"name": "Normal", "mu": -2.3, "sigma": 1.0},
        #         "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.2}}
        #     }
        # }
    )

# =====================================================================================
# Non-Centered Model Specification
# =====================================================================================

def make_model_noncentered(data, mname_full):
    """
    Creates an HSSM with a NON-CENTERED parameterization.

    - Uses the formula '~ 1 + regressors + (1 + regressors | group)'
    - Fixed effects (e.g., "Intercept") and Random effects (e.g., "1|participant_id")
      are specified with separate priors.
    - `noncentered=True` is typically used (HSSM's default).
    """
    print('Making HSSM model (Non-centered specification)')

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

    nohist_spec = [
        {"name": "v", "formula": "v ~ 1 + signed_contrast + (1 + signed_contrast | participant_id)", "link": "identity",
         "prior": {
            "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
            "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
            "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.2}},
            "signed_contrast|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.2}}
         }},
        {"name": "z", "formula": "z ~ 1 + (1 | participant_id)", "link": "logit",
         "prior": {
            "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.3},
            "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
         }},
        {"name": "t", "formula": "t ~ 1 + (1 | participant_id)", "link": "log",
         "prior": {
            "Intercept": {"name": "Normal", "mu": -1.9, "sigma": 0.3},
            "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.3}}
         }},
        {"name": "a", "formula": "a ~ 1 + (1 | participant_id)", "link": "log",
         "prior": {
            "Intercept": {"name": "Normal", "mu": -0.1, "sigma": 0.3},
            "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
         }}
    ]

    model_specs = {
        'nohist': nohist_spec,
        'catnohist': [
            {"name": "v", "formula": "v ~ 1 + C(signed_contrast) + (1 + C(signed_contrast) | participant_id)", "link": "identity",
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "C(signed_contrast)": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "C(signed_contrast)|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[1], nohist_spec[2], nohist_spec[3]
        ],
        'prevresp_v': [
            {"name": "v", "formula": "v ~ 1 + prevresp_cat + signed_contrast + (1 + prevresp_cat + signed_contrast | participant_id)", "link": "identity",
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "prevresp_cat": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "prevresp_cat|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "signed_contrast|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[1], nohist_spec[2], nohist_spec[3]
        ],
        'prevresp_z': [
            nohist_spec[0],
            {"name": "z", "formula": "z ~ 1 + prevresp_cat + (1 + prevresp_cat | participant_id)", "link": "logit",
             "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.3},
                "prevresp_cat": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}},
                "prevresp_cat|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
             }},
            nohist_spec[2], nohist_spec[3]
        ]
    }
    model_specs['prevresp_zv'] = [model_specs['prevresp_v'][0], model_specs['prevresp_z'][1], nohist_spec[2], nohist_spec[3]]

    if mname not in model_specs: raise ValueError('Model name not recognized!')
    current_model_spec = model_specs[mname]

    if "angle" in base_model:
        current_model_spec.append({
            "name": "theta", "formula": "theta ~ 1 + (1 | participant_id)", "link": "identity",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "Weibull", "alpha": 1.5, "beta": 0.5}}
            }
        })

    return hssm.HSSM(
        data=data, model=base_model, loglik_kind=spec_loglik_kind,
        include=current_model_spec, noncentered=True,
        lapse=None,
        p_outlier=None
        # lapse=bmb.Prior("Uniform", lower=0.0, upper=5.0),
        # p_outlier={
        #     "formula": "p_outlier ~ 1 + (1 | participant_id)", "link": "logit",
        #     "prior": {
        #         "Intercept": {"name": "Normal", "mu": -1.6, "sigma": 1.0},
        #         "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.2}}
        #     }
        # }
    )
