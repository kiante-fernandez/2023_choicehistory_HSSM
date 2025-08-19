# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2023/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2023/17/05      Kianté Fernandez<kiantefernan@gmail.com>   added make_model draft
# 2024/06/10      Kianté Fernandez<kiantefernan@gmail.com>   updated with models with hierarchical
# 2025/08/18      Kianté Fernandez<kiantefernan@gmail.com>   Implemented A-centered parameterization

import pandas as pd
import hssm
import bambi as bmb

# %% make model function
def make_model(data, mname_full):
    """
    Create a Hierarchical Sequential Sampling Model (HSSM) with a centered parameterization.

    This function implements the "A-centered model" or "0+" specification where subject-specific
    coefficients are drawn directly from a hierarchical prior whose parameters (mu, sigma) are
    estimated from the data. This is achieved by changing the formula structure to
    `~ 0 + (1 + regressor | group)` and setting `noncentered=False`.

    Parameters:
    data (DataFrame): The data to be used in the model.
    mname_full (str): The full name of the model specifying its configuration.

    Returns:
    HSSM: The constructed HSSM model.
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

    # Define the base 'nohist' model specification with A-centered parameterization
    nohist_spec = [
        {"name": "v",
         "formula": "v ~ 0 + (1 + signed_contrast | participant_id)", # Centered formula
         "link": "identity",
         "prior": {
            "1|participant_id": {
                "name": "Normal",
                "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, # Hyperprior on the mean (fixed effect)
                "sigma": {"name": "HalfNormal", "sigma": 0.5}
            },
            "signed_contrast|participant_id": {
                "name": "Normal",
                "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0}, # Hyperprior on the mean (fixed effect)
                "sigma": {"name": "HalfNormal", "sigma": 0.5}
            }
         }},
        {"name": "z",
         "formula": "z ~ 0 + (1 | participant_id)", # Centered formula
         "link": "logit",
         "prior": {
            "1|participant_id": {
                "name": "Normal",
                "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.3}, # Hyperprior
                "sigma": {"name": "HalfNormal", "sigma": 0.5}
            }
         }},
        {"name": "t",
         "formula": "t ~ 0 + (1 | participant_id)", # Centered formula
         "link": "log",
         "prior": {
            "1|participant_id": {
                "name": "Normal",
                "mu": {"name": "Normal", "mu": -1.9, "sigma": 0.3}, # Hyperprior
                "sigma": {"name": "HalfNormal", "sigma": 0.3}
            }
         }},
        {"name": "a",
         "formula": "a ~ 0 + (1 | participant_id)", # Centered formula
         "link": "log",
         "prior": {
            "1|participant_id": {
                "name": "Normal",
                "mu": {"name": "Normal", "mu": -0.1, "sigma": 0.3}, # Hyperprior
                "sigma": {"name": "HalfNormal", "sigma": 0.5}
            }
         }}
    ]

    # Now build the full dictionary of model specifications
    model_specs = {
        'nohist': nohist_spec,
        'catnohist': [
            {"name": "v",
             "formula": "v ~ 0 + (1 + C(signed_contrast) | participant_id)", # Centered
             "link": "identity",
             "prior": {
                "1|participant_id": {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                },
                "C(signed_contrast)|participant_id": {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                }
             }},
            nohist_spec[1], # z spec
            nohist_spec[2], # t spec
            nohist_spec[3]  # a spec
        ],
        'prevresp_v': [
            {"name": "v",
             "formula": "v ~ 0 + (1 + prevresp_cat + signed_contrast | participant_id)", # Centered
             "link": "identity",
             "prior": {
                "1|participant_id": {
                    "name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                },
                "prevresp_cat|participant_id": {
                    "name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                },
                "signed_contrast|participant_id": {
                    "name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                }
             }},
            nohist_spec[1], # z spec
            nohist_spec[2], # t spec
            nohist_spec[3]  # a spec
        ],
        'prevresp_z': [
            nohist_spec[0], # v spec
            {"name": "z",
             "formula": "z ~ 0 + (1 + prevresp_cat | participant_id)", # Centered
             "link": "logit",
             "prior": {
                "1|participant_id": {
                    "name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.3},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                },
                "prevresp_cat|participant_id": {
                    "name": "Normal", "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                }
             }},
            nohist_spec[2], # t spec
            nohist_spec[3]  # a spec
        ]
    }
    # Build the 'prevresp_zv' spec from the others
    model_specs['prevresp_zv'] = [
        model_specs['prevresp_v'][0], # v spec
        model_specs['prevresp_z'][1], # z spec
        nohist_spec[2], # t spec
        nohist_spec[3]  # a spec
    ]


    if mname not in model_specs:
        raise ValueError('Model name not recognized!')

    current_model_spec = model_specs[mname]

    # Add theta parameter if needed
    if "angle" in base_model:
        current_model_spec.append({
            "name": "theta", "formula": "theta ~ 0 + (1 | participant_id)", "link": "identity", # Centered
            "prior": {
                "1|participant_id": {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5}
                }
            }})

    hssm_model = hssm.HSSM(
        data=data,
        model=base_model,
        loglik_kind=spec_loglik_kind,
        include=current_model_spec,
        noncentered=False, # Set to False for centered parameterization
        lapse=bmb.Prior("Uniform", lower=0.0, upper=5.0),
        p_outlier={
            "formula": "p_outlier ~ 1 + (1 | participant_id)", # Standard formula for non-hierarchical priors
            "link": "logit",
            "prior": {
                "Intercept": {"name": "Normal", "mu": -1.6, "sigma": 1.0},
                "1|participant_id": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 0.2}}
            }
        }
    )
    return hssm_model