# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2022/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2022/17/05      Kianté Fernandez<kiantefernan@gmail.com>   added make_model draft


import pandas as pd
import hssm

hssm.set_floatX("float32")

# %% write make model function
def make_model(data, mname_full):
    """
    Create a Hierarchical Sequential Sampling Model (HSSM) based on the specified model name.

    Parameters:
    data: DataFrame
        The data to be used in the model.
    mname_full: str
        The full name of the model specifying its configuration.

    Returns:
    hssm_model: HSSM object
        The constructed HSSM model.
    """
 
    print('making HSSM model')

    if mname_full.startswith('full_ddm_'):
        base_model = 'full_ddm'
        mname = mname_full[len('full_ddm_'):]
        spec_loglik_kind="blackbox" 
    else:
        mname_split = mname_full.split('_')
        base_model = mname_split[0]
        mname = "_".join(mname_split[1:])
        spec_loglik_kind="approx_differentiable" 

    print(f'Base model: {base_model}')
    print(f'Model name: {mname}')

    model_specs = {
    'nohist_stimcat': [
        {"name": "v", 
         "formula": "v ~ 0 + C(coherence) + (0+ C(coherence)|subj_idx)", #use for compare with continuous get a jumbo running
         "link": "identity"},
        # {"name": "a", 
        #  "formula": "a ~ 1 + (1|subj_idx)", 
        #  "link": "identity"},
        # {"name": "t", 
        #  "formula": "t ~ 1 + (1|subj_idx)", 
        #  "link": "identity"}
    ],
    # 'nohist_stimcat_dummycode': [
    #     {"name": "v", 
    #      "formula": "v ~ 0 + C(coherence, Treatment) + (0 + C(coherence, Treatment)|subj_idx)", 
    #      "link": "identity"},
    #     {"name": "z", 
    #      "formula": "z ~ 1 + (1|subj_idx)", 
    #      "link": "identity"}
    # ],
    # 'nohist_stimcat_reducedrankcode': [
    #     {"name": "v", 
    #      "formula": "v ~ 0 + C(coherence, Treatment(0)) + (0 + C(coherence, Treatment(0))|subj_idx)", 
    #      "link": "identity"},
    #     {"name": "z", 
    #      "formula": "z ~ 1 + (1|subj_idx)", 
    #      "link": "identity"}
    # ],
    # TODO add the model with continuous coherence and stimulus 
    
    'nohist': [
        {"name": "v", 
         "formula": "v ~ 1 + stimulus + (1 + stimulus|subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + (1|subj_idx)", 
         "link": "identity"},
        # {"name": "a", 
        #  "formula": "a ~ 1 + (1|subj_idx)", 
        #  "link": "identity"}
        # {"name": "t", 
        #  "formula": "t ~ 1 + (1|subj_idx)", 
        #  "link": "identity"}
    ],
    'prevresp_v': [
        {"name": "v", 
         "formula": "v ~ 1 + stimulus + prevresp + (1 + stimulus + prevresp|subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + (1|subj_idx)", 
         "link": "identity"},
        {"name": "a", 
         "formula": "a ~ 1 + (1|subj_idx)", 
         "link": "identity"}
    ],
    'prevresp_z': [
        {"name": "v", 
         "formula": "v ~ 1 + stimulus + (1 + stimulus |subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + prevresp + (1 + prevresp|subj_idx)", 
         "link": "identity"},
        {"name": "a", 
         "formula": "a ~ 1 + (1|subj_idx)", 
         "link": "identity"}
        # {"name": "a", 
        #  "formula": "a ~ 1 + (1|subj_idx)", 
        #  "link": "identity"},
        # {"name": "t", 
        #  "formula": "t ~ 1 + (1|subj_idx)", 
        #  "link": "identity"}
    ],
    # 'stimcat_prevresp_zv': [
    #     {"name": "v", 
    #      "formula": "v ~ C(coherence) + prevresp + (C(coherence) + prevresp |subj_idx)", 
    #      "link": "identity"},
    #     {"name": "z", 
    #      "formula": "z ~ 1 + prevresp + (1 + prevresp|subj_idx)",
    #      "link": "identity"},
    #     {"name": "a", 
    #      "formula": "a ~ 1 + (1|subj_idx)", 
    #      "link": "identity"}
    #     # {"name": "t", 
    #     #  "formula": "t ~ 1 + (1|subj_idx)", 
    #     #  "link": "identity"}
    # ],
    'prevresp_zv': [
        {"name": "v", 
         "formula": "v ~ stimulus + prevresp + (stimulus + prevresp |subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + prevresp + (1 + prevresp|subj_idx)", 
         "link": "identity"},
        {"name": "a", 
         "formula": "a ~ 1 + (1|subj_idx)", 
         "link": "identity"}
        # {"name": "a", 
        #  "formula": "a ~ 1 + (1|subj_idx)", 
        #  "link": "identity"},
        # {"name": "t", 
        #  "formula": "t ~ 1 + (1|subj_idx)", 
        #  "link": "identity"}
    ]
    }
    if mname in model_specs:
        hssm_model = hssm.HSSM(data, 
                               model=base_model,
                               loglik_kind=spec_loglik_kind, #note so we can use ddm, angle, and weibull 
                               include=model_specs[mname],
                               prior_settings="safe")
    else:
        raise ValueError('Model name not recognized!')

    return hssm_model
