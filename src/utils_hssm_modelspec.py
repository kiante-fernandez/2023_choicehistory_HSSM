# hssm_modelspec.py - sandbox for MODEL SPECIFICATION for SSM using the hssm syntax
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2022/12/05      Kianté Fernandez<kiantefernan@gmail.com>   coded up version one
# 2022/17/05      Kianté Fernandez<kiantefernan@gmail.com>   added make_model draft


#TODO these still do not work (irrelgular sampling). Need to invetigate further 

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
    'nohist': [
        {"name": "v", 
         "formula": "v ~ 1 + signed_contrast + (1 + signed_contrast|subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + (1|subj_idx)", 
         "link": "identity"},
    ],
    'prevresp_v': [
        {"name": "v", 
         "formula": "v ~ 1 + signed_contrast + prevresp + (1 + signed_contrast + prevresp|subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + (1|subj_idx)", 
         "link": "identity"}
    ],
    'prevresp_z': [
        {"name": "v", 
         "formula": "v ~ 1 + signed_contrast + (1 + signed_contrast |subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + prevresp + (1 + prevresp|subj_idx)", 
         "link": "identity"}
    ],
    'prevresp_zv': [
        {"name": "v", 
         "formula": "v ~ signed_contrast + prevresp + (signed_contrast + prevresp |subj_idx)", 
         "link": "identity"},
        {"name": "z", 
         "formula": "z ~ 1 + prevresp + (1 + prevresp|subj_idx)", 
         "link": "identity"}
    ]
    }
    if mname in model_specs:
        hssm_model = hssm.HSSM(data, 
                               model=base_model,
                               loglik_kind=spec_loglik_kind, #note so we can use ddm, angle, and weibull 
                               include=model_specs[mname],
                               prior_settings= "safe",
                               hierarchical=True,
                               link_settings = "log_logit")
    else:
        raise ValueError('Model name not recognized!')

    return hssm_model


#%% draft of better version? 

# param_v = hssm.Param(
#     "v",
#     formula="v ~ 1 + signed_contrast + (1 + signed_contrast|participant_id)",
#     link="identity",
#     prior={
#         "Intercept": hssm.Prior("Normal", mu=0.0, sigma=3),
#         "1|participant_id": hssm.Prior(
#             "Normal",
#             mu=0.0,
#             sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
#         ),
#         "signed_contrast": hssm.Prior("Normal", mu=0.0, sigma=2),
#         "signed_contrast|participant_id": hssm.Prior(
#             "Normal",
#             mu=0.0,
#             sigma=hssm.Prior("Weibull",alpha = 1, beta = 0.2), 
#         ),
#     },
# )

# param_z = hssm.Param(
#     "z",
#     formula="z ~ 1 + (1|participant_id)",
#     link="identity",
#     prior={
#         "Intercept": hssm.Prior("Gamma", mu=10, sigma=10, bounds=(0.01,.99)),
#         "1|participant_id": hssm.Prior(
#             "Normal",
#             mu=0.0,
#             sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
#         )
#     },
#     bounds= (0.01,.99)
# )

# param_a = hssm.Param(
#     "a",
#     formula="a ~ 1 + (1|participant_id)",
#     link="identity",
#     prior={
#         "Intercept": hssm.Prior("Gamma", mu=1.5, sigma=1),
#         "1|participant_id": hssm.Prior(
#             "Normal",
#             mu=0.0,
#             sigma=hssm.Prior("Weibull",alpha = 1.5, beta = 0.3), 
#         )
#     },
#     bounds=(0.01, np.inf)
# )

# param_t = hssm.Param(
#     "t",
#     formula="t ~ 1 + (1|participant_id)",
#     link="identity",
#     prior={
#         "Intercept": hssm.Prior("Weibull", alpha=1, beta=0.2),
#         "1|participant_id": hssm.Prior(
#             "Normal",
#             mu=0.0,
#             sigma=hssm.Prior("Weibull", alpha = 1.5, beta = 0.3), 
#         )
#     },
#     bounds=(0.100, np.inf)
# )

# model_specs = [param_v, param_z, param_a, param_t]

# hssm_model = hssm.HSSM(data = dataset,
#                        model= "ddm",
#                        include=model_specs,
#                        loglik_kind="analytical")

# # hssm_model.model.plot_priors()

# hssm_model.sample(sampler="nuts_numpyro",
#                   cores=4,
#                   chains=4,
#                   draws=3000,
#                   tune=3000,
#                   idata_kwargs=dict(log_likelihood=True)
    
# )
