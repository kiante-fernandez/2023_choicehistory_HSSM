   
import pandas as pd
import numpy as np
   

def compute_choice_history(trials):

    print('adding choice history columns to database...')

    # append choice history 
    trials['prevresp']      = trials.response.shift(1)
    trials['prevfb']        = trials.correct.shift(1)
    trials['prevcontrast']  = np.abs(trials.signed_contrast.shift(1))

    # # also append choice future (for correction a la Lak et al.)
    # trials['nextresp']      = trials.response.shift(-1)
    # trials['nextfb']        = trials.correct.shift(-1)
    # trials['nextcontrast']  = np.abs(trials.signed_contrast.shift(-1))

    # remove when not consecutive based on trial_index
    trials_not_consecutive       = (trials.trialnum - trials.trialnum.shift(1)) != 1.
    for col in ['prevresp', 'prevfb', 'prevcontrast']:
        trials.loc[trials_not_consecutive, col] = np.nan

    # add some more history measures
    trials['repeat'] = np.where(trials.response == trials.prevresp, 1, 0)

    return trials

def clean_rts(rt, cutoff=[0.08, 2],
              compare_with=None, comparison_cutoff=None):

    assert (0 < np.nanmedian(rt) < 3) # median RT should be within some reasonable bounds

    print('cleaning RTs...')
    # remove RTs below and above cutoff, for HDDM 
    rt_clean = rt.copy()
    rt_clean[rt_clean < cutoff[0]] = np.nan 
    rt_clean[rt_clean > cutoff[1]] = np.nan 

    # only keep RTs when they are close to the trial duration
    if compare_with is not None:
        timing_difference = compare_with - rt
        # assert all(timing_difference > 0) # all RTs should be smaller than trial duration
        rt_clean[timing_difference > comparison_cutoff] = np.nan

    return rt_clean

def rescale_contrast(x):
    """
    Since signed contrast does not linearly map onto drift rate, rescale it (with a basic tanh function)
    to approximate linearity (so that we can include a single, linear 'contrast' term in the regression models)

    See plot_contrast_rescale.py for a tanh fit, which generates the parameters below
    """

    a = 2.13731484
    b = 0.05322221
    
    return a * np.tanh( b * x )