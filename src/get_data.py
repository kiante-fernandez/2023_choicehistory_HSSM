"""
get data from IBL mice
Anne Urai, Leiden University, 2023

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

# %%
import pandas as pd
import numpy as np
import sys, os, time
import seaborn as sns
import matplotlib.pyplot as plt
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from brainbox.behavior.training import query_criterion
from brainbox.io.one import load_wheel_reaction_times
from tqdm import tqdm

from one.api import ONE # use ONE instead of DJ! more future-proof

one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')

# define path to save the data and figures
datapath = 'data'

#%% 0. GET LIST OF ALL POTENTIAL SUBJECTS 
# find the full cache with subjects + protocols (but not training status)
cache_df = pd.DataFrame(one._cache.sessions).reindex().sort_values(by=['projects', 'lab', 'subject', 'date'])
# cache_df = cache_df[cache_df['projects'] == 'ibl_neuropixel_brainwide_01']

# find subjects through Alyx REST -- only those subjects who made it to biased
subjects = one.alyx.rest('sessions', 'list', 
    # tag='2021_Q1_IBL_et_al_Behaviour',
    dataset_types='trials.table',
    task_protocol='biased')
subject_names = np.unique([s['subject'] for s in subjects])
print(len(subject_names))

#%% 1. QUERY SESSIONS
# on which day did these animals reach biasedCW? get these eids
eids_to_use = []
for subject in tqdm(subject_names):

    # first find the day when trained_1a was reached
    try: eid, n_sessions, n_days = query_criterion(subject, 'trained_1a')  
    except: print('cannot retrieve training status for %s'%subject); continue  # if some animals don't allow criterion query, skip for now
    
    # no trained_1a? some animals may have gone straight to trained_1b
    if eid is None: eid, n_sessions, n_days = query_criterion(subject, 'trained_1b') 
    if eid is None: print('no trained_1a or trained_1b found for %s'%subject); continue

    # now find the 3 days before this
    # find the 3 sessions before n_sessions (when trained_1a was reached)
    cache_sj = cache_df[cache_df['subject'] == subject].reset_index()
    three_sessions_before_trained1a = cache_sj.iloc[n_sessions-3:n_sessions, :]

    # assert that all of these have the task protocol including trainingchoiceworld
    if not (all(three_sessions_before_trained1a['task_protocol'].str.contains('trainingChoiceWorld'))): \
        print('not all sessions have trainingChoiceWorld for %s'%subject); \
            print(three_sessions_before_trained1a); continue

    # assert that this returns 3 sessions per mouse
    if not (len(three_sessions_before_trained1a)) == 3: print('not 3 sessions for %s'%subject); continue

    # also double check that for each of those sessions, trials can be loaded
    add_to_list = True
    for eid_to_check in three_sessions_before_trained1a['id'].tolist():
        try: trials_obj = one.load_object(eid_to_check, 'trials')
        except: add_to_list = False; print('could not load trials for %s'%subject); continue
    if not add_to_list: continue

    #print('adding sessions for %s'%subject)
    for eid in three_sessions_before_trained1a['id'].tolist():
        eids_to_use.append(eid) # append

# PRINT number of sessions
assert(len(eids_to_use) % 3 == 0) # there should be 3 sessions per mouse
print('%d sessions'%(len(eids_to_use)))

# %% 2. LOAD TRIALS
behav = []
for eid in tqdm(eids_to_use): # dont use all for now...

    trials_obj = one.load_object(eid, 'trials')
    trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
    # use a very crude measure of RT: trial duration 
    trials_obj['trial_duration'] = trials_obj['response_times'] - trials_obj['goCue_times']

    trials = trials_obj.to_df() # to dataframe
    trials['trialnum'] = trials.index # to keep track of choice history
    trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1})

    # better way to define RT: based on wheelMoves, Miles' code from Brainbox
    try: trials['firstmove_time'] = load_wheel_reaction_times(eid)
        # # also: was this movement the final one?
        # TODO: ask Miles why the public database does not contain wheel data for these training sessions?
        # # https://int-brain-lab.github.io/iblenv/notebooks_external/docs_wheel_moves.html#Finding-reaction-time-and-'determined'-movements
        # wheel = one.load_object(eid, 'wheel')
        # moves = extract_wheel_moves(wheel['timestamps'], wheel['position'])
        # assert moves, 'unable to load trials and wheelMoves data'
        # trials['firstmove_time_wheel'], trials['wheel_is_final_movement'], ids \
        #     = extract_first_movement_times(moves, trials_obj)
    except: print('skipping %s, no RTs'%eid); continue

    # retrieve the mouse name, session etc
    ref_dict = one.eid2ref(eid)
    trials['eid'] = eid
    trials['subj_idx'] = ref_dict.subject
    trials['date'] = ref_dict.date
    behav.append(trials)

# continue only with some columns we need
df = pd.concat(behav)
df = df[['eid', 'subj_idx', 'date', 'signed_contrast', 
         'response', 'trial_duration', 'firstmove_time','feedbackType', 'trialnum']]

# %% 4. REFORMAT AND SAVE TRIALS
df.to_csv(os.path.join(datapath, 'ibl_trainingchoiceworld_raw.csv'))
print(os.path.join(datapath, 'ibl_trainingchoiceworld_raw.csv'))
print('%d mice, %d trials'%(df.subj_idx.nunique(),  df.subj_idx.count()))

# %%
df.describe()