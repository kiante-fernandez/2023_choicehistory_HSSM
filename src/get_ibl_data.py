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
import utils_choice_history as more_tools
import brainbox.behavior.training as training
from brainbox.behavior.training import query_criterion
from datetime import datetime

#from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
#from brainbox.io.one import load_wheel_reaction_times
from tqdm import tqdm
from pathlib import Path

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
assert not one.offline

if not one.offline:
    # print(one.alyx.user)
    print(one.alyx.base_url)

# define path to save the data and figures
datapath = 'data'

#  ============================================ #
# %% FIRST, FIND SUBJECTS OF INTEREST
whichTask  = 'trainingChoiceWorld' # 'trainingChoiceWorld' or 'ephysChoiceWorld'
eids, info = one.search(project='brainwide', dataset='trials.table', 
                        task_protocol = whichTask,
                        details=True)
df_info = pd.DataFrame(info).sort_values(by=['lab', 'subject', 'date', 'number'])

subjects = np.sort(np.unique([i['subject'] for i in info]))
print('number of subjects: ')
print(len(subjects))

# for now, restrict to keep manageable
subjects = subjects[0:10]
print(subjects)

#%% SECOND, FIND WHICH SESSIONS WE'D LIKE TO USE
eids_to_use = []
for subject in subjects:

    eids, info = one.search(subject=subject, 
                            dataset=['trials.table'], 
                            details=True)
    df_info = pd.DataFrame(info).sort_values(by=['lab', 'subject', 'date', 'number'])
    df_info['date'] = pd.to_datetime(df_info['date'], utc = True)

    # what was the training status at each session?
    training_status = one.load_aggregate('subjects', subject, 
                                         '_ibl_subjectTraining.table').reset_index()
    training_status['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in training_status['date']]
    training_status['date'] = pd.to_datetime(training_status['date'], utc = True)

    # Join to sessions table table - some wrangling needed to match on datess
    #df_info['date2'] = df_info['date']
    df_info = df_info.set_index('date').join(training_status.set_index('date'))
    df_info['training_status'] = df_info.training_status.fillna(method='ffill')
    df_info = df_info.reset_index()

    # now get all the sessions for this subject
    if whichTask == 'trainingChoiceWorld':

        # find the sessions where the animal was trained_1a OR trained_1b
        first_day_trained = df_info[('trained' in df_info['training_status'])]

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
        
        
        # take the last 3 sessions of this animal for now - later, improve
    three_sessions_before_trained1a = eids[-3:]
    assert len(three_sessions_before_trained1a) == 3

    # training_status = training.get_subject_training_status(subject, one=one)

    # # from this, find the 3 sessions we'd like
    # whichsessions = 'trainingCW'
    # if whichsessions == 'trainingCW':
    #     use_sessions = 'trained_1a'
    # elif whichsessions == 'ephys':
    #     use_sessions = 'ephys'



    #print('adding sessions for %s'%subject)
    for eid in three_sessions_before_trained1a:
        eids_to_use.append(eid) # append

# %% THIRD, GET TRIALS TABLE FOR ALL OF THESE
behav = [] # all behavior for all animals
for eid in tqdm(eids_to_use): # dont use all for now...

    trials_obj = one.load_object(eid, 'trials', collection='alf')

    # trials_obj = one.load_object(eid, 'trials')
    trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
    # use a very crude measure of RT: trial duration 
    trials_obj['rt'] = trials_obj['response_times'] - trials_obj['goCue_times']
    # additionally, save the movement onset
    trials_obj['movement_onset'] = trials_obj['firstMovement_times'] - trials_obj['goCue_times']

    # for trainingChoiceWorld, the probabilityLeft is always 0.5
    if whichTask == 'trainingChoiceWorld':
        trials_obj['probabilityLeft'] = 0.5

    trials = trials_obj.to_df() # to dataframe
    trials['trialnum'] = trials.index # to keep track of choice history
    trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1})
    trials['correct']  = trials['feedbackType']

    # retrieve the mouse name, session etc
    ref_dict = one.eid2ref(eid)
    trials['eid'] = eid
    trials['subj_idx'] = ref_dict.subject
    trials['date'] = ref_dict.date
    behav.append(trials)

# continue only with some columns we need
data = pd.concat(behav)
data = data[['eid', 'subj_idx', 'date', 'signed_contrast', 
         'response', 'rt', 'movement_onset', 'correct', 'trialnum']]

# %% FOURTH, CLEAN AND PREPROCESS THE DATA

# remove RTs that sit outside the cutoff window
# define how we quantify RTs
data['rt_raw'] = data['rt'].copy()
data['rt'] = more_tools.clean_rts(data['rt'], cutoff=[0.08, 2], 
                                  compare_with=None)

# add choice history information
data = more_tools.compute_choice_history(data)

# # add some more history measures
data['repeat'] = np.where(data.response == data.prevresp, 1, 0)

# save to csv   
data.to_csv(os.path.join(datapath, 'ibl_%s_clean.csv'%whichTask), 
            index=False)
