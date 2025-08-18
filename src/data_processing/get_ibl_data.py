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
import os
from datetime import datetime
import re 
from tqdm import tqdm
import utils_choice_history as more_tools
import seaborn

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
assert not one.offline

if not one.offline:
    # print(one.alyx.user)
    print('You are connected to: ' + one.alyx.base_url)

# define path to save the data and figures
# Get the directory of the script being run
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
fig_folder_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
data_folder_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')

whichTask = 'trainingChoiceWorld' # 'trainingChoiceWorld' or 'ephysChoiceWorld'

#  ============================================ #
# %% FIRST, FIND SUBJECTS OF INTEREST
# from https://www.biorxiv.org/content/10.1101/2023.12.22.573001v1.full
# use the Bruijns et al tag to grab only subjects that have the subjectTraining.table released
regexp = re.compile(r'Subjects/\w*/((\w|-)+)/_ibl') 
datasets = one.alyx.rest('datasets', 'list', tag='2023_Q4_Bruijns_et_al') # extract subject names 
subjects = np.unique(np.sort([regexp.search(ds['file_records'][0]['relative_path']).group(1) for ds in datasets])) # reduce to list of unique names 

print('number of subjects: %s'%len(subjects))
print(subjects)

#  ============================================ #
# %% SECOND, FIND WHICH SESSIONS WE'D LIKE TO USE
big_trials_df = []
for subject in tqdm(subjects):

    # Load training status and join to full trials table, https://int-brain-lab.github.io/iblenv/notebooks_external/loading_trials_data.html#Loading-all-the-sessions'-trials-for-a-single-subject-at-once
    trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
    training_table = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')
    trials = (trials
            .set_index('session')
            .join(training_table.set_index('session'))
            .sort_values(by=['session_start_time', 'intervals_0']))
    trials['training_status'] = trials.training_status.ffill()

    # now find only the data we'd like to keep
    if whichTask == 'trainingChoiceWorld':

        # find the sessions where the animal was trained_1a OR trained_1b
        try: first_day_trained = trials[trials['training_status'].str.contains("trained")]['session_start_time'].iloc[0]
        except: print('skipping %s, could not identify first day trained'%subject); continue
        all_dates = np.sort(trials['session_start_time'].unique())
        first_trained_loc = np.where(all_dates == first_day_trained)[0][0]

        # extra check: assert that this mouse has never seen biased or ephys tasks before
        try:
            assert not trials.iloc[:first_trained_loc]['task_protocol'].str.contains('bias').any()
            assert not trials.iloc[:first_trained_loc]['task_protocol'].str.contains('ephys').any()
        except: print ('skippint %s, has seen biased or ephys before getting trained'%subject); continue
        
        # now find the num_sessions before this
        num_sessions = 3
        if num_sessions > 1:
            dates_before_trained = all_dates[first_trained_loc-num_sessions:first_trained_loc]
        elif num_sessions == 1:
            # try to get only 1 session instead - ensure a list
            dates_before_trained = [all_dates[first_trained_loc-1]]
        else:
            print('number of sessions must be 1 or more!')

        # subset the dataframe to only include these sessions
        trials = trials[trials['session_start_time'].isin(dates_before_trained)]   

        # assert that this returns num_sessions per mouse
        try: assert (trials['session_start_time'].nunique()) == num_sessions
        except: print('skipping %s, did not find %d sessions before trained'%(subject, num_sessions)); continue 

        # assert that all of these have the task protocol including trainingchoiceworld
        try: assert (all(trials['task_protocol'].str.contains('trainingChoiceWorld')))
        except: print('skipping %s, task not trainingCW'%subject)

        # complete the df to include all we need
        trials['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials['contrastLeft'], 
                                                        trials['contrastRight']]))
        trials['abs_contrast'] = np.abs(trials['signed_contrast'])
        trials['trialnum'] = trials.groupby(['session_start_time']).cumcount()
        trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1})
        trials['correct']  = trials['feedbackType'].map({-1: 0, 1: 1})
        # use a very crude measure of RT: trial duration - see https://docs.google.com/document/d/1s1huCm6eap2cdI6e-3cEnMpH8fP7KvXd16h_350WCU8/edit
        trials['rt'] = trials['response_times'] - trials['stimOn_times']
        # additionally, save the movement onset
        trials['movement_onset'] = trials['firstMovement_times'] - trials['stimOn_times']
        # TODO consider using only trials where the mouse movemnet time is similar to response time. 
        trials['prior_bias'] = trials['probabilityLeft']
        # for trainingChoiceWorld, the probabilityLeft is always 0.5
        if whichTask == 'trainingChoiceWorld': trials['prior_bias'] = 0.5
        # retrieve the mouse name, session etc
        trials['subj_idx'] = subject
        trials['eid'] = trials.index

        # TODO: Add is_final_movement to trial selection: 
        # https://int-brain-lab.github.io/iblenv/notebooks_external/docs_wheel_moves.html#Finding-reaction-time-and-'determined'-movements

        # 1. check that we have all contrasts introduced in each session
        num_contrast = trials['abs_contrast'].unique()
        try: assert len(num_contrast) >= 5 # there should be 5 or more
        except: print('skipping %s, not all contrasts introduced'%subject); print(num_contrast); continue

        # 2. check that we have > 400 trials in each session
        num_trials = trials.groupby(['session_start_time'])['trialnum'].count().reset_index()
        try: assert all(num_trials['trialnum'] > 400)
        except: print('skipping %s, not enough trials '%subject); continue

        # 3. check that performance is above 80% for all easy trials
        # assert that we have good enough behavior in each session
        trials['high_contrast_level'] = trials['abs_contrast'] >= 50
        perf = trials.groupby(['high_contrast_level'])['correct'].mean().reset_index()
        try: assert all(perf[(perf['high_contrast_level'] == True)]['correct'] > 0.80)
        except: print('skipping %s, performance too low'%subject); print(perf); continue

        # # 4. check that median RT is below some reasonable value in each session
        # median_rt = trials.groupby(['session_start_time'])['rt'].median().reset_index()
        # # try: assert all(median_rt['rt'] < 1)
        # # except: print('skipping %s, RT too slow '%subject); continue

        # # Check if we still have enough trials after RT exclusion
        # trials_per_session = trials.groupby(['session_start_time'])['trialnum'].count()
        # print(trials_per_session)
        # # try: assert all(trials_per_session > 300)  # reduced from 400 to account for excluded trials
        # # except: print('skipping %s, not enough trials after RT exclusion'%subject); continue
        
        # # 5. check that we don't have negative RTs
        # #try: assert all(trials.rt > 0)
        # #except: print('skipping %s, negative RTs '%subject); continue

        # continue only with some columns we need
        trials = trials[['eid', 'subj_idx', 'session_start_time', 'signed_contrast', 'prior_bias',
         'response', 'rt', 'movement_onset', 'correct', 'trialnum']]

    # add to the big df for saving later
    big_trials_df.append(trials) # append

# concatenate across all subjects
data = pd.concat(big_trials_df).reset_index()
# data_orig = data.copy()

print('# sessions included:')
print(len(np.unique(data.session_start_time)))
print('# subjects included:')
print(len(np.unique(data.subj_idx)))

# add choice history information
data = more_tools.compute_choice_history(data)

# Get current date for filename
current_date = datetime.now().strftime("%Y%m%d")

# save to csv
output_file = os.path.join(data_folder_path, f'ibl_{whichTask}_{current_date}.csv')
data.to_csv(output_file, index=False)
print('Saved file to:')
print(output_file)
# %%

#First apply the lower bound RT criterion (200ms = 0.200s)
trials = trials[trials['rt'] > 0.200]

# 4.1 Apply the IQR-based exclusion per session
def apply_iqr_exclusion(group, iqr_multiplier=2):
    Q1 = group['rt'].quantile(0.25)
    Q3 = group['rt'].quantile(0.75)
    IQR = Q3 - Q1
    return group[
        (group['rt'] > (Q1 - iqr_multiplier * IQR)) & 
        (group['rt'] < (Q3 + iqr_multiplier * IQR))
    ]

# Apply the IQR exclusion for each session
trials = trials.groupby(['session_start_time']).apply(apply_iqr_exclusion).reset_index(drop=True)
