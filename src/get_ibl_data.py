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
# from brainbox.behavior.training import query_criterion
from datetime import datetime
import re 

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
# Get the directory of the script being run
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
fig_folder_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
data_folder_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')

whichTask  = 'trainingChoiceWorld' # 'trainingChoiceWorld' or 'ephysChoiceWorld'

#  ============================================ #
# %% FIRST, FIND SUBJECTS OF INTEREST
# from https://www.biorxiv.org/content/10.1101/2023.12.22.573001v1.full
# use the Bruijns et al tag to grab only subjects that have the subjectTraining.table released
regexp = re.compile(r'Subjects/\w*/((\w|-)+)/_ibl') 
datasets = one.alyx.rest('datasets', 'list', tag='2023_Q4_Bruijns_et_al') # extract subject names 
subjects = np.unique(np.sort([regexp.search(ds['file_records'][0]['relative_path']).group(1) for ds in datasets])) # reduce to list of unique names 

print('number of subjects: ')
print(len(subjects))
print(subjects)

#%% SECOND, FIND WHICH SESSIONS WE'D LIKE TO USE
# date_check = []
# for subject in tqdm(subjects):

#     # what was the training status at each session?
#     eids, info = one.search(subject=subject, 
#                             dataset=['trials.table'], 
#                             details=True)
#     try: assert(len(info) > 0); #check we get something back that makes sense
#     except: continue
#     df_info = pd.DataFrame(info).sort_values(by=['lab', 'subject', 'date', 'number'])
#     df_info['date'] = pd.to_datetime(df_info['date'], utc = True)
#     date_check.append({'subject': np.repeat(subject, df_info['date'].nunique()),
#                         'source':np.repeat('trials_table', df_info['date'].nunique()),
#                         'date': df_info['date'].unique()})

#     #print('info from trials.table')
#     #print(df_info['date'].describe())
#     trials_agg = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
#     #date_check.append({'subject': subject, 'source':'subjecttrials_table','date': trials_agg['session_start_time'].unique()})

#     date_check.append({'subject': np.repeat(subject, trials_agg['session_start_time'].nunique()),
#                         'source':np.repeat('subjecttrials_table', trials_agg['session_start_time'].nunique()),
#                         'date': trials_agg['session_start_time'].unique()})
    
#     #print('info from subjectTrials.table')
#     #print(trials_agg['session_start_time'].describe())
#     training_status = one.load_aggregate('subjects', subject, 
#                                         '_ibl_subjectTraining.table').reset_index()
#     training_status['date'] = pd.to_datetime(training_status['date'], utc = True)
#     #date_check.append({'subject': subject, 'source':'subjecttraining_table','date': training_status['date'].unique()})

#     date_check.append({'subject': np.repeat(subject, training_status['date'].nunique()),
#                         'source':np.repeat('subjecttraining_table', training_status['date'].nunique()),
#                         'date': training_status['date'].unique()})

    
#     #print('info from subjectTraining.table')
#     #print(training_status['date'].describe())

# date_df = pd.concat([pd.DataFrame(d) for d in date_check])
# fig = sns.FacetGrid(date_df, col='subject', col_wrap=8, sharey=False, hue='source')
# fig.map(sns.swarmplot, 'source', 'date').add_legend()
# plt.savefig(os.path.join(fig_folder_path, 'ibl_date_check.png'))

#     #df_info_orig = df_info.copy()


    
# assert(1==0)
#sess = trials_agg['session_start_time'].unique()

#%%

big_trials_df = []
for subject in tqdm(subjects):

    # # what was the training status at each session?
    # eids, info = one.search(subject=subject, 
    #                         dataset=['trials.table'], 
    #                         details=True)
    # #try: assert(len(info) > 0); #check we get something back that makes sense
    # #except: continue
    # df_info = pd.DataFrame(info).sort_values(by=['lab', 'subject', 'date', 'number'])
    # df_info['date'] = pd.to_datetime(df_info['date'], utc = True)
    # df_info['eid'] = eids

    # try: assert (any(df_info['task_protocol'].str.contains('trainingChoiceWorld')))
    # except: print('skipping %s, no trainingChoiceWorld for '%subject); continue

    # # grab this subjects trainingTable
    # try: training_status = one.load_aggregate('subjects', subject, 
    #                                     '_ibl_subjectTraining.table').reset_index()
    # except: print('skipping %s, could not load subjectTraining.table'%subject); continue
    # training_status['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in training_status['date']]
    # training_status['date'] = pd.to_datetime(training_status['date'], utc = True)
    # # confirm that this animal reached 'trained'
    # try: assert training_status['training_status'].str.contains('trained').any()
    # except: print('skipping %s, no trained status'%subject); continue

    # # now get all of the trials for this subject as well
    # trials_agg = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

    # # Join to sessions table table - some wrangling needed to match on dates
    # df_info = df_info.set_index('date').join(training_status.set_index('date'))
    # df_info['training_status'] = df_info.training_status.fillna(method='ffill')
    # df_info = df_info.reset_index()

    ####
    # Load training status and join to trials table, https://int-brain-lab.github.io/iblenv/notebooks_external/loading_trials_data.html#Loading-all-the-sessions'-trials-for-a-single-subject-at-once
    trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
    training_table = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')
    trials = (trials
            .set_index('session')
            .join(training_table.set_index('session'))
            .sort_values(by=['session_start_time', 'intervals_0']))
    trials['training_status'] = trials.training_status.fillna(method='ffill')

    # now get all the sessions for this subject
    if whichTask == 'trainingChoiceWorld':

        # find the sessions where the animal was trained_1a OR trained_1b
        try: first_day_trained = trials[trials['training_status'].str.contains("trained")]['session_start_time'].iloc[0]
        except: print('skipping %s, could not identify first day trained'%subject); continue

        # now find the 3 days before this
        all_dates = np.sort(trials['session_start_time'].unique())
        first_trained_loc = np.where(all_dates == first_day_trained)[0][0]
        three_dates_before_trained = all_dates[first_trained_loc-3:first_trained_loc]

        # subset the dataframe to only include these sessions
        trials = trials[trials['session_start_time'].isin(three_dates_before_trained)]   

        # assert that this returns 3 sessions per mouse
        try: assert (trials['session_start_time'].nunique()) == 3
        except: print('skipping %s, did not find 3 sessions before trained'%subject); continue 

        # assert that all of these have the task protocol including trainingchoiceworld
        assert (all(trials['task_protocol'].str.contains('trainingChoiceWorld')))

        # complete the df to include all we need
        trials['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials['contrastLeft'], 
                                                        trials['contrastRight']]))
        trials['abs_contrast'] = np.abs(trials['signed_contrast'])
        trials['trialnum'] = trials.groupby(['session_start_time']).cumcount()
        # use a very crude measure of RT: trial duration 
        trials['rt'] = trials['response_times'] - trials['goCue_times']
        # additionally, save the movement onset
        trials['movement_onset'] = trials['firstMovement_times'] - trials['goCue_times']
        trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1})
        trials['correct']  = trials['feedbackType']
        trials['prior_bias'] = trials['probabilityLeft']
        # for trainingChoiceWorld, the probabilityLeft is always 0.5
        if whichTask == 'trainingChoiceWorld':
            trials['prior_bias'] = 0.5
        # retrieve the mouse name, session etc
        trials['subj_idx'] = subject
        trials['eid'] = trials.index

        # assert that we have good enough behavior in each session
        perf = trials.groupby(['session_start_time', 'abs_contrast'])['feedbackType'].mean()

        # check that performance is above 90% for all easy trials
        # check that we have all contrasts introduced in each session
        # check that we have > 400 trials in each session

        # add_to_list = True
        # for eid_to_check in trials['eid'].tolist():
        #     trials_obj = one.load_object(eid_to_check, 'trials')
        #     performance_easy = training.compute_performance_easy(trials_obj)
        #     try: assert performance_easy > 0.9
        #     except: add_to_list = False; print('skipping %s, performance too low'%subject); continue
    


    #print('adding sessions for %s'%subject)
    #print('adding %s to df'%subject)
    big_trials_df.append(trials) # append

# continue only with some columns we need
data = pd.concat(big_trials_df).reset_index()
data_orig = data.copy()
data = data[['eid', 'subj_idx', 'session_start_time', 'signed_contrast', #'prior_bias',
         'response', 'rt', 'movement_onset', 'correct', 'trialnum']]

print('# sessions included:')
print(len(np.unique(data.session_start_time)))
print('# subjects included:')
print(len(np.unique(data.subj_idx)))

# %% FOURTH, CLEAN AND PREPROCESS THE DATA

# remove RTs that sit outside the cutoff window
# define how we quantify RTs
# data['rt_raw'] = data['rt'].copy()
# data['rt'] = more_tools.clean_rts(data['rt'], cutoff=[0.08, 2], 
#                                   compare_with=None)

# add choice history information
data = more_tools.compute_choice_history(data)

# # add some more history measures
data['repeat'] = np.where(data.response == data.prevresp, 1, 0)

# save to csv   
data.to_csv(os.path.join(data_folder_path, 'ibl_%s_clean.csv'%whichTask), 
            index=False)
print('saved file to:')
print(os.path.join(data_folder_path, 'ibl_%s_clean.csv'%whichTask))

# %%
