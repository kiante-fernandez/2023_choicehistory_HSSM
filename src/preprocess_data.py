"""
Preprocess RTs and add choice history information
Anne Urai, Leiden University, 2023

"""
# %%
import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import brainbox as bb
import utils_plot as tools
import utils_choice_history as more_tools


# Get the directory of the script being run
script_dir = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the data file
fig_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')

tools.seaborn_style()

# set some thresholds
rt_variable_name = 'trial_duration' # trial_duration or firstmove_time... decide
rt_cutoff = [0.120, 2] # 80ms, 2s - from BWM paper

# %% ================================= #

data = pd.read_csv(os.path.join(data_file_path, 'ibl_trainingchoiceworld_raw.csv'))

# remove RTs that sit outside the cutoff window
# define how we quantify RTs
data['rt_raw'] = data[rt_variable_name].copy()
data['rt'] = more_tools.clean_rts(data[rt_variable_name], cutoff=rt_cutoff, 
                                  compare_with=None)

# add choice history information
data_clean = more_tools.compute_choice_history(data)

# rescale contrast, so that we can enter as a linear term for the drift rate
data_clean['stimulus'] = more_tools.rescale_contrast(data_clean['signed_contrast'])

# save to csv   
data_clean.to_csv(os.path.join(data_file_path, 'ibl_trainingchoiceworld_clean.csv'), index=False)

# %% ================================= #
# DISTRIBUTION OF RESULTING RTS

data['rt_raw'] = data[rt_variable_name].copy()

# make strings for bin labels based on rt_cutoff values
bin_labels = ['< %dms'%int(rt_cutoff[0]*1000), 
              '%dms - %ds'%(int(rt_cutoff[0]*1000), int(rt_cutoff[1])), 
              '> %ds'%int(rt_cutoff[1])]
data_clean['rt_raw_category'] = pd.cut(data['rt_raw'], 
                                       bins=[data.rt_raw.min(), rt_cutoff[0], rt_cutoff[1], data.rt_raw.max()],
                                       labels=bin_labels, right=True)

# squash for easier plotting - to show all slow trials as 1 bin 
data_clean.rt_raw[data_clean.rt_raw > rt_cutoff[1]] = rt_cutoff[1] 

# use FacetGrid to ensure the same figure size (approximately)
fig = sns.FacetGrid(data=data_clean, hue='rt_raw_category',
                  palette=['lightgrey', 'darkblue', 'lightgrey'])
fig.map(sns.histplot, "rt_raw", multiple='stack', legend=False, binwidth=0.06)

if rt_variable_name == 'trial_duration':
    xlabel = 'Trial duration (s)'
elif rt_variable_name == 'firstmove_time':
    xlabel = 'Time of first movement (s)'

for axidx, ax in enumerate(fig.axes.flat):
    ax.set(xlabel=xlabel, xlim=[-0.1, rt_cutoff[1] + 0.05], 
        yticklabels=[],
        title='a. Reaction time distributions')
sns.despine(trim=True)

# annotate: how many trials are below the lower cutoff, and how many are above the higher cutoff?
percent_below = (data_clean.rt_raw < rt_cutoff[0]).mean() * 100
percent_above = (data_clean.rt_raw >= rt_cutoff[1]).mean() * 100
plt.annotate('%d%%'%percent_below, xy=(rt_cutoff[0]/2, 2000), ha='center', fontsize=7)
plt.annotate('%d%%'%percent_above, xy=(rt_cutoff[1]-0.1, 3000), ha='center', fontsize=7)

plt.savefig(os.path.join(fig_file_path, "rt_raw_distributions.png"))
plt.savefig(os.path.join(fig_file_path, "rt_raw_distributions.pdf"))

#%% now plot the same, but one panel per mouse
plt.savefig(os.path.join(figpath, "rt_raw_distributions_allsj.png"))
fig = sns.FacetGrid(data=data_clean, col='subj_idx', hue='rt_raw_category', 
                    palette=['lightgrey', 'darkblue', 'lightgrey'],
                    col_wrap=6, sharex=True, sharey=False)
fig.map(sns.histplot, 'rt_raw', binwidth=0.075)
fig.set(xlabel=xlabel, xlim=[-0.1, rt_cutoff[1] + 0.05], 
        yticklabels=[])
sns.despine(trim=True)
plt.savefig(os.path.join(fig_file_path, "rt_raw_distributions_allsj.png"))
plt.savefig(os.path.join(fig_file_path, "rt_raw_distributions_allsj.pdf"))

# # %% ================================= #
# #  SUPP: RT CDF
# rt_cutoff = 10

# f, ax = plt.subplots(1,1,figsize=[3,3])
# hist, bins = np.histogram(data.rt, bins=1000)
# logbins = np.append(0, np.logspace(np.log10(bins[1]), np.log10(bins[-1]), len(bins)))
# ax.hist(data.rt, bins=logbins, cumulative=True, 
#     density=True, histtype='step')
# ax.set_xscale('log')

# # indicate the cutoff we use here
# ax.set_xlabel("RT (s)")
# ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: (
# 	'{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
# ax.set_ylabel('CDF')
# # indicate the percentage of trials excluded by rt cutoff
# perc = (data.rt < rt_cutoff).mean()
# sns.lineplot(x=[0, rt_cutoff], y=[perc, perc], style=0, color='k', 
#             dashes={0: (2, 1)}, lw=1, legend=False)
# sns.lineplot(x=[rt_cutoff, rt_cutoff], y=[0, perc], style=0, color='k', 
#             dashes={0: (2, 1)}, lw=1, legend=False)
# ax.set(ylim=[-0.01, 1], xlim=[0.02, 59])
# sns.despine()
# plt.tight_layout()
# f.savefig(os.path.join(figpath, "rt_cdf.png"))
# # ToDo: where in the session do slow RTs occur? presumably at the end

# %% ================================= #
# # PLOT THE RESULTS OF THIS PREPROCESSING

# # how much are the first movement time and the trial duration related?
# data_clean['abs_contrast']  = np.abs(data_clean['signed_contrast'])
# rt_summ = data_clean.groupby(['subj_idx','abs_contrast'])[['rt_raw', 'trial_duration']].median()
# fig = sns.scatterplot(data=rt_summ, x='rt_raw', y='trial_duration', hue='abs_contrast', marker='.')
# fig.set(xscale="log", yscale="log")

# data_clean['rt_diff'] = data_clean['trial_duration'] - data_clean['rt_raw']
# data_clean.rt_diff[data_clean.rt_diff > 3] = 3 # squash for easier plotting
# fig = sns.displot(data=data_clean, x='rt_diff')
# fig.set(xscale="log")
