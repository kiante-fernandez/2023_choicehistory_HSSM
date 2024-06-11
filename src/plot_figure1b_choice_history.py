"""
basic choice history data on IBL trainingChoiceWorld
Anne Urai, Leiden University, 2023

"""
# %%
import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import psychofit as psy
import utils_plot as tools
import utils_choice_history as more_tools

## INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", palette="colorblind")
tools.seaborn_style()

script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
dataset = 'ibl_trainingChoiceWorld_clean.csv'
fig_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', dataset)

#load data
data = pd.read_csv(data_file_path)

# %% ================================= #
# USE THE SAME FILE AS FOR HDDM FITS
# ================================= #

data.head(n=10)
data['previous_trial'] = 100*data.prevfb + 10*data.prevresp  # for color coding
cmap = sns.color_palette("Paired")
cmap = cmap[4:]

# %% ================================= #
# simple choice history psychfuncs
# ================================= #

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(data, hue='previous_trial', palette=cmap,
					hue_order=[-90., +110.,  -100., +100.])
fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for axidx, ax in enumerate(fig.axes.flat):
        ax.set_title('c. History-dependent psychometric')
fig.despine(trim=True)
fig.savefig(os.path.join(fig_file_path, "%s_psychfuncs_history.png"%dataset), dpi=300)
plt.close('all')

# #%%  plot one curve for each animal, one panel per lab
# data['previous_outcome_name'] = data['previous_outcome'].map({1.0:'After rewarded trial',
#                                                               -1.0:'After unrewarded trial'})
# data['previous_choice_name'] = data['previous_choice'].map({1.0:'right',
#                                                               0.0:'left'})
# fig = sns.FacetGrid(data, hue='previous_choice_name',
#                     row='previous_outcome_name', row_order=['After unrewarded trial', 'After rewarded trial'])
# fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx")
# fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
# fig.set_titles("{row_name}")
# #fig._legend.set_title('Previous choice')
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "psychfuncs_history_2cols.png"))
# plt.close('all')

# #%%extra
# fig = sns.FacetGrid(data, hue='previous_choice_name')
# fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx").add_legend()
# fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
# fig.set_titles("{row_name}")
# fig._legend.set_title('Previous choice')
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "psychfuncs_history_prevchoice.png"))
# plt.close('all')

# # plot one curve for each animal, one panel per lab
# fig = sns.FacetGrid(data, col='subj_idx', col_wrap=7,
# 					hue='previous_trial',
# 					palette='Paired', hue_order=[-90., +110.,  -100., +100.])
# fig.map(tools.plot_psychometric, "signed_contrast",
#         "response", "subj_idx")
# fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "psychfuncs_history_permouse.png"))


#%% also previous history chronometric 
# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(data, hue='previous_trial', palette=cmap,
					hue_order=[-90., +110.,  -100., +100.])
fig.map(tools.plot_chronometric, "signed_contrast", "rt", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
for axidx, ax in enumerate(fig.axes.flat):
        ax.set_title('d. History-dependent chronometric')
        ax.set_ylim([0, 0.7])
fig.despine(trim=True)
fig.savefig(os.path.join(fig_file_path, "%s_chronfuncs_history.png"%dataset), dpi=300)
plt.close('all')

# %%
