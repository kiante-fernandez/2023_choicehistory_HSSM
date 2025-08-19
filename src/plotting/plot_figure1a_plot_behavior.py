"""
basic choice history data on IBL trainingChoiceWorld
Anne Urai, Leiden University, 2023

"""
# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils_plot as tools

## INITIALIZE A FEW THINGS
tools.seaborn_style()

# Get the directory of the script being run
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
fig_folder_path = os.path.join(script_dir, '..', '..', 'results', 'figures')
data_folder_path = os.path.join(script_dir, '..', '..', 'data')

# Construct the path to the data file
dataset = 'ibl_trainingChoiceWorld_20250819'
# dataset = 'visual_motion_2afc_fd'

#load data
data = pd.read_csv(os.path.join(data_folder_path, '%s.csv'%dataset))
if dataset == 'visual_motion_2afc_fd':
      data['signed_contrast'] = data['stimulus'] * data['coherence']

# keep the scaling by 100 to match original y-axis for plotting
data['signed_contrast'] = data['signed_contrast'] * 100 


# %% ================================= #
# REGULAR PSYCHFUNCS
# ================================= #

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_psychometric, "signed_contrast", "response",
        "subj_idx", color='lightgrey', alpha=0.3)
# add means on top
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_psychometric(data.signed_contrast, data.response,
                      data.subj_idx, ax=ax, legend=False, color='darkblue', linewidth=2)
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Choice (% right)')
#ax.set_title('Psychometric function (n = %d)'%data.subj_idx.nunique())
#ax.set_title('Psychometric function')
ax.text(20, .10, 'n = %d'%data.subj_idx.nunique())

fig.savefig(os.path.join(fig_folder_path, "%s_psychfuncs.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_psychfuncs.pdf"%dataset), dpi=300)

# %% ================================= #
# CHRONFUNCS on good RTs
# ================================= #

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_chronometric, "signed_contrast", "rt", 
    "subj_idx", color='lightgray', alpha=0.3)
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_chronometric(data.signed_contrast, data.rt,
                      data.subj_idx, ax=ax, legend=False, color='darkblue', linewidth=2)
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
#ax.set_title('b. Chronometric function (n = %d)'%data.subj_idx.nunique())
#ax.set_title('Chronometric function')

fig.savefig(os.path.join(fig_folder_path, "%s_chronfuncs.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_chronfuncs.pdf"%dataset), dpi=300)

# and RT distributions
fig = sns.FacetGrid(data, hue="subj_idx", aspect=1.5)
fig.map(sns.histplot, "rt", binwidth=0.02, element='step', fill=False,
        stat='probability',
        color='lightgray', alpha=0.7)
for axidx, ax in enumerate(fig.axes.flat):
    sns.histplot(data, x='rt', ax=ax, element='step', fill=False,
             stat='probability', binwidth=0.02, legend=False, color='darkblue', linewidth=2)
    ax.set_xlim([0, 2])
fig.despine(trim=True, offset=1)
fig.set_axis_labels('RT (s)', ' ')
fig.set_yticklabels('')
#ax.set_title('RT distributions')
fig.savefig(os.path.join(fig_folder_path, "%s_rtdist.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_rtdist.pdf"%dataset), dpi=300)

# %% ================================= #
# RT distributions per subject
# ================================= #

fig = sns.FacetGrid(data, col="subj_idx", col_wrap=np.ceil(np.sqrt(data.subj_idx.nunique())).astype(int),
                        sharex=True, sharey=False)
fig.map(sns.histplot, "rt", binwidth=0.01, element='step', color='darkblue')
fig.savefig(os.path.join(fig_folder_path, "%s_rtdist_allsj.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_rtdist_allsj.pdf"%dataset), dpi=300)

# %% ================================= #
# USE THE SAME FILE AS FOR HDDM FITS
# ================================= #

data.head(n=10)
#data['prevfb'] = (data.prevresp == data.prevstim)
data['prevresp'] = data['prevresp'].map({-1: 0, 1: 1})
data['previous_trial'] = 100*data.prevfb + 10*data.prevresp  # for color coding
print(data.groupby(['previous_trial'])[['prevfb', 'prevresp']].mean().reset_index())
cmap = sns.color_palette("Paired")
cmap = cmap[4:]
hue_order = [0., +100.,  +10., +110.]

# %% ================================= #
# simple choice history psychfuncs + chronfuncs
# ================================= #

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(data, hue='previous_trial', palette=cmap, hue_order=hue_order)
fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'Choice (% right)')
#for axidx, ax in enumerate(fig.axes.flat):
        #ax.set_title('History bias')
fig.despine(trim=True)
fig.savefig(os.path.join(fig_folder_path, "%s_psychfuncs_history.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_psychfuncs_history.pdf"%dataset), dpi=300)

plt.close('all')

#%% also previous history chronometric 
fig = sns.FacetGrid(data, hue='previous_trial', palette=cmap, hue_order=hue_order)
fig.map(tools.plot_chronometric, "signed_contrast", "rt", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
for axidx, ax in enumerate(fig.axes.flat):
        #ax.set_title('History-dependent chronometric')
        ax.set_ylim([0, 1])
fig.despine(trim=True)
fig.savefig(os.path.join(fig_folder_path, "%s_chronfuncs_history.png"%dataset), dpi=300)
fig.savefig(os.path.join(fig_folder_path, "%s_chronfuncs_history.pdf"%dataset), dpi=300)

plt.close('all')

# %%
