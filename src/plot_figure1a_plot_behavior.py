"""
basic choice history data on IBL trainingChoiceWorld
Anne Urai, Leiden University, 2023

"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import utils_plot as tools

## INITIALIZE A FEW THINGS
tools.seaborn_style()

# Get the directory of the script being run
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
dataset = 'ibl_trainingChoiceWorld_clean'
fig_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM','results', 'figures')
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data', dataset + '.csv')

#load data
data = pd.read_csv(data_file_path)

# %% ================================= #
# REGULAR PSYCHFUNCS
# ================================= #

#data['signed_contrast'] = data['coherence'] * data['stimulus']

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_psychometric, "signed_contrast", "response",
        "subj_idx", color='lightgrey', alpha=0.3)
# add means on top
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_psychometric(data.signed_contrast, data.response,
                      data.subj_idx, ax=ax, legend=False, color='darkblue', linewidth=2)

#fig.map(sns.lineplot, "signed_contrast", "response", color='gray', alpha=0.7)     
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
ax.set_title('a. Psychometric function (n = %d)'%data.subj_idx.nunique())

fig.savefig(os.path.join(fig_file_path, "%s_psychfuncs.png"%dataset), dpi=300)
# fig.savefig(os.path.join(fig_file_path, "psychfuncs_allmice.pdf"))

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
ax.set_title('b. Chronometric function (n = %d)'%data.subj_idx.nunique())
fig.savefig(os.path.join(fig_file_path, "%s_chronfuncs.png"%dataset), dpi=300)
# fig.savefig(os.path.join(fig_file_path, "chronfuncs_allmice.pdf"))

# and RT distributions
fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(sns.histplot, "rt", binwidth=0.01, element='step', fill=False,
        stat='probability',
        color='lightgray', alpha=0.7)
for axidx, ax in enumerate(fig.axes.flat):
    sns.histplot(data, x='rt', ax=ax, element='step', fill=False,
             stat='probability', binwidth=0.01, legend=False, color='darkblue', linewidth=2)
    ax.set_xlim([0, 1.5])
fig.despine(trim=True, offset=1)
fig.set_axis_labels('RT (s)', ' ')
ax.set_title('RT distributions')
fig.savefig(os.path.join(fig_file_path, "%s_rtdist.png"%dataset), dpi=300)

# DISTRIBUTION OF RESULTING RTS
data['rt_raw'] = data['rt'].copy()
rt_cutoff = [0.08, 2]
# make strings for bin labels based on rt_cutoff values
bin_labels = ['< %.2fs'%(rt_cutoff[0]), 
              '%.2fs - %ds'%((rt_cutoff[0]), int(rt_cutoff[1])), 
              '> %ds'%int(rt_cutoff[1])]
data['rt_raw_category'] = pd.cut(data['rt_raw'], 
                                       bins=[data.rt_raw.min(), rt_cutoff[0], rt_cutoff[1], data.rt_raw.max()],
                                       labels=bin_labels, right=True)

# squash for easier plotting - to show all slow trials as 1 bin 
data.loc[data.rt_raw > rt_cutoff[1], 'rt_raw'] = rt_cutoff[1] 

# use FacetGrid to ensure the same figure size (approximately)
fig = sns.FacetGrid(data)
for axidx, ax in enumerate(fig.axes.flat):
        sns.histplot(data, x="rt_raw", hue='rt_raw_category',common_bins=True,
                palette=['lightgrey', 'darkblue', 'lightgrey'], legend=False, ax=ax)
        ax.set(xlabel='RT (s)', xlim=[-0.1, rt_cutoff[1] + 0.05], 
                yticklabels=[],
                title='RT exclusion')
sns.despine(trim=True)

# annotate: how many trials are below the lower cutoff, and how many are above the higher cutoff?
percent_below = (data.rt_raw < rt_cutoff[0]).mean() * 100
percent_above = (data.rt_raw >= rt_cutoff[1]).mean() * 100
plt.annotate('%d%%'%percent_below, xy=(rt_cutoff[0]/2, 2000), ha='center', fontsize=7)
plt.annotate('%d%%'%percent_above, xy=(rt_cutoff[1]-0.1, 3000), ha='center', fontsize=7)
fig.savefig(os.path.join(fig_file_path, "%s_rtdist_cleaned.png"%dataset), dpi=300)


# %% ================================= #
# USE THE SAME FILE AS FOR HDDM FITS
# ================================= #

data.head(n=10)
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
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for axidx, ax in enumerate(fig.axes.flat):
        ax.set_title('c. History-dependent psychometric')
fig.despine(trim=True)
fig.savefig(os.path.join(fig_file_path, "%s_psychfuncs_history.png"%dataset), dpi=300)
plt.close('all')

#%% also previous history chronometric 
# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(data, hue='previous_trial', palette=cmap, hue_order=hue_order)
fig.map(tools.plot_chronometric, "signed_contrast", "rt", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
for axidx, ax in enumerate(fig.axes.flat):
        ax.set_title('d. History-dependent chronometric')
        ax.set_ylim([0, 3])
fig.despine(trim=True)
fig.savefig(os.path.join(fig_file_path, "%s_chronfuncs_history.png"%dataset), dpi=300)
plt.close('all')

# %%
