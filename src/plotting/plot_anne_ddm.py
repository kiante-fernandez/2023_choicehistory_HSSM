"""
plot HDDM model, with history terms, to data from IBL mice
Anne Urai, 2018 CSHL

"""
# ============================================ #
# GETTING STARTED
# ============================================ #

import pandas as pd
import scipy as sp
import os
import numpy as np

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns

# more handy imports
from utils.utils_plot import results_long2wide_hddmnn, seaborn_style, corrfunc
# import corrstats
seaborn_style()

# find path depending on location and dataset
usr = os.environ['USER']
if 'uraiae' in usr: # ALICE
    modelpath = '/home/uraiae/data1/HDDMnn/ibl_trainingchoiceworld_clean' # on ALICE
figpath = 'figures'
datapath = 'data'

# MAKE THE FIGURE, divide subplots using gridspec
pal = sns.color_palette("Paired")
pal2 = pal[2:4] + pal[0:2] + pal[8:10]

mdl_type = 'ddm'

# ============================================ #
# FIRST, MODEL COMPARISON - DDM, angle, weibull
# ============================================ #

models = ['ddm_nohist', 'angle_nohist', 'weibull_nohist']
mdcomp = pd.DataFrame()
for mod in models:
    print(mod)
    tmp_md = pd.read_csv(os.path.join(modelpath, mod, 'model_comparison.csv'))
    tmp_md['model'] = mod
    mdcomp = pd.concat([mdcomp, tmp_md])

# subtract baseline model
# cols = ['aic', 'bic', 'dic']
# for c in cols:
#     mdcomp[c] = mdcomp[c] - mdcomp.loc[mdcomp.model == 'ddm_nohist', c]
# mdcomp = mdcomp[mdcomp.model != 'ddm_nohist']

fig, ax = plt.subplots(ncols=3, nrows=1)
pal3 = pal[3:4] + pal[1:2] + pal[9:10]
for ax_idx, mdl_comp_idx in enumerate(['dic', 'aic', 'bic']):
    sns.barplot(x="model", y=mdl_comp_idx, ax=ax[ax_idx], data=mdcomp, 
                #bottom=mdcomp.loc[mdcomp.model == 'ddm_nohist', mdl_comp_idx].item(), 
                palette=pal3)
    ax[ax_idx].set(ylabel=r'$\Delta$' + mdl_comp_idx.upper(), xlabel='', xticklabels=['DDM', 'angle', 'weibull'])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=40, ha='right')
sns.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(figpath, 'modelcomp_ddm_angle_weibull.png'))

# ============================================ #
# COMPUTE HISTORY SHIFT AND CORRELATE WITH BEHAVIOR
# ============================================ #

data = pd.read_csv(os.path.join(datapath, 'ibl_trainingchoiceworld_clean.csv'))

data['repeat'] = (data.response == data.prevresp)
rep = data.groupby(['subj_idx', 'prevfb'])['repeat'].mean().reset_index()
rep = rep.pivot(index='subj_idx', columns='prevfb', values='repeat').reset_index()
rep = rep.rename(columns={1.0: 'repeat_prevcorrect', -1.0: 'repeat_preverror'})
# also add a measure of repetition without previous outcome
rep2 = data.groupby(['subj_idx'])['repeat'].mean().reset_index()
rep = pd.merge(rep, rep2, on='subj_idx')
rep = rep.sort_values(by=['repeat'])

# ============================================ #
# plot overall repetition barplot
# ============================================ #

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,4))
g = sns.scatterplot(y="subj_idx", x="repeat",
                data=rep, color="0.3", ax=ax)
#sns.despine(trim=True)
ax.set(ylabel='# Mouse', xlabel='P(repeat)', yticklabels=[])
ax.axvline(x=0.5, color='darkgrey')
#fig.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(figpath, 'choice_repetition.png'))

# ============================================ #
# FIRST, MODEL COMPARISON
# ============================================ #

models = ['nohist',  'prevresp_z',
          'prevresp_v',
          'prevresp_zv']
mdcomp = pd.DataFrame()
for mod in models:
    print(mod)
    tmp_md = pd.read_csv(os.path.join(modelpath, mdl_type + '_' + mod, 'model_comparison.csv'))
    tmp_md['model'] = mdl_type + '_' + mod
    mdcomp = pd.concat([mdcomp, tmp_md])

# subtract baseline model
cols = ['aic', 'bic', 'dic']
for c in cols:
    mdcomp[c] = mdcomp[c] - mdcomp.loc[mdcomp.model == mdl_type + '_nohist', c].item()
mdcomp = mdcomp[mdcomp.model != mdl_type + '_nohist']

fig, ax = plt.subplots(ncols=3, nrows=1)
pal3 = pal[3:4] + pal[1:2] + pal[9:10]
# sns.barplot(x="model", y="dic", ax=ax[0], data=mdcomp, palette=pal3)
# ax[0].set(ylabel=r'$\Delta$DIC', xlabel='', xticklabels=['z', 'vbias', 'both'])
# sns.barplot(x="model", y="aic", ax=ax[1], data=mdcomp, palette=pal3)
# ax[1].set(ylabel=r'$\Delta$AIC', xlabel='', xticklabels=['z', 'vbias', 'both'])
# sns.barplot(x="model", y="bic", ax=ax[2], data=mdcomp, palette=pal3)
# ax[2].set(ylabel=r'$\Delta$BIC', xlabel='', xticklabels=['z', 'vbias', 'both'])

for ax_idx, mdl_comp_idx in enumerate(['dic', 'aic', 'bic']):
    sns.barplot(x="model", y=mdl_comp_idx, ax=ax[ax_idx], data=mdcomp, 
                #bottom=mdcomp.loc[mdcomp.model == 'ddm_nohist', mdl_comp_idx].item(), 
                palette=pal3)
    ax[ax_idx].set(ylabel=r'$\Delta$' + mdl_comp_idx.upper(), xlabel='', xticklabels=['z', 'vbias', 'both'])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=40, ha='right')
fig.tight_layout()
fig.savefig(os.path.join(figpath, 'modelcomp_%s.png'%mdl_type))


# ============================================ #
# now load in the models we need
# ============================================ #

md_wide = results_long2wide_hddmnn(pd.read_csv(os.path.join(modelpath, mdl_type + '_prevresp_zv', 'results_combined.csv')))

# COMPUTE THE SAME THING FROM HDDM COLUMN NAMES
# md_wide['dcshift'] = md_wide['dc']['1.0'] - md_wide['dc']['0.0']
# md_wide['zshift'] = md_wide['z']['1.0'] - md_wide['z']['0.0']
# md_wide = md_wide[['subj_idx', 'dcshift', 'zshift']]
# md_wide.columns = md_wide.columns.droplevel(1)
md_wide = pd.merge(md_wide, rep, on='subj_idx')

fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(6,3))
corrfunc(x=md_wide.z_prevresp, y=md_wide.repeat, ax=ax[0], color=pal2[1])
ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')
corrfunc(x=md_wide.v_prevresp, y=md_wide.repeat, ax=ax[1], color=pal2[3])
ax[1].set(xlabel='History shift in drift bias', ylabel=' ')

# ADD STEIGERS TEST ON TOP
# x = repeat, y = zshift, z = dcshift
tstat, pval = corrstats.dependent_corr(sp.stats.spearmanr(md_wide.z_prevresp, md_wide.repeat, nan_policy='omit')[0],
                                       sp.stats.spearmanr(md_wide.v_prevresp, md_wide.repeat, nan_policy='omit')[0],
                                       sp.stats.spearmanr(md_wide.z_prevresp, md_wide.v_prevresp, nan_policy='omit')[0],
                                        len(md_wide),
                                       twotailed=True, conf_level=0.95, method='steiger')
deltarho = sp.stats.spearmanr(md_wide.z_prevresp, md_wide.repeat, nan_policy='omit')[0] - \
           sp.stats.spearmanr(md_wide.v_prevresp, md_wide.repeat, nan_policy='omit')[0]
if pval < 0.0001:
    fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho), fontsize=10)
else:
    fig.suptitle(r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, 'scatterplot_trainingchoiceworld_%s_prevchoice_zv.png'%mdl_type))

# ============================================ #
# ADD MODELS WITH PREVIOUS OUTCOME MODULATION
# ============================================ #

# md = pd.read_csv(os.path.join(modelpath, 'prevchoiceoutcome_dcz', 'results_combined.csv'))
# md_wide = results_long2wide_hddmnn(md)

# # COMPUTE THE SAME THING FROM HDDM COLUMN NAMES
# md_wide['dcshift_preverror']   = md_wide['dc']['1.0.-1.0'] - md_wide['dc']['0.0.-1.0']
# md_wide['dcshift_prevcorrect'] = md_wide['dc']['1.0.1.0'] - md_wide['dc']['0.0.1.0']
# md_wide['zshift_preverror']    = md_wide['z']['1.0.-1.0'] - md_wide['z']['0.0.-1.0']
# md_wide['zshift_prevcorrect']  = md_wide['z']['1.0.1.0'] - md_wide['z']['0.0.1.0']
# md_wide = md_wide[['subj_idx', 'dcshift_preverror', 'dcshift_prevcorrect', 'zshift_preverror', 'zshift_prevcorrect']]
# md_wide.columns = md_wide.columns.droplevel(1)

# # md_wide.dropna(inplace=True)
# md_wide = pd.merge(md_wide, rep, on='subj_idx')

# fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=False)
# corrfunc(x=md_wide.zshift_prevcorrect, y=md_wide.repeat_prevcorrect, ax=ax[0,0], color='0.3')
# ax[0,0].set(xlabel='', ylabel='P(repeat) after correct')
# corrfunc(x=md_wide.dcshift_prevcorrect, y=md_wide.repeat_prevcorrect, ax=ax[0,1], color='0.3')
# ax[0,1].set(xlabel='', ylabel=' ')
# corrfunc(x=md_wide.zshift_preverror, y=md_wide.repeat_preverror, ax=ax[1,0], color='firebrick')
# ax[1,0].set(xlabel='History shift in z', ylabel='P(repeat) after error')
# corrfunc(x=md_wide.dcshift_preverror, y=md_wide.repeat_preverror, ax=ax[1,1], color='firebrick')
# ax[1,1].set(xlabel='History shift in drift bias', ylabel='')

# sns.despine(trim=True)
# plt.tight_layout()
# fig.savefig(os.path.join(figpath, 'scatterplot_trainingchoiceworld_prevchoiceoutcome_dcz.pdf'))

# # =============== #
# # ADD STATS BETWEEN THE TWO CORRELATION COEFFICIENTS
# # =============== #

# # huge correlation plot
# g = sns.pairplot(md_wide, vars=['repeat_prevcorrect', 'repeat_preverror',
#                           'zshift_prevcorrect', 'zshift_preverror',
#                          'dcshift_prevcorrect', 'dcshift_preverror'],
#                  kind='reg')
# g.savefig(os.path.join(figpath, 'scatterplot_prevchoiceoutcome_dcz_allvars.pdf'))

# print(md_wide[['repeat_prevcorrect', 'repeat_preverror',
#                           'zshift_prevcorrect', 'zshift_preverror',
#                          'dcshift_prevcorrect', 'dcshift_preverror']].corr())

# # ============================================ #
# # CONDITIONAL BIAS FUNCTIONS
# # ============================================ #


