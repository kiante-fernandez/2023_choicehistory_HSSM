# -*- coding: utf-8 -*-
"""
General functions and queries for the analysis of behavioral data from the IBL task

Guido Meijer, Anne Urai, Alejandro Pan Vazquez & Miles Wells
16 Jan 2020
"""
import os
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "lines.linewidth": 1,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": False,
                "savefig.dpi": 300,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def figpath():
    # Retrieve absolute path of paper-behavior dir
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    # Make figure directory
    fig_dir = os.path.join(repo_dir, 'exported_figs')
    # If doesn't already exist, create
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    return fig_dir


def datapath():
    """
    Return the location of data directory
    """
   # Retrieve absolute path of paper-behavior dir
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    # Make figure directory
    data_dir = os.path.join(repo_dir, 'data')
    # If doesn't already exist, create
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir

# ================================================================== #
# DEFINE PSYCHFUNCFIT TO WORK WITH FACETGRID IN SEABORN
# ================================================================== #

def fit_psychfunc(df):

    import psychofit as psy

    choicedat = df.groupby('signed_contrast').agg(
        {'choice': 'count', 'choice2': 'mean'}).reset_index()
    if len(choicedat) >= 4: # need some minimum number of unique x-values
        pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [choicedat['signed_contrast'].min(), 5, 0., 0.]),
                                 parmax=np.array([choicedat['signed_contrast'].max(), 40., 1, 1]))
    else:
        pars = [np.nan, np.nan, np.nan, np.nan]

    df2 = {'bias': pars[0], 'threshold': pars[1],
           'lapselow': pars[2], 'lapsehigh': pars[3]}
    df2 = pd.DataFrame(df2, index=[0])

    df2['ntrials'] = df['choice'].count()

    return df2


def plot_psychometric(x, y, subj, **kwargs):

    import psychofit as psy

    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y, 'subject_nickname': subj})
    df2 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()
    df2.rename(columns={"choice2": "ntrials",
                        "choice": "fraction"}, inplace=True)
    df2 = df2.groupby(['signed_contrast'])[['ntrials', 'fraction']].mean().reset_index()
    #df2 = df2[['signed_contrast', 'ntrials', 'fraction']]

    # only 'break' the x-axis and remove 50% contrast when 0% is present
    # print(df2.signed_contrast.unique())
    if 0. in df2.signed_contrast.values and 100. in df2.signed_contrast.values:
        brokenXaxis = True
    else:
        brokenXaxis = False

    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['signed_contrast'].min(), 5, 0., 0.]),
                                 parmax=np.array([df2['signed_contrast'].max(), 40., 1, 1]))

    if brokenXaxis:
        # plot psychfunc
        g = sns.lineplot(x=np.arange(-27, 27),
                         y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)), **kwargs)

        # plot psychfunc: -100, +100
        sns.lineplot(x=np.arange(-36, -31),
                     y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)), **kwargs)
        sns.lineplot(x=np.arange(31, 36),
                     y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)), **kwargs)

        # if there are any points at -50, 50 left, remove those
        if 50 in df.signed_contrast.values or -50 in df.signed_contrast.values:
            df.drop(df[(df['signed_contrast'] == -50.) | (df['signed_contrast'] == 50)].index,
                    inplace=True)

        # now break the x-axis
        df['signed_contrast'] = df['signed_contrast'].replace(-100, -35)
        df['signed_contrast'] = df['signed_contrast'].replace(100, 35)

    else:
        # plot psychfunc
        g = sns.lineplot(x=np.arange(-103, 103),
                         y=psy.erf_psycho_2gammas(pars, np.arange(-103, 103)), **kwargs)

    df3 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()

    # plot datapoints with errorbars on top
    if df['subject_nickname'].nunique() > 1:
        # put the kwargs into a merged dict, so that overriding does not cause an error
        sns.lineplot(x=df3['signed_contrast'], y=df3['choice'],
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'errorbar':('ci', 95)}, **kwargs})

    if brokenXaxis:
        g.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
        g.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                          size='small', rotation=60)
        g.set_xlim([-40, 40])
        break_xaxis()

    else:
        g.set_xticks([-100, -50, 0, 50, 100])
        g.set_xticklabels(['-100', '-50', '0', '50', '100'],
                          size='small', rotation=60)
        g.set_xlim([-110, 110])

    g.set_ylim([0, 1.02])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1])
    g.set_yticklabels(['0', '25', '50', '75', '100'])


def plot_chronometric(x, y, subj, **kwargs):

    df = pd.DataFrame(
        {'signed_contrast': x, 'rt': y, 'subject_nickname': subj})
    df.dropna(inplace=True)  # ignore NaN RTs
    df2 = df.groupby(['signed_contrast', 'subject_nickname']
                     ).agg({'rt': 'median'}).reset_index()
    # df2 = df2.groupby(['signed_contrast']).mean().reset_index()
    df2 = df2[['signed_contrast', 'rt', 'subject_nickname']]

    # only 'break' the x-axis and remove 50% contrast when 0% is present
    # print(df2.signed_contrast.unique())
    if 0. in df2.signed_contrast.values and 100. in df2.signed_contrast.values:
        brokenXaxis = True

        df2['signed_contrast'] = df2['signed_contrast'].replace(-100, -35)
        df2['signed_contrast'] = df2['signed_contrast'].replace(100, 35)
        df2 = df2.loc[np.abs(df2.signed_contrast) != 50, :] # remove those

    else:
        brokenXaxis = False

    ax = sns.lineplot(x='signed_contrast', y='rt', err_style="bars", mew=0.5,
                      errorbar=('ci', 95), data=df2, **kwargs)

    # all the points
    if df['subject_nickname'].nunique() > 1:
        sns.lineplot(
            x='signed_contrast',
            y='rt',
            data=df2,
            **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'errorbar':('ci', 95)}, **kwargs})

    if brokenXaxis:
        ax.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
        ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                          size='small', rotation=60)
        ax.set_xlim([-40, 40])
        break_xaxis()
        ax.set_ylim([0, df2['rt'].max()*1.1])

    else:
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(['-100', '-50', '0', '50', '100'],
                          size='small', rotation=60)
        ax.set_xlim([-110, 110])


def break_xaxis(y=0, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')


def add_n(x, y, sj, **kwargs):

    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y, 'subject_nickname': sj})

    # ADD TEXT ABOUT NUMBER OF ANIMALS AND TRIALS
    plt.text(
        15,
        0.2,
        '%d mice, %d trials' %
        (df.subject_nickname.nunique(),
         df.choice.count()),
        fontweight='normal',
        fontsize=6,
        color='k')


def num_star(pvalue):
    if pvalue < 0.0001:
        stars = '**** p < 0.0001'
    elif pvalue < 0.001:
        stars = '*** p < 0.001'
    elif pvalue < 0.01:
        stars = '** p < 0.01'
    elif pvalue < 0.05:
        stars = '* p < 0.05'
    else:
        stars = ''
    return stars


# ============================================ #
# ANNOTATE THE CORRELATION PLOT
# ============================================ #

def corrfunc(x, y, **kws):

    # compute spearmans correlation across age groups
    r, pval = sp.stats.spearmanr(x, y, nan_policy='omit')
    print('%s, %s, %.2f, %.3f'%(x.name, y.name, r, pval))

    if 'ax' in kws.keys():
        ax = kws['ax']
    else:
        ax = plt.gca()

    # if this correlates, draw a regression line across groups
    if pval < 0.05/4:
        data = pd.DataFrame({'x':x, 'y':y})
        sns.regplot(data=data, x='x', y='y', truncate=True, color='gray',
                    scatter=False, ci=None, robust=True, ax=ax)

    # now plot the datapoint, with age groups
    if 'yerr' in kws.keys():
        ax.errorbar(x, y, yerr=kws['yerr'].values, fmt='none', zorder=0, ecolor='silver', elinewidth=0.5)
        kws.pop('yerr', None)
    sns.scatterplot(x=x, y=y, legend=False, **kws)

    # annotate with the correlation coefficient + n-2 degrees of freedom
    txt = r"$\rho$({}) = {:.3f}".format(len(x)-2, r) + "\n" + "p = {:.4f}".format(pval)
    if pval < 0.0001:
        txt = r"$\rho$({}) = {:.3f}".format(len(x)-2, r) + "\n" + "p < 0.0001"
    ax.annotate(txt, xy=(.7, .1), xycoords='axes fraction', fontsize='small')



def results_long2wide(md):

    # recode to something more useful
    # 0. replace x_subj(yy).ZZZZ with x(yy)_subj.ZZZZ
    md["colname_tmp"] = md["index"].replace('.+\_subj\(.+\)\..+', '.+\(.+\)\_subj\..+', regex=True)

    # 1. separate the subject from the parameter
    new = md["index"].str.split("_subj.", n=1, expand=True)
    md["parameter"] = new[0]
    md["subj_idx"] = new[1]
    new = md["subj_idx"].str.split("\)\.", n=1, expand=True)

    # separate out subject idx and parameter value
    for index, row in new.iterrows():
        if row[1] == None:
            row[1] = row[0]
            row[0] = None

    md["parameter_condition"] = new[0]
    md["subj_idx"] = new[1]

    # pivot to put parameters as column names and subjects as row names
    md = md.drop('index', axis=1)
    md = md.drop('Unnamed: 0', axis=1)
    md_wide = md.pivot_table(index=['subj_idx'], values='mean',
                             columns=['parameter', 'parameter_condition']).reset_index()
    return md_wide

def results_long2wide_hddmnn(md, name_col="index", val_col='mean'):
    # Anne Urai, 2022: include a little parser that returns a more manageable output
    # can be used on full_parameter_dict from hddm_dataset_generators.simulator_h_c
    # or on the output of gen_stats()
    import re # regexp

    # recode to something more useful
    # 0. replace x_subj(yy).ZZZZ with x(yy)_subj.ZZZZ
    md["colname_tmp"] = [re.sub(".+\_subj\(.+\)\..+", ".+\(.+\)\_subj\..+", i) for i in list(md[name_col])]

    # 1. separate the subject from the parameter
    new = md[name_col].str.split("_subj.", n=1, expand=True)
    md["parameter"] = new[0]
    md["subj_idx"] = new[1]

    # only run this below if it's not a regression model!
    if not any(md[name_col].str.contains('Intercept', case=False)) \
        and not any(md[name_col].str.contains('indirect', case=False)):
        new = md["subj_idx"].str.split("\)\.", n=1, expand=True)
        # separate out subject idx and parameter value
        for index, row in new.iterrows():
            if row[1] == None:
                row[1] = row[0]
                row[0] = None

        md["parameter_condition"] = new[0]
        md["subj_idx"] = new[1]

        # pivot to put parameters as column names and subjects as row names
        md = md.drop(name_col, axis=1)
        md_wide = md.pivot_table(index=['subj_idx'], values=val_col,
                                 columns=['parameter', 'parameter_condition']).reset_index()
    else:
        # pivot to put parameters as column names and subjects as row names
        md = md.drop(name_col, axis=1)
        md_wide = md.pivot_table(index=['subj_idx'], values=val_col,
                                 columns=['parameter']).reset_index()
        
    return md_wide