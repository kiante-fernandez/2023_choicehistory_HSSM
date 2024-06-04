# compare_hssm_hddm_2afc_urai_2019.py - computational correspondece reproducibility anaysis of 2afc fd from 
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 

# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2024/06/03      Kianté Fernandez<kiantefernan@gmail.com>   moved code from draft version

# %% load packages
import sys, os, time
import arviz as az
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from hssm import set_floatX
#import hssm 

#%matplotlib inline
set_floatX("float32")
#%% load data
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
results_dir = os.path.join(script_dir, '..', 'results', 'figures')

elife_data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd.csv'))

elife_data['signed_contrast'] = elife_data['coherence'] * elife_data['stimulus']
elife_data['response'] = elife_data['response'].replace({0: -1, 1: 1})
elife_data['rt'] = elife_data['rt'].round(5)
elife_data['participant_id'] = elife_data['subj_idx']
#%% single subject parameter estimation
participants = elife_data['participant_id'].unique()

# Initialize a dictionary to store summaries
summaries = {}

# Loop over each participant
for participant in participants:
    # Subset data for the current participant
    subj_dataset = elife_data[elife_data['participant_id'] == participant]

    # Define model
    sub_hssm_model = hssm.HSSM(data=subj_dataset,
                               model="ddm",
                            #    model="full_ddm",
                               include=[
        {
            "name": "v",
            "formula":"v ~ 1 + signed_contrast + prevresp"
        },
        {
            "name": "z",
            "formula":"z ~ 1 + prevresp",
        }
    ],                          #change depends on which comparison (could create larger loop for each)
                                # loglik_kind="blackbox",
                                # sv = 0, sz = 0, st = 0)      
                                # loglik_kind="analytical")
                               loglik_kind="approx_differentiable")

    # Estimate model
    subj_sample_res = sub_hssm_model.sample(
        sampler="nuts_numpyro",
        # sampler="mcmc",
        cores=4,
        chains=4,
        draws=1000,
        tune=1000,
        # step = pm.Slice(model=sub_hssm_model.pymc_model)
    )
    # Create summary and store it in the dictionary
    summaries[participant] = az.summary(subj_sample_res)

 #save results
for participants, dataframe in summaries.items():
    # filename = f'HSSM_M4_fit_{participants}_MCMC.csv'
    #filename = f'HSSM_M4_fit_{participants}_nuts_numpyro_approx_differentiable.csv'
    # filename = f'HSSM_M4_fit_{participants}_MCMC_analytical.csv'
    # filename = f'HSSM_M4_fit_{participants}_MCMC_SLICE.csv'

    dataframe.to_csv(filename, index=True)
    print(f'Saved data for participant {participants} as {filename}')

# %% after parameter estimation, load files above
def load_files(base_path):
    dataframes_mcmc = {}
    dataframes_slice = {}
    dataframes_nuts_analytical = {}
    dataframes_approx_differentiable = {}

    # Find all MCMC and SLICE files
    mcmc_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC.csv'))
    slice_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC_SLICE.csv'))
    analytical_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_MCMC_analytical.csv'))
    approx_differentiable_files = glob.glob(os.path.join(base_path, 'HSSM_M4_fit_*_nuts_numpyro_approx_differentiable.csv'))

    # Load SLICE files and add participant_id, retain index
    for file in slice_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_slice = pd.read_csv(file, index_col=0)
        df_slice['participant_id'] = subject_number
        df_slice['sampler'] = "slice"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_slice[subject_number] = df_slice.reset_index().rename(columns={'index': 'parameter'})

    # Load MCMC files and add participant_id, set parameter based on corresponding SLICE file
    for file in mcmc_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_mcmc = pd.read_csv(file)
        df_mcmc['participant_id'] = subject_number
        df_mcmc['sampler'] = "nuts"
        # Assign parameter column from corresponding SLICE DataFrame
        if subject_number in dataframes_slice:
            df_mcmc['parameter'] = dataframes_slice[subject_number]['parameter'].values
        dataframes_mcmc[subject_number] = df_mcmc

    for file in analytical_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_ana = pd.read_csv(file, index_col=0)
        df_ana['participant_id'] = subject_number
        df_ana['sampler'] = "nuts_analytical"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_nuts_analytical[subject_number] = df_ana.reset_index().rename(columns={'index': 'parameter'})
        
    for file in approx_differentiable_files:
        subject_number = os.path.basename(file).split('_')[3]
        df_LAN = pd.read_csv(file, index_col=0)
        df_LAN['participant_id'] = subject_number
        df_LAN['sampler'] = "numpyro_approx_differentiable"
        # Reset index to make 'parameter' column and store it in dictionary
        dataframes_approx_differentiable[subject_number] = df_LAN.reset_index().rename(columns={'index': 'parameter'})
      
      
    return dataframes_mcmc, dataframes_slice, dataframes_nuts_analytical, dataframes_approx_differentiable

base_path = script_dir 
dataframes_mcmc, dataframes_slice, dataframes_nuts_analytical, dataframes_approx_differentiable = load_files(base_path)

# Concatenate all the DataFrames into a single DataFrame
all_data = pd.concat([pd.concat(dataframes_slice.values(), ignore_index=True),
                      pd.concat(dataframes_mcmc.values(), ignore_index=True),
                      pd.concat(dataframes_nuts_analytical.values(), ignore_index=True),
                      pd.concat(dataframes_approx_differentiable.values(), ignore_index=True)], ignore_index=True)

required_parameters = ['z_Intercept', 'z_prevresp', 'a', 't', 'v_prevresp', 'v_Intercept', 'v_signed_contrast']
filtered_data = all_data[all_data['parameter'].isin(required_parameters)]
#save 
filtered_data.to_csv('individual_fits_combined_data.csv', index=False)
# %% plot the results of subject-level estimate contrasts
# Generate the path to the results folder
results_dir = os.path.join(script_dir, '..', 'results', 'figures')

contrast_results_dir = os.path.join(script_dir, '..', 'results', 'contrasting_estimation_exercise_20240514')
contrast_results_dir = os.path.normpath(contrast_results_dir)
csv_file_path = os.path.join(contrast_results_dir, 'individual_fits_combined_data.csv')
filtered_data = pd.read_csv(csv_file_path)

average_r_hat = filtered_data.groupby(['sampler', 'parameter'])['r_hat'].mean().reset_index()
average_estimate = filtered_data.groupby(['sampler', 'parameter'])['mean'].mean().reset_index()

# Define the color palette
palette = sns.color_palette("viridis", as_cmap=False)

# Adjust overall aesthetics for publication quality
sns.set(style='ticks', context='talk', palette=palette)

# Plotting Average r_hat for each sampler by parameter
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='parameter', y='r_hat', hue='sampler', data=average_r_hat, palette=palette)
plt.title('Average $\\hat{R}$ by Parameter and Sampler')
plt.xticks(rotation=45)
plt.xlabel('Parameter')
plt.ylabel('Average $\\hat{R}$')
plt.legend(title='Sampler')
sns.despine()  # Remove the top and right spines
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Average_R_hat_by_Parameter_and_Sampler.png'))
plt.show()

# Plotting Mean estimate for each sampler by parameter
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='parameter', y='mean', hue='sampler', data=average_estimate, palette=palette)
plt.title('Mean Estimates by Parameter and Sampler')
plt.xticks(rotation=45)
plt.xlabel('Parameter')
plt.ylabel('Estimate')
plt.legend(title='Sampler')
sns.despine()  # Remove the top and right spines
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Mean_Estimates_by_Parameter_and_Sampler.png'))
plt.show()

# %% Identity plots 
# Load the dataset
data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd_hddmfits.csv'))
# Select columns that include the subject number and columns with 'regressdczprevresplag1' in their names
selected_data = data.filter(regex='subjnr|regressdczprevresplag1|t__stimcodingdczprevresp')

rename_dict = {
    'subjnr': 'participant_id',
    'a__regressdczprevresplag1': 'a',
    't__stimcodingdczprevresp': 't',
    'v_Intercept__regressdczprevresplag1': 'v_Intercept',
    'v_stimulus__regressdczprevresplag1': 'v_signed_contrast',
    'v_prevresp__regressdczprevresplag1': 'v_prevresp',
    'z_Intercept__regressdczprevresplag1': 'z_Intercept',
    'z_prevresp__regressdczprevresplag1': 'z_prevresp'
}
selected_data = data.rename(columns=rename_dict)

columns_to_keep = list(rename_dict.values())  # List of new names based on the renaming dictionary
selected_data = selected_data[columns_to_keep]

# selected_data = data.filter(regex='subjnr|stimcodingdczprevresp') # noted from Anne as the parameters that I used in the main paper. File has mutiple copies of each need to follow up.
# 't__regressdczprevresplag1': 't',
#t__regressdclag1
selected_data['sampler'] = 'old_slice'

melted_df = selected_data.melt(id_vars=["participant_id", "sampler"], var_name="parameter", value_name="old_slice")
melted_df['participant_id'] = melted_df['participant_id'].astype(int)
pivot_data['participant_id'] = pivot_data['participant_id'].astype(int)

combined_df = pd.merge(pivot_data, melted_df, on=["participant_id", "parameter"], how="outer")
# %% plot identiy lines 
# Define palettes for each FacetGrid
palette_deep = sns.color_palette('deep', n_colors=len(combined_df['parameter'].unique()))
palette_pastel = sns.color_palette('pastel', n_colors=len(combined_df['parameter'].unique()))
palette_set2 = sns.color_palette('Set2', n_colors=len(combined_df['parameter'].unique()))

# First FacetGrid: Comparing 'slice' and 'nuts_analytical'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_deep)
g.map_dataframe(sns.scatterplot, x='slice', y='nuts_analytical')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['slice'].min(), subdata['nuts_analytical'].min())
#     max_val = max(subdata['slice'].max(), subdata['nuts_analytical'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('Slice Mean Estimate', 'Nuts Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig(os.path.join(results_dir, 'Slice_vs_Nuts_Analytical.png'))
plt.show()

# Second FacetGrid: Comparing 'slice' and 'old_slice'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_pastel)
g.map_dataframe(sns.scatterplot, x='slice', y='old_slice')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['slice'].min(), subdata['old_slice'].min())
#     max_val = max(subdata['slice'].max(), subdata['old_slice'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('New Slice Mean Estimate', 'Old Slice Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig(os.path.join(results_dir, 'New_vs_Old_Slice.png'))
plt.show()

# Third FacetGrid: Comparing 'nuts_analytical' and 'old_slice'
g = sns.FacetGrid(combined_df, col='parameter', col_wrap=4, height=4, sharex=False, sharey=False, hue='parameter', palette=palette_set2)
g.map_dataframe(sns.scatterplot, x='nuts_analytical', y='old_slice')

# for ax, (_, subdata) in zip(g.axes.flat, combined_df.groupby('parameter')):
#     min_val = min(subdata['nuts_analytical'].min(), subdata['old_slice'].min())
#     max_val = max(subdata['nuts_analytical'].max(), subdata['old_slice'].max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'gray', ls="--")
#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

g.set_axis_labels('Nuts Analytical Mean Estimate', 'Old Slice Mean Estimate')
g.set_titles(col_template="{col_name}")
plt.savefig(os.path.join(results_dir, 'Nuts_Analytical_vs_Old_Slice.png'))
plt.show()

# %% investiate really low estimates
data = pd.read_csv(os.path.join(data_file_path, 'visual_motion_2afc_fd_hddmfits.csv'))
# Select columns that include the subject number and columns with 'regressdczprevresplag1' in their names
# selected_data_ndt = data.filter(regex='subjnr|t_')
# selected_data_t = data.filter(regex='prevresp')
selected_data_t = data.filter(regex='^t_')

# Define number of columns per row
cols_per_row = 4  # Adjust based on how many plots you want per row
num_rows = (len(selected_data_t.columns) + cols_per_row - 1) // cols_per_row  # Calculate the required number of rows

# Plotting
plt.figure(figsize=(20, 4 * num_rows))  # Adjust figure size based on number of rows
for i, column in enumerate(selected_data_t.columns):
    plt.subplot(num_rows, cols_per_row, i + 1)  # Create a subplot for each column
    sns.histplot(selected_data_t[column], kde=False)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
