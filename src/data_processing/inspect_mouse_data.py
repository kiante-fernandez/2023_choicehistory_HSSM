#%%load package
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import hssm
import bambi as bmb

#%%
#script_dir = os.path.dirname(os.path.realpath(__file__))
#data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')


# Load and preprocess mouse data
mouse_data_path = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
mouse_data = pd.read_csv(mouse_data_path)

#%%
# Group by both subject ID and session, then limit to first 350 trials per session
def limit_trials(group):
    return group.head(350)  # Take only first 350 trials

# Apply the function to each session group
mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(limit_trials).reset_index(drop=True)

# Check how many trials were removed
before_limiting = len(mouse_data)
after_limiting = len(mouse_data_limited)
trials_removed = before_limiting - after_limiting

print(f"Total trials before limiting: {before_limiting}")
print(f"Total trials after limiting to 350 per session: {after_limiting}")
print(f"Number of trials removed: {trials_removed}")
print(f"Percentage of trials removed: {trials_removed / before_limiting * 100:.2f}%")

# %%
#get 5 mice 
#five_mouse_id = mouse_data['subj_idx'].unique()[0:10]

first_mouse_id = mouse_data['subj_idx'].unique()[1]

single_mouse = mouse_data[mouse_data['subj_idx'] == first_mouse_id].copy()
#single_mouse = single_mouse[['subj_idx', 'signed_contrast', 'response', 'rt', 'movement_onset', 'prevresp']]
#single_mouse = mouse_data[['subj_idx', 'signed_contrast', 'response', 'rt', 'movement_onset', 'prevresp']]

print(single_mouse.head())

#%% how does movement time look like reletive to the repsonse time. 
valid_data = single_mouse.dropna(subset=['movement_onset', 'rt'])

#look at valid data less than 10 seconds
valid_data = valid_data[(valid_data['movement_onset'] < 5) & (valid_data['rt'] < 5)]
valid_data = valid_data[(valid_data['movement_onset'] > 0.08) & (valid_data['rt'] > 0.08)]
# valid_data = valid_data[(valid_data['rt'] > 0.08)]
# valid_data = valid_data[(valid_data['rt'] < 10)]
# valid_data = valid_data[(valid_data['movement_onset'] < 10)]

# valid_data = valid_data[(valid_data['rt'] > .8) & (valid_data['rt'] < 2)]
# valid_data = valid_data[(valid_data['movement_onset'] > .8) & (valid_data['movement_onset'] < 2)]

#%%
#quantify how much data removed based on 10 second cutoff
total_data_points = len(single_mouse)
valid_data_points = len(valid_data)
print(f"Total data points: {total_data_points}")
print(f"Valid data points: {valid_data_points}")
print(f"Percentage of valid data points: {valid_data_points / total_data_points * 100:.2f}%")

#remove any cases where movement onset is negative
# valid_data = valid_data[valid_data['movement_onset'] >= 0]

plt.figure(figsize=(10, 6))
plt.scatter(valid_data['rt'], valid_data['movement_onset'], alpha=0.5)
#add identity line
min_val = min(valid_data['movement_onset'].min(), valid_data['rt'].min())
max_val = max(valid_data['movement_onset'].max(), valid_data['rt'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line')
plt.xlabel('Reaction Time (rt)', fontsize=12)
plt.ylabel('Movement Onset', fontsize=12)
plt.title(f'Reaction Time vs Movement Onset for Mouse {valid_data["subj_idx"].iloc[0]}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Ensure the axes have the same scale
plt.axis('equal')

# Optional: Calculate correlation coefficient
correlation = valid_data['rt'].corr(valid_data['movement_onset'])
plt.annotate(f"Correlation: {correlation:.4f}", 
             xy=(0.05, 0.95), 
             xycoords='axes fraction', 
             fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
#plot density of reaction time and movement time
plt.figure(figsize=(10, 6))
sns.kdeplot(valid_data['movement_onset'], label='Movement Onset', color='g', fill=True, alpha=0.5)
sns.kdeplot(valid_data['rt'], label='Reaction Time', color='b', fill=True, alpha=0.4)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Density of Movement Onset and Reaction Time for Mouse {valid_data["subj_idx"].iloc[0]}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
#min and max of movement time
movement_onset_min = valid_data['movement_onset'].min()
movement_onset_max = valid_data['movement_onset'].max()
print(f"Movement Onset Min: {movement_onset_min}")
print(f"Movement Onset Max: {movement_onset_max}")
#min and max of reaction time
rt_min = valid_data['rt'].min()
rt_max = valid_data['rt'].max()
print(f"Reaction Time Min: {rt_min}")
print(f"Reaction Time Max: {rt_max}")
# %%
#5 numbers summary of movement time
movement_onset_summary = valid_data['movement_onset'].describe()
print("Movement Onset Summary:")
print(movement_onset_summary)
#5 numbers summary of reaction time
rt_summary = valid_data['rt'].describe()
print("Reaction Time Summary:")
print(rt_summary)
# %%
#recode response to be -1 and 1 rather than 0 and 1
valid_data['response'] = valid_data['response'].replace({0: -1, 1: 1})
#remove instances where prevresp is NaN
valid_data = valid_data.dropna(subset=['prevresp'])
#%%
#first clip the data on the lower end of movement_onset
# valid_data = valid_data[valid_data['movement_onset'] > 0.0100]
simple_ddm_model = hssm.HSSM(data=valid_data,
                            model="ddm",
                            loglik_kind="approx_differentiable",
                            p_outlier={"name": "Beta", "alpha": 4, "beta": 10},
                            prior_settings="safe",
                            lapse=bmb.Prior("Uniform", lower=0.100, upper=10.0)
)

simple_ddm_model_v_regression = hssm.HSSM(
    data=valid_data,
    model="ddm",
    loglik_kind="approx_differentiable",
    prior_settings="safe",
    p_outlier={"name": "Beta", "alpha": 4, "beta": 10},
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "signed_contrast": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
            },
            "formula": "v ~ 1 + signed_contrast",
            "link": "identity"
        },
        
    ],
)

simple_ddm_model_v_regression = hssm.HSSM(
    data=valid_data,
    model="ddm",
    loglik_kind="approx_differentiable",
    prior_settings="safe",
    p_outlier={"name": "Beta", "alpha": 4, "beta": 10},
    include=[
        {
            "name": "v",
            "formula": "1 + C(signed_contrast)",
        },
        
    ],
)

infer_data_simple_ddm_model = simple_ddm_model.sample(
    # cores=4,  # how many cores to use
    # chains=4,  # how many chains to run
    # draws=100,  # number of draws from the markov chain
    # tune=400,  # number of burn-in samples
    # idata_kwargs=dict(log_likelihood=True),  # return log likelihood
    initvals={
        'p_outlier': np.array(0.10),
        # 'a': np.array(1.0),
        # 'z': np.array(0.5),
        # 'theta': np.array(0),
        # 't': np.array(0.100),
        # 'v': np.array(0.0)
    },
    # discard_tuned_samples=False
    # target_accept=0.99
)

az.summary(infer_data_simple_ddm_model)

az.plot_trace(infer_data_simple_ddm_model)
plt.tight_layout()
plt.show()

ax = hssm.plotting.plot_posterior_predictive(simple_ddm_model, range=(-2,2))
sns.despine()
ax.set_ylabel("")
plt.title("Posterior Predictive Plot")
plt.tight_layout()
plt.show()
#%% run the same thing but with movement time. but todo so we need to rename it rt as that is what hssm expects
#make a new copy of data with the rename
valid_data_movement_time = valid_data.copy()
#remove rt column
valid_data_movement_time = valid_data_movement_time.drop(columns=['rt'])
#rename movement_onset to rt
valid_data_movement_time = valid_data_movement_time.rename(columns={"movement_onset": "rt"})

simple_ddm_model = hssm.HSSM(data=valid_data_movement_time,
                             model="angle",
                             loglik_kind="approx_differentiable",
                             prior_settings="safe",
                             lapse=bmb.Prior("Uniform", lower=0.0500, upper=10.0),
                             p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.4},
)

simple_ddm_model = hssm.HSSM(
    data=valid_data_movement_time,
    model="angle",
    loglik_kind="approx_differentiable",
    prior_settings="safe",
    lapse=bmb.Prior("Uniform", lower=0.0500, upper=10.0),
    p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.4},
    # p_outlier= 0.02,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + signed_contrast",
            "link": "identity",
        }
    ],
    discard_tuned_samples=False, 
)

infer_data_simple_ddm_model = simple_ddm_model.sample(
    cores=3,  # how many cores to use
    chains=3,  # how many chains to run
    draws=100,  # number of draws from the markov chain
    tune=200,  # number of burn-in samples
    idata_kwargs=dict(log_likelihood=True),  # return log likelihood
    initvals={
        'z': np.array(0.5),
        'a': np.array(0.5),
        'p_outlier': np.array(0.01),
        't': np.array(0.025),
        'theta': np.array(0.6),
        'v': np.array(0.0)
    }
)

az.plot_trace(infer_data_simple_ddm_model)
plt.tight_layout()
plt.show()

ax = hssm.plotting.plot_posterior_predictive(simple_ddm_model, range=(-2,2))
sns.despine()
ax.set_ylabel("")
plt.title("Posterior Predictive Plot with movement time")
plt.tight_layout()
plt.show()

# %%

#get the five_mouse data ready for estimations
#make participant_id col 
valid_data_five_mouse = mouse_data[mouse_data['subj_idx'].isin(five_mouse_id)].copy()
valid_data_five_mouse['participant_id'] = valid_data_five_mouse['subj_idx']

simple_ddm_model = hssm.HSSM(data=valid_data_five_mouse,
                            model="angle",
                            loglik_kind="approx_differentiable",
                            # p_outlier={"name": "Uniform", "lower": 0.01, "upper": 0.4},
                            p_outlier={
                                "formula": "p_outlier ~ 1 + (1 | mouse_id)",
                                "prior": {"Intercept": {"name": "Normal", "mu": -2, "sigma": 0.5}},
                                "link": "logit"
                                },
                            initval_jitter =0.01,
                            prior_settings="safe",
                            include=[
                            {"name": "v", 
                             "formula": "v ~ 1 + (1 |participant_id)"},
                            {"name": "z",
                             "formula": "z ~ 1 + (1 |participant_id)"},
                            {"name": "t", 
                             "formula": "t ~ 1 + (1 |participant_id)"},
                            {"name": "a", 
                             "formula": "a ~ 1 + (1 |participant_id)"}
                            ]
)

#%%
# --- IMPORTS ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import hssm
import bambi as bmb
import time
import signal
import sys
from contextlib import contextmanager
#%%
# --- TIMEOUT HANDLER ---
class TimeoutException(Exception):
    """Exception raised when a function execution times out"""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutException if execution exceeds specified time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# --- DATA LOADING AND PREPROCESSING ---
# Load mouse behavioral data
mouse_data_path = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
mouse_data = pd.read_csv(mouse_data_path)

# Limit to first 350 trials per session for each mouse
def limit_trials(group):
    return group.head(350)  # Take only first 350 trials

# Apply the function to each session group
mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(limit_trials).reset_index(drop=True)

# --- MOUSE SELECTION ---
# Get unique mouse IDs
mouse_ids = mouse_data_limited['subj_idx'].unique()
print(f"Found {len(mouse_ids)} mice initially")

# Exclude specific mice from analysis (for various reasons including data quality issues)
excluded_mice = [
    'CSHL059', 'CSHL055', 'CSHL060', 'CSHL_015', 'CSH_ZAD_017', 'DY_018', 
    'KS043', 'KS044', 'KS045', 'KS046', 'KS086', 'KS091', 'MFD_07', 
    'NR_0020', 'NYU-47', 'PL015', 'PL016', 'PL037', 'PL050', 'SWC_022', 
    'SWC_038', 'SWC_058', 'UCLA033', 'UCLA048', 'ZFM-01577', 'ZM_1898',
    'PL024', 'PL031', 'SWC_021', 'ZFM-01935', 'ZFM-04308', 'CSHL045', 
    'CSHL052', 'CSH_ZAD_022', 'CSH_ZAD_024', 'DY_008', 'DY_020', 'NR_0027',
    'CSHL049', 'CSHL053', 'CSHL047', 'KS017', 'KS094', 'NR_0019', 'ZM_2245', 
    'ibl_witten_27','DY_014', 'KS084', 'NYU-11', 'NYU-37', 'SWC_061', 
    'UCLA011', 'UCLA014', 'ZFM-02369', 'ibl_witten_14', 'ibl_witten_16',
    'CSHL054', 'SWC_053', 'SWC_054', 'UCLA017', 'ZFM-01592', 'UCLA017', 
    'ibl_witten_13'
]

# Apply exclusions
mouse_ids = np.array([m for m in mouse_ids if m not in excluded_mice])
print(f"Found {len(mouse_ids)} mice after exclusions")

# --- MODEL FITTING ---
# Create a dictionary to store results
results = {}

# Set timeout limit in seconds (15 minutes)
timeout_limit = 900
#%%
# Loop through each mouse
for mouse_id in mouse_ids:
    print(f"\nProcessing mouse {mouse_id}")
    
    # Get data for this mouse
    single_mouse = mouse_data_limited[mouse_data_limited['subj_idx'] == mouse_id].copy()
    
    # Basic preprocessing
    # Filter out trials with invalid movement onset or reaction time
    valid_data = single_mouse.dropna(subset=['movement_onset', 'rt'])
    valid_data = valid_data[(valid_data['movement_onset'] < 5) & (valid_data['rt'] < 5)]
    valid_data = valid_data[(valid_data['movement_onset'] > 0.08) & (valid_data['rt'] > 0.08)]
    
    # Recode response to be -1 and 1 rather than 0 and 1
    valid_data['response'] = valid_data['response'].replace({0: -1, 1: 1})
    
    # Clean data for model fitting - drop rows with missing values in key columns
    valid_data_clean = valid_data.dropna(subset=['prevresp','signed_contrast', 'rt', 'response'])
    
    # Skip if not enough data
    if len(valid_data_clean) < 100:
        print(f"Skipping mouse {mouse_id} - not enough valid data ({len(valid_data_clean)} trials)")
        results[mouse_id] = {'success': False, 'reason': 'insufficient data'}
        continue
    
    print(f"Fitting model for mouse {mouse_id} with {len(valid_data_clean)} trials")
    
    try:
        # Create contrast categories and prepare data
        contrast_values = sorted(valid_data_clean['signed_contrast'].unique())
        contrast_mapping = {value: f'c_{value}' for value in contrast_values}
        valid_data_clean['contrast_category'] = valid_data_clean['signed_contrast'].map(contrast_mapping)
        valid_data_clean['prevresp'] = valid_data_clean['prevresp'].replace({0: -1, 1: 1})
        valid_data_clean['prevresp_cat'] = valid_data_clean['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
        valid_data_clean['prevresp_cat'] = valid_data_clean['prevresp_cat'].astype('category')
        valid_data_clean['contrast_category'] = valid_data_clean['contrast_category'].astype('category')

        # --- MODEL DEFINITIONS ---
        # Define 8 different models:
        # 1a-1d: Drift-diffusion models (DDM) with different parameters
        # 2a-2d: Angle models with different parameters
        
        # DDM Models
        # Model 1a: DDM with contrast affecting drift rate
        model1a = hssm.HSSM(
            data=valid_data_clean,
            model="ddm",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
            ],
        )
        
        # Model 1b: DDM with contrast and previous response affecting drift rate
        model1b = hssm.HSSM(
            data=valid_data_clean,
            model="ddm",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + prevresp_cat + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
            ],
        )
        
        # Model 1c: DDM with contrast affecting drift rate and previous response affecting starting point
        model1c = hssm.HSSM(
            data=valid_data_clean,
            model="ddm",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat"
                },
            ],
        )
        
        # Model 1d: DDM with both contrast and previous response affecting drift rate AND previous response affecting starting point
        model1d = hssm.HSSM(
            data=valid_data_clean,
            model="ddm",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + prevresp_cat + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat"
                },
            ],
        )
        
        # Angle Models (similar structure to DDM models but using "angle" model type)
        # Model 2a: Angle model with contrast affecting drift rate
        model2a = hssm.HSSM(
            data=valid_data_clean,
            model="angle",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
            ],
        )
        
        # Model 2b: Angle model with contrast and previous response affecting drift rate
        model2b = hssm.HSSM(
            data=valid_data_clean,
            model="angle",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + prevresp_cat + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
            ],
        )
        
        # Model 2c: Angle model with contrast affecting drift rate and previous response affecting starting point
        model2c = hssm.HSSM(
            data=valid_data_clean,
            model="angle",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat"
                },
            ],
        )
        
        # Model 2d: Angle model with both contrast and previous response affecting drift rate AND previous response affecting starting point
        model2d = hssm.HSSM(
            data=valid_data_clean,
            model="angle",
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 0 + prevresp_cat + C(contrast_category, Treatment('c_0.0'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat"
                },
            ],
        )
        
        # --- MODEL FITTING ---
        # Try to fit the models with a timeout
        with time_limit(timeout_limit):
            start_time = time.time()
            
            # Fit all 8 models with limited sampling for speed
            # For each model, draw 500 samples after 1000 burn-in samples
            infer_data1a = model1a.sample(draws=1000, tune=2000)
            infer_data1b = model1b.sample(draws=1000, tune=2000)
            infer_data1c = model1c.sample(draws=1000, tune=2000)
            infer_data1d = model1d.sample(draws=1000, tune=2000)
            infer_data2a = model2a.sample(draws=1000, tune=2000)
            infer_data2b = model2b.sample(draws=1000, tune=2000)
            infer_data2c = model2c.sample(draws=1000, tune=2000)
            infer_data2d = model2d.sample(draws=1000, tune=2000)
            
            # Compare all models
            compare_data = az.compare({
                "ddma": model1a.traces,
                "ddmb": model1b.traces,
                "ddmc": model1c.traces,
                "ddmd": model1d.traces,
                "anglea": model2a.traces,
                "angleb": model2b.traces,
                "anglec": model2c.traces,
                "angled": model2d.traces
            })
            
            # Plot model comparison
            az.plot_compare(compare_data)
            plt.tight_layout()
            plt.show()
            
            # Get summaries for all models
            summary1a = az.summary(infer_data1a)
            summary1b = az.summary(infer_data1b)
            summary1c = az.summary(infer_data1c)
            summary1d = az.summary(infer_data1d)
            summary2a = az.summary(infer_data2a)
            summary2b = az.summary(infer_data2b)
            summary2c = az.summary(infer_data2c)
            summary2d = az.summary(infer_data2d)

            # Store all results for this mouse
            results[mouse_id] = {
                'time_taken': time.time() - start_time,
                'success': True,
                'contrast_values': contrast_values,
                'n_trials': len(valid_data_clean),
                'summaries': {
                    'ddma': summary1a,
                    'ddmb': summary1b,
                    'ddmc': summary1c,
                    'ddmd': summary1d,
                    'anglea': summary2a,
                    'angleb': summary2b,
                    'anglec': summary2c,
                    'angled': summary2d
                },
                'comparison': compare_data
            }
            
            print(f"Successfully fit model for mouse {mouse_id} in {results[mouse_id]['time_taken']:.2f} seconds")
                
    except TimeoutException:
        print(f"Timed out after {timeout_limit} seconds for mouse {mouse_id}")
        results[mouse_id] = {'success': False, 'reason': 'timeout'}
        
    except Exception as e:
        print(f"Error fitting model for mouse {mouse_id}: {str(e)}")
        results[mouse_id] = {'success': False, 'reason': str(e)}

# --- RESULTS REPORTING ---
# Print overall results
print("\n\nOverall Results:")
for mouse_id, result in results.items():
    if result['success']:
        print(f"Mouse {mouse_id}: SUCCESS - {result['time_taken']:.2f} seconds - {result.get('n_trials', 0)} trials")
    else:
        print(f"Mouse {mouse_id}: FAILED - {result['reason']}")

# Save results in a CSV
results_df = pd.DataFrame.from_dict(
    {mouse_id: {
        'success': result['success'],
        'time_taken': result.get('time_taken', None),
        'n_trials': result.get('n_trials', None),
        'reason': result.get('reason', None)
    } for mouse_id, result in results.items()},
    orient='index'
)
results_df.to_csv('mouse_model_fit_results.csv')


# %%



# # Collect results for successful fits
# successful_summaries = {}
# for mouse_id, result in results.items():
#     if result['success']:
#         successful_summaries[mouse_id] = {
#             'summary': result['summary'],
#             'contrast_values': result['contrast_values']
#         }

# # Optional: Save a visualization of the drift rates for each contrast value
# if successful_summaries:
#     print("\nGenerating drift rate plots for successful fits...")
    
#     for mouse_id, result in successful_summaries.items():
#         try:
#             # Extract drift rates for different contrast values
#             summary = result['summary']
#             contrast_values = result['contrast_values']
            
#             # Get the drift rate parameters (they start with 'v_contrast_category')
#             drift_params = [col for col in summary.index if col.startswith('v_contrast_category')]
            
#             if drift_params:
#                 # Create a dataframe with contrast values and drift rates
#                 drift_data = []
#                 for param in drift_params:
#                     # Extract the contrast value from the parameter name
#                     contrast_label = param.replace('v_contrast_category[', '').replace(']', '')
#                     # Find the corresponding numerical contrast value
#                     for c_val, c_label in contrast_mapping.items():
#                         if c_label == contrast_label:
#                             contrast_value = c_val
#                             break
                    
#                     drift_data.append({
#                         'contrast': contrast_value,
#                         'drift_rate': summary.loc[param, 'mean'],
#                         'drift_rate_sd': summary.loc[param, 'sd']
#                     })
                
#                 drift_df = pd.DataFrame(drift_data)
                
#                 # Sort by contrast value
#                 drift_df = drift_df.sort_values('contrast')
                
#                 # Plot
#                 plt.figure(figsize=(10, 6))
#                 plt.errorbar(
#                     drift_df['contrast'], 
#                     drift_df['drift_rate'], 
#                     yerr=drift_df['drift_rate_sd'],
#                     marker='o',
#                     linestyle='-',
#                     capsize=5
#                 )
#                 plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
#                 plt.xlabel('Contrast Value')
#                 plt.ylabel('Drift Rate (v)')
#                 plt.title(f'Drift Rates by Contrast for Mouse {mouse_id}')
#                 plt.grid(True, alpha=0.3)
                
#                 # Save the plot
#                 plt.savefig(f'mouse_{mouse_id}_drift_rates.png')
#                 plt.close()
                
#                 print(f"Generated drift rate plot for mouse {mouse_id}")
                
#         except Exception as e:
#             print(f"Error creating plot for mouse {mouse_id}: {str(e)}")
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import hssm
import bambi as bmb
import time
import signal
import sys
from contextlib import contextmanager

# Timeout handler for model fitting
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Load and preprocess mouse data
mouse_data_path = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
mouse_data = pd.read_csv(mouse_data_path)

# Group by both subject ID and session, then limit to first 350 trials per session
def limit_trials(group):
    return group.head(350)  # Take only first 350 trials

# Apply the function to each session group
mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(limit_trials).reset_index(drop=True)

# Get unique mouse IDs
mouse_ids = mouse_data_limited['subj_idx'].unique()
print(f"Found {len(mouse_ids)} mice")

#remove mouse with ID  CSHL059
mouse_ids = mouse_ids[mouse_ids != 'CSHL059']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSHL055']  #?  Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSHL060']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSHL_015']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'CSH_ZAD_017']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'DY_018']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS043']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS044']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS045']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS046']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS086']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS091']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'MFD_07']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NR_0020']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NYU-47']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'PL015']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'PL016']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'PL037']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'PL050']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_022']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_038']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_058']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'UCLA033']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'UCLA048']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZFM-01577']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZM_1898']  # Exclude the unwanted mouse
#?UCLA049
#PL024
#PL031
#SWC_021
mouse_ids = mouse_ids[mouse_ids != 'PL024']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'PL031']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_021']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZFM-01935']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZFM-04308']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'CSHL045']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSHL052']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSH_ZAD_022']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSH_ZAD_024']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'DY_008']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'DY_020']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NR_0027']  # Exclude the unwanted mouse

#NR_0019

mouse_ids = mouse_ids[mouse_ids != 'CSHL049']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'CSHL053']  # Exclude the unwanted mouse

#werid intilialization
mouse_ids = mouse_ids[mouse_ids != 'CSHL047']  # Exclude the unwanted mouse
#more timeouts 
mouse_ids = mouse_ids[mouse_ids != 'KS017']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS094']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NR_0019']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZM_2245']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ibl_witten_27']  # Exclude the unwanted mouse

#other exlcuions reference not in levels
mouse_ids = mouse_ids[mouse_ids != 'DY_014']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'KS084']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NYU-11']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'NYU-37']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'SWC_061']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'UCLA011']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'UCLA014']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'ZFM-02369']  # Exclude the unwanted mouse

mouse_ids = mouse_ids[mouse_ids != 'ibl_witten_14']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ibl_witten_16']  # Exclude the unwanted mouse


#even more based on not obiovios postior geomoetires

mouse_ids = mouse_ids[mouse_ids != 'CSHL054']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_053']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'SWC_054']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'UCLA017']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ZFM-01592']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'UCLA017']  # Exclude the unwanted mouse
mouse_ids = mouse_ids[mouse_ids != 'ibl_witten_13']  # Exclude the unwanted mouse

#get mouse ids after frist 8 mice
#mouse_ids = mouse_ids[22:]  # Limit to first 8 mice
print(mouse_ids)
print(f"Found {len(mouse_ids)} mice")


# Define subset of mice to include
# subset_mice = [
#     'CSHL054', 'CSH_ZAD_001', 'CSH_ZAD_011', 'CSH_ZAD_019', 
#     'CSH_ZAD_029', 'DY_014', 'KS017', 'KS019', 'KS021',
#     'KS052', 'KS055', 'KS084', 'KS094', 'MFD_06',
#     'MFD_08', 'NR_0017'
# ]
#define new subset
# 'CSHL054' 'CSH_ZAD_001' 'CSH_ZAD_011' 'CSH_ZAD_019' 'CSH_ZAD_029' 'KS019'
#  'KS021' 'KS052' 'KS055' 'MFD_06' 'MFD_08' 'NR_0017' 'NR_0028' 'NR_0029'
#  'NYU-12' 'NYU-27' 'NYU-30' 'SWC_023' 'SWC_039' 'SWC_042' 'SWC_043'
#  'SWC_052' 'SWC_053' 'SWC_054' 'SWC_060' 'SWC_066' 'UCLA005' 'UCLA006'
#  'UCLA012' 'UCLA015' 'UCLA017' 'UCLA030' 'UCLA034' 'UCLA035' 'UCLA036'
#  'UCLA049' 'UCLA052' 'ZFM-01576' 'ZFM-01592' 'ZFM-01936' 'ZFM-01937'
#  'ZFM-05236' 'ZM_1897' 'ZM_2240' 'ZM_2241' 'ZM_3003' 'ibl_witten_13'
#  'ibl_witten_19' 'ibl_witten_25' 'ibl_witten_26' 'ibl_witten_29'
#  'ibl_witten_32'

subset_mice = ['CSH_ZAD_001','CSH_ZAD_011' ,'CSH_ZAD_019' ,'CSH_ZAD_029' ,'KS019','KS052','KS055','MFD_06','MFD_08','NR_0017']

# Filter and preprocess data
mouse_data_limited = mouse_data_limited[mouse_data_limited['subj_idx'].isin(subset_mice)]
mouse_data_limited['participant_id'] = mouse_data_limited['subj_idx']

# Basic preprocessing
mouse_data_limited = mouse_data_limited.dropna(subset=['movement_onset', 'rt', 'prevresp', 'signed_contrast', 'response'])
mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] < 5) & (mouse_data_limited['rt'] < 5)]
mouse_data_limited = mouse_data_limited[(mouse_data_limited['movement_onset'] > 0.08) & (mouse_data_limited['rt'] > 0.08)]

# Recode response to be -1 and 1 rather than 0 and 1
mouse_data_limited['response'] = mouse_data_limited['response'].replace({0: -1, 1: 1})
mouse_data_limited['prevresp'] = mouse_data_limited['prevresp'].replace({0: -1, 1: 1})

# Create categorical variables
contrast_values = sorted(mouse_data_limited['signed_contrast'].unique())
contrast_mapping = {value: f'c_{value}' for value in contrast_values}
mouse_data_limited['contrast_category'] = mouse_data_limited['signed_contrast'].map(contrast_mapping)
mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
mouse_data_limited['prevresp_cat'] = mouse_data_limited['prevresp_cat'].astype('category')
mouse_data_limited['contrast_category'] = mouse_data_limited['contrast_category'].astype('category')

# Print summary of data
print(f"Data contains {len(mouse_data_limited)} trials across {mouse_data_limited['participant_id'].nunique()} mice")
print(f"Contrast values: {contrast_values}")

# Create the hierarchical model
model = hssm.HSSM(
    data=mouse_data_limited,
    model="ddm",
    loglik_kind="approx_differentiable",
    prior_settings="safe",
    p_outlier={"name": "Beta", "alpha": 4, "beta": 36},
    # lapse=bmb.Prior("Uniform", lower=0.0100, upper=10.0),
    include=[
        {
            "name": "v",
            "formula": "v ~ 0 + prevresp_cat + C(contrast_category, Treatment('c_0.0')) + (0 + prevresp_cat + C(contrast_category, Treatment('c_0.0')) | participant_id)",
        },
        {
            "name": "z",
            "formula": "z ~ 0 + prevresp_cat + (0 + prevresp_cat | participant_id)",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + (1 | participant_id)",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + (1 | participant_id)",
        },
        
    ]
)
                    # {"name": "v", 
                    #          "formula": "v ~ 1 + (1 |participant_id)"},
                    #         {"name": "z",
                    #          "formula": "z ~ 1 + (1 |participant_id)"},
                    #         {"name": "t", 
                    #          "formula": "t ~ 1 + (1 |participant_id)"},
                    #         {"name": "a", 
                    #          "formula": "a ~ 1 + (1 |participant_id)"}
# Sample from the posterior
samples = model.sample()

# Check model summary
summary = model.summary()
print(summary)

# Plot trace of parameters
model.plot_trace()
plt.tight_layout()
plt.show()


# %%
