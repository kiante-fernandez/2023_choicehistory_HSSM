#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import hssm
import bambi as bmb
import time
import pickle
from pathlib import Path
import json

# Define output directory structure
OUTPUT_DIR = Path("./model_results")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)
(OUTPUT_DIR / "summaries").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "traces").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "posterior_predictive").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots" / "model_comparison").mkdir(exist_ok=True)

# Load and preprocess mouse data
mouse_data_path = os.path.join('/Users/kiante/Documents/2023_choicehistory_HSSM/data/ibl_trainingChoiceWorld_raw_20250310.csv')
mouse_data = pd.read_csv(mouse_data_path)

# Limit to first 350 trials per session
mouse_data_limited = mouse_data.groupby(['subj_idx', 'session']).apply(
    lambda group: group.head(350)
).reset_index(drop=True)

# Define mice to exclude
excluded_mice = [
    'CSHL059', 'CSHL060', 'CSHL_015', 'CSH_ZAD_017', 'DY_018', 
    'KS043', 'KS044', 'KS045', 'KS046', 'KS086', 'KS091', 'MFD_07', 
    'NR_0020', 'NYU-47', 'PL015', 'PL016', 'PL037', 'PL050', 'SWC_022', 
    'SWC_038', 'SWC_058', 'UCLA033', 'UCLA048', 'ZFM-01577', 'ZM_1898',
    'PL024', 'PL031', 'SWC_021', 'ZFM-01935', 'ZFM-04308', 'CSHL045', 
    'CSHL052', 'CSH_ZAD_022', 'CSH_ZAD_024', 'DY_008', 'DY_020', 'NR_0027',
    'CSHL049', 'CSHL053', 'CSHL047', 'KS017', 'KS094', 'NR_0019', 'ZM_2245', 
    'ibl_witten_27', 'DY_014', 'KS084', 'NYU-11', 'NYU-37', 'SWC_061', 
    'UCLA011', 'UCLA014', 'ZFM-02369', 'ibl_witten_14', 'ibl_witten_16',
    'CSHL054', 'SWC_053', 'SWC_054', 'UCLA017', 'ZFM-01592', 'UCLA017', 
    'ibl_witten_13','SWC_066'
]
# excluded_mice = [
#     'CSHL059', 'CSHL055', 'CSHL060', 'CSHL_015', 'CSH_ZAD_017', 'DY_018'
# ]
#other mice to consider to exclude
#['ibl_witten_19','ZM_2241','ZFM-01937','UCLA049','UCLA036','SWC_066','SWC_060','SWC_023','NYU-27','MFD_06','KS055']

# Filter data to exclude the specified mice
all_mouse_ids = mouse_data_limited['subj_idx'].unique()
included_mice = [m for m in all_mouse_ids if m not in excluded_mice]

#included_mice = ['SWC_043', 'ZFM-01937', 'SWC_066', 'KS019', 'ibl_witten_29', 'CSH_ZAD_029']
#included_mice = ['ZFM-01937', 'KS019', 'ibl_witten_29']


print(f"Found {len(all_mouse_ids)} mice total")
print(f"Excluding {len(excluded_mice)} mice")
print(f"Including {len(included_mice)} mice in analysis")
#%%

# Global results container
global_results = {
    'mice': {},
    'model_wins': {
        'ddma': 0, 'ddmb': 0, 'ddmc': 0, 'ddmd': 0,
        'anglea': 0, 'angleb': 0, 'anglec': 0, 'angled': 0
    },
    'model_ranks': {
        'ddma': [], 'ddmb': [], 'ddmc': [], 'ddmd': [],
        'anglea': [], 'angleb': [], 'anglec': [], 'angled': []
    },
    'parameter_estimates': {}
}
# mouse_id = included_mice[0]  # For debugging, start with the first mouse
# mouse_id = 'ZM_2241'  # For debugging, start with the first mouse
# mouse_id = 'ibl_witten_29'  # For debugging, start with the first mouse
# mouse_id = 'CSH_ZAD_019'  # For debugging, start with the first mouse
# mouse_id = 'KS019'  # For debugging, start with the first mouse
# mouse_id = 'NR_0029'  # For debugging, start with the first mouse

mouse_id = 'CSHL060'  # For debugging, start with the first mouse

# Define function to process each mouse
def process_mouse(mouse_id):
    print(f"\nProcessing mouse {mouse_id}")
    
    # Get data for this mouse
    mouse_data = mouse_data_limited[mouse_data_limited['subj_idx'] == mouse_id].copy()
    
    # Basic preprocessing
    valid_data = mouse_data.dropna(subset=['movement_onset', 'rt', 'prevresp', 'signed_contrast', 'response'])
    valid_data = valid_data[(valid_data['movement_onset'] < 5) & (valid_data['rt'] < 5)]
    valid_data = valid_data[(valid_data['movement_onset'] > 0.08) & (valid_data['rt'] > 0.08)]
    
    # Recode response to be -1 and 1 rather than 0 and 1
    valid_data['response'] = valid_data['response'].replace({0: -1, 1: 1})
    valid_data['prevresp'] = valid_data['prevresp'].replace({0: -1, 1: 1})
    
    # Create categorical variables
    contrast_values = sorted(valid_data['signed_contrast'].unique())
    print(f"Available contrast values for mouse {mouse_id}: {contrast_values}")
    
    contrast_mapping = {value: f'c_{value}' for value in contrast_values}
    valid_data['contrast_category'] = valid_data['signed_contrast'].map(contrast_mapping)
    valid_data['prevresp_cat'] = valid_data['prevresp'].map({-1.0: 'prev_left', 1.0: 'prev_right'})
    valid_data['prevresp_cat'] = valid_data['prevresp_cat'].astype('category')
    valid_data['contrast_category'] = valid_data['contrast_category'].astype('category')
    
    # Skip if not enough data
    if len(valid_data) < 100:
        print(f"Skipping mouse {mouse_id} - not enough valid data ({len(valid_data)} trials)")
        return None
        
    print(f"Fitting models for mouse {mouse_id} with {len(valid_data)} trials")
    
    # Determine reference contrast level (use 0.0 if available, otherwise use the minimum contrast)
    if 0.0 in contrast_values:
        reference_contrast = 'c_0.0'
    else:
        min_contrast = min(contrast_values)
        reference_contrast = f'c_{min_contrast}'
    
    print(f"Using {reference_contrast} as reference contrast level")
    
    # Define all 8 models with dynamic reference contrast
    models = {}
    
    try:
        # DDM Models
        # Model 1a: DDM with contrast affecting drift rate
        # models['ddma'] = hssm.HSSM(
        #     data=valid_data,
        #     model="ddm",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #     ],
        # )
        
        # # # Model 1b: DDM with contrast and previous response affecting drift rate
        # models['ddmb'] = hssm.HSSM(
        #     data=valid_data,
        #     model="ddm",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + prevresp_cat + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #     ],
        # )
        
        # # # Model 1c: DDM with contrast affecting drift rate and previous response affecting starting point
        # models['ddmc'] = hssm.HSSM(
        #     data=valid_data,
        #     model="ddm",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #         {
        #             "name": "z",
        #             "formula": "z ~ 0 + prevresp_cat"
        #         },
        #     ],
        # )
        
        # # Model 1d: DDM with both contrast and previous response affecting drift rate AND previous response affecting starting point
        models['ddmd'] = hssm.HSSM(
            data=valid_data,
            model="ddm",
            # initval_jitter=0.001,
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
            lapse=bmb.Prior("Uniform", lower=0.00, upper=10.0),
            include=[
                {
                    "name": "v",
                    "formula": f"v ~ 0 + prevresp_cat + C(contrast_category, Treatment('{reference_contrast}'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat",
                    # "bounds": (0.1, 0.9)  # Ensure starting point is between 0 and 1
                },
            ],
        )
        
        # # Angle Models (similar structure to DDM models but using "angle" model type)
        # # Model 2a: Angle model with contrast affecting drift rate
        # models['anglea'] = hssm.HSSM(
        #     data=valid_data,
        #     model="angle",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #         {
        #             "name": "theta",
        #             "bounds": (-0.1, 0.1)
        #         },
        #     ],
        # )
        
        # # Model 2b: Angle model with contrast and previous response affecting drift rate
        # models['angleb'] = hssm.HSSM(
        #     data=valid_data,
        #     model="angle",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 24},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + prevresp_cat + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #         {
        #             "name": "theta",
        #             "bounds": (-0.1, 0.1)
        #         },
        #     ],
        # )
        
        # # Model 2c: Angle model with contrast affecting drift rate and previous response affecting starting point
        # models['anglec'] = hssm.HSSM(
        #     data=valid_data,
        #     model="angle",
        #     # initval_jitter=0.001,
        #     loglik_kind="approx_differentiable",
        #     prior_settings="safe",
        #     p_outlier={"name": "Beta", "alpha": 6, "beta": 26},
        #     include=[
        #         {
        #             "name": "v",
        #             "formula": f"v ~ 0 + C(contrast_category, Treatment('{reference_contrast}'))",
        #             "link": "identity"
        #         },
        #         {
        #             "name": "theta",
        #             "bounds": (-0.1, 0.1)
        #         },
        #         {
        #             "name": "z",
        #             "formula": "z ~ 0 + prevresp_cat"
        #         },
        #     ],
        # )
        
        # Model 2d: Angle model with both contrast and previous response affecting drift rate AND previous response affecting starting point
        models['angled'] = hssm.HSSM(
            data=valid_data,
            model="angle",
            # initval_jitter=0.001,
            loglik_kind="approx_differentiable",
            prior_settings="safe",
            p_outlier={"name": "Beta", "alpha": 6, "beta": 26},
            lapse=bmb.Prior("Uniform", lower=0.00, upper=10.0),
            # p_outlier=None,
            include=[
                {
                    "name": "v",
                    "formula": f"v ~ 0 + prevresp_cat + C(contrast_category, Treatment('{reference_contrast}'))",
                    "link": "identity"
                },
                {
                    "name": "z",
                    "formula": "z ~ 0 + prevresp_cat"
                },
            ],
        )
    except Exception as e:
        print(f"Error initializing models for mouse {mouse_id}: {str(e)}")
        return None
    
    # Fit all models
    traces = {}
    summaries = {}
    
    #FOR mouse_id = 'ZM_2241'  # For debugging, start with the first mouse
    initvals = {'p_outlier': np.array(0.05),
             't': np.array(.20),
             'a': np.array(0.6),
             'v_prevresp_cat': np.array([0., 0.]),
             "v_C(contrast_category, Treatment('c_0.0'))": np.array([0., 0., 0., 0., 0., 0., 0., 0.]), #needs to be fixed to be the inti values given from model.initvals
             'z_prevresp_cat': np.array([0.5, 0.5])}
    #FOR mouse_id = 'ibl_witten_29'  # For debugging, start with the first mouse
    initvals = {'theta': np.array(0.),
                'a': np.array(1),
                't': np.array(0.025),
                'p_outlier': np.array(0.01),
                'v_prevresp_cat': np.array([-0.0066423 ,  0.00308775]),
                "v_C(contrast_category, Treatment('c_0.0'))": np.array([-0.00238993, -0.00071808,  0.00442804, -0.00867372, -0.00638684,
         0.00177897, -0.00992672,  0.00901758]),
                'z_prevresp_cat': np.array([0.5, 0.5])}
    #for mouse_id = 'CSH_ZAD_019'  # For debugging, start with the first mouse
    initvals = {'theta': np.array(0.),
                'a': np.array(0.6),
                't': np.array(0.20),
                'p_outlier': np.array(0.15),
                'v_prevresp_cat': np.array([0.4,0.4]),
                "v_C(contrast_category, Treatment('c_0.0'))": np.array([0.,0., 0.,0.,0.,0.,0.,0.,0.,0.]),
                'z_prevresp_cat': np.array([0.5, 0.5])}
    #for mouse_id = "KS019"  # For debugging, start with the first mouse
    initvals = {'theta': np.array(0),
                'a': np.array(1.),
                't': np.array(0.1),
                'p_outlier': np.array(0.01),
                'v_prevresp_cat': np.array([-0.00923532, -0.00134261]),
                "v_C(contrast_category, Treatment('c_0.0'))": np.array([ 2.29921461e-05,  6.36260957e-03, -3.38879507e-03,  7.01784017e-03,
         1.82533311e-03, -9.47639439e-03, -6.66751573e-03,  5.10514155e-03,
        -2.07301276e-03,  8.21345253e-04]),
                'z_prevresp_cat': np.array([0.5, 0.5])}
    #for mouse_id - "NR_0029"
    initvals = {'theta': np.array(0),
                'a': np.array(1.),
                't': np.array(0.1),
                'p_outlier': np.array(0.01),
                'v_prevresp_cat': np.array([0.00264335, -0.00594655]),
                "v_C(contrast_category, Treatment('c_0.0'))": np.array([ 0.00803371,  0.00485673,  0.00064022, -0.00874875,  0.00693306,
         0.00626095, -0.00744023, -0.00744097]),
                'z_prevresp_cat': np.array([0.5, 0.5])}
    
    #model_name = 'angled'  # For debugging, start with the last model
    for model_name, model in models.items():
        print(f"Fitting {model_name} for mouse {mouse_id}")
        try:
            # Sample from the model
            start_time = time.time()
            traces[model_name] = model.sample(
                draws=200,
                tune=1000,
                chains=3,
                cores=3,
                # target_accept=0.99,   
                initvals=initvals
    #             # 'z': np.array(0.5),
    #             # 'a': np.array(0.5),
    #             # 'v': np.array(0.0)
    #             # 'p_outlier': np.array(0.1),
    #             't': np.array(0.100),
    #             'theta': np.array(0.0)
    # }
            )
            
            end_time = time.time()
            # Get summary statistics
            summaries[model_name] = az.summary(traces[model_name])
        
            # Save the summary
            summaries[model_name].to_csv(OUTPUT_DIR / "summaries" / f"{mouse_id}_{model_name}_summary.csv")
            
            # try:
            #     print(f"Generating posterior predictive samples for {model_name}...")
            #     pp_idata = model.sample_posterior_predictive(
            #     idata=traces[model_name],
            #     data=valid_data,  # Use the same data used for fitting
            #     )
            # except Exception as e:
            #     print(f"Error generating posterior predictive samples for {model_name}: {str(e)}")
        
            # Save the model
            # with open(OUTPUT_DIR / "models" / f"{mouse_id}_{model_name}.pkl", "wb") as f:
            #     pickle.dump(model, f)
            
#            model_save_dir = OUTPUT_DIR / "models" / f"{mouse_id}_{model_name}"
#            model.save_model(model_name=f"{mouse_id}_{model_name}", path=str(OUTPUT_DIR / "models"))
#            print(f"Saved model to: {model_save_dir}")

            # Generate and save trace plot
            plt.figure(figsize=(12, 8))
            az.plot_trace(traces[model_name])
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plots" / "traces" / f"{mouse_id}_{model_name}_trace.png")
            plt.close()
            
            # Generate and save posterior predictive plot
            plt.figure(figsize=(10, 6))
            ax = hssm.plotting.plot_posterior_predictive(model, range=(-2, 2))
            sns.despine()
            ax.set_ylabel("")
            plt.title(f"{model_name} Posterior Predictive Plot for {mouse_id}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plots" / "posterior_predictive" / f"{mouse_id}_{model_name}_pp.png")
            plt.close()
            
            print(f"Successfully fit {model_name} for mouse {mouse_id} in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error fitting {model_name} for mouse {mouse_id}: {str(e)}")
            continue    
    
    # If we have at least 2 models successfully fit, do model comparison
    if len(traces) >= 2:
        try:
            # Compare models
            comparison = az.compare({model_name: trace for model_name, trace in traces.items()})
            
            # Save comparison results
            comparison.to_csv(OUTPUT_DIR / "summaries" / f"{mouse_id}_model_comparison.csv")
            
            # Plot model comparison
            plt.figure(figsize=(10, 6))
            az.plot_compare(comparison)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plots" / "model_comparison" / f"{mouse_id}_model_comparison.png")
            plt.close()
            
            # Update global results
            winning_model = comparison.index[0]
            global_results['model_wins'][winning_model] += 1
            
            for i, model_name in enumerate(comparison.index):
                global_results['model_ranks'][model_name].append(i + 1)
                
            # Extract key parameters from the winning model
            best_model_summary = summaries[winning_model]
            global_results['mice'][mouse_id] = {
                'winning_model': winning_model,
                'model_comparison': comparison.to_dict(),
                'best_model_summary': best_model_summary.to_dict(),
                'contrast_values': contrast_values
            }
            
            # Extract drift rate parameters by contrast
            drift_params = {}
            for param in best_model_summary.index:
                if param.startswith('v_contrast_category['):
                    contrast_label = param.replace('v_contrast_category[', '').replace(']', '')
                    for c_val, c_label in contrast_mapping.items():
                        if c_label == contrast_label:
                            drift_params[c_val] = {
                                'mean': best_model_summary.loc[param, 'mean'],
                                'sd': best_model_summary.loc[param, 'sd'],
                                'hdi_3%': best_model_summary.loc[param, 'hdi_3%'],
                                'hdi_97%': best_model_summary.loc[param, 'hdi_97%']
                            }
                            
            global_results['mice'][mouse_id]['drift_params'] = drift_params
            
            # If the winning model has previous response parameters, extract those too
            prev_resp_params = {}
            for param in best_model_summary.index:
                if 'prevresp_cat[prev_left]' in param or 'prevresp_cat[prev_right]' in param:
                    prev_resp_params[param] = {
                        'mean': best_model_summary.loc[param, 'mean'],
                        'sd': best_model_summary.loc[param, 'sd'],
                        'hdi_3%': best_model_summary.loc[param, 'hdi_3%'],
                        'hdi_97%': best_model_summary.loc[param, 'hdi_97%']
                    }
            
            if prev_resp_params:
                global_results['mice'][mouse_id]['prev_resp_params'] = prev_resp_params
            
            print(f"Model comparison complete for mouse {mouse_id}. Winning model: {winning_model}")
            
        except Exception as e:
            print(f"Error in model comparison for mouse {mouse_id}: {str(e)}")
    
    return True

# Process each included mouse
# for mouse_id in reversed(included_mice):
#     process_mouse(mouse_id)

for mouse_id in included_mice:
    process_mouse(mouse_id)
    
# Generate global summary reports
# 1. Model winning frequency
model_win_df = pd.DataFrame(global_results['model_wins'].items(), columns=['model', 'wins'])
model_win_df['proportion'] = model_win_df['wins'] / model_win_df['wins'].sum() if model_win_df['wins'].sum() > 0 else 0
model_win_df.to_csv(OUTPUT_DIR / "model_wins.csv")

# Plot model win frequency
if model_win_df['wins'].sum() > 0:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='proportion', data=model_win_df)
    plt.title("Proportion of Mice Best Fit by Each Model")
    plt.ylabel("Proportion")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "model_win_frequency.png")
    plt.close()

# 2. Average model ranks
model_ranks = {}
for model_name, ranks in global_results['model_ranks'].items():
    if ranks:
        model_ranks[model_name] = np.mean(ranks)

if model_ranks:
    model_rank_df = pd.DataFrame(model_ranks.items(), columns=['model', 'avg_rank'])
    model_rank_df = model_rank_df.sort_values('avg_rank')
    model_rank_df.to_csv(OUTPUT_DIR / "model_ranks.csv")

    # Plot model ranks
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='avg_rank', data=model_rank_df)
    plt.title("Average Rank of Each Model Across Mice")
    plt.ylabel("Average Rank (lower is better)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "model_ranks.png")
    plt.close()

# 3. Drift rate by contrast across mice
contrast_drift = {}
for mouse_id, mouse_data in global_results['mice'].items():
    if 'drift_params' in mouse_data:
        for contrast, params in mouse_data['drift_params'].items():
            if contrast not in contrast_drift:
                contrast_drift[contrast] = []
            contrast_drift[contrast].append(params['mean'])

# Create dataframe for plotting
drift_data = []
for contrast, values in contrast_drift.items():
    for value in values:
        drift_data.append({
            'contrast': contrast,
            'drift_rate': value
        })

if drift_data:
    drift_df = pd.DataFrame(drift_data)

    # Plot drift rate by contrast across mice
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='contrast', y='drift_rate', data=drift_df)
    plt.title("Drift Rate by Contrast Across Mice")
    plt.ylabel("Drift Rate (v)")
    plt.xlabel("Contrast")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "drift_by_contrast.png")
    plt.close()

# 4. Previous response effects across mice
prev_resp_data = []
for mouse_id, mouse_data in global_results['mice'].items():
    if 'prev_resp_params' in mouse_data:
        for param, values in mouse_data['prev_resp_params'].items():
            if 'v_prevresp_cat' in param:
                param_type = 'drift'
            elif 'z_prevresp_cat' in param:
                param_type = 'starting_point'
            else:
                continue
                
            if 'prev_left' in param:
                direction = 'left'
            elif 'prev_right' in param:
                direction = 'right'
            else:
                continue
                
            prev_resp_data.append({
                'mouse_id': mouse_id,
                'parameter_type': param_type,
                'previous_response': direction,
                'effect': values['mean']
            })

if prev_resp_data:
    prev_resp_df = pd.DataFrame(prev_resp_data)

    # Plot previous response effects
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='parameter_type', y='effect', hue='previous_response', data=prev_resp_df)
    plt.title("Previous Response Effects Across Mice")
    plt.ylabel("Parameter Effect")
    plt.xlabel("Parameter Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "prev_resp_effects.png")
    plt.close()

# Save global results in JSON format
with open(OUTPUT_DIR / "global_results.json", "w") as f:
    # Handle numpy types by conversion
    def serialize_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    json.dump(global_results, f, default=serialize_numpy)

print("Analysis complete!")
# %%
model_name = 'angled' 
#model_name = 'ddmd'
# Example model name to save summary for
az.summary(models[model_name].traces).to_csv(OUTPUT_DIR / "tune_by_hand" / f"{mouse_id}_{model_name}_summary.csv")

plt.figure(figsize=(12, 8))
az.plot_trace(models[model_name].traces)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tune_by_hand" / f"{mouse_id}_{model_name}_trace.png")
plt.show()
plt.close()
# Save the summary

# %%