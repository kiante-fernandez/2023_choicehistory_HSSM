#TODO old code to organize to create for the categorical signed contrast drift rate figures this contiains code from the old 
#2023_choicehistory_HSSM/src/plotting/plot_figure2_drift_c_contrast.py and the old plot_figure3_drift_c.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define contrast levels globally
CONTRAST_LEVELS = [-100.0, -50.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 50.0, 100.0]

def load_and_process_data():
    # Read the data
    df = pd.read_csv("/Users/kiante/Documents/2023_choicehistory_HSSM/src/ddm_catnohist_results_combined.csv")
    
    # Extract group baseline (v_Intercept) and its SD
    v_intercept = df[df['index'] == 'v_Intercept']['mean'].iloc[0]
    v_intercept_sd = df[df['index'] == 'v_Intercept']['sd'].iloc[0]
    print(f"v_Intercept: {v_intercept}")
    
    # Get group level effects and their SDs
    group_effects = {}
    group_sds = {}
    
    # Reference level is now handled differently
    for contrast in CONTRAST_LEVELS:
        if contrast == -100.0:
            group_effects[contrast] = v_intercept
            group_sds[contrast] = v_intercept_sd
        else:
            mask = df['index'] == f'v_C(signed_contrast)[{contrast}]'
            if any(mask):
                effect = df[mask]['mean'].iloc[0]
                effect_sd = df[mask]['sd'].iloc[0]
                group_effects[contrast] = effect
                group_sds[contrast] = effect_sd
                print(f"Found group effect for contrast {contrast}: {effect}")
    
    # Initialize subject estimates
    subject_estimates = {c: [] for c in CONTRAST_LEVELS}
    
    # Extract subject baselines
    subject_baselines = {}
    for subject in range(1, 70):
        mask = df['index'] == f'v_1|participant_id[{subject}]'
        if any(mask):
            subject_baselines[subject] = df[mask]['mean'].iloc[0]
        else:
            subject_baselines[subject] = 0
    
    # For each subject
    for subject in range(1, 70):
        # For each contrast level
        for contrast in CONTRAST_LEVELS:
            if contrast == -100.0:
                subject_estimates[contrast].append(v_intercept + subject_baselines[subject])
            else:
                mask = df['index'] == f'v_C(signed_contrast)|participant_id_offset[{contrast}, {subject}]'
                if any(mask):
                    offset = df[mask]['mean'].iloc[0]
                    estimate = offset
                    subject_estimates[contrast].append(estimate)
    
    return group_effects, group_sds, subject_estimates

def create_publication_plot(group_effects, group_sds, subject_estimates):
    plt.figure(figsize=(14, 8))
    
    # Set publication-ready style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 2,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot individual subject data points with jitter
    jitter_width = 2.0  # Adjust this value to control amount of jitter
    for contrast in subject_estimates.keys():
        if subject_estimates[contrast]:
            # Create jittered x positions
            n_points = len(subject_estimates[contrast])
            jittered_x = np.random.normal(contrast, jitter_width, n_points)
            
            plt.scatter(jittered_x, 
                       subject_estimates[contrast], 
                       color='black', alpha=0.3, s=50,
                       zorder=1)
    
    # Plot group mean line with error bars
    contrasts_sorted = sorted(group_effects.keys())
    estimates_sorted = [group_effects[c] for c in contrasts_sorted]
    sds_sorted = [group_sds[c] for c in contrasts_sorted]
    
    # Plot mean line with thicker line
    plt.plot(contrasts_sorted, estimates_sorted, 'k-', 
             linewidth=4, zorder=3)
    
    # Add error bars (95% CI = 2*SD)
    plt.errorbar(contrasts_sorted, estimates_sorted, 
                yerr=np.array(sds_sorted)*2,
                fmt='none', color='black', 
                capsize=6, capthick=2.5,
                elinewidth=2.5, zorder=2)
    
    plt.xlabel('Signed contrast (%)')
    plt.ylabel('Drift rate (v)')
    
    # Add reference lines (lighter)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, zorder=0)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3, zorder=0)
    
    # Set axis limits
    plt.ylim(-3, 3)
    plt.xlim(-120, 120)
    
    # Set x-ticks with custom spacing
    plt.xticks(CONTRAST_LEVELS)
    
    # Adjust subplot parameters to give specified padding
    plt.tight_layout(pad=2)
    
    return plt

def print_summary_statistics(group_effects, subject_estimates):
    print("\nGroup Level Estimates:")
    for contrast in sorted(group_effects.keys()):
        print(f"Contrast {contrast:>6}: {group_effects[contrast]:.3f}")
    
    print("\nSubject Level Summary Statistics:")
    for contrast in sorted(subject_estimates.keys()):
        values = subject_estimates[contrast]
        if values:
            print(f"\nContrast {contrast}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  SD: {np.std(values):.3f}")
            print(f"  Min: {np.min(values):.3f}")
            print(f"  Max: {np.max(values):.3f}")
            print(f"  N subjects: {len(values)}")

def save_figure(plt):
    # Get the directory of the script being run
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the path to the figures folder
    fig_folder_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'results', 'figures')
    
    # Create the figures directory if it doesn't exist
    os.makedirs(fig_folder_path, exist_ok=True)
    
    # Save the figure
    filename = f"catnohist_drift_rates.png"
    filepath = os.path.join(fig_folder_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {filepath}")

# Set random seed for reproducible jitter
np.random.seed(2024)
group_effects, group_sds, subject_estimates = load_and_process_data()
print_summary_statistics(group_effects, subject_estimates)
plot = create_publication_plot(group_effects, group_sds, subject_estimates)
save_figure(plot)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Define paths and constants
SUMMARY_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/model_results/summaries'
PLOT_DIR = '/Users/kiante/Documents/2023_choicehistory_HSSM/results/figures'
MODEL_TYPE = 'angled'  # or 'angled'

# Create plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

# Define common contrast levels
CONTRAST_LEVELS = [-100.0, -50.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 50.0, 100.0]

def extract_full_mouse_id(filename):
    """Extract the complete mouse ID from the filename, removing model type suffixes."""
    # First, remove model suffixes if present
    clean_name = filename.replace(f'_{MODEL_TYPE}_summary.csv', '')
    clean_name = clean_name.replace(f'_{MODEL_TYPE}', '')
    
    # Handle different naming patterns
    if clean_name.startswith('CSH_ZAD_'):
        return '_'.join(clean_name.split('_')[:3])  # CSH_ZAD_001
    elif clean_name.startswith('MFD_'):
        return '_'.join(clean_name.split('_')[:2])  # MFD_06
    elif clean_name.startswith('NR_'):
        return '_'.join(clean_name.split('_')[:2])  # NR_0017
    elif clean_name.startswith('SWC_'):
        return '_'.join(clean_name.split('_')[:2])  # SWC_023
    elif clean_name.startswith('NYU-'):
        # NYU has dashes instead of underscores
        match = re.match(r'(NYU-\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('UCLA'):
        # Extract UCLA ID
        match = re.match(r'(UCLA\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('KS'):
        # Extract KS ID
        match = re.match(r'(KS\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ZFM-'):
        # Extract ZFM ID
        match = re.match(r'(ZFM-\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ZM_'):
        # Extract ZM ID
        match = re.match(r'(ZM_\d+)', clean_name)
        if match:
            return match.group(1)
    elif clean_name.startswith('ibl_witten_'):
        # Extract ibl_witten ID
        match = re.match(r'(ibl_witten_\d+)', clean_name)
        if match:
            return match.group(1)
    
    # Return cleaned name
    return clean_name

def extract_contrast_drift_rates(file_path):
    """Extract drift rates for different contrast levels from a model summary file."""
    # Load the summary file without headers - first column is parameter name
    df = pd.read_csv(file_path, header=None)
    
    df.columns = ['parameter', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 
                     'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat']
    # Ensure 'mean' column is numeric
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    
    # Get filename and extract mouse ID
    filename = os.path.basename(file_path)
    mouse_id = extract_full_mouse_id(filename)
    
    # Initialize dictionary to store drift rates
    drift_rates = {'mouse_id': mouse_id}
    
    # Extract drift rates for each contrast level
    for contrast in CONTRAST_LEVELS:
        # For contrast = 0, use 0 as default (will be updated if found)
        if contrast == 0.0:
            drift_rates[f'c_{contrast}'] = 0.0
        
        # Define patterns to look for in parameter names
        contrast_str = str(contrast) if contrast == int(contrast) else f"{contrast:.1f}"
        patterns = [
            f"v_C\\(contrast_category.*\\)\\[c_{contrast_str}\\]",
            f"v_C\\(contrast_category.*\\)\\[c_{contrast}\\]",
            f"v_contrast_category\\[c_{contrast_str}\\]",
            f"v_signed_contrast\\[{contrast_str}\\]"
        ]
        
        found = False
        for pattern in patterns:
            # Use str.contains with regex to find matching parameters
            mask = df['parameter'].astype(str).str.contains(pattern, regex=True, na=False)
            if any(mask):
                drift = df[mask]['mean'].iloc[0]
                if not pd.isna(drift):  # Ensure we have a valid numeric value
                    drift_rates[f'c_{contrast}'] = float(drift)
                    found = True
                    break
        
        if not found and contrast != 0.0:
            drift_rates[f'c_{contrast}'] = np.nan
    
    return drift_rates

# def plot_contrast_drift_rates(all_drift_rates, mouse_id=None):
#     """Create a plot of drift rates across contrast levels."""
#     # Set up the figure
#     plt.figure(figsize=(10, 6))
    
#     # Use seaborn for better aesthetics
#     sns.set_style('whitegrid')
    
#     # Convert to DataFrame for easier plotting
#     df = pd.DataFrame(all_drift_rates)
    
#     # Ensure all columns with contrast values are numeric
#     for col in df.columns:
#         if col.startswith('c_'):
#             df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     if mouse_id is not None:
#         # Plot for a single mouse
#         mouse_data = df[df['mouse_id'] == mouse_id]
#         if len(mouse_data) == 0:
#             print(f"No data found for mouse {mouse_id}")
#             return None
        
#         # Extract contrast values and drift rates
#         contrasts = []
#         drift_rates = []
        
#         for contrast in CONTRAST_LEVELS:
#             col = f'c_{contrast}'
#             if col in mouse_data.columns and not pd.isna(mouse_data[col].iloc[0]):
#                 contrasts.append(contrast)
#                 drift_rates.append(float(mouse_data[col].iloc[0]))
        
#         # Plot the drift rates
#         plt.plot(contrasts, drift_rates, 'o-', linewidth=2, markersize=8, color='blue')
        
#         plt.title(f'Drift Rates by Contrast for Mouse {mouse_id}')
#     else:
#         # Plot for all mice
#         # First get the average across mice for each contrast
#         mean_rates = []
#         std_rates = []
#         contrasts = []
        
#         for contrast in CONTRAST_LEVELS:
#             col = f'c_{contrast}'
#             if col in df.columns:
#                 # Remove NaN values
#                 valid_rates = df[col].dropna()
#                 if len(valid_rates) > 0:
#                     mean_rates.append(float(valid_rates.mean()))
#                     std_rates.append(float(valid_rates.std()))
#                     contrasts.append(contrast)
        
#         # Plot individual mice with low opacity
#         for idx, row in df.iterrows():
#             mouse_contrasts = []
#             mouse_rates = []
            
#             for contrast in contrasts:
#                 col = f'c_{contrast}'
#                 if col in df.columns and not pd.isna(row[col]):
#                     mouse_contrasts.append(contrast)
#                     mouse_rates.append(float(row[col]))
            
#             if mouse_contrasts:
#                 plt.plot(mouse_contrasts, mouse_rates, 'o-', alpha=0.2, linewidth=1, markersize=4, color='gray')
        
#         # Plot the average with error bars
#         plt.errorbar(contrasts, mean_rates, yerr=std_rates, fmt='o-', linewidth=2, 
#                      markersize=8, color='blue', capsize=5, label='Group Average')
        
#         plt.title(f'Drift Rates by Contrast ({MODEL_TYPE.upper()} Model)')
    
#     # Add reference line at y=0
#     plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
#     # Set axis labels and limits
#     plt.xlabel('Contrast Value')
#     plt.ylabel('Drift Rate (v)')
#     plt.ylim(-2.5, 2.5)  # Adjusted to accommodate most data
    
#     # Add grid
#     plt.grid(True, alpha=0.3)
    
#     # Improve layout
#     plt.tight_layout()
    
#     return plt

def plot_contrast_drift_rates(all_drift_rates, mouse_id=None):
    """Create a plot of drift rates across contrast levels with jittered points."""
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Use seaborn for better aesthetics
    sns.set_style('whitegrid')
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(all_drift_rates)
    
    # Ensure all columns with contrast values are numeric
    for col in df.columns:
        if col.startswith('c_'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if mouse_id is not None:
        # Plot for a single mouse
        mouse_data = df[df['mouse_id'] == mouse_id]
        if len(mouse_data) == 0:
            print(f"No data found for mouse {mouse_id}")
            return None
        
        # Extract contrast values and drift rates
        contrasts = []
        drift_rates = []
        
        for contrast in CONTRAST_LEVELS:
            col = f'c_{contrast}'
            if col in mouse_data.columns and not pd.isna(mouse_data[col].iloc[0]):
                contrasts.append(contrast)
                drift_rates.append(float(mouse_data[col].iloc[0]))
        
        # Plot the drift rates
        plt.plot(contrasts, drift_rates, 'o-', linewidth=2, markersize=8, color='black')
        
        plt.title(f'Drift Rates by Contrast for Mouse {mouse_id}')
    else:
        # Plot for all mice
        # First get the average across mice for each contrast
        mean_rates = []
        std_rates = []
        contrasts = []
        
        for contrast in CONTRAST_LEVELS:
            col = f'c_{contrast}'
            if col in df.columns:
                # Remove NaN values
                valid_rates = df[col].dropna()
                if len(valid_rates) > 0:
                    mean_rates.append(float(valid_rates.mean()))
                    std_rates.append(float(valid_rates.std()))
                    contrasts.append(contrast)
        
        # Plot individual mice as jittered points at each contrast level (no lines)
        for contrast in contrasts:
            col = f'c_{contrast}'
            valid_rates = df[col].dropna()
            
            if len(valid_rates) > 0:
                # Create jitter for x-axis
                jitter = np.random.normal(0, 1.0, size=len(valid_rates))
                jitter_scale = (contrasts[1] - contrasts[0]) * 0.05 if len(contrasts) > 1 else 1.0
                jittered_x = np.full(len(valid_rates), contrast) + jitter * jitter_scale
                
                # Plot jittered points
                plt.scatter(jittered_x, valid_rates, alpha=0.3, s=30, color='gray', edgecolor='none')
        
        # Plot the average with error bars in black
        plt.errorbar(contrasts, mean_rates, yerr=std_rates, fmt='o-', linewidth=2, 
                     markersize=8, color='black', capsize=5, label='Group Average')
        
        plt.title(f'Drift Rates by Contrast ({MODEL_TYPE.upper()} Model)')
    
    # Add reference line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Set axis labels and limits
    plt.xlabel('Contrast Value')
    plt.ylabel('Drift Rate (v)')
    plt.ylim(-2.5, 2.5)  # Adjusted to accommodate most data
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    return plt

def main():
    # Get all summary files for the specified model type
    summary_files = sorted([f for f in os.listdir(SUMMARY_DIR) if f.endswith(f'_{MODEL_TYPE}_summary.csv')])
    
    print(f"Found {len(summary_files)} {MODEL_TYPE} summary files")
    
    # Extract drift rates for each mouse
    all_drift_rates = []
    for filename in summary_files:
        try:
            file_path = os.path.join(SUMMARY_DIR, filename)
            print(f"Processing {filename}...")
            drift_rates = extract_contrast_drift_rates(file_path)
            all_drift_rates.append(drift_rates)
            print(f"  Successfully extracted drift rates for {drift_rates['mouse_id']}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Successfully processed {len(all_drift_rates)} files")
    
    if all_drift_rates:
        # Debug: print the first few items to check structure
        print("\nSample of extracted drift rates:")
        for i, rates in enumerate(all_drift_rates[:3]):
            print(f"Mouse {i+1}: {rates['mouse_id']}")
            for k, v in rates.items():
                if k != 'mouse_id':
                    print(f"  {k}: {v}")
        
        # Plot for all mice
        try:
            plt = plot_contrast_drift_rates(all_drift_rates)
            if plt:
                all_plot_path = os.path.join(PLOT_DIR, f"{MODEL_TYPE}_contrast_drift_rates.png")
                plt.savefig(all_plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved group plot to {all_plot_path}")
                plt.close()
        except Exception as e:
            print(f"Error creating group plot: {str(e)}")
        
        # Plot for each individual mouse
        for drift_rates in all_drift_rates:
            mouse_id = drift_rates['mouse_id']
            try:
                plt = plot_contrast_drift_rates([drift_rates], mouse_id)
                if plt:
                    mouse_plot_path = os.path.join(PLOT_DIR, f"{MODEL_TYPE}_{mouse_id}_contrast_drift_rates.png")
                    plt.savefig(mouse_plot_path, dpi=300, bbox_inches='tight')
                    print(f"Saved individual plot for {mouse_id}")
                    plt.close()
            except Exception as e:
                print(f"Error creating plot for {mouse_id}: {str(e)}")
    else:
        print("No data to plot")

if __name__ == "__main__":
    main()