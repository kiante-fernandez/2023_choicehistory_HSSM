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