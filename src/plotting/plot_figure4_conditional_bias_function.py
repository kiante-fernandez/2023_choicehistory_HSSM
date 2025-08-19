
#old stuff for getting the conditional bias funciton plot
#%% Conditional bias function computation
def compute_conditional_bias_function(data, n_quantiles=5):
    """
    Compute conditional bias function with quantiles
    Returns both summary statistics and subject-level data
    """
    # Create lists to store data for each quantile
    quantile_means = []
    quantile_sems = []
    rt_means = []
    
    # Store subject-level data for plotting individual points
    subject_data = []
    
    # Process each RT quantile separately
    for q in range(n_quantiles):
        # For each subject, get their data for this quantile
        subject_means = []
        subject_rt_means = []
        
        for subj_id in data['participant_id'].unique():
            subj_data = data[data['participant_id'] == subj_id].copy()
            
            # Get quantile boundaries for this subject's RT
            quantile_edges = np.percentile(subj_data['rt'], 
                                          np.linspace(0, 100, n_quantiles+1))
            
            # Select data in this quantile
            if q < n_quantiles-1:
                q_data = subj_data[(subj_data['rt'] >= quantile_edges[q]) & 
                                  (subj_data['rt'] < quantile_edges[q+1])]
            else:
                q_data = subj_data[subj_data['rt'] >= quantile_edges[q]]
            
            if len(q_data) > 0:
                repeat_mean = q_data['repeat'].mean()
                rt_mean = q_data['rt'].mean()
                
                subject_means.append(repeat_mean)
                subject_rt_means.append(rt_mean)
                
                # Store individual subject data for later plotting
                subject_data.append({
                    'subject_id': subj_id,
                    'rt_quantile': q,
                    'repeat_mean': repeat_mean,
                    'rt_mean': rt_mean
                })
        
        # Calculate mean and SEM across subjects for this quantile
        quantile_means.append(np.mean(subject_means))
        quantile_sems.append(stats.sem(subject_means))
        rt_means.append(np.mean(subject_rt_means))
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'rt_quantile': range(n_quantiles),
        'repeat_mean': quantile_means,
        'repeat_sem': quantile_sems,
        'rt_mean': rt_means
    })
    subject_df = pd.DataFrame(subject_data)
    
    return summary, subject_df

#%% Professional plotting function with jitter
def plot_conditional_bias_function(summary, subject_df, n_quantiles=5):
    """
    Create conditional bias function plot with jittered data points
    """
    # Set figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual subject data points with jitter
    for q in range(n_quantiles):
        q_data = subject_df[subject_df['rt_quantile'] == q]
        
        # Add horizontal jitter to x-coordinates
        jitter_amount = 0.15  # Adjust this value to control jitter width
        x_jittered = np.array([q] * len(q_data)) + np.random.uniform(-jitter_amount, jitter_amount, size=len(q_data))
        
        ax.scatter(
            x_jittered, 
            q_data['repeat_mean'], 
            color='gray', 
            alpha=0.2,  # Slightly increased alpha for better visibility
            s=25,       # Slightly larger points
            zorder=1
        )
    
    # Plot main line with error bars - thicker and more prominent
    ax.errorbar(
        range(n_quantiles), 
        summary['repeat_mean'], 
        yerr=summary['repeat_sem'],
        fmt='-o', 
        color='black', 
        ecolor='black', 
        capsize=6, 
        linewidth=2.5, 
        markersize=9,
        zorder=3
    )
    
    # Add reference line at 0.5 (no bias)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, zorder=2)
    
    # Set labels with better font
    ax.set_xlabel('Response time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Choice bias (fraction)', fontsize=16, fontweight='bold')
    
    # Set axis limits
    ax.set_ylim(0.48, 0.61)
    ax.set_xlim(-0.5, n_quantiles-0.5)
    
    # Set custom x-tick labels to show actual RT values
    ax.set_xticks(range(n_quantiles))
    ax.set_xticklabels([f"{rt:.2f}" for rt in summary['rt_mean']], fontsize=13)
    
    # Set y-ticks and make them more readable
    ax.set_yticks(np.arange(0.50, 0.61, 0.04))
    ax.tick_params(axis='y', labelsize=12)
    
    # Remove top and right spines
    sns.despine(ax=ax)
    
    ax.grid(False)

    plt.tight_layout()
    
    return fig, ax