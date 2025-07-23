import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def calculate_intra_cluster_variance_trajectory(trajectories, labels):
    """
    Calculates the Average Intra-Cluster Variance for each time step in a trajectory.

    Args:
        trajectories (np.array): Shape (N, num_epochs, 2). The 2D embeddings over time.
        labels (np.array): Shape (N,). The class labels for each point.

    Returns:
        np.array: A 1D array of variance scores, one for each epoch.
    """
    num_epochs = trajectories.shape[1]
    variance_scores = []
    print("Calculating Intra-Cluster Variance trajectory...")
    for t in range(num_epochs):
        current_embedding = trajectories[:, t, :]
        class_variances = []
        for c in np.unique(labels):
            class_points = current_embedding[labels == c]
            if len(class_points) > 1:
                centroid = class_points.mean(axis=0)
                variance = np.mean(np.sum((class_points - centroid)**2, axis=1))
                class_variances.append(variance)
        
        # Average the variance across all classes for this epoch
        if class_variances:
            variance_scores.append(np.mean(class_variances))
        else:
            variance_scores.append(variance_scores[-1] if variance_scores else 0)
            
    return np.array(variance_scores)

def calculate_inter_cluster_distance_trajectory(trajectories, labels):
    """
    Calculates the Average Inter-Cluster Distance for each time step.
    A higher value is better. A sharp decrease indicates collapse.
    """
    num_epochs = trajectories.shape[1]
    distance_scores = []
    print("Calculating Inter-Cluster Distance trajectory...")
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return np.zeros(num_epochs) # Metric is undefined for a single cluster

    for t in range(num_epochs):
        current_embedding = trajectories[:, t, :]
        centroids = np.array([current_embedding[labels == c].mean(axis=0) for c in unique_labels])
        
        # Calculate pairwise distances between centroids and take the mean
        if len(centroids) > 1:
            dist_matrix = squareform(pdist(centroids, 'euclidean'))
            # Get sum of upper triangle (to avoid double counting) and normalize
            avg_dist = np.sum(np.triu(dist_matrix)) / (len(centroids) * (len(centroids) - 1) / 2)
            distance_scores.append(avg_dist)
        else:
            distance_scores.append(0)
            
    return np.array(distance_scores)

def find_trigger_epoch_with_margin(
    metric_values, 
    epochs, 
    higher_is_better=False, 
    burn_in=5,
    k_consecutive=2,        # How many consecutive epochs of bad trend to require
    std_fraction_margin=0.2, # The margin: change must be > this * std_dev
    std_window_size = 10
):
    """
    Finds an anomaly trigger epoch based on a sustained trend that exceeds a dynamic,
    standard-deviation-based margin.
    """
    # Not enough history to evaluate a trend or the burn_in period
    if len(metric_values) < burn_in + k_consecutive:
        return -1
    
    # Iterate through the epochs, starting after the burn-in period
    for i in range(burn_in, len(metric_values)):
        
        # 1. Check for a sustained trend over the last k epochs
        # --------------------------------------------------------
        is_sustained_bad_trend = True
        # Ensure we have enough history for the check
        if i < k_consecutive -1: 
            continue
            
        for j in range(k_consecutive):
            # The indices to check are from i-j down to i-j-1
            # e.g., for k=2 at i=10: check (10 vs 9) and (9 vs 8)
            current_val = metric_values[i - j]
            prev_val = metric_values[i - j - 1]
            
            is_bad_direction = (current_val < prev_val) if higher_is_better else (current_val > prev_val)
            if not is_bad_direction:
                is_sustained_bad_trend = False
                break
        
        if not is_sustained_bad_trend:
            continue # The trend was not consistently bad, check next epoch

        # 2. If trend is sustained, check if the change is significant (exceeds margin)
        # --------------------------------------------------------------------------
        
        # Calculate the dynamic margin based on recent volatility
        # Ensure we don't go out of bounds at the start
        window_start_idx = max(0, i - std_window_size)
        recent_window = metric_values[window_start_idx : i]
        
        if len(recent_window) < 2:
            std_dev = 0
        else:
            std_dev = np.std(recent_window)

        margin = std_dev * std_fraction_margin
        
        # We only need to check if the *last* step in the trend was significant
        last_step_change = abs(metric_values[i] - metric_values[i-1])
        
        if last_step_change > margin:
            # Both conditions are met: sustained trend AND significant change
            return epochs[i]
            
    # If the loop finishes without finding a trigger
    return -1

def analyze_and_plot_trajectories(
    sentrycam_trajectories, 
    high_trajectory,
    all_labels, 
    loss_trajectory, 
    valid_loss_trajectory,
    scenario_type, 
    trigger_strategy='derivative',
    burn_in=5,
    save_name='unstable_training.pdf'
):
    """
    Analyzes pre-computed trajectories using a 2D health space (Inter vs. Intra cluster metrics)
    and generates a comprehensive diagnostic plot.
    """
    num_epochs = sentrycam_trajectories.shape[1]
    epochs = np.arange(1, num_epochs + 1)
    
    print(f"\n--- Analyzing Scenario: '{scenario_type}' with 2D Health Metrics ---")

    # --- Step 1: Calculate both core geometric metric trajectories ---
    intra_variance_traj = calculate_intra_cluster_variance_trajectory(sentrycam_trajectories, all_labels)
    inter_distance_traj = calculate_inter_cluster_distance_trajectory(sentrycam_trajectories, all_labels)

    high_intra_variance_traj = calculate_intra_cluster_variance_trajectory(high_trajectory, all_labels)
    high_inter_distance_traj = calculate_inter_cluster_distance_trajectory(high_trajectory, all_labels)

    # --- Step 2: Find trigger epochs with the new margin-aware function ---
    t_sentry_trigger_1 = find_trigger_epoch_with_margin(inter_distance_traj, epochs, higher_is_better=True, burn_in=burn_in)
    t_sentry_trigger_2 = find_trigger_epoch_with_margin(intra_variance_traj, epochs, higher_is_better=False, burn_in=burn_in)
    
    # Handle the case where one trigger is -1 (not found)
    if t_sentry_trigger_1 == -1: t_sentry_trigger_1 = float('inf')
    if t_sentry_trigger_2 == -1: t_sentry_trigger_2 = float('inf')
    t_sentry_trigger = min(t_sentry_trigger_1, t_sentry_trigger_2)
    if t_sentry_trigger == float('inf'): t_sentry_trigger = -1 # Reset if neither triggered
    
    t_high_sentry_trigger_1 = find_trigger_epoch_with_margin(high_inter_distance_traj, epochs, higher_is_better=True, burn_in=burn_in)
    t_high_sentry_trigger_2 = find_trigger_epoch_with_margin(high_intra_variance_traj, epochs, higher_is_better=False, burn_in=burn_in)

    if t_high_sentry_trigger_1 == -1: t_high_sentry_trigger_1 = float('inf')
    if t_high_sentry_trigger_2 == -1: t_high_sentry_trigger_2 = float('inf')
    t_high_sentry_trigger = min(t_high_sentry_trigger_1, t_high_sentry_trigger_2)
    if t_high_sentry_trigger == float('inf'): t_high_sentry_trigger = -1
    
    # Use the validation loss for the loss trigger
    t_loss_trigger = find_trigger_epoch_with_margin(valid_loss_trajectory, epochs, higher_is_better=False, burn_in=burn_in)

    # --- Step 3: Print Quantitative Results ---
    print(f"\n--- Anomaly Detection Results ('{trigger_strategy}' strategy) ---")
    print(f"SentryCam trigger at Epoch: {t_sentry_trigger}")
    print(f"High-quality trigger at Epoch: {t_high_sentry_trigger}")
    print(f"Loss-based trigger (starts rising) at Epoch: {t_loss_trigger}")
    if t_sentry_trigger != -1 and t_loss_trigger != -1:
        early_warning_epochs = t_loss_trigger - t_sentry_trigger
        if early_warning_epochs > 0:
            print(f"SentryCam provided a {early_warning_epochs}-epoch early warning!")

    # --- Step 4: Generate the 2D Cluster Health Trajectory Plot ---
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2)
    
    # Set font sizes for better paper readability
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    # === Panel (a): The new 2D Cluster Health Trajectory ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Panel (a): 2D Cluster Health Trajectory", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Intra-Cluster Variance", fontsize=12)
    ax1.set_ylabel("Inter-Cluster Distance", fontsize=12)

    # Combine metrics into a list of (x, y) points for plotting
    health_points = np.column_stack((intra_variance_traj, inter_distance_traj))

    if t_sentry_trigger != -1:
        health_points = health_points[:t_sentry_trigger, :]

    
    # Plot the trajectory path, color-coded by epoch
    scatter = ax1.scatter(health_points[:, 0], health_points[:, 1], c=epochs[:t_sentry_trigger] if t_sentry_trigger != -1 else epochs, cmap='viridis', s=40, zorder=3)
    cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical')
    cbar.set_label("Epoch", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Draw arrows to show the direction of training
    for i in range(1, len(health_points)):
        ax1.annotate("",
                     xy=health_points[i], xycoords='data',
                     xytext=health_points[i-1], textcoords='data',
                     arrowprops=dict(arrowstyle="->", color="black", alpha=0.6,
                                     shrinkA=5, shrinkB=5,
                                     patchA=None, patchB=None,
                                     connectionstyle="arc3,rad=0.1"))

    # Highlight and label key points
    ax1.plot(health_points[0, 0], health_points[0, 1], 'o', c='red', markersize=10, label='Start (Epoch 1)', zorder=4)
    if t_sentry_trigger != -1:
        trigger_idx = int(t_sentry_trigger - 1)
        ax1.plot(health_points[trigger_idx, 0], health_points[trigger_idx, 1], 'X', c='orange', markersize=12, label=f'SentryCam Alert (Epoch {t_sentry_trigger})', zorder=5)
    
    # Mark the ideal "goal" region
    ideal_x_thresh = np.percentile(intra_variance_traj, 20)
    ideal_y_thresh = np.percentile(inter_distance_traj, 80)
    ax1.axvspan(0, ideal_x_thresh, color='green', alpha=0.1, zorder=0, label='Ideal Region')
    ax1.axhspan(ideal_y_thresh, health_points[:, 1].max() * 1.1, color='green', alpha=0.1, zorder=0)
    
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(labelsize=10)

    # # === Panel (b): The high-quality Cluster Health Trajectory ===
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.set_title("Panel (b): High-quality Cluster Health Trajectory", fontweight='bold')
    # ax2.set_xlabel("Intra-Cluster Variance")
    # ax2.set_ylabel("Inter-Cluster Distance")

    # # Combine metrics into a list of (x, y) points for plotting
    # high_health_points = np.column_stack((high_intra_variance_traj, high_inter_distance_traj))

    # if t_high_sentry_trigger != -1:
    #     high_health_points = high_health_points[:t_high_sentry_trigger, :]
    
    # # Plot the trajectory path, color-coded by epoch
    # scatter = ax2.scatter(high_health_points[:, 0], high_health_points[:, 1], c=epochs[:t_high_sentry_trigger] if t_high_sentry_trigger != -1 else epochs, cmap='viridis', s=40, zorder=3)
    # cbar = fig.colorbar(scatter, ax=ax2, orientation='vertical')
    # cbar.set_label("Epoch")

    # # Draw arrows to show the direction of training
    # for i in range(1, len(high_health_points)):
    #     ax2.annotate("",
    #                  xy=high_health_points[i], xycoords='data',
    #                  xytext=high_health_points[i-1], textcoords='data',
    #                  arrowprops=dict(arrowstyle="->", color="black", alpha=0.4,
    #                                  shrinkA=5, shrinkB=5,
    #                                  patchA=None, patchB=None,
    #                                  connectionstyle="arc3,rad=0.1"))

    # # Highlight and label key points
    # ax2.plot(high_health_points[0, 0], high_health_points[0, 1], 'o', c='red', markersize=10, label='Start (Epoch 1)', zorder=4)
    # if t_high_sentry_trigger != -1:
    #     trigger_idx = t_high_sentry_trigger - 1
    #     ax2.plot(high_health_points[trigger_idx, 0], high_health_points[trigger_idx, 1], 'X', c='orange', markersize=12, label=f'High-quality Alert (Epoch {t_high_sentry_trigger})', zorder=5)
    
    # # Mark the ideal "goal" region
    # ideal_x_thresh = np.percentile(high_intra_variance_traj, 20)
    # ideal_y_thresh = np.percentile(high_inter_distance_traj, 80)
    # ax2.axvspan(0, ideal_x_thresh, color='green', alpha=0.1, zorder=0, label='Ideal Region')
    # ax2.axhspan(ideal_y_thresh, high_health_points[:, 1].max() * 1.1, color='green', alpha=0.1, zorder=0)
    
    # ax2.legend()
    # ax2.grid(True, linestyle='--', alpha=0.6)

    # === Panel (c): Traditional Loss Curve for Context ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Panel (b): Validation Loss Curve", fontweight='bold', fontsize=14)
    ax2.plot(epochs, loss_trajectory, '-o', color='red', label='Training Loss')
    ax2.plot(epochs, valid_loss_trajectory, '-o', color='blue', label='Validation Loss')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    
    # Annotate with triggers for comparison
    if t_loss_trigger != -1:
        ax2.axvline(x=t_loss_trigger, color='red', linestyle='--', label=f'Loss Alert (Epoch {t_loss_trigger})')
    if t_sentry_trigger != -1:
        ax2.axvline(x=t_sentry_trigger, color='blue', linestyle='--', label=f'SentryCam trajectory Alert (Epoch {t_sentry_trigger})')
    if t_high_sentry_trigger != -1:
        ax2.axvline(x=t_high_sentry_trigger, color='green', linestyle='--', label=f'High dimensional trajectory Alert (Epoch {t_high_sentry_trigger})')
    
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--')
    ax2.tick_params(labelsize=10)

    # # === Panel (d): SentryCam View ===
    # ax = fig.add_subplot(gs[0, 3])
    # emb = sentrycam_trajectories[:, t_sentry_trigger - 1, :]
    # scatter = ax.scatter(emb[:, 0], emb[:, 1], c=all_labels, cmap='Spectral', s=5, alpha=0.7)
    # ax.set_title(f"SentryCam View (Epoch {t_sentry_trigger})")
    # ax.set_xticks([]); ax.set_yticks([])

    # ax = fig.add_subplot(gs[0, 4])
    # emb = sentrycam_trajectories[:, t_high_sentry_trigger - 1, :]
    # scatter = ax.scatter(emb[:, 0], emb[:, 1], c=all_labels, cmap='Spectral', s=5, alpha=0.7)
    # ax.set_title(f"SentryCam View (Epoch {t_high_sentry_trigger})")
    # ax.set_xticks([]); ax.set_yticks([])

    # fig.colorbar(scatter, ax=ax, label='Class Label')
    # fig.tight_layout()
    # plt.suptitle(f"Diagnostic Analysis for Scenario: {scenario_type.title()}", fontsize=16, y=1.02)
    # plt.show()
    
    # plt.suptitle(f"SentryCam Diagnostic for '{scenario_type.title()}' Scenario", fontsize=18, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig("unstable_training.pdf", bbox_inches='tight', dpi=300)