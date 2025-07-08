import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import pdist, squareform

# --- 1. METRIC CALCULATION FUNCTIONS (Input: Trajectory Data) ---
def calculate_dbi_trajectory(trajectories, labels):
    """
    Calculates the Davies-Bouldin Index for each time step in a trajectory.

    Args:
        trajectories (np.array): Shape (N, num_epochs, 2). The 2D embeddings over time.
        labels (np.array): Shape (N,). The class labels for each point.

    Returns:
        np.array: A 1D array of DBI scores, one for each epoch.
    """
    num_epochs = trajectories.shape[1]
    dbi_scores = []
    print("Calculating Davies-Bouldin Index trajectory...")
    for t in range(num_epochs):
        current_embedding = trajectories[:, t, :]
        # Check for edge case of a single cluster, which would crash DBI
        if len(np.unique(labels)) > 1:
            score = davies_bouldin_score(current_embedding, labels)
            dbi_scores.append(score)
        else: # If only one cluster, score is not well-defined. Append previous or 0.
            dbi_scores.append(dbi_scores[-1] if dbi_scores else 0)
    return np.array(dbi_scores)

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

# --- 2. DYNAMIC THRESHOLDING (This function is unchanged) ---

def find_trigger_epoch(metric_values, epochs, strategy='derivative', higher_is_better=False, burn_in=5, **kwargs):
    """
    Finds the anomaly trigger epoch. Now handles both "lower is better" and "higher is better" metrics.
    """
    if len(metric_values) <= burn_in:
        return -1
    trigger_epoch = -1
    smoothed_metric = np.convolve(metric_values, np.ones(3)/3, mode='same')

    # Condition for an anomaly depends on whether higher or lower is better
    is_anomaly = (lambda current, prev: current < prev) if higher_is_better else (lambda current, prev: current > prev)

    if strategy == 'derivative':
        for i in range(burn_in, len(smoothed_metric)):
            if is_anomaly(smoothed_metric[i], smoothed_metric[i-1]) and is_anomaly(smoothed_metric[i], smoothed_metric[i-2]):
                trigger_epoch = epochs[i]
                break
    
    elif strategy == 'relative_to_best':
        tolerance = kwargs.get('tolerance', 0.2)
        # Find the best score during burn-in
        best_score = np.max(metric_values[:burn_in]) if higher_is_better else np.min(metric_values[:burn_in])
        if abs(best_score) < 1e-9: best_score = 1e-9 if best_score >= 0 else -1e-9

        # Threshold is a degradation from the best score
        threshold = best_score * (1 - tolerance) if higher_is_better else best_score * (1 + tolerance)
        
        for i in range(burn_in, len(metric_values)):
            if is_anomaly(metric_values[i], threshold):
                trigger_epoch = epochs[i]
                break
                
    elif strategy == 'std_dev_zscore':
        window_size = kwargs.get('window_size', 5)
        num_std_devs = kwargs.get('num_std_devs', 2.0)
        if len(metric_values) < burn_in + window_size: return -1
        
        for i in range(burn_in + window_size, len(metric_values)):
            window = metric_values[i-window_size : i]
            mean, std = np.mean(window), np.std(window)
            if std < 1e-9: continue
            
            # Anomaly is being too many std devs away from the mean in the "bad" direction
            threshold = mean - num_std_devs * std if higher_is_better else mean + num_std_devs * std
            
            if is_anomaly(metric_values[i], threshold):
                trigger_epoch = epochs[i]
                break

    return trigger_epoch

# --- 3. MAIN ANALYSIS & PLOTTING FUNCTION ---

def analyze_and_plot_trajectories(
    sentrycam_trajectories, 
    all_labels, 
    loss_trajectory, 
    valid_loss_trajectory,
    scenario_type, 
    trigger_strategy='derivative'
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

    # --- Step 2: Find trigger epochs for early warning calculation ---
    # The primary anomaly signal for collapse/instability is a drop in inter-cluster distance.
    # We use this as our main SentryCam trigger.
    t_sentry_trigger = find_trigger_epoch(inter_distance_traj, epochs, strategy=trigger_strategy, higher_is_better=True)
    t_loss_trigger = find_trigger_epoch(loss_trajectory, epochs, strategy=trigger_strategy, higher_is_better=False)

    # --- Step 3: Print Quantitative Results ---
    print(f"\n--- Anomaly Detection Results ('{trigger_strategy}' strategy) ---")
    print(f"Loss-based trigger (starts rising) at Epoch: {t_loss_trigger}")
    print(f"SentryCam trigger (Inter-Cluster Distance starts dropping) at Epoch: {t_sentry_trigger}")
    if t_sentry_trigger != -1 and t_loss_trigger != -1:
        early_warning_epochs = t_loss_trigger - t_sentry_trigger
        if early_warning_epochs > 0:
            print(f"SentryCam provided a {early_warning_epochs}-epoch early warning!")

    # --- Step 4: Generate the 2D Cluster Health Trajectory Plot ---
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2)
    
    # === Panel (a): The new 2D Cluster Health Trajectory ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Panel (a): 2D Cluster Health Trajectory", fontweight='bold')
    ax1.set_xlabel("Intra-Cluster Variance (Worse ->)")
    ax1.set_ylabel("Inter-Cluster Distance (<- Better)")

    # Combine metrics into a list of (x, y) points for plotting
    health_points = np.column_stack((intra_variance_traj, inter_distance_traj))
    
    # Plot the trajectory path, color-coded by epoch
    scatter = ax1.scatter(health_points[:, 0], health_points[:, 1], c=epochs, cmap='viridis', s=40, zorder=3)
    cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical')
    cbar.set_label("Epoch")

    # Draw arrows to show the direction of training
    for i in range(1, len(health_points)):
        ax1.annotate("",
                     xy=health_points[i], xycoords='data',
                     xytext=health_points[i-1], textcoords='data',
                     arrowprops=dict(arrowstyle="->", color="black", alpha=0.4,
                                     shrinkA=5, shrinkB=5,
                                     patchA=None, patchB=None,
                                     connectionstyle="arc3,rad=0.1"))

    # Highlight and label key points
    ax1.plot(health_points[0, 0], health_points[0, 1], 'o', c='red', markersize=10, label='Start (Epoch 1)', zorder=4)
    if t_sentry_trigger != -1:
        trigger_idx = t_sentry_trigger - 1
        ax1.plot(health_points[trigger_idx, 0], health_points[trigger_idx, 1], 'X', c='orange', markersize=12, label=f'SentryCam Alert (Epoch {t_sentry_trigger})', zorder=5)
    
    # Mark the ideal "goal" region
    ideal_x_thresh = np.percentile(intra_variance_traj, 25)
    ideal_y_thresh = np.percentile(inter_distance_traj, 75)
    ax1.axvspan(0, ideal_x_thresh, color='green', alpha=0.1, zorder=0, label='Ideal Region')
    ax1.axhspan(ideal_y_thresh, health_points[:, 1].max() * 1.1, color='green', alpha=0.1, zorder=0)
    
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # === Panel (b): Traditional Loss Curve for Context ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Panel (b): Validation Loss Curve", fontweight='bold')
    ax2.plot(epochs, loss_trajectory, '-o', color='red', label='Training Loss')
    ax2.plot(epochs, valid_loss_trajectory, '-o', color='blue', label='Validation Loss')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")

    # # === Panel (c): SentryCam View ===
    # epoch_to_plot=7
    # ax = fig.add_subplot(gs[1, 0])
    # emb = sentrycam_trajectories[:, epoch_to_plot - 1, :]
    # scatter = ax.scatter(emb[:, 0], emb[:, 1], c=all_labels, cmap='Spectral', s=5, alpha=0.7)
    # ax.set_title(f"SentryCam View (Epoch {epoch_to_plot})")
    # ax.set_xticks([]); ax.set_yticks([])

    # epoch_to_plot=13
    # ax = fig.add_subplot(gs[1, 1])
    # emb = sentrycam_trajectories[:, epoch_to_plot - 1, :]
    # scatter = ax.scatter(emb[:, 0], emb[:, 1], c=all_labels, cmap='Spectral', s=5, alpha=0.7)
    # ax.set_title(f"SentryCam View (Epoch {epoch_to_plot})")
    # ax.set_xticks([]); ax.set_yticks([])

    # fig.colorbar(scatter, ax=ax, label='Class Label')
    # fig.tight_layout()
    # plt.suptitle(f"Diagnostic Analysis for Scenario: {scenario_type.title()}", fontsize=16, y=1.02)
    # plt.show()
    
    # Annotate with triggers for comparison
    if t_loss_trigger != -1:
        ax2.axvline(x=t_loss_trigger, color='red', linestyle='--', label=f'Loss Alert (Epoch {t_loss_trigger})')
    if t_sentry_trigger != -1:
        ax2.axvline(x=t_sentry_trigger, color='blue', linestyle='--', label=f'SentryCam Alert (Epoch {t_sentry_trigger})')
    
    ax2.legend()
    ax2.grid(True, linestyle='--')
    
    plt.suptitle(f"SentryCam Diagnostic for '{scenario_type.title()}' Scenario", fontsize=16, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()