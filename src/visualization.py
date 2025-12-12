import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .utils import print_info, print_warning, Colors
from .analysis import filter_by_time

# Try imports
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def plot_optimization_metrics(k_range, inertia, silhouette, rejected=[]):
    """Plots Elbow (Inertia) and Silhouette scores with Rejection Flags."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Inertia
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(k_range, inertia, color=color, marker='o', label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Silhouette
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette, color=color, marker='s', linestyle='--', label='Silhouette')
    
    # Mark Rejected
    if rejected:
        for k in rejected:
            if k in k_range:
                idx = k_range.index(k)
                val = silhouette[idx]
                ax2.scatter([k], [val], color='red', marker='x', s=150, linewidth=3, zorder=20, label='Rejected (Size)')

    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Optimal K Analysis: Elbow Method & Silhouette Score')
    fig.tight_layout()
    
    # Save for Researcher Mode if dir exists
    import os
    if os.path.exists("reports/researcher"):
        plt.savefig("reports/researcher/kselect_METRICS.png")
    
    plt.show() # Show for interactive users still

def plot_cluster_analysis(results_df, centroids_df, label_map, sensor_name="Global"):
    """Comprehensive 3-panel visualization of clustering results."""
    
    if not SKLEARN_AVAILABLE:
        print_warning("Sklearn required for PCA visualization.")
        return

    n_clusters = len(centroids_df)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # --- Plot 1: PCA Projection (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    pca = PCA(n_components=2)
    feature_cols = centroids_df.columns # These correspond to X columns
    
    # Drop rows with NAs in features
    valid_data = results_df.dropna(subset=feature_cols)
    X_pca = valid_data[feature_cols]
    
    if len(X_pca) > 0:
        coords = pca.fit_transform(X_pca)
        
        scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=valid_data['Cluster'], cmap='viridis', alpha=0.6)
        ax1.set_title(f"PCA Visual Separation (Saved {pca.explained_variance_ratio_.sum()*100:.1f}% Variance)")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        # Legend
        handles, _ = scatter.legend_elements()
        legend_labels = [label_map.get(i, f"Cluster {i}") for i in range(n_clusters)]
        ax1.legend(handles, legend_labels, loc="best")
    
    # --- Plot 2: Centroid Heatmap (Top Right) ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    im = ax2.imshow(centroids_df, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax2.set_yticks(range(n_clusters))
    ax2.set_yticklabels([label_map.get(i, f"C{i}") for i in range(n_clusters)])
    ax2.set_xticks(range(len(feature_cols)))
    ax2.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax2.set_title("Cluster Feature Signatures (Z-Scores)")
    plt.colorbar(im, ax=ax2, label='Z-Score')
    
    # --- Plot 3: Time Series (Bottom) ---
    ax3 = fig.add_subplot(gs[1, :])
    
    # Scatter plot of time
    # Y-axis = Cluster ID
    # Use 'SendDate' column if exists, else index
    if 'SendDate' in results_df.columns:
        times = pd.to_datetime(results_df['SendDate'])
    else:
        times = results_df.index
        
    y_vals = results_df['Cluster']
    
    ax3.scatter(times, y_vals, c=y_vals, cmap='viridis', s=20, alpha=0.8)
    ax3.set_yticks(range(n_clusters))
    ax3.set_yticklabels([label_map.get(i, f"C{i}") for i in range(n_clusters)])
    ax3.set_title(f"{sensor_name} - Activity States Over Time")
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax3.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    # Check if we are in non-interactive mode (e.g. producing report)
    # We might want to save instead of show?
    # For now, let's try strict save if reports dir exists
    import os
    if os.path.exists("reports"):
        plt.savefig(f"reports/cluster_analysis_{sensor_name}.png")
        print_info("Visualization", f"Saved plot to reports/cluster_analysis_{sensor_name}.png")
    else:
        plt.show()
    plt.close()

def plot_sensor_data(df, sensor_name, sensor_name_col, target_label, target_id, time_option, period_option, location_name="", start_date=None, end_date=None):
    """Plots Value vs SendDate."""
    print_info("Plotting", f"'{target_label}' for '{sensor_name}'")
    
    mask = (df[sensor_name_col] == sensor_name) & (df['SensorId'].astype(str) == str(target_id))
    subset = df[mask].copy()
    
    if subset.empty:
        print_warning("No data found.")
        return

    subset['SendDate'] = pd.to_datetime(subset['SendDate'], errors='coerce')
    subset = subset.dropna(subset=['SendDate']).sort_values('SendDate')
    
    subset = filter_by_time(subset, time_option, start_date, end_date)
    
    if subset.empty:
        print_warning("No data in time range.")
        return

    # Resampling
    if period_option and period_option != 'none':
        print_info("Resampling", period_option)
        subset = subset.set_index('SendDate')
        subset['Value'] = pd.to_numeric(subset['Value'], errors='coerce')
        subset = subset['Value'].resample(period_option).mean().reset_index()

    if subset.empty or subset['Value'].dropna().empty:
         print_warning("No valid data points to plot.")
         return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(subset['SendDate'], subset['Value'], color='#000080', linewidth=2)
    ax.fill_between(subset['SendDate'], subset['Value'], color='#ADD8E6', alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#E0E0E0') 
    ax.yaxis.set_visible(False)
    ax.grid(False)

    title_text = f"{sensor_name} - {location_name}" if location_name else sensor_name
    ax.text(0.0, 1.10, title_text, transform=ax.transAxes, fontsize=16, fontweight='bold', color='black', ha='left')
    ax.text(0.0, 1.05, target_label, transform=ax.transAxes, fontsize=12, color='gray', ha='left')
    
    plt.tight_layout()
    plt.show()
    print_info("Status", "Plot displayed.")


def plot_feature_profiles(centroids_df, label_map, sensor_name):
    """Plots a bar chart of feature profiles for each cluster."""
    if centroids_df.empty: return
    
    n_clusters = len(centroids_df)
    features = centroids_df.columns
    n_features = len(features)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(n_features)
    width = 0.8 / n_clusters
    
    # Plot bars
    for i in range(n_clusters):
        offset = (i - n_clusters/2) * width + width/2
        cluster_label = label_map.get(i, f"Cluster {i}")
        
        # Color: Use a distinct color for each cluster
        # viridis or tab10
        color = plt.cm.tab10(i % 10)
        
        ax.bar(x + offset, centroids_df.iloc[i], width, label=cluster_label, color=color, alpha=0.9, edgecolor='white')

    ax.set_ylabel('Z-Score (Standardized Value)')
    ax.set_title(f'Cluster Feature Profiles: {sensor_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    if os.path.exists("reports/researcher"):
        plt.savefig(f"reports/researcher/feature_profiles_{sensor_name}.png")
        print("Saved feature_profiles.png")
    else:
        plt.show()

# --- RESEARCHER MODE VISUALIZATIONS ---
import os 

def plot_coverage_timeline(df, input_period, title_suffix=""):
    """V1: Coverage Timeline (Valid Bins)."""
    # Helper to scan for valid bins (NaN check) per column
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # We want a heatmap: Rows=Rooms, X=Time
    # 1. Inspect NAs
    is_valid = df.notna().astype(int)
    
    # Plot imshow
    # Aspect auto to stretch time
    im = ax.imshow(is_valid.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Labels
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    
    # Time X-Axis (Rough index ticks)
    n_ticks = 10
    idxs = np.linspace(0, len(df)-1, n_ticks).astype(int)
    labels = [df.index[i].strftime('%Y-%m-%d') for i in idxs]
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels, rotation=45)
    
    ax.set_title(f"Data Coverage Timeline (Green=Valid, Red=Missing/Invalid) {title_suffix}")
    plt.tight_layout()
    
    # Save (Researcher always saves)
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/coverage_TIMELINE_{title_suffix}.png")
    plt.close()

def plot_stacked_validation(level_series, feat_series, labels, title_suffix=""):
    """V2/V3: Stacked Level vs Feature."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Level
    ax1.plot(level_series.index, level_series.values, color='navy', lw=1.5)
    ax1.set_title(f"{labels[0]} (Level)")
    ax1.fill_between(level_series.index, level_series.values, color='skyblue', alpha=0.3)
    
    # Plot 2: Feature
    ax2.plot(feat_series.index, feat_series.values, color='darkorange', lw=1.5)
    ax2.set_title(f"{labels[1]} (Derived Feature)")
    ax2.fill_between(feat_series.index, feat_series.values, color='orange', alpha=0.3)
    
    _style_res_axis(ax1)
    _style_res_axis(ax2)
    
    fig.suptitle(f"Feature Validation: {title_suffix}")
    plt.tight_layout()
    
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/feature_{title_suffix}.png")
    plt.close()

def plot_correlation_heatmap(matrix, title_suffix=""):
    """V6: Feature Correlation."""
    check_df = matrix.dropna()
    corr = check_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Ticks
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(f"Feature Correlation Matrix {title_suffix}")
    plt.tight_layout()
    
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/qc_CORR_{title_suffix}.png")
    plt.close()

def plot_state_share(labeled_df, title_suffix=""):
    """V12: State Share Stacked Bar."""
    # Crosstab
    if "activity_label" not in labeled_df.columns: return
    ct = pd.crosstab(labeled_df.index.get_level_values("NodeId"), labeled_df["activity_label"], normalize='index') * 100
    
    ax = ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.9)
    plt.title(f"State Share per Room {title_suffix}")
    plt.ylabel("% Time")
    plt.xlabel("Room") # Fix Label
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/usage_SHARE_{title_suffix}.png")
    plt.close()

def plot_hourly_heatmap(labeled_df, node_id, title_suffix=""):
    """V13: Hour-of-Day Heatmap for a node."""
    # Filter node
    subset = labeled_df[labeled_df.index.get_level_values("NodeId") == node_id]
    if subset.empty: return
    
    # Extract Hour
    times = subset.index.get_level_values("TimeBin")
    hours = times.hour
    
    # Pivot
    ct = pd.crosstab(subset["activity_label"], hours)
    ct_norm = ct.div(ct.sum(axis=0), axis=1).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(ct_norm, aspect='auto', cmap='plasma', vmin=0, vmax=1)
    
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_yticks(range(len(ct_norm.index)))
    ax.set_yticklabels(ct_norm.index)
    ax.set_ylabel("Activity State")
    
    # Fix Title Duplication
    clean_suffix = title_suffix.replace(node_id, "").strip("_")
    ax.set_title(f"Hourly Activity Profile: {node_id} {clean_suffix}")
    
    plt.colorbar(im, ax=ax, label="Prob")
    plt.tight_layout()
    
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/usage_HEATMAP_{node_id}_{title_suffix}.png")
    plt.close()
    
def plot_transition_matrix(trans_matrix, title_suffix=""):
    """V15: Transition Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1)
    
    # Add numbers
    for i in range(len(trans_matrix)):
        for j in range(len(trans_matrix)):
            text = ax.text(j, i, f"{trans_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if trans_matrix.iloc[i, j] < 0.5 else "white")
            
    ax.set_xticks(range(len(trans_matrix.columns)))
    ax.set_yticks(range(len(trans_matrix.index)))
    ax.set_xticklabels(trans_matrix.columns)
    ax.set_yticklabels(trans_matrix.index)
    
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    
    plt.colorbar(im, ax=ax, label="Transition Prob")
    ax.set_title(f"Transition Matrix {title_suffix}")
    plt.tight_layout()
    
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    plt.savefig(f"reports/researcher/usage_TRANSITION_{title_suffix}.png")
    plt.close()

def _style_res_axis(ax):
    """Helper for researcher plots."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)
    # Ensure Date Formatting covers Day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
