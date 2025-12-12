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

def plot_optimization_metrics(k_range, inertia, silhouette):
    """Plots Elbow (Inertia) and Silhouette scores."""
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
    # Use bar for silhouette? or line
    ax2.plot(k_range, silhouette, color=color, marker='s', linestyle='--', label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Optimal K Analysis: Elbow Method & Silhouette Score')
    fig.tight_layout()
    plt.show()

def plot_cluster_analysis(results_df, centroids_df, label_map, sensor_name):
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
    # Results DF has 'Cluster', 'ActivityLabel' and feature cols.
    # We need to drop metadata to get features.
    feature_cols = centroids_df.columns # These correspond to X columns
    X_pca = results_df[feature_cols]
    
    coords = pca.fit_transform(X_pca)
    
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=results_df['Cluster'], cmap='viridis', alpha=0.6)
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
    # Use index
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
    plt.show()

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
    plt.show()
