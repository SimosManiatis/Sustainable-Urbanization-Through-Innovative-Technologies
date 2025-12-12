import pandas as pd
import numpy as np
from datetime import timedelta
from .utils import print_header, print_info, print_warning, print_success, Colors

# Try importing sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def get_feature_columns(df):
    """Returns columns containing 'norm_' (Wide Format: Measurement_norm_Feature)."""
    return [c for c in df.columns if 'norm_' in c]


def find_optimal_k_global(df, max_k=10):
    """
    Runs K-Means sweep on the global dataset (wide format).
    Returns metrics to help decide K.
    """
    if not SKLEARN_AVAILABLE:
        print_warning("Scikit-learn required for clustering.")
        return None, None, None

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        print_warning("No normalized features (norm_*) found for clustering.")
        return None, None, None

    X = df[feature_cols].dropna()
    if len(X) < 50:
        print_warning("Not enough data points for clustering.")
        return None, None, None

    print_info("Optimization", f"Sweeping K=2..{max_k} on {len(feature_cols)} features...")
    
    inertia_values = []
    silhouette_values = []
    k_range = range(2, min(max_k + 1, len(X)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        # Silhouette can be slow on large data, sample if needed
        sil = silhouette_score(X, labels, sample_size=5000) 
        
        inertia_values.append(inertia)
        silhouette_values.append(sil)
        print(f"  -> k={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}")
        
    return k_range, inertia_values, silhouette_values


def perform_global_clustering(df, n_clusters=4):
    """
    Performs global K-Means clustering on the wide DataFrame.
    """
    if not SKLEARN_AVAILABLE:
        return df, None, {}

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        return df, None, {}

    # We need to cluster on valid rows only, but return a DF with Cluster ID matching original indices
    # Drop NAs for fitting
    data_to_fit = df.dropna(subset=feature_cols)
    X = data_to_fit[feature_cols]

    print_info("Clustering", f"Fitting KMeans (K={n_clusters}) on {len(X)} samples...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Assign back to a copy of the original DF
    result_df = df.copy()
    result_df['Cluster'] = np.nan
    # Align indices
    result_df.loc[data_to_fit.index, 'Cluster'] = clusters
    
    # Fill NaN clusters? No, keep as NaN (these were dropped rows)
    # Cast to Int (nullable)
    result_df['Cluster'] = result_df['Cluster'].astype("Int64")

    # Centroids
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)

    # Activity Labeling
    # Logic: Calculate "Magnitude" of each cluster centroid (mean absolute Z-score)
    # Higher magnitude usually means more deviation from mean (active) if norm is robust.
    # Alternatively: Check magnitude of 'std' or 'delta' features specifically if they exist.
    # Let's use mean of all norm features for now, assuming they represent "activity/variance".
    
    # Better heuristic: 
    # Usually "Inactive" = Low variance (low std), Low change (low delta).
    # Since we normalized, Inactive might be negative Z-score (below average variability)? 
    # Or near 0 if average is inactive?
    # Actually, Z-score 0 is the Mean.
    # If the dataset is mostly inactive, Mean might be "Low Activity".
    # High Z-score = High Activity.
    # Low Z-score (negative) = Very stable?
    
    # Let's sum the centroids' values (assuming features are "amount of variation").
    # If features are just "Temperature Mean", then high != active.
    # But user said "Activity-state inference". 
    # Features: "ValueUsed_mean" (raw value?), "std_5min", "delta_15min".
    # We normalized ALL of them.
    # We should focus on *variability* features for activity labeling.
    var_cols = [c for c in feature_cols if 'std' in c or 'delta' in c]
    if not var_cols: var_cols = feature_cols # Fallback
    
    # Calculate magnitude based on variability features
    centroid_activity_score = centroids[var_cols].mean(axis=1)
    
    # Sort clusters by score
    sorted_map = centroid_activity_score.sort_values().index.tolist()
    
    label_map = {}
    labels = ["Inactive", "Low Activity", "Medium Activity", "High Activity", "Very High Activity"]
    
    # Distribute labels
    # If K=4, map to Inactive, Low, Medium, High
    # If K=3, Inactive, Medium, High
    
    step = len(labels) / n_clusters
    for i, cluster_id in enumerate(sorted_map):
        # Pick label
        label_idx = int(i * step)
        label_name = labels[min(label_idx, len(labels)-1)]
        label_map[cluster_id] = label_name

    result_df['ActivityLabel'] = result_df['Cluster'].map(label_map)
    
    print_success(f"Clustering complete. Mapped {n_clusters} clusters to activity labels.")
    
    return result_df, centroids, label_map

# Try importing sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def filter_by_time(df, time_option, start_date=None, end_date=None):
    """Filters dataframe based on time option."""
    if time_option == 'all':
        return df
    
    if not pd.api.types.is_datetime64_any_dtype(df['SendDate']):
        df['SendDate'] = pd.to_datetime(df['SendDate'], errors='coerce')
        
    max_date = df['SendDate'].max()
    if pd.isna(max_date):
        return df
        
    calc_start = None
    
    if time_option == '24h':
        calc_start = max_date - timedelta(hours=24)
    elif time_option == '7d':
        calc_start = max_date - timedelta(days=7)
    elif time_option == 'custom' and start_date:
        # Use provided custom range
        # Fix: Ensure timezone compatibility
        col_tz = df['SendDate'].dt.tz
        if col_tz is not None:
             if start_date and start_date.tzinfo is None:
                 start_date = pd.Timestamp(start_date).tz_localize(col_tz)
             if end_date and end_date.tzinfo is None:
                 end_date = pd.Timestamp(end_date).tz_localize(col_tz)
                 
        mask = (df['SendDate'] >= start_date)
        if end_date:
            mask &= (df['SendDate'] <= end_date)
        return df[mask]

    if calc_start:
        return df[df['SendDate'] >= calc_start]
        
    return df

def prepare_clustering_data(df, sensor_name, sensor_name_col, time_option, start_date=None, end_date=None):
    """Common data preparation for clustering tasks."""
    mask = (df[sensor_name_col] == sensor_name)
    subset = df[mask].copy()
    
    if subset.empty:
        print_warning("No data found for this sensor.")
        return None, None

    subset = filter_by_time(subset, time_option, start_date, end_date)
    if subset.empty:
        print_warning("No data in selected time range.")
        return None, None

    # Pivot/Resample
    subset['SendDate'] = pd.to_datetime(subset['SendDate'])
    subset['Value'] = pd.to_numeric(subset['Value'], errors='coerce')
    
    # We cluster on FEATURES calculated during preprocessing (std, delta, norm_*)
    # But those are columns in the aggregated df.
    # We should use the aggregated data directly if df is already aggregated.
    # The 'df' passed here is usually the 15-min aggregated one from main.
    
    # Check if we have 'norm_' columns
    norm_cols = [c for c in df.columns if c.startswith('norm_')]
    if not norm_cols:
         print_warning("Normalized features not found. Running simple Value clustering.")
         features_to_use = ['Value']
    else:
         features_to_use = norm_cols
         
    # Filter columns
    data_for_clustering = subset.set_index('SendDate')[features_to_use].dropna()
    
    if len(data_for_clustering) < 50:
         print_warning(f"Not enough data points ({len(data_for_clustering)}) for reliable analysis.")
         return None, None
         
    return data_for_clustering, subset


def find_optimal_k(df, sensor_name, sensor_name_col, time_option, start_date=None, end_date=None, max_k=10):
    """Calculates Inertia and Silhouette scores for k=2..max_k."""
    if not SKLEARN_AVAILABLE:
        print_warning("Scikit-learn required.")
        return None, None

    print_info("Status", f"Optimizing 'K' for {sensor_name}...")
    
    X, _ = prepare_clustering_data(df, sensor_name, sensor_name_col, time_option, start_date, end_date)
    if X is None: return None, None
    
    inertia_values = []
    silhouette_values = []
    k_range = range(2, min(max_k + 1, len(X)))
    
    if len(k_range) == 0:
        print_warning("Data too small to cluster.")
        return None, None
        
    print(f"  {Colors.CYAN}Training K-Means models...{Colors.ENDC}")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels)
        
        inertia_values.append(inertia)
        silhouette_values.append(sil)
        print(f"  -> k={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}")
        
    return k_range, inertia_values, silhouette_values


def perform_clustering(df, sensor_name, sensor_name_col, id_to_label, time_option, n_clusters=4, location_name="", start_date=None, end_date=None):
    """Performs K-Means clustering and returns results."""
    if not SKLEARN_AVAILABLE:
        print_warning("Scikit-learn is required.")
        return None, None, None

    print_info("Status", f"Clustering '{sensor_name}' with K={n_clusters}...")
    
    X, original_subset = prepare_clustering_data(df, sensor_name, sensor_name_col, time_option, start_date, end_date)
    if X is None: return None, None, None

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Attach clusters to data
    results = X.copy()
    results['Cluster'] = clusters
    
    # We also want the time series back. Index is SendDate.
    # original_subset has other columns we might want?
    
    print_success(f"Clustered {len(results)} points.")
    
    # Centroids
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_) if 'scaler' in locals() else kmeans.cluster_centers_, columns=X.columns)
    # Actually X was already normalized features. Centroids are in Z-score space.
    # That is good for heatmap.
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    
    # Auto-Labeling logic
    # Find feature with highest variance between clusters?
    # Or just label by average magnitude (0=Low, 1=Med... if sorted)
    # Let's sort clusters by their overall magnitude (L2 norm of centroid)
    cluster_magnitude = centroids.abs().mean(axis=1)
    # Create mapping: Old Cluster ID -> New Label
    sorted_indices = cluster_magnitude.sort_values().index
    
    label_map = {}
    activity_levels = ["Very Low", "Low", "Medium", "High", "Very High"]
    
    for rank, old_id in enumerate(sorted_indices):
        name = activity_levels[rank] if rank < len(activity_levels) else f"Level {rank+1}"
        label_map[old_id] = f"{name} Activity"
        
    results['ActivityLabel'] = results['Cluster'].map(label_map)
    
    # Generate Textual Report
    generate_insights_report(results, label_map, sensor_name)
    
    return results, centroids, label_map

def generate_insights_report(df, label_map, sensor_name):
    """Prints a textual summary of activity states."""
    print_header(f"Usage Insights Report: {sensor_name}")
    
    total_time = len(df) * 15 # minutes
    total_hours = total_time / 60
    
    print(f"Total Analyzed Time: {total_hours:.1f} hours ({len(df)} intervals)")
    print("-" * 40)
    
    # % Time in each state
    cluster_counts = df['Cluster'].value_counts()
    
    for cluster_id in sorted(label_map.keys()):
        label = label_map[cluster_id]
        count = cluster_counts.get(cluster_id, 0)
        pct = (count / len(df)) * 100
        
        # Peak Hour
        cluster_subset = df[df['Cluster'] == cluster_id]
        if not cluster_subset.empty:
            # Assumes 'SendDate' is index
            hours = cluster_subset.index.hour
            peak_hour = hours.value_counts().idxmax()
            peak_str = f"{peak_hour:02d}:00"
        else:
            peak_str = "N/A"
            
        print(f"• {Colors.CYAN}{label:<15}{Colors.ENDC}: {pct:5.1f}% time | Peak Activity: {peak_str}")
        
    print("-" * 40)
    
    # Simple Insight
    high_activity_clusters = [k for k,v in label_map.items() if "High" in v]
    high_time = df[df['Cluster'].isin(high_activity_clusters)]
    high_pct = (len(high_time) / len(df)) * 100
    
    print(f"{Colors.GREEN}Insight:{Colors.ENDC} This space is in a highly active state {high_pct:.1f}% of the time.")
    
    # Actionable Recommendations
    print(f"\n{Colors.HEADER}--- Actionable Recommendations ---{Colors.ENDC}")
    
    # 1. Underutilization
    low_activity_clusters = [k for k,v in label_map.items() if "Very Low" in v or "Low" in v]
    low_time_pct = 0
    if low_activity_clusters:
         low_time = df[df['Cluster'].isin(low_activity_clusters)]
         low_time_pct = (len(low_time) / len(df)) * 100
         
    if low_time_pct > 60:
        print(f"• {Colors.WARNING}High Inactivity ({low_time_pct:.1f}%):{Colors.ENDC} Consider optimizing HVAC schedules to save energy during these long idle periods.")
        print(f"• Space Consolidation: If this pattern persists across work hours, consider repurposing this space.")
        
    # 2. High Density
    if high_pct > 30:
         print(f"• {Colors.WARNING}High Traffic ({high_pct:.1f}%):{Colors.ENDC} Ensure adequate ventilation rates (ACH) during peak hours.")
         
    if low_time_pct < 60 and high_pct < 10:
         print(f"• Balanced Usage: Space seems well-utilized without extremes.")
         
    print("-" * 40)


def save_clustering_results(df, label_map, sensor_name):
    """Saves the clustered dataframe to CSV."""
    filename = f"clustered_{sensor_name}_{datetime.now().strftime('%Y%m%d')}.csv"
    try:
        # Save essential columns
        cols_to_save = ['SendDate', 'Value', 'ActivityLabel', 'Cluster'] + [c for c in df.columns if 'norm_' in c]
        output = df[cols_to_save].copy()
        output.to_csv(filename, index=True)
        print_success(f"Results saved to: {filename}")
    except Exception as e:
        print_warning(f"Could not save CSV: {e}")
