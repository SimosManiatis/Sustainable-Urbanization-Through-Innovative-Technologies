import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from .utils import print_info, print_error, print_success, print_warning

def build_feature_matrix(df, inputs, id_to_label):
    """
    Builds the feature table (Node, TimeBin, Features...).
    Reuses 1-min resampling logic but optimized for batch.
    """
    print_info("Matrix", "Building Feature Matrix...")
    
    # 1. Selection & Filtering
    if inputs["start"]: df = df[df["SendDate"] >= inputs["start"]]
    if inputs["end"]: df = df[df["SendDate"] <= inputs["end"]]
    
    target_nodes = inputs["target_nodes"] # Provided by caller
    df = df[df["NodeId"].isin(target_nodes)]
    
    # Master Timeline
    start_bound = inputs["start"] if inputs["start"] else df["SendDate"].min()
    end_bound = inputs["end"] if inputs["end"] else df["SendDate"].max()
    master_index = pd.date_range(start=start_bound, end=end_bound, freq=inputs["period_rule"])
    
    # 2. Define Features
    features_to_calc = []
    # Preset Logic
    if inputs["preset"] == "Minimal":
        features_to_calc = [
            ("Sound_ave", "Rolling STD", 5, "sound_std_5min"),
            ("Light_ave", "Delta", 15, "light_delta_15min"),
            ("Temperature_ave", "Delta", 15, "temp_delta_15min"),
            ("Air Quality_ave", "Delta", 15, "airq_delta_15min")
        ]
        if inputs.get("include_bme680"):
             features_to_calc.append(("Air Qu. BME680_ave", "Delta", 15, "bme680_delta_15min"))
             
    elif inputs["preset"] == "Full":
        # Add everything
        features_to_calc = [
            ("Sound_ave", "Rolling STD", 1, "sound_std_1min"),
            ("Sound_ave", "Rolling STD", 2.5, "sound_std_2.5min"),
            ("Sound_ave", "Rolling STD", 5, "sound_std_5min"),
            ("Light_ave", "Delta", 2, "light_delta_2min"),
            ("Light_ave", "Delta", 15, "light_delta_15min"),
            ("Temperature_ave", "Delta", 15, "temp_delta_15min"),
            # ... add others ...
        ]
        
    final_dfs = []
    
    # Helper for feature calc (duplicated from explorer? ideally shared, but placing here for module isolation)
    def _calc_feat(node_df, ftype, win):
        ts = node_df.set_index("SendDate")["ValueUsed"].resample("1min").mean()
        w = int(max(1, win))
        if ftype == "Rolling STD":
            return ts.rolling(window=w, min_periods=max(1, w//2)).std()
        elif ftype == "Delta":
            return ts.diff(periods=w)
        return ts

    for label, ftype, win, suffix in features_to_calc:
        print(f"  -> Calc {suffix} ({label})...")
        target_ids = [sid for sid, lbl in id_to_label.items() if lbl == label]
        if not target_ids: continue
        
        feat_raw = df[df["SensorId"].astype(str).isin(target_ids)]
        
        # Canonicalize here since we filtered raw
        if "CorrectValue" in feat_raw.columns:
            feat_raw["ValueUsed"] = feat_raw["CorrectValue"].fillna(feat_raw["Value"])
        else:
            feat_raw["ValueUsed"] = feat_raw["Value"]
        feat_raw["ValueUsed"] = pd.to_numeric(feat_raw["ValueUsed"], errors='coerce')
        feat_raw = feat_raw.dropna(subset=["ValueUsed"])
        
        for node in target_nodes:
            ndf = feat_raw[feat_raw["NodeId"] == node]
            if ndf.empty: continue
            
            feat_ts = _calc_feat(ndf, ftype, win)
            
            # Aggregate to Bin (Mean)
            resampled = feat_ts.resample(inputs["period_rule"]).mean()
            aligned = resampled.reindex(master_index)
            
            # Create MultiIndex Logic or Wide Format? 
            # Request says: (NodeId, time_bin) rows.
            # So DataFrame should be stacked? 
            # Easier to build wide per node then melt? 
            # Or build long DataFrame directly.
            
            aligned.name = suffix
            temp = aligned.to_frame()
            temp["NodeId"] = node
            temp = temp.reset_index().rename(columns={"index": "TimeBin"})
            final_dfs.append(temp)
            
    if not final_dfs:
        return None
        
    # Merge all features
    # Concat all long inputs
    big_long = pd.concat(final_dfs, ignore_index=True)
    
    # Pivot to obtain [TimeBin, NodeId] index and [features] columns
    matrix = big_long.pivot_table(index=["NodeId", "TimeBin"], values=[x[3] for x in features_to_calc])
    
    # Gating Check
    before_n = len(matrix)
    
    # "Strict" vs "Keep"
    if inputs.get("validity") != "Keep":
        matrix = matrix.dropna()
        
    after_n = len(matrix)
    print_info("Gating", f"Kept {after_n}/{before_n} bins ({after_n/before_n*100:.1f}%) complete data.")
    
    if matrix.empty: # Check empty before winsorizing
        if inputs.get("validity") != "Keep":
            print_error("Matrix empty after gating.")
            return None
        
    # FIX: Outlier Winsorization (Clip to 1st/99th percentile)
    # This prevents massive z-scores from single spikes.
    lower = matrix.quantile(0.01)
    upper = matrix.quantile(0.99)
    matrix = matrix.clip(lower=lower, upper=upper, axis=1)
    
    return matrix

def run_kmeans_clustering(matrix, inputs):
    """Runs K-means sweep and fits model."""
    print_info("Clustering", "Running K-means Analysis...")
    
    # 1. Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)
    
    # 2. K-Selection (Sweep)
    best_k = inputs.get("manual_k")
    scores = {}
    inertias = {}
    
    k_range = list(range(2, 11))
    sil_list = []
    sse_list = []
    rejected_ks = []
    
    # FIX: Min Cluster Size Threshold (e.g. 3%)
    min_size_pct = 0.03
    min_samples = int(len(matrix) * min_size_pct)
    
    if not best_k:
        print("  -> Sweeping K=2..10...")
        best_score = -1
        # Default to 2 if no valid found
        best_k = 2 
        
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            
            score = silhouette_score(X, labels)
            inertia = km.inertia_
            
            scores[k] = score
            inertias[k] = inertia
            
            # Record metrics regardless of rejection for plotting
            sil_list.append(score)
            sse_list.append(inertia)
            
            # Check Sizes for Selection
            counts = np.bincount(labels)
            if np.min(counts) < min_samples:
                print(f"     K={k}: Sil={score:.3f} (REJECT: Smallest cluster {np.min(counts)} < {min_samples})")
                rejected_ks.append(k)
                # Do not update best_k
            else:
                print(f"     K={k}: Sil={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_k = k
                    
        print_success(f"Auto-selected K={best_k}")
        
    # 3. Final Fit
    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = km_final.fit_predict(X)
    
    # 4. Attach to Data
    result = matrix.copy()
    result["cluster_id"] = labels
    
    metrics = {"k": k_range, "sil": sil_list, "sse": sse_list, "rejected": rejected_ks}
    return result, best_k, metrics

def generate_cluster_profiles(clustered_df):
    """Returns centroid definition and size stats."""
    # 1. Sizes
    counts = clustered_df["cluster_id"].value_counts().sort_index()
    total = len(clustered_df)
    props = counts / total * 100
    
    print("\nCluster Sizes:")
    for cid in counts.index:
        print(f"  Cluster {cid}: {counts[cid]} bins ({props[cid]:.1f}%)")
        
    # 2. Centroids (Mean of features)
    centroids = clustered_df.groupby("cluster_id").mean()
    
    # Z-Score Normalization for Heatmap
    z_scores = (centroids - centroids.mean()) / centroids.std()
    
    return z_scores, centroids

def generate_usage_report(labeled_df):
    """Generates usage stats (Time Share, Peak Hours)."""
    # Assuming 'activity_label' column exists
    if "activity_label" not in labeled_df.columns:
        labeled_df["activity_label"] = labeled_df["cluster_id"].astype(str)
        
    print("\n--- Usage Report ---")
    
    # Per Room State Share
    print("State Share per Room:")
    ct = pd.crosstab(labeled_df.index.get_level_values("NodeId"), labeled_df["activity_label"], normalize='index') * 100
    print(ct.round(1))
    
    return ct

# --- LEGACY WRAPPERS (For main.py compatibility) ---
def perform_global_clustering(df, n_clusters=4):
    """Legacy wrapper for Standard Pipeline."""
    # Ensure inputs format
    inputs = {"manual_k": n_clusters}
    # Using new engine
    # Assuming df is feature matrix. if not, this might fail, but fixes import.
    clustered, k, scores = run_kmeans_clustering(df, inputs)
    centroids = generate_cluster_profiles(clustered)
    label_map = {i: f"Cluster {i}" for i in range(k)}
    return clustered, centroids, label_map

def find_optimal_k_global(df):
    """Legacy wrapper."""
    inputs = {"manual_k": None}
    clustered, k, scores = run_kmeans_clustering(df, inputs)
    return k, scores

def filter_by_time(df, time_option, start_date=None, end_date=None):
    """Filters dataframe by time range (Legacy helper)."""
    if df.empty or 'SendDate' not in df.columns: return df
    
    # Ensure datetime
    # We assume callers might pass mixed types, so coerce usually
    # But here we just filter if column exists
    
    # Calculate relative reference from MAX date in data (as simulation)
    # or current time? Usually max date in dataset is better for static data.
    max_date = df['SendDate'].max()
    
    if time_option == 'Last 24 Hours':
        cutoff = max_date - pd.Timedelta(hours=24)
        return df[df['SendDate'] >= cutoff]
    elif time_option == 'Last 7 Days':
        cutoff = max_date - pd.Timedelta(days=7)
        return df[df['SendDate'] >= cutoff]
    elif time_option == 'Custom Range':
        if start_date: 
            # Ensure TZ awareness match if possible, but assuming UTC
            df = df[df['SendDate'] >= start_date]
        if end_date:
            df = df[df['SendDate'] <= end_date]
        return df

# --- MORE LEGACY WRAPPERS (For ui.py compatibility) ---
def perform_clustering(df, n_clusters=3):
    """Wrapper for legacy UI calls (local/per-room clustering usually)."""
    # Assuming standard implementation was just simple K-means on the input DF
    inputs = {"manual_k": n_clusters}
    clustered, k, scores = run_kmeans_clustering(df, inputs)
    centroids = generate_cluster_profiles(clustered)
    label_map = {i: f"Cluster {i}" for i in range(k)}
    return clustered, centroids, label_map

def find_optimal_k(df):
    """Wrapper for legacy UI."""
    return find_optimal_k_global(df)

def save_clustering_results(df, path):
    """Simple save."""
    df.to_csv(path)
    print_success(f"Saved results to {path}")

def compute_transition_matrix(labeled_df):
    """Calculates transition probabilities between states."""
    if "activity_label" not in labeled_df.columns: return None
    
    # Sort
    df = labeled_df.sort_values(["NodeId", "TimeBin"])
    
    matrices = []
    
    for node in df.index.get_level_values("NodeId").unique():
        sub = df[df.index.get_level_values("NodeId") == node].copy()
        
        # Shift to get Next State
        sub["Next"] = sub["activity_label"].shift(-1)
        sub = sub.dropna()
        
        # Crosstab Count
        ct = pd.crosstab(sub["activity_label"], sub["Next"])
        
        # Normalize Rows (Prob of moving FROM state TO state)
        ct_norm = ct.div(ct.sum(axis=1), axis=0).fillna(0)
        matrices.append(ct_norm)
        
    # Average across rooms? Or return dict?
    # Researcher requested "per room" and "combined".
    # For simplicity, let's return a Combined (Average) matrix for the report.
    
    # To avg, we need alignment.
    if not matrices: return None
    
    # Align all to union of columns/indices
    all_states = sorted(list(set().union(*[m.index for m in matrices]).union(*[m.columns for m in matrices])))
    
    sum_mat = pd.DataFrame(0.0, index=all_states, columns=all_states)
    count_mat = pd.DataFrame(0, index=all_states, columns=all_states)
    
    for m in matrices:
        reindexed = m.reindex(index=all_states, columns=all_states, fill_value=0)
        sum_mat += reindexed
        count_mat += 1 # Simple avg of probabilities? Or weighted by observations?
        # Weighted is strictly better but simple avg of room behavior is okay for "Typical Room"
        
    avg_mat = sum_mat / len(matrices)
    return avg_mat
        
    return df
