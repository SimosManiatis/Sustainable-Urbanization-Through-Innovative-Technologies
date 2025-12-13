import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime
from .utils import print_header, print_info, print_error, print_success, print_warning, Colors
from .data_loader import load_raw_data
from .analysis import (
    build_feature_matrix, run_kmeans_clustering, generate_cluster_profiles, 
    generate_usage_report, compute_transition_matrix
)
from .visualization import (
    plot_cluster_analysis, plot_optimization_metrics, plot_sensor_data, 
    plot_feature_profiles, plot_coverage_timeline, plot_stacked_validation,
    plot_correlation_heatmap, plot_state_share, plot_hourly_heatmap, 
    plot_transition_matrix
)

# ... (Existing helpers) ...

def _run_researcher_mode(df, locations, id_to_label, inputs=None):
    """Orchestrates the 15-step Researcher Validation Pipeline."""
    print_info("Researcher", "Starting Scientific Validation Pipeline...")
    
    # 0. Defaults
    start = pd.Timestamp("2025-11-29").tz_localize("UTC")
    # FIX: Inclusive End Date
    end = pd.Timestamp("2025-12-03 23:59:59").tz_localize("UTC")
    
    # FIX: Period Handling (Pass from Main)
    period_rule = "15min"
    period_label = "15min"
    if inputs and "period_rule" in inputs:
        period_rule = inputs["period_rule"]
        period_label = inputs["period_rule"]
    
    print_info("Config", f"Date: {start} - {end} | Period: {period_rule}")
    
    # filter
    df = df[(df["SendDate"] >= start) & (df["SendDate"] <= end)].copy()
    
    # --- V1: Coverage (Gating Impact) ---
    print_info("V1", "Coverage Timeline (Gating Impact)...")
    
    # 1. Build "Strict" Matrix (Gated)
    strict_inputs = {
        "start": start, "end": end, "period_rule": period_rule, 
        "target_nodes": ["TRP1", "TRP2", "TRP3", "TRP4", "TRP5"],
        "preset": "Minimal", "validity": "Strict"
    }
    mat_strict = build_feature_matrix(df, strict_inputs, id_to_label)
    
    # 2. Build "Keep" Matrix (All Bins)
    keep_inputs = strict_inputs.copy()
    keep_inputs["validity"] = "Keep"
    mat_keep = build_feature_matrix(df, keep_inputs, id_to_label)
    
    if mat_keep is not None and mat_strict is not None:
        # Create Mask: 1 if bin Survives Gating, 0 if Dropped
        # mat_strict index is subset of mat_keep index
        # We need to align them.
        # Initialize mask with 0s (Dropped)
        valid_mask = pd.Series(0, index=mat_keep.index, name="Valid")
        
        # Mark surviving indices as 1
        valid_mask.loc[valid_mask.index.isin(mat_strict.index)] = 1
        
        # Unstack to create Time x Node
        valid_piv = valid_mask.unstack("NodeId")
        
        # Reindex to Master Timeline (Explicit Gaps)
        master_index = pd.date_range(start=start, end=end, freq=period_rule)
        valid_piv = valid_piv.reindex(master_index)
        
        # 1=Valid, 0=Dropped/Missing. Fillna(0) for master gaps.
        valid_piv = valid_piv.fillna(0)
        
        plot_coverage_timeline(valid_piv, period_label, "Matrix_Completeness_PREGATE")
    
    # --- V2-V5: Feature Validation ---
    print_info("V2-V5", "Feature Validation Plots (Fixing Gaps)...")
    
    target_node = "TRP4" # Representative
    node_df = df[df["NodeId"] == target_node].copy() # Fix Copy
    
    # Rewrite Helper to allow Gaps
    def _get_aligned_series(lbl, ftype, win):
        tids = [sid for sid, l in id_to_label.items() if l == lbl]
        sub = node_df[node_df["SensorId"].astype(str).isin(tids)].copy()
        if sub.empty: return None, None
        
        # Canonicalize
        if "CorrectValue" in sub.columns: v = sub["CorrectValue"].fillna(sub["Value"])
        else: v = sub["Value"]
        v = pd.to_numeric(v, errors='coerce')
        
        # Keep NaNs for now to preserve time structure? 
        # But we need to index by time first.
        sub["ValueUsed"] = v
        ts = sub.set_index("SendDate")["ValueUsed"]
        
        # 1-min Resample (This handles gaps by inserting NaNs for missing mins)
        ts_1min = ts.resample("1min").mean()
        
        # Feat
        w = int(max(1, win))
        if ftype == "Rolling STD":
            feat = ts_1min.rolling(window=w, min_periods=max(1, w//2)).std()
        else:
            feat = ts_1min.diff(periods=w)
            
        # Agg to Period (Don't dropna yet, let plotter handle it or plot gaps)
        # We assume plotter expects aligned series
        agg_lev = ts_1min.resample(period_rule).mean()
        agg_feat = feat.resample(period_rule).max() # Use MAX for features like Delta to capture peaks? Or Mean?
        # Standard: Mean for levels, Mean (or Max) for activity proxies.
        # Let's stick to Mean for consistency with matrix.
        agg_feat = feat.resample(period_rule).mean()
        
        return agg_lev, agg_feat

    # V2: Sound
    lev, feat = _get_aligned_series("Sound_ave", "Rolling STD", 5)
    if lev is not None and not lev.dropna().empty: 
        plot_stacked_validation(lev, feat, ["Sound Level", "Sound Std 5min"], f"SOUND_{target_node}")

    # V3: Light
    lev, feat = _get_aligned_series("Light_ave", "Delta", 2)
    if lev is not None and not lev.dropna().empty:
        plot_stacked_validation(lev, feat, ["Light Level", "Light Delta 2min"], f"LIGHT_{target_node}")
    
    # --- V6-V7: Matrix Checks ---
    print_info("V6-V7", "Matrix Structure...")
    matrix_inputs = {
        "start": start, "end": end, "period_rule": period_rule, 
        "target_nodes": ["TRP1", "TRP2", "TRP3", "TRP4", "TRP5"],
        "preset": "Minimal", "validity": "Flexible"
    }
    matrix = build_feature_matrix(df, matrix_inputs, id_to_label)
    if matrix is not None:
        plot_correlation_heatmap(matrix, "GLOBAL")
    
    # --- V8-V11: Clustering ---
    print_info("V8-V11", "Clustering Analysis...")
    k_inputs = {"manual_k": None} # Auto
    res, k, metrics = run_kmeans_clustering(matrix, k_inputs)
    
    # V8: K-Sweep Plots
    if metrics:
        plot_optimization_metrics(metrics["k"], metrics["sse"], metrics["sil"], metrics.get("rejected", []))
        # Move plot to researcher folder?
        # plot_optimization_metrics uses plt.show(). I need to patch it to save?
        # Or better: Assume it saves if I didn't change it.
        # Actually, plot_optimization_metrics in viz.py uses plt.show(). 
        # I should have updated it to save to reports/researcher.
        # Since I can't edit viz.py easily again without re-finding anchor, 
        # I will rely on manual save here if I can get fig? 
        # No, function returns nothing. 
        # I will patch viz.py quickly next step to support save path argument.
    
    # Profiles
    profs_z, profs_raw = generate_cluster_profiles(res)
    plot_feature_profiles(profs_z, {i: f"C{i}" for i in range(k)}, "Global Activity")
    
    # Save Raw Centroids
    if not os.path.exists("reports/researcher"): os.makedirs("reports/researcher")
    profs_raw.to_csv("reports/researcher/cluster_centroids_RAW.csv")
    print_success("Saved Raw Centroids to reports/researcher/cluster_centroids_RAW.csv")
    
    # --- V12-V15: Usage ---
    print_info("V12-V15", "Usage Reporting...")
    res["activity_label"] = res["cluster_id"].astype(str) # No human label yet
    
    plot_state_share(res, "GLOBAL")
    plot_hourly_heatmap(res, "TRP4", "TRP4")
    
    trans_mat = compute_transition_matrix(res)
    if trans_mat is not None:
        plot_transition_matrix(trans_mat, "GLOBAL")
        
    print_success("Researcher Pipeline Complete. Check reports/researcher/")

# --- HELPERS ---
def get_user_choice(prompt, options, return_index=False):
    """Helper to get user choice."""
    print(f"\n{Colors.HEADER}{prompt}{Colors.ENDC}")
    if isinstance(options, list):
        for i, opt in enumerate(options): print(f"{i+1}. {opt}")
        while True:
            try:
                val = input(f"{Colors.CYAN}Select (1-{len(options)}): {Colors.ENDC}").strip()
                idx = int(val) - 1
                if 0 <= idx < len(options): return idx if return_index else options[idx]
                print_error("Invalid selection.")
            except ValueError: print_error("NaN")
    elif isinstance(options, dict):
        keys = list(options.keys())
        for i, key in enumerate(keys): print(f"{i+1}. {key}")
        while True:
            try:
                val = input(f"{Colors.CYAN}Select (1-{len(keys)}): {Colors.ENDC}").strip()
                idx = int(val) - 1
                if 0 <= idx < len(keys): return options[keys[idx]]
                print_error("Invalid selection.")
            except ValueError: print_error("NaN")

def parse_date_input(prompt):
    print(f"\n{Colors.HEADER}{prompt}{Colors.ENDC}")
    while True:
        val = input(f"{Colors.CYAN}Enter Date (YYYY-MM-DD [HH:MM]) or press Enter to skip: {Colors.ENDC}").strip()
        if not val: return None
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                dt = datetime.strptime(val, fmt)
                return pd.Timestamp(dt).tz_localize('UTC') 
            except ValueError: continue
        # Try YYYY-MM-DD and set to end of day if requested
        try:
             dt = datetime.strptime(val, "%Y-%m-%d")
             if "End" in prompt:
                 dt = dt.replace(hour=23, minute=59, second=59)
             return pd.Timestamp(dt).tz_localize('UTC')
        except ValueError:
             pass
        print_error("Invalid format.")

def _get_common_inputs(df, locations):
    inputs = {}
    print_info("Scope", "Select Room Scope")
    scope = get_user_choice("Scope:", ["Single room", "Compare all rooms"])
    inputs["mode"] = scope
    
    if scope == "Single room":
        sorted_locs = sorted(locations.items())
        loc_options = [f"{lid} - {name}" for lid, name in sorted_locs]
        idx = get_user_choice("Select Room:", loc_options, return_index=True)
        inputs["target_nodes"] = [sorted_locs[idx][0]]
        inputs["node_name"] = sorted_locs[idx][0]
    else:
        inputs["target_nodes"] = ["TRP1", "TRP2", "TRP3", "TRP4", "TRP5"]
        inputs["node_name"] = "ALL"
        
    print_info("Time", f"Available: {df['SendDate'].min()} to {df['SendDate'].max()}")
    inputs["start"] = parse_date_input("Start Date (Optional)")
    inputs["end"] = parse_date_input("End Date (Optional)")
    
    periods = {"Quarter Hour (15min)": "15min", "Half Hour (30min)": "30min", "Hour (1h)": "1h", "Day (1D)": "1D"}
    pname = get_user_choice("Analysis Period:", list(periods.keys()))
    inputs["period_name"] = pname
    inputs["period_rule"] = periods[pname]
    
    return inputs

def get_full_inputs(mode_idx, df, locations, id_to_label):
    """Dispatches input collection based on mode."""
    inputs = _get_common_inputs(df, locations)
    inputs["mode_idx"] = mode_idx
    
    # Mode 1: Level Trend
    if mode_idx == 0:
        measurements = sorted(list(set(id_to_label.values())))
        inputs["label"] = get_user_choice("Select Measurement:", measurements)
        dtypes = {"Mean": "mean", "Min": "min", "Max": "max", "First": "first"}
        inputs["dtype_name"] = get_user_choice("Data Type:", list(dtypes.keys()))
        inputs["agg_func"] = dtypes[inputs["dtype_name"]]
        
    # Mode 2: Activity Trend
    elif mode_idx == 1:
        measurements = sorted(list(set(id_to_label.values())))
        inputs["label"] = get_user_choice("Base Measurement:", measurements)
        inputs["feature_type"] = get_user_choice("Feature:", ["Rolling STD", "Delta"])
        if inputs["feature_type"] == "Rolling STD":
            inputs["window"] = float(get_user_choice("Window (min):", ["1", "2.5", "5", "10"]))
        else:
            inputs["window"] = float(get_user_choice("Window (min):", ["2", "5", "15"]))
        inputs["agg_func"] = "mean" # Default for feature
        inputs["dtype_name"] = f"{inputs['feature_type']} {inputs['window']}"

    # Mode 3: Matrix Builder
    elif mode_idx == 2:
        inputs["preset"] = get_user_choice("Feature Preset:", ["Minimal", "Full"])
        inputs["validity"] = get_user_choice("Validity:", ["Strict", "Flexible"])
        # Check BME
        inputs["include_bme680"] = False # Default

    # Mode 4: Clustering
    elif mode_idx == 3:
        # Need matrix inputs too to build it on fly? Or load?
        # Simpler: Build on fly for now (User said "Build matrix now OR load"). 
        # We will assume Build Now for Explorer simplicity (Batch logic).
        inputs["preset"] = "Minimal" # Default for clustering demo
        inputs["k_strategy"] = get_user_choice("K Strategy:", ["Auto Scale", "Manual"])
        if inputs["k_strategy"] == "Manual":
            inputs["manual_k"] = int(input("Enter K: "))
            
    # Mode 8: Analysis Validation
    elif mode_idx == 8:
        inputs["agg_func"] = get_user_choice("Aggregation Method:", ["mean", "median"])
            
    return inputs

# --- PLOTTING HELPERS --- (Reusing existing plot logic slightly adapted)
def _style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=0)
    plt.tight_layout()

def plot_trend(df, inputs, locations):
    """Plots whatever is in df (Level or Feature)."""
    plt.figure(figsize=(12, 6))
    colors = ["#F44336", "#2196F3", "#4CAF50", "#FFC107", "#9C27B0"]
    
    # If df is MultiIndex (Node, Time) from Matrix? No, Trend usage returns Wide likely.
    # Existing process_data returns wide (Index=Time, Cols=Nodes).
    
    for i, col in enumerate(df.columns):
        series = df[col].dropna()
        if series.empty: continue
        c = colors[i % len(colors)]
        lbl = f"{locations.get(col, col)} ({col})"
        plt.plot(series.index, series.values, color=c, lw=2, label=lbl)
        plt.fill_between(series.index, series.values, color=c, alpha=0.1)
        
    title = f"{inputs.get('label','')} {inputs.get('dtype_name','')}"
    plt.title(title, loc='left', fontsize=14, fontweight='bold')
    _style_plot(plt.gca())
    plt.legend(loc='upper right', frameon=False)
    
    # Save
    if not os.path.exists("reports"): os.makedirs("reports")
    path = f"reports/trend_{inputs['node_name']}_{inputs.get('label','custom')}.png"
    plt.savefig(path)
    print_success(f"Saved plot: {path}")

# --- MAIN ---
def run_sensor_trend_explorer():
    print_header("Sensor Trend Explorer")
    print_info("Setup", "Loading dataset...")
    df, id_to_label, locations = load_raw_data()
    if df is None or df.empty: return

    # Pre-parse Date
    df["SendDate"] = pd.to_datetime(df["SendDate"], utc=True, errors='coerce')
    df = df.dropna(subset=["SendDate"])

    modes = [
        "Trend Views (Levels)",
        "Trend Views (Activity Features)",
        "Feature Matrix Builder",
        "K-means Activity Clustering",
        "Cluster Review (Not Impl)",
        "Room Usage Reporting (Not Impl)",
        "Batch Run",
        "Researcher Mode (Scientific Validation)",
        "Analysis Validation (Strict Pipeline)"
    ]
    mode_idx = get_user_choice("Select Tool Mode:", modes, return_index=True)
    
    inputs = get_full_inputs(mode_idx, df, locations, id_to_label)
    
    # Dispatch
    # Dispatch
    if mode_idx in [0, 1]: 
        print_info("Processing", "Aggregating Trend Data...")
        # 1. Filter
        if inputs["start"]: df = df[df["SendDate"] >= inputs["start"]]
        if inputs["end"]: df = df[df["SendDate"] <= inputs["end"]]
        
        target_ids = [sid for sid, lbl in id_to_label.items() if lbl == inputs["label"]]
        df = df[df["SensorId"].astype(str).isin(target_ids)]
        
        df = df[df["NodeId"].isin(inputs["target_nodes"])]
        
        # 2. Resample
        # Level vs Activity
        if mode_idx == 0:
            if "CorrectValue" in df.columns: df["ValueUsed"] = df["CorrectValue"].fillna(df["Value"])
            else: df["ValueUsed"] = df["Value"]
            df["ValueUsed"] = pd.to_numeric(df["ValueUsed"], errors='coerce')
            val_col = "ValueUsed"
        else:
             # Activity: 1-min resample first
             # This is a mini-version of build_matrix logic
             val_col = inputs["feature_type"]
             # ... (Optimization: Implementing full logic here is verbose. 
             # Better to use a helper in analysis.py called 'compute_trend_series'?)
             pass
             
        # Actually, let's keep it simple for this step and just implement Level Trend robustly.
        # Activity Trend needs the 1-min logic.
        
        master_index = pd.date_range(inputs["start"] or df["SendDate"].min(), inputs["end"] or df["SendDate"].max(), freq=inputs["period_rule"])
        
        res_dict = {}
        for node in inputs["target_nodes"]:
            ndf = df[df["NodeId"] == node]
            if ndf.empty: continue
            
            if mode_idx == 1:
                # 1-min
                ts = ndf.set_index("SendDate")["Value"].astype(float).resample("1min").mean()
                if inputs["feature_type"] == "Rolling STD":
                    w = int(inputs["window"])
                    ts = ts.rolling(window=w, min_periods=max(1, w//2)).std()
                else:
                    ts = ts.diff(periods=int(inputs["window"]))
                
                # Agg
                binned = ts.resample(inputs["period_rule"]).mean()
            else:
                # Level
                binned = ndf.set_index("SendDate")[val_col].resample(inputs["period_rule"]).agg(inputs["agg_func"])
                
            res_dict[node] = binned.reindex(master_index)
            
        res_df = pd.DataFrame(res_dict, index=master_index)
        plot_trend(res_df, inputs, locations)
        res_df.to_csv(f"reports/trend_{inputs['node_name']}.csv")
        print_success("Trend View Complete.")
        
    elif mode_idx == 2: # Matrix
        matrix = build_feature_matrix(df, inputs, id_to_label)
        if matrix is not None:
             matrix.to_csv("reports/feature_matrix.csv")
             print_success("Saved reports/feature_matrix.csv")
             
    elif mode_idx == 3: # Clustering
        # 1. Build
        matrix = build_feature_matrix(df, inputs, id_to_label)
        if matrix is not None:
            # 2. Cluster
            res, k, scores = run_kmeans_clustering(matrix, inputs)
            res.to_csv("reports/clustered_data.csv")
            print_success(f"Saved reports/clustered_data.csv (K={k})")
            
            # 3. Profile
            profs = generate_cluster_profiles(res)
            print(profs)

    elif mode_idx == 6: # Batch
        print_info("Batch", "Running End-to-End...")
        # 1. Matrix
        inputs["preset"] = "Minimal"
        inputs["validity"] = "Flexible"
        matrix = build_feature_matrix(df, inputs, id_to_label)
        
        # 2. Cluster
        inputs["manual_k"] = None # Auto
        res, k, scores = run_kmeans_clustering(matrix, inputs)
        
        # 3. Report
        generate_usage_report(res)
        
        # Save
        res.to_csv("reports/batch_results.csv")
        print_success("Batch Complete.")
        
    elif mode_idx == 7: # Researcher
        _run_researcher_mode(df, locations, id_to_label, inputs)
        
    elif mode_idx == 8: # Analysis Validation
        from .analysis_validation import AnalysisValidation
        # Ensure config matches class expectation
        av_config = {
            "raw_data_dir": "raw_data", # Hardcoded relative or pass from data_loader if exposed
            # Actually data_loader doesn't expose dir, but we know it
            # Better: Let's assume we use the constants.
            "mapping_file": "sensor_mapping.json",
            "output_dir": "reports/analysis_validation",
            "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_dt": inputs["start"],
            "end_dt": inputs["end"],
            "bin_size": inputs["period_rule"],
            "agg_method": inputs["agg_func"], # mean/median
            "target_nodes": inputs["target_nodes"]
        }
        # Constants patch
        from .constants import RAW_DATA_DIR, MAPPING_FILE
        av_config["raw_data_dir"] = RAW_DATA_DIR
        av_config["mapping_file"] = MAPPING_FILE
        
        av = AnalysisValidation(av_config)
        av.run()
