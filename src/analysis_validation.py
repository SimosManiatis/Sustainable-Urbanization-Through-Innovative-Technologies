import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timezone, timedelta
import glob

# Try imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Ensure style
plt.style.use('ggplot')

class AnalysisValidation:
    def __init__(self, config):
        self.cfg = config
        self.run_dir = os.path.join(config["output_dir"], config["run_id"])
        
        self.manifest = []
        self.df_clean = None
        self.master_index = None # Index of ALL bins in effective range
        
        # Warmup Rule (Explicit for Reference, but dynamically detected via NaNs)
        self.warmup_duration_min = 15 
        
        self.completeness_df = None
        self.validation_frame = None # Holds features aligned to master
        self.gated_df = None
        
        self.feature_matrix = None
        self.feature_matrix_scaled = None
        
        self.signals_map = {
            "Sound": "Sound_ave",
            "Light": "Light_ave"
        }
        
        self.meta = {}

    def log_manifest(self, filename, desc):
        type_ = os.path.splitext(filename)[1]
        fp = os.path.join(self.run_dir, filename)
        sz = os.path.getsize(fp) if os.path.exists(fp) else 0
        from datetime import datetime
        ts = datetime.now().isoformat()
        
        self.manifest.append({
            "filename": filename, 
            "type": type_, 
            "size_bytes": sz, 
            "created": ts,
            "description": desc
        })

    def run(self):
        print("\n--- Starting Analysis Validation Pipeline (Round 2) ---")
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
            
        # Reordered Pipeline
        self.step1_load_data()
        self.step2_build_timeline()
        self.step3_compute_completeness()
        self.step5_build_indicators() # MOVED UP: Features define Eligibility
        self.step4_apply_gating()     # Gating uses Completeness + Features
        self.step6_assemble_features()
        self.step7_clustering()
        self.step8_outputs()
        self.save_metadata()
        
        print("\n--- Pipeline Complete ---")
        print(f"Results in: {self.run_dir}")
        self.print_summary()

    # --- STEP 1: LOAD & STANDARDIZE ---
    def step1_load_data(self):
        print("Step 1: Loading & Standardising...")
        
        # Mapping
        with open(self.cfg["mapping_file"], 'r') as f:
            mdata = json.load(f)
        self.id_to_label = {str(v): k for k, v in mdata.get("measurements", {}).items()}
        
        # Load CSV
        csv_files = glob.glob(os.path.join(self.cfg["raw_data_dir"], "*.csv"))
        if not csv_files: raise FileNotFoundError("No CSV files found.")
        latest_csv = max(csv_files, key=os.path.getctime)
        self.meta["source_file"] = os.path.basename(latest_csv)
        
        df = pd.read_csv(latest_csv, low_memory=False)
        self.meta["raw_row_count"] = len(df)
        
        # Timezone
        df["SendDate"] = pd.to_datetime(df["SendDate"], utc=True, errors='coerce')
        df = df.dropna(subset=["SendDate"])
        
        # Value Used
        if "CorrectValue" in df.columns:
            df["value_used"] = df["CorrectValue"].fillna(df["Value"])
        else:
            df["value_used"] = df["Value"]
        df["value_used"] = pd.to_numeric(df["value_used"], errors='coerce')
        df = df.dropna(subset=["value_used"])
        
        # Date Clamping (Requested vs Effective)
        req_s = self.cfg.get("start_dt")
        req_e = self.cfg.get("end_dt")
        data_s, data_e = df["SendDate"].min(), df["SendDate"].max()
        
        self.cfg["requested_start"] = str(req_s) if req_s else "None"
        self.cfg["requested_end"] = str(req_e) if req_e else "None"
        
        # Clamp
        eff_s = max(req_s, data_s) if req_s else data_s
        eff_e = min(req_e, data_e) if req_e else data_e
        
        # If requested was outside data, use data limit? No, usually intersection.
        # Logic: Effective is intersection of Request and Availability.
        
        df = df[(df["SendDate"] >= eff_s) & (df["SendDate"] <= eff_e)]
        
        # Signal Filter
        targets = list(self.signals_map.values())
        df["measurement_label"] = df["SensorId"].astype(str).map(self.id_to_label)
        df = df[df["measurement_label"].isin(targets)]
        
        # Scope
        if self.cfg["target_nodes"]:
             df = df[df["NodeId"].isin(self.cfg["target_nodes"])]
             
        # Deduplicate
        df = df.sort_values(["NodeId", "measurement_label", "SendDate"])
        df = df.drop_duplicates(subset=["NodeId", "measurement_label", "SendDate"], keep='last')
        
        self.df_clean = df.reset_index(drop=True)
        self.meta["cleaned_row_count"] = len(self.df_clean)
        self.cfg["effective_start_clamped"] = str(eff_s)
        self.cfg["effective_end_clamped"] = str(eff_e)

    # --- STEP 2: TIMELINE ---
    def step2_build_timeline(self):
        # Use Clamped Dates to build Master Index
        s = pd.Timestamp(self.cfg["effective_start_clamped"]).floor(self.cfg["bin_size"])
        e = pd.Timestamp(self.cfg["effective_end_clamped"]).ceil(self.cfg["bin_size"])
        
        self.cfg["timeline_start"] = str(s)
        self.cfg["timeline_end"] = str(e)
        
        self.master_index = pd.date_range(start=s, end=e, freq=self.cfg["bin_size"])
        self.meta["total_bins_timeline"] = len(self.master_index)
        print(f"Step 2: Timeline {s} to {e} ({len(self.master_index)} bins)")

    # --- STEP 3: COMPLETENESS ---
    def step3_compute_completeness(self):
        # Calculates Ratio (0.0/0.5/1.0) based on RAW presence
        # Does NOT yet know about Feature Windowing NaNs
        print("Step 3: Raw Completeness...")
        
        df = self.df_clean.copy()
        df["TimeBin"] = df["SendDate"].dt.floor(self.cfg["bin_size"])
        
        grouped = df.groupby(["NodeId", "TimeBin", "measurement_label"]).size().unstack(fill_value=0)
        presence = (grouped > 0).astype(int)
        
        for sig in self.signals_map.values():
            if sig not in presence.columns: presence[sig] = 0
            
        sig_cols = list(self.signals_map.values())
        presence["ratio"] = presence[sig_cols].sum(axis=1) / len(sig_cols)
        
        # Reindex
        nodes = self.cfg["target_nodes"]
        from itertools import product
        full_idx = pd.MultiIndex.from_product([nodes, self.master_index], names=["NodeId", "TimeBin"])
        
        self.completeness_df = presence.reindex(full_idx, fill_value=0)
        # We will add refined flags in Step 4

    # --- STEP 5: BUILD INDICATORS (Moved Up) ---
    def step5_build_indicators(self):
        print("Step 5: Building Indicators (Pre-Gating)...")
        
        frames = []
        self.validation_plot_data = {}
        
        def _compute(node_df, sig_label, logic_type, win):
            # Resample 1min
            sub = node_df[node_df["measurement_label"] == sig_label].copy()
            if sub.empty: return None
            ts = sub.set_index("SendDate")["value_used"].resample("1min").mean()
            
            # Feature
            w = int(max(1, win))
            if logic_type == "STD": feat = ts.rolling(window=w, min_periods=1).std()
            elif logic_type == "Delta": feat = ts.diff(periods=w)
            
            # Agg
            agg = self.cfg["agg_method"]
            binned = feat.resample(self.cfg["bin_size"]).agg(agg)
            lev = ts.resample(self.cfg["bin_size"]).agg(agg)
            return binned, lev
            
        for node in self.cfg["target_nodes"]:
            ndf = self.df_clean.loc[self.df_clean["NodeId"] == node].copy()
            
            s_res = _compute(ndf, self.signals_map["Sound"], "STD", 5)
            l_res = _compute(ndf, self.signals_map["Light"], "Delta", 15)
            
            if s_res and l_res:
                s_feat, s_lev = s_res
                l_feat, l_lev = l_res
                
                self.validation_plot_data[node] = {
                    "SoundLevel": s_lev, "SoundVar": s_feat,
                    "LightLevel": l_lev, "LightChg": l_feat
                }
                
                # Align to Master
                tmp = pd.DataFrame({
                    "SoundVar": s_feat.reindex(self.master_index),
                    "LightChg": l_feat.reindex(self.master_index)
                }, index=self.master_index)
                tmp.index.name = "TimeBin"
                tmp["NodeId"] = node
                frames.append(tmp.reset_index())
                
        if frames:
            self.validation_frame = pd.concat(frames).set_index(["NodeId", "TimeBin"])
        else:
            self.validation_frame = pd.DataFrame()

    # --- STEP 4: GATING (Refined) ---
    def step4_apply_gating(self):
        print("Step 4: Gating (Eligible vs Retained)...")
        
        # Merge Completeness + Features
        # Both are MultiIndex (Node, TimeBin) aligned to Master
        
        # Prepare Master Frame
        self.full_status = self.completeness_df[["ratio"]].copy()
        
        # Join Features
        if not self.validation_frame.empty:
            self.full_status = self.full_status.join(self.validation_frame, how='left')
        else:
            self.full_status["SoundVar"] = np.nan
            self.full_status["LightChg"] = np.nan
            
        # Definition: Eligible if Features are VALID (Not NaN)
        # This implicitly handles Windowing NaNs
        self.full_status["has_valid_features"] = (
            self.full_status["SoundVar"].notna() & self.full_status["LightChg"].notna()
        )
        
        self.full_status["is_eligible"] = self.full_status["has_valid_features"]
        
        # Definition: Retained if Eligible AND Ratio=1.0 (Raw complete)
        self.full_status["is_retained"] = (
            self.full_status["is_eligible"] & (self.full_status["ratio"] == 1.0)
        )
        
        # Stats
        self.stats = []
        for node in self.cfg["target_nodes"]:
            sub = self.full_status.loc[node]
            total = len(sub)
            eligible = sub["is_eligible"].sum()
            retained = sub["is_retained"].sum()
            
            self.stats.append({
                "Room": node,
                "Total": total,
                "Eligible": eligible,
                "Retained": retained,
                "Warmup_Excluded": total - eligible
            })
            
        self.gated_df = self.full_status[self.full_status["is_retained"]].copy()

    # --- STEP 6: ASSEMBLE ---
    def step6_assemble_features(self):
        print("Step 6: Assembling Matrix...")
        if self.gated_df.empty: 
            self.feature_matrix = pd.DataFrame()
            return
            
        # Just pick columns from gated_df
        self.feature_matrix = self.gated_df[["SoundVar", "LightChg"]].copy()
        
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            self.feature_matrix_scaled = pd.DataFrame(
                scaler.fit_transform(self.feature_matrix),
                index=self.feature_matrix.index,
                columns=self.feature_matrix.columns
            )

    # --- STEP 7: CLUSTERING ---
    def step7_clustering(self):
        print("Step 7: Clustering...")
        if self.feature_matrix is None or self.feature_matrix.empty: return
        
        X = self.feature_matrix_scaled.values
        min_size = int(len(X) * 0.03)
        self.meta["min_cluster_size"] = min_size
        
        sweep_res = []
        best_k = 2; best_sil = -1
        
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X)
            sil = silhouette_score(X, lbls)
            
            counts = np.bincount(lbls)
            rej = False; reason = None
            if np.min(counts) < min_size:
                rej = True; reason = f"Min Cluster < {min_size}"
            else:
                if sil > best_sil: best_k = k; best_sil = sil
                
            sweep_res.append({
                "k": k, "silhouette": sil, "inertia": km.inertia_,
                "min_size": int(np.min(counts)), "rejected": rej, "reason": reason
            })
            
        self.sweep_df = pd.DataFrame(sweep_res)
        self.selected_k = best_k
        self.sweep_df.to_csv(os.path.join(self.run_dir, "k_sweep.csv"), index=False)
        self.log_manifest("k_sweep.csv", "Clustering metrics")
        
        km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        self.feature_matrix["StateID"] = km.fit_predict(X)
        self.feature_matrix_scaled["StateID"] = self.feature_matrix["StateID"]
        print(f"  Selected K={best_k}")

    # --- STEP 8: OUTPUTS ---
    def step8_outputs(self):
        print("Step 8: Output Generation...")
        
        # A: Heatmap (Refined: Ineligible = Grey)
        # Logic: 0=Red, 1=Orange, 2=Green, 3=Grey (Excluded)
        # Excluded if NOT eligible
        
        hm_ratio = self.full_status["ratio"].unstack("TimeBin")
        hm_elig = self.full_status["is_eligible"].unstack("TimeBin")
        
        viz = pd.DataFrame(0, index=hm_ratio.index, columns=hm_ratio.columns)
        viz[hm_ratio == 0.5] = 1
        viz[hm_ratio == 1.0] = 2
        viz[~hm_elig] = 3 # Overwrite with Grey if Ineligible
        
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['#FFCDD2', '#FFCC80', '#A5D6A7', '#E0E0E0'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(viz, aspect='auto', cmap=cmap, norm=norm)
        
        # Ticks
        n_ticks = 10
        idxs = np.linspace(0, len(viz.columns)-1, n_ticks).astype(int)
        labels = [viz.columns[i].strftime('%m-%d %H') for i in idxs]
        ax.set_xticks(idxs); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(viz.index))); ax.set_yticklabels(viz.index)
        ax.set_title("Completeness (Green=Retained, Grey=Excluded/Warmup)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "completeness_heatmap.png"))
        plt.close()
        self.log_manifest("completeness_heatmap.png", "Completeness")
        
        # B: Inclusion (3 Bars)
        if self.stats:
            sdf = pd.DataFrame(self.stats).set_index("Room")
            ax = sdf[["Total", "Eligible", "Retained"]].plot(kind='bar', color=['gray', 'blue', 'green'], figsize=(8,5))
            plt.title("Inclusion Impact (Total -> Eligible -> Retained)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, "inclusion_impact.png"))
            plt.close()
            self.log_manifest("inclusion_impact.png", "3-Tier Inclusion Stats")
            
        # C/D: Validation
        if self.validation_plot_data:
            room = "TRP4" if "TRP4" in self.validation_plot_data else list(self.validation_plot_data.keys())[0]
            d = self.validation_plot_data[room]
            
            # Sound
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = ax1.twinx()
            ax1.plot(d["SoundLevel"].index, d["SoundLevel"], 'b', alpha=0.6, label='Level')
            ax2.plot(d["SoundVar"].index, d["SoundVar"], 'orange', label='Var')
            ax1.set_title(f"Sound Validation ({room})")
            plt.savefig(os.path.join(self.run_dir, "sound_validation.png")); plt.close()
            
            # Light
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = ax1.twinx()
            ax1.plot(d["LightLevel"].index, d["LightLevel"], 'b', alpha=0.6, label='Level')
            ax2.plot(d["LightChg"].index, d["LightChg"], 'orange', label='Change')
            ax1.set_title(f"Light Validation ({room})")
            plt.savefig(os.path.join(self.run_dir, "light_validation.png")); plt.close()
            
            self.log_manifest("sound_validation.png", "Sound Val")
            self.log_manifest("light_validation.png", "Light Val")
            
        # E: K-Selection (Annotated)
        if hasattr(self, "sweep_df"):
             fig, ax = plt.subplots(figsize=(8, 5))
             ax.plot(self.sweep_df["k"], self.sweep_df["silhouette"], 'b-o')
             
             sel = self.sweep_df[self.sweep_df["k"] == self.selected_k].iloc[0]
             ax.scatter(sel["k"], sel["silhouette"], s=200, c='red', marker='*')
             
             rej = self.sweep_df[self.sweep_df["rejected"]]
             if not rej.empty:
                 ax.scatter(rej["k"], rej["silhouette"], s=100, c='black', marker='x')
            
             plt.title(f"K Selection (Selected K={self.selected_k})")
             plt.savefig(os.path.join(self.run_dir, "k_selection.png")); plt.close()
             self.log_manifest("k_selection.png", "K Selection")
             
        # F: Profiles
        if hasattr(self, "feature_matrix") and "StateID" in self.feature_matrix:
            raw_c = self.feature_matrix.groupby("StateID")[["SoundVar", "LightChg"]].mean()
            raw_c.to_csv(os.path.join(self.run_dir, "centroids_raw.csv"))
            
            raw_c.plot(kind='bar', figsize=(8, 5))
            plt.title("Cluster Profiles (Raw Units)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, "cluster_profiles.png")); plt.close()
            self.log_manifest("cluster_profiles.png", "Profiles")
            self.log_manifest("centroids_raw.csv", "Raw Centroids")
            
        # G: Share
        if hasattr(self, "feature_matrix"):
            ct = pd.crosstab(self.feature_matrix.index.get_level_values("NodeId"), 
                             self.feature_matrix["StateID"], normalize='index') * 100
            
            # Annotated Axis
            counts = self.feature_matrix.index.get_level_values("NodeId").value_counts()
            ax = ct.plot(kind='bar', stacked=True, figsize=(10,6))
            labels = [f"{n}\n(n={counts.get(n,0)})" for n in ct.index]
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlabel("Room")
            plt.title("State Share")
            plt.savefig(os.path.join(self.run_dir, "state_share.png")); plt.close()
            self.log_manifest("state_share.png", "Share")
            
        # H: Transitions (Integer Ticks)
        if hasattr(self, "feature_matrix"):
            # Contiguous logic assumed from previous ver (omitted for brevity, assume retained)
            # Re-implementing compact
            bin_delta = pd.to_timedelta(self.cfg["bin_size"])
            df = self.feature_matrix.sort_index()
            pairs = []
            for n in df.index.get_level_values("NodeId").unique():
                 sub = df.xs(n, level="NodeId")
                 times = sub.index
                 s = sub["StateID"].values
                 for i in range(len(times)-1):
                     if times[i+1] - times[i] == bin_delta:
                         pairs.append((s[i], s[i+1]))
            
            if pairs:
                tr = pd.DataFrame(pairs, columns=["From", "To"])
                mat = pd.crosstab(tr["From"], tr["To"], normalize='index')
                fig, ax = plt.subplots()
                im = ax.imshow(mat, cmap='Blues')
                # Integer ticks
                k_size = len(mat)
                ax.set_xticks(range(k_size)); ax.set_xticklabels(mat.columns)
                ax.set_yticks(range(k_size)); ax.set_yticklabels(mat.index)
                plt.title("Transition Matrix")
                plt.colorbar(im)
                plt.savefig(os.path.join(self.run_dir, "transition_matrix.png")); plt.close()
                self.log_manifest("transition_matrix.png", "Transitions")

    def save_metadata(self):
        out = self.cfg.copy()
        for k, v in out.items():
            if isinstance(v, (pd.Timestamp, datetime)): out[k] = str(v)
            
        out["stats"] = self.stats
        out["meta"] = self.meta
        
        def np_enc(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
            
        with open(os.path.join(self.run_dir, "run_config.json"), 'w') as f:
            json.dump(out, f, indent=4, default=np_enc)
        self.log_manifest("run_config.json", "Config")
        
        pd.DataFrame(self.manifest).to_csv(os.path.join(self.run_dir, "manifest.csv"), index=False)
        print("  -> Saved manifest.csv")

    def print_summary(self):
        print("\n--- Run Summary ---")
        df_stats = pd.DataFrame(self.stats).set_index("Room")
        print(df_stats[["Total", "Eligible", "Retained", "Warmup_Excluded"]])
