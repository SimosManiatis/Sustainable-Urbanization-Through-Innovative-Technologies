import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from datetime import datetime
import glob

# Try imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    SKLEARN_AVAILABLE = True
    from itertools import combinations
except ImportError:
    SKLEARN_AVAILABLE = False
    from itertools import combinations # Fallback if sklearn fails but itertools is stdlib

# Aesthetic Setup
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

class AnalysisValidation:
    def __init__(self, config):
        self.cfg = config
        self.run_dir = os.path.join(config["output_dir"], config["run_id"])
        self.manifest = []
        self.meta = {
            "clipping_stats": {},
            "dropped_features": []
        }
        
        # Signals Definitions
        # Signals Definitions
        self.signals_map = {
            "Sound": "Sound_ave",
            "Light": "Light_ave"
        }
        
        # Load Room Locations
        self.locations = {}
        try:
            with open(self.cfg["mapping_file"], 'r') as f:
                d = json.load(f)
                self.locations = d.get("locations", {})
        except Exception:
            pass
            
        self.colors = ['#34495e', '#1abc9c', '#d35400', '#9b59b6', '#f1c40f', '#e74c3c']

        self.df_clean = None
        self.master_index = None

    def _get_name(self, node):
        return f"{self.locations.get(node, node)} ({node})"

    def log_manifest(self, filename, desc, scope="Global"):
        type_ = os.path.splitext(filename)[1]
        fp = os.path.join(self.run_dir, filename)
        sz = os.path.getsize(fp) if os.path.exists(fp) else 0
        self.manifest.append({
            "filename": filename, "type": type_, "size_bytes": sz, 
            "created": datetime.now().isoformat(), "description": desc, "scope": scope
        })

    def run(self):
        print("\n--- Starting Full Analysis Pipeline ---")
        if not os.path.exists(self.run_dir): os.makedirs(self.run_dir)
        
        # 1. Load (Common)
        self.step_load_data()
        self.step_build_timeline()
        self.save_metadata(initial=True)
        
        # 2. Branch
        targets = self.cfg["target_nodes"]
        if len(targets) == 1:
            print(f"-> Phase A: Per-Room Analysis ({targets[0]})")
            self.run_phase_a(targets[0])
        else:
            print(f"-> Phase B: Global Analysis ({len(targets)} rooms)")
            self.run_phase_b(targets)
            
        print("\n--- Pipeline Complete ---")
        print(f"Results in: {self.run_dir}")

    # --- COMMON SETUP ---
    def step_load_data(self):
        print("Step 1: Loading & Cleaning...")
        with open(self.cfg["mapping_file"], 'r') as f:
            mdata = json.load(f)
        self.id_to_label = {str(v): k for k, v in mdata.get("measurements", {}).items()}
        
        csv_files = glob.glob(os.path.join(self.cfg["raw_data_dir"], "*.csv"))
        if not csv_files: raise FileNotFoundError("No CSVs.")
        latest = max(csv_files, key=os.path.getctime)
        self.meta["source_file"] = os.path.basename(latest)
        
        df = pd.read_csv(latest, low_memory=False)
        df["SendDate"] = pd.to_datetime(df["SendDate"], utc=True, errors='coerce')
        df = df.dropna(subset=["SendDate"]).sort_values("SendDate")
        
        # Value Logic
        if "CorrectValue" in df.columns:
            df["val"] = df["CorrectValue"].fillna(df["Value"])
        else:
            df["val"] = df["Value"]
        df["val"] = pd.to_numeric(df["val"], errors='coerce')
        df = df.dropna(subset=["val"])
        
        # Clamping
        req_s, req_e = self.cfg.get("start_dt"), self.cfg.get("end_dt")
        eff_s = str(max(req_s, df["SendDate"].min())) if req_s else str(df["SendDate"].min())
        eff_e = str(min(req_e, df["SendDate"].max())) if req_e else str(df["SendDate"].max())
        
        df = df[(df["SendDate"] >= eff_s) & (df["SendDate"] <= eff_e)]
        self.cfg["effective_range"] = {"start": eff_s, "end": eff_e}
        
        # Scope
        if self.cfg["target_nodes"]:
             df = df[df["NodeId"].isin(self.cfg["target_nodes"])]
             
        # Detect Env
        lbls = df["SensorId"].astype(str).map(self.id_to_label).unique()
        if "Temperature_ave" in lbls: self.signals_map["Temp"] = "Temperature_ave"
        if "Air Quality_ave" in lbls: self.signals_map["AQ"] = "Air Quality_ave"
        if "Humidity_ave" in lbls: self.signals_map["Hum"] = "Humidity_ave"
        if "Pressure_ave" in lbls: self.signals_map["Pres"] = "Pressure_ave"
        
        # Filter
        targs = list(self.signals_map.values())
        df["label"] = df["SensorId"].astype(str).map(self.id_to_label)
        df = df[df["label"].isin(targs)]
        self.df_clean = df.drop_duplicates(["NodeId","label","SendDate"], keep='last')

    def step_build_timeline(self):
        er = self.cfg["effective_range"]
        s = pd.Timestamp(er["start"]).floor(self.cfg["bin_size"])
        e = pd.Timestamp(er["end"]).ceil(self.cfg["bin_size"])
        self.master_index = pd.date_range(s, e, freq=self.cfg["bin_size"])
        self.meta["n_bins"] = len(self.master_index)

    # --- HELPERS ---
    def enforce_temporal_continuity(self, labels, time_index, min_duration_mins):
        """Merges short segments < min_duration into neighbors."""
        if len(labels) == 0: return labels
        
        # Convert to Series for easier handling
        s = pd.Series(labels, index=time_index)
        
        # Iterative merge until stable
        max_iters = 5
        for _ in range(max_iters):
            # Identify segments: (Value, Start, End, Count)
            # Group by value change
            grp = (s != s.shift()).cumsum()
            segs = s.groupby(grp).agg(['first', 'count'])
            # Calc duration? Index is TimeBin. 
            # We can approximate duration by count * bin_size (assuming continuity)
            # Or use exact time diff if gaps exist.
            # Here we assume 'retained' bins might have gaps.
            # If retained bins are sparse, "duration" is conceptual.
            # Let's use Count * BinSize for now as "Effective Duration".
            
            bin_mins = pd.to_timedelta(self.cfg["bin_size"]).total_seconds() / 60
            segs["duration"] = segs["count"] * bin_mins
            
            # Find short segments
            short_mask = segs["duration"] < min_duration_mins
            if not short_mask.any(): break
            
            # Merge logic: Rebuild array
            # Simplest: For each short segment, look at prev/next segment
            # Merge into the one with more samples? Or closest centroid? 
            # We don't have centroids here easily. Merge into LARGER neighbor.
            
            # Map group_id -> new_label
            # Iterate through groups
            new_vals = []
            param_groups = s.groupby(grp)
            
            # Convert to list of dicts for easier linear scan
            seg_list = []
            for gid, sub in param_groups:
                seg_list.append({"gid": gid, "val": sub.iloc[0], "count": len(sub), "dur": len(sub)*bin_mins})
                
            # Forward scan to merge
            # This is complex to do vectorized. Linear scan:
            merged_list = []
            if not seg_list: return labels
            
            curr = seg_list[0]
            for i in range(1, len(seg_list)):
                next_seg = seg_list[i]
                
                # Check if curr is short
                if curr["dur"] < min_duration_mins:
                    # Merge into next? (Assign curr's pixels to next's label? Or prev?)
                    # Strategy: If curr short, take Next's label if Next is long?
                    # Or keep accumulating until > min?
                    # "Merge into valid neighbor".
                    
                    # If we have a 'prev' (merged_list is not empty), merge with prev?
                    if merged_list:
                        # Merge with prev
                        prev = merged_list[-1]
                        # Update prev count/dur
                        # But wait, we change label to prev's label
                        # curr is absorbed.
                        # We don't Append curr. We just extend prev.
                        # But label equality check? 
                        # We just force curr's bins to take prev's label.
                        # Actually logic: Labels array update.
                        pass
                        # This linear scan is tricky to implement robustly in one pass.
                        # Let's stick to "Smallest Cluster Removal" approach if segments are hard?
                        # No, User asked for "Run Length Encoding... Merge into neighbor".
                    
                    # Let's do a simple pass:
                    # If segment i is short:
                    #   Identify neighbors i-1 and i+1.
                    #   Pick neighbor with max(duration).
                    #   Change segment i's label to match that neighbor.
                    #   Stop (one merge per iter to be safe? or can do all independent?)
                    #   Doing all independent might conflict.
                    pass
            
            # Vectorized approach using fillna?
            # Replace short segment values with NaN, then ffill/bfill?
            # 1. Mask short segments
            # 2. Re-label
            
            # Get integer indices of short groups
            short_ids = segs.index[short_mask]
            
            # Mask in original series
            mask_indices = grp.isin(short_ids)
            
            if mask_indices.any():
                # Set to NaN
                # Ensure float for NaN
                s_mod = s.astype(float)
                s_mod[mask_indices] = np.nan
                
                # Interpolate (Nearest? or FFill?)
                # "Merge into neighbor". Nearest implies temporal distance?
                # FFill then BFill covers gaps.
                s_mod = s_mod.ffill().bfill()
                
                # FIX: Handle residual NaNs (if all segments were short/masked)
                if s_mod.isna().any():
                    s_mod = s_mod.fillna(s.astype(float))
                
                s = s_mod.astype(int)
            else:
                break
                
        return s.values

    # --- CORE WORKER FUNCTIONS ---
    def process_room_data(self, node):
        """Prep 1: Returns raw_aligned, indicators_aligned, clipping_stats."""
        ndf = self.df_clean[self.df_clean["NodeId"] == node].copy()
        if ndf.empty: return None
        
        res = {}; raw_plots = {}
        
        # Helper
        def _resample(sig_key, feat_type, win):
            lbl = self.signals_map.get(sig_key)
            if not lbl: return None
            sub = ndf[ndf["label"] == lbl]
            if sub.empty: return None
            
            # 1. Raw 1min
            ts = sub.set_index("SendDate")["val"].resample("1min").mean()
            
            # 2. Indicator
            w = int(max(1, win))
            if feat_type == "STD": feat = ts.rolling(w, min_periods=max(1, w//2)).std()
            elif feat_type == "Delta": feat = ts.diff(w)
            
            # 3. Agg
            agg = self.cfg["agg_method"]
            b_feat = feat.resample(self.cfg["bin_size"]).agg(agg)
            b_raw = ts.resample(self.cfg["bin_size"]).agg(agg)
            
            return b_raw, b_feat
            
        # calc
        s = _resample("Sound", "STD", 5)
        l = _resample("Light", "Delta", 15)
        if s: 
            res["SoundVar"] = s[1]
            res["SoundMean"] = s[0]
            raw_plots["Sound"] = s[0]
        if l: 
            res["LightChg"] = l[1]
            res["LightMean"] = l[0]
            raw_plots["Light"] = l[0]
        
        if "Temp" in self.signals_map:
             t = _resample("Temp", "Delta", 15)
             if t: 
                res["TempDelta"] = t[1]
                res["TempMean"] = t[0]
             
        if "AQ" in self.signals_map:
             aq = _resample("AQ", "Delta", 15)
             if aq: 
                res["AQDelta"] = aq[1]
                res["AQMean"] = aq[0]

        if "Hum" in self.signals_map:
             h = _resample("Hum", "Delta", 15)
             if h: 
                res["HumDelta"] = h[1]
                res["HumMean"] = h[0]

        if "Pres" in self.signals_map:
             p = _resample("Pres", "Delta", 15)
             if p: 
                res["PresDelta"] = p[1]
                res["PresMean"] = p[0]
             
        # Align
        if not res: return None
        
        idf = pd.DataFrame(res, index=self.master_index)
        idf.index.name = "TimeBin"
        raf = pd.DataFrame(raw_plots, index=self.master_index)
        
        # Outlier Clip
        for c in idf.columns:
            lo = idf[c].quantile(0.01); hi = idf[c].quantile(0.99)
            idf[c] = idf[c].clip(lo, hi)
            self.meta["clipping_stats"][f"{node}_{c}"] = {"lo": lo, "hi": hi}
            
        return raf, idf

    def assess_inclusion(self, node, raw_df, ind_df):
        """Prep 2: 4-Level Accounting."""
        # 1. Total
        total_bins = len(self.master_index)
        
        # 2. Raw Available (Sound & Light Required)
        # Check raw non-null
        req = ["Sound", "Light"]
        # Add dummy if missing
        check = raw_df.copy()
        for r in req: 
            if r not in check.columns: check[r] = np.nan
        has_raw = check[req].notna().all(axis=1)
        
        # 3. Indicator Valid
        # Check all calculated indicators non-null
        has_ind = ind_df.notna().all(axis=1)
        
        # 4. Retained
        # Must have Raw AND Ind
        is_retained = has_raw & has_ind
        
        stats = {
            "Total": total_bins,
            "Raw_Avail": has_raw.sum(),
            "Ind_Valid": has_ind.sum(), # Note: Ind valid usually implies raw, but windowing kills start
            "Retained": is_retained.sum()
        }
        
        # Flags df
        flags = pd.DataFrame({
            "has_raw": has_raw, "has_ind": has_ind, "is_retained": is_retained
        }, index=self.master_index)
        
        return stats, flags, ind_df[is_retained]

    def perform_clustering(self, feat_df, name_prefix):
        """Standardize -> Sweep (ARI + Temporal) -> Select -> Fit."""
        if feat_df.empty: return None, 0
        
        # Drop constant
        df = feat_df.copy()
        df = df.loc[:, df.std() > 0]
        if df.empty: return None, 0
        
        if SKLEARN_AVAILABLE:
            X = StandardScaler().fit_transform(df)
            
            # Constraints
            bin_mins = pd.to_timedelta(self.cfg["bin_size"]).total_seconds() / 60
            min_samples = int(np.ceil(120 / bin_mins))
            self.meta["clustering_params"] = {"min_duration_min": 120, "min_samples": min_samples}
            
            sweep_data = []
            best_k = None
            
            for k in range(2, 11):
                # Run 10 seeds
                seed_res = []
                for seed in range(10):
                    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                    lbs = km.fit_predict(X)
                    sil = silhouette_score(X, lbs)
                    
                    # Apply Temporal Merge to check effective size
                    lbs_merged = self.enforce_temporal_continuity(lbs, df.index, 120)
                    min_sz = np.min(np.bincount(lbs_merged)) if len(lbs_merged) > 0 else 0
                    
                    seed_res.append({
                        "lbs": lbs, "sil": sil, "min_sz_post": min_sz
                    })
                
                # ARI Stability
                aris = []
                for i, j in combinations(range(10), 2):
                    aris.append(adjusted_rand_score(seed_res[i]["lbs"], seed_res[j]["lbs"]))
                ari_mean = np.mean(aris) if aris else 0
                
                # Stats
                sils = [r["sil"] for r in seed_res]
                sil_mean = np.mean(sils)
                sil_std = np.std(sils)
                
                # Worst-case min size across seeds (Safety)
                min_sz_min = np.min([r["min_sz_post"] for r in seed_res])
                
                # Validity: All seeds generated valid structure? Or Average?
                # User: "Report min_sz_min... Strict safety check".
                is_valid = min_sz_min >= min_samples
                
                # Score: Sil - Penalty + ARI Bonus
                score = sil_mean - (0.5 * sil_std) + (0.1 * ari_mean)
                
                sweep_data.append({
                    "k": k, "sil_mean": sil_mean, "sil_std": sil_std, "ari_mean": ari_mean,
                    "min_sz_min": min_sz_min, "valid": is_valid, "score": score
                })

            sdf = pd.DataFrame(sweep_data)
            
            # Selection (Richer K Rule)
            valid_cands = sdf[sdf["valid"]].copy()
            
            if valid_cands.empty:
                print(f"  [{name_prefix}] Skipped (Degenerate/Unstable).")
                best_k = None
            else:
                top_score = valid_cands["score"].max()
                # Candidates within epsilon
                epsilon = 0.02
                candidates = valid_cands[valid_cands["score"] >= top_score - epsilon]
                
                # Select LARGEST K among good candidates (Richer Structure)
                best_row = candidates.sort_values("k", ascending=False).iloc[0]
                best_k = int(best_row["k"])
                
                if len(candidates) > 1:
                     print(f"  [{name_prefix}] Selected Richer K={best_k} (Score {best_row['score']:.3f} vs Max {top_score:.3f})")

            # Save Sweep
            # sdf.to_csv(os.path.join(self.run_dir, f"{name_prefix}_k_sweep.csv"), index=False)
            print(f"\n--- K Sweep Results ({name_prefix}) ---")
            print(sdf.to_string())
            self.log_manifest(f"{name_prefix}_k_sweep.csv", "Sweep Data", name_prefix)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8,5))
            val = sdf[sdf["valid"]]; inv = sdf[~sdf["valid"]]
            
            if not val.empty: 
                ax.errorbar(val["k"], val["sil_mean"], yerr=val["sil_std"], 
                           fmt='o-', color='#34495e', ecolor='gray', capsize=3, label='Valid')
            if not inv.empty: 
                ax.errorbar(inv["k"], inv["sil_mean"], yerr=inv["sil_std"], 
                           fmt='x', color='#e74c3c', alpha=0.6, label='Rejected')
            
            if best_k:
                sel = sdf[sdf["k"]==best_k].iloc[0]
                ax.scatter(sel["k"], sel["sil_mean"], c='#f1c40f', marker='*', s=250, zorder=10, 
                          edgecolors='black', label=f"Selected K={best_k}")
                
            ax.set_title(f"Optimal K Selection", fontsize=12, fontweight='bold')
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_k_selection.png"), dpi=100)
            plt.close()
            
            # Final Fit & Merge
            if best_k:
                # Retrain
                km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
                raw_lbs = km.fit_predict(X)
                
                # Apply Temporal Merge
                final_lbs = self.enforce_temporal_continuity(raw_lbs, df.index, 120)
                df["StateID"] = final_lbs
                
                # Centroids (Recalculated on merged labels)
                # Note: Merge might eliminate a StateID if it was entirely short segments.
                # So we check unique states.
                centroids = df.groupby("StateID").mean()
                # centroids.to_csv(os.path.join(self.run_dir, f"{name_prefix}_centroids_raw.csv"))
                print(f"\n--- Raw Centroids ({name_prefix}) ---")
                print(centroids.to_string())
                
                return df, best_k
            
        return None, 0

    def _get_state_names(self, df):
        """
        Heuristic to semantic state naming.
        Returns: {0: '0 - Quiet/Dark', 1: '1 - Active', ...}
        """
        if "StateID" not in df.columns: return {}
        
        # Calc Centroids
        stats = df.groupby("StateID")[["LightMean", "SoundMean"]].mean()
        
        # Determine Thresholds (Dynamic)
        # Or simple rank?
        # Rank by Light
        
        names = {}
        for sid, row in stats.iterrows():
            l = row.get("LightMean", 0)
            s = row.get("SoundMean", 0)
            
            # Simple Logic
            if l < 50 and s < 45: tag = "Quiet"
            elif l > 100: tag = "Active"
            else: tag = "Moderate"
            
            names[sid] = f"{sid} - {tag}"
            
        return names

    def generate_usage_plots(self, tagged_df, name_prefix, state_names=None):
        # Resolve Name
        display_name = name_prefix
        if name_prefix in self.locations: display_name = self._get_name(name_prefix)
        elif name_prefix == "global": display_name = "Global Analysis"
    
        # Share
        cts = tagged_df["StateID"].value_counts(normalize=True).sort_index()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        # Map colors to index
        cols = [self.colors[i % len(self.colors)] for i in cts.index]
        cts.plot(kind='bar', color=cols, ax=ax, width=0.6)
        
        # Labels
        labels = [state_names.get(i, str(i)) for i in cts.index] if state_names else cts.index
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_title(f"State Distribution: {display_name}", fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xlabel("State", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_state_share.png"), dpi=100)
        plt.close()
        
        # Transitions
        # Explicit Gap Check
        df = tagged_df.sort_index() # Index is MultiIndex (Node, Time) or just Time?
        # If Single Room, Index is TimeBin
        # If Global, Index is Multi (Node, Time)
        
        pairs = []
        delta = pd.to_timedelta(self.cfg["bin_size"])
        
        if "NodeId" in df.index.names:
            nodes = df.index.get_level_values("NodeId").unique()
            for n in nodes:
                 sub = df.xs(n, level="NodeId")
                 t = sub.index; s = sub["StateID"].values
                 for i in range(len(t)-1):
                     if t[i+1]-t[i] == delta: pairs.append((s[i], s[i+1]))
        else:
             t = df.index; s = df["StateID"].values
             for i in range(len(t)-1):
                 if t[i+1]-t[i] == delta: pairs.append((s[i], s[i+1]))
                 
        if pairs:
             tr = pd.DataFrame(pairs, columns=["From","To"])
             mat = pd.crosstab(tr["From"], tr["To"], normalize='index')
             
             fig, ax = plt.subplots(figsize=(6,5))
             im = ax.imshow(mat, cmap='Blues', aspect='auto')
             
             # Annotate
             for i in range(len(mat)):
                 for j in range(len(mat.columns)):
                     text = ax.text(j, i, f"{mat.iloc[i, j]:.2f}",
                                    ha="center", va="center", color="black" if mat.iloc[i, j] < 0.7 else "white")

             # Axis Labels with Names
             from_labels = [state_names.get(i, str(i)) for i in mat.index] if state_names else mat.index
             to_labels = [state_names.get(i, str(i)) for i in mat.columns] if state_names else mat.columns
             
             ax.set_xticks(range(len(mat))); ax.set_xticklabels(to_labels, rotation=45, ha='right')
             ax.set_yticks(range(len(mat))); ax.set_yticklabels(from_labels)
             ax.set_title(f"Transition Probabilities: {display_name}", fontsize=12, fontweight='bold', pad=15)
             ax.set_xlabel("To State")
             ax.set_ylabel("From State")
             ax.grid(False) # No grid on heatmap
             
             cbar = plt.colorbar(im)
             cbar.outline.set_visible(False)
             
             plt.tight_layout()
             plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_transition.png"), dpi=100)
             plt.close()

    def analyze_substructure(self, df, name_prefix):
        """
        Recursive Substructure Check:
        If a dominant baseline state exists (>70%), isolate the REST and re-cluster.
        """
        if "StateID" not in df.columns: return
        
        counts = df["StateID"].value_counts(normalize=True)
        if counts.empty: return
        
        dom_id = counts.idxmax()
        dom_share = counts.max()
        
        if dom_share > 0.70:
            print(f"\n[Substructure] Dominant State {dom_id} found ({dom_share*100:.1f}%). Checking hidden structure in residue...")
            
            # Filter matches only
            sub_df = df[df["StateID"] != dom_id].copy()
            
            # Check size (Reuse constants logic?)
            # Hardcoded safer limit for substructure?
            # 120 bins = 30 hours.
            if len(sub_df) < 120:
                print(f"[Substructure] Residue too small ({len(sub_df)} bins). Skipping.")
                return

            # Remove StateID col to prep for re-clustering
            # Also drop NodeId index levels if they mess up? 
            # perform_clustering handles cleaning.
            feat_df = sub_df.drop(columns=["StateID"], errors='ignore')
            
            # Recurse
            sub_res, sub_k = self.perform_clustering(feat_df, f"{name_prefix}_sub")
            
            if sub_res is not None and sub_k > 1:
                print(f"!!! HIDDEN STRUCTURE FOUND in {name_prefix} residue (K={sub_k}) !!!")
                # Plot for the hidden structure?
                self.generate_usage_plots(sub_res, f"{name_prefix}_sub")
            else:
                 print(f"[Substructure] Residue is homogeneous (No hidden structure found).")


    # --- ADVANCED MODULES ---
    def plot_weekly_occupancy(self, df, active_state, name_prefix, state_names=None):
        """Heatmap of Active State Probability (Day x Hour)."""
        dn = self._get_name(name_prefix)
        
        # Active Name
        active_label = state_names.get(active_state, f"State {active_state}") if state_names else f"State {active_state}"
        
        # Prep Data
        # Filter for active state
        df["IsActive"] = (df["StateID"] == active_state).astype(int)
        
        # Group by DayOfWeek and Hour
        # Handle MultiIndex (Node, Time) vs Single Index (Time)
        if isinstance(df.index, pd.MultiIndex):
            # Assuming Level 1 is Time (based on run_phase_b extraction)
            # Try to find a level that is datetime-like
            idx = df.index.get_level_values(-1) # Usually the last one
            if not isinstance(idx, pd.DatetimeIndex):
                 # Fallback search
                 for i in range(df.index.nlevels):
                     if isinstance(df.index.get_level_values(i), pd.DatetimeIndex):
                         idx = df.index.get_level_values(i)
                         break
        else:
            idx = df.index

        grp = df.groupby([idx.dayofweek, idx.hour])["IsActive"].mean().unstack()
        # Reindex to ensure full grid
        grp = grp.reindex(index=range(7), columns=range(24)).fillna(0)
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(grp, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        
        # Axis
        ax.set_xticks(range(24))
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)
        ax.set_xlabel("Hour of Day")
        ax.set_title(f"Typical Weekly Occupancy: {dn}\n(Prob of {active_label})", fontsize=12, fontweight='bold')
        ax.grid(False)
        
        # Colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Occupancy Prob", rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_weekly_occupancy.png"), dpi=100)
        plt.close()

    def plot_comfort_zone(self, df, active_state, name_prefix, state_names=None):
        """Scatter of Temp vs Hum for Active periods."""
        dn = self._get_name(name_prefix)
        
        if "TempMean" not in df.columns or "HumMean" not in df.columns: return
        
        # Filter Active
        active = df[df["StateID"] == active_state]
        if active.empty: return

        active_label = state_names.get(active_state, "Occupied") if state_names else "Occupied"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Comfort Box (ASHRAE approx: 20-26C, 30-60%)
        # Rectangle(xy, width, height)
        # xy = (20, 30)
        rect = Rectangle((20, 30), 6, 30, linewidth=1, edgecolor='green', facecolor='green', alpha=0.1, label='Comfort Zone')
        ax.add_patch(rect)
        
        # Scatter
        sc = ax.scatter(active["TempMean"], active["HumMean"], c='#d35400', alpha=0.6, edgecolors='w', s=40, label=active_label)
        
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Humidity (%)")
        ax.set_title(f"Comfort Analysis: {dn}\n({active_label} Periods Only)", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        
        ax.set_xlim(15, 30)
        ax.set_ylim(10, 80)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_comfort_zone.png"), dpi=100)
        plt.close()

    def detect_anomalies(self, df, name_prefix, state_names=None):
        """Simple distance-based anomaly detection."""
        # For each point, calc distance to its state centroid.
        # 1. Calc Centroids
        cols = [c for c in df.columns if c not in ["StateID", "IsActive", "AnomalyScore"]]
        if not cols: return
        
        X = df[cols]
        # Standardize for distance
        if SKLEARN_AVAILABLE:
            try:
                X_scaled = StandardScaler().fit_transform(X)
                # Re-attach state
                sdf = pd.DataFrame(X_scaled, index=df.index, columns=cols)
                sdf["StateID"] = df["StateID"]
                
                # Center of each state
                centers = sdf.groupby("StateID").mean()
                
                # Calc Dist
                dists = []
                # Vectorized distance using apply is faster?
                # or merge
                # Let's iterate if not too huge
                # Or vector:
                # Map centers to rows
                # diff = sdf - centers
                # dist = norm(diff)
                
                # Doing iter for safety on index matching
                for idx, row in sdf.iterrows():
                    sid = int(row["StateID"])
                    # Check if sid in centers (should be)
                    if sid in centers.index:
                        ctr = centers.loc[sid]
                        d = np.linalg.norm(row[cols] - ctr)
                        dists.append(d)
                    else:
                        dists.append(0)
                    
                df["AnomalyScore"] = dists
                
                # Threshold (e.g., 3 sigma)
                mean_d = np.mean(dists)
                std_d = np.std(dists)
                thresh = mean_d + (3 * std_d)
                
                anomalies = df[df["AnomalyScore"] > thresh]
                
                if not anomalies.empty:
                    print(f"\n[Anomaly Report] {name_prefix}: Found {len(anomalies)} anomalies (>3 std dev).")
                    
                    # Augment report with names
                    if state_names:
                        anomalies["StateName"] = anomalies["StateID"].map(state_names)
                        print(anomalies[["StateName", "AnomalyScore"]].sort_values("AnomalyScore", ascending=False).head(5).to_string())
                    else:
                        print(anomalies[["StateID", "AnomalyScore"]].sort_values("AnomalyScore", ascending=False).head(5).to_string())
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    x_vals = np.arange(len(df))
                    y_scores = df["AnomalyScore"].values
                    is_anom = df["AnomalyScore"] > thresh
                    x_anom = x_vals[is_anom]
                    y_anom = y_scores[is_anom]
                    
                    ax.plot(x_vals, y_scores, color='#95a5a6', lw=1, label="Score")
                    ax.scatter(x_anom, y_anom, color='#e74c3c', s=20, label="Anomaly")
                    ax.axhline(thresh, color='r', linestyle='--', alpha=0.5, label="Threshold")
                    ax.set_title(f"Anomaly Detection: {self._get_name(name_prefix)}")
                    
                    if isinstance(df.index, pd.MultiIndex):
                         ax.set_xticks([])
                         ax.set_xlabel("Sample Index (Global)")
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_anomalies.png"), dpi=100)
                    plt.close()
                else:
                    print(f"\n[Anomaly Report] {name_prefix}: No significant anomalies found.")
            except Exception as e:
                print(f"[Anomaly Error] {e}")

    def plot_parameter_metrics(self, df, name_prefix):
        """
        Plots a stacked time-series of available environmental parameters
        matching the User's requested 'Minimalist' aesthetic, but with distinct colors.
        """
        dn = self._get_name(name_prefix)
        
        # Color Map
        colors = {
            "TempMean": "#FF9800", # Orange
            "HumMean": "#2196F3",  # Blue
            "LightMean": "#FDD835", # Yellow
            "SoundMean": "#9C27B0", # Purple
            "AQMean": "#4CAF50"    # Green
        }
        
        # Identify available 'Mean' columns
        params = []
        if "TempMean" in df.columns: params.append(("TempMean", "Temperature"))
        if "HumMean" in df.columns: params.append(("HumMean", "Humidity"))
        if "LightMean" in df.columns: params.append(("LightMean", "Light"))
        if "SoundMean" in df.columns: params.append(("SoundMean", "Sound"))
        if "AQMean" in df.columns: params.append(("AQMean", "Air Quality"))
        
        if not params: return

        # Create subplots
        fig, axes = plt.subplots(len(params), 1, figsize=(12, 3*len(params)), sharex=True)
        if len(params) == 1: axes = [axes]
        
        # Handle X-axis for plotting
        x_vals = np.arange(len(df))
        
        # Date String for Subtitle
        date_str = ""
        if isinstance(df.index, pd.DatetimeIndex):
            start_d = df.index.min().strftime("%Y-%m-%d")
            end_d = df.index.max().strftime("%Y-%m-%d")
            date_str = f"{start_d} / {end_d}"
        
        for i, (ax, (col, label)) in enumerate(zip(axes, params)):
            y_vals = df[col].values
            c = colors.get(col, "#607D8B") # Default Grey
            
            # Plot Line & Fill
            ax.plot(x_vals, y_vals, color=c, linewidth=2.5)
            
            y_min = np.min(y_vals)
            y_max = np.max(y_vals)
            padding = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
            fill_base = y_min - padding
            
            ax.fill_between(x_vals, y_vals, fill_base, color=c, alpha=0.3)
            
            # Clean Styling (Minimalist)
            ax.set_ylim(bottom=fill_base, top=y_max + padding)
            
            # Remove Grid & Spines
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color(c)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Remove Ticks
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            if i == len(axes)-1:
                 # Bottom Plot
                 ax.spines['bottom'].set_visible(True)
                 ax.set_xticks([]) 
            else:
                 ax.spines['bottom'].set_visible(False)
                 ax.set_xticks([])

            # Title / Typography
            ax.text(0.0, 0.9, label, transform=ax.transAxes, fontsize=12, fontweight='bold', color='#37474F')
            ax.text(0.0, 0.8, date_str, transform=ax.transAxes, fontsize=9, color='#78909C')

        # Overall Title (Room Name)
        axes[0].text(0.0, 1.1, dn, transform=axes[0].transAxes, fontsize=16, fontweight='bold', color='black')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_param_trends.png"), dpi=100)
        plt.close()

    def plot_correlations(self, df, name_prefix):
        """
        Plots a correlation heatmap for environmental parameters.
        """
        dn = self._get_name(name_prefix)
        
        # Select Mean/Var columns derived from sensors
        cols = [c for c in df.columns if "Mean" in c or "Var" in c]
        if len(cols) < 2: return
        
        corr = df[cols].corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        
        ticks = np.arange(len(cols))
        plt.xticks(ticks, cols, rotation=45, ha='right')
        plt.yticks(ticks, cols)
        plt.title(f"Parameter Correlation: {dn}", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_correlations.png"), dpi=100)
        plt.close()

    def plot_activity_comparison(self, df, active_state_map, name_prefix, state_names=None):
        """
        Plots a single line chart comparing the activity status (Active/Inactive) 
        of all rooms over time. 
        active_state_map: {NodeId: ActiveStateID} 
        """
        if "StateID" not in df.columns: return

        # 1. Prepare Data
        # We need (Time, Room) -> IsActive
        if isinstance(df.index, pd.MultiIndex):
            # Assuming index is (NodeId, Time)
            temp = df.copy()
            temp = temp.reset_index(level=0) # Move NodeId to col
            pivot = temp.pivot(columns="NodeId", values="StateID")
        else:
            return

        if pivot.empty: return
        
        # Reindex to handle gaps
        freq = self.cfg.get("bin_size", "15min")
        full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq)
        pivot = pivot.reindex(full_idx)

        # 2. Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rooms = pivot.columns
        offset_step = 1.1 # Separation
        
        yticks = []
        yticklabels = []
        
        for i, room in enumerate(rooms):
            series = pivot[room] # Series of StateIDs (with NaNs)
            active_target = active_state_map.get(str(room), -1) # Ensure str key
            
            # Convert to Binary Active/Inactive (preserving NaNs)
            def binary_map(val):
                if pd.isna(val): return np.nan
                return 1.0 if int(val) == active_target else 0.0
            
            bin_series = series.map(binary_map)
            
            base_y = i * offset_step
            # Plot 'Active' as step line
            y_vals = bin_series.map(lambda x: base_y + (0.8 if x == 1.0 else 0.0) if pd.notna(x) else np.nan)
            
            # Using plot with NaNs breaks the line
            ax.plot(pivot.index, y_vals, label=self._get_name(room), drawstyle='steps-post', lw=1.5)
            
            yticks.append(base_y + 0.4)
            
            # Label Construction
            room_name = self._get_name(room)
            if len(room_name) > 15: room_name = room_name[:15] + "..."
            
            # Add state info?
            state_label = ""
            if state_names and active_target in state_names:
                # e.g. "State 5 - Active" -> "Active (S5)"
                # or just use full name?
                # User asked: "where and which activites the rooms reach"
                # Let's use the full state name provided by state_names logic
                s_name = state_names[active_target]
                # Simplify? "5 - Active"
                state_label = f" [{s_name}]"
                
            yticklabels.append(f"{room_name}{state_label}")

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title(f"Activity Comparison: All Rooms (Per-Room Active State)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_activity_comparison.png"), dpi=100)
        plt.close()

    def plot_combined_parameters(self, df, name_prefix):
        """
        Plots a single chart with lines for all environmental parameters,
        normalized to 0-1 range to fit on one axis.
        """
        dn = self._get_name(name_prefix)
        
        # Params to include
        check_cols = ["TempMean", "HumMean", "LightMean", "SoundMean", "AQMean"]
        cols = [c for c in check_cols if c in df.columns]
        
        if not cols: return
        
        # Reindex for Gaps
        freq = self.cfg.get("bin_size", "15min")
        if isinstance(df.index, pd.DatetimeIndex):
            # dedupe index if needed
            df_dedup = df[~df.index.duplicated(keep='last')]
            full_idx = pd.date_range(df_dedup.index.min(), df_dedup.index.max(), freq=freq)
            df_plot = df_dedup.reindex(full_idx)
        else:
            df_plot = df
        
        # Prepare Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            "TempMean": "#e67e22", # Orange
            "HumMean": "#3498db",  # Blue
            "LightMean": "#f1c40f", # Yellow
            "SoundMean": "#9b59b6", # Purple
            "AQMean": "#2ecc71"    # Green
        }
        
        labels = {
            "TempMean": "Temp",
            "HumMean": "Hum",
            "LightMean": "Light",
            "SoundMean": "Sound",
            "AQMean": "AQ"
        }
        
        for col in cols:
            series = df_plot[col]
            # Normalize Min-Max (computed on plot df so Min can be min of available)
            s_min = series.min()
            s_max = series.max()
            denom = s_max - s_min
            
            if pd.isna(s_min) or pd.isna(s_max): continue

            if denom == 0:
                norm = series - s_min 
            else:
                norm = (series - s_min) / denom
                
            ax.plot(df_plot.index, norm, label=f"{labels[col]} (Norm)", color=colors.get(col, "grey"), lw=1.5, alpha=0.8)

        ax.set_title(f"Multi-Parameter Trends (Normalized): {dn}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Normalized Range (0-1)")
        ax.set_yticks([0, 0.5, 1])
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_param_overlay.png"), dpi=100)
        plt.close()

    # --- PHASE A: SINGLE ROOM ---
    def run_phase_a(self, node):
        # 1. Prep
        raw, ind = self.process_room_data(node)
        
        # 2. Accounting
        stats, flags, retained_df = self.assess_inclusion(node, raw, ind)
        
        # 3. Plots (Diagnostic & Inclusion)
        # Inclusion
        fig, ax = plt.subplots(figsize=(6, 4))
        dn = self._get_name(node)
        bars = ax.bar(["Total", "Raw", "Ind", "Retained"], 
               [stats["Total"], stats["Raw_Avail"], stats["Ind_Valid"], stats["Retained"]],
               color=['#bdc3c7', '#f39c12', '#3498db', '#2ecc71'], width=0.6)
        ax.bar_label(bars, fmt='%d', padding=3)
        ax.set_title(f"Data Health: {dn}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Time Bins")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{node}_inclusion_impact.png"), dpi=100)
        plt.close()
        
        # Diagnostic
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax2 = ax1.twinx()
        if "Sound" in raw: ax1.plot(raw.index, raw["Sound"], color='#3498db', lw=1, alpha=0.4, label="Sound")
        if "SoundVar" in ind: ax2.plot(ind.index, ind["SoundVar"], color='#2c3e50', lw=1.5, ls='-', label="Var")
        if "Light" in raw: ax1.plot(raw.index, raw["Light"], color='#f1c40f', lw=1, alpha=0.4, label="Light")
        if "LightChg" in ind: ax2.plot(ind.index, ind["LightChg"], color='#e67e22', lw=1.5, ls='-', label="Chg")
        ax1.set_title(f"Diagnostics: {dn}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{node}_diagnostic_timeline.png"), dpi=100)
        plt.close()

        # 4. Clustering & Advanced Analysis
        if not retained_df.empty:
            res_df, k = self.perform_clustering(retained_df, node)
            if res_df is not None:
                names = self._get_state_names(res_df)
                self.generate_usage_plots(res_df, node, state_names=names)
                print(f"\n--- Labelled Data Sample ({node}) ---")
                print(res_df.head(10).to_string())
                
                # Determine Active State (Highest LightMean or SoundVar)
                # Heuristic: The state with max mean(LightMean) assumed to be 'Active'
                # If LightMean not avail, use SoundVar.
                try:
                    stats = res_df.groupby("StateID").mean()
                    if "LightMean" in stats.columns:
                        active_id = stats["LightMean"].idxmax()
                    elif "SoundVar" in stats.columns:
                        active_id = stats["SoundVar"].idxmax()
                    else:
                        active_id = stats.index[-1] # Fallback
                except:
                    active_id = 0
                
                print(f"  [Analysis] Identified State {active_id} as 'Active' (for Heatmap/Comfort).")
                
                # Advanced Modules
                self.plot_weekly_occupancy(res_df.copy(), active_id, node, state_names=names)
                self.plot_comfort_zone(res_df.copy(), active_id, node, state_names=names)
                self.detect_anomalies(res_df.copy(), node, state_names=names)
                self.plot_parameter_metrics(res_df.copy(), node)
                self.plot_correlations(res_df.copy(), node)

                # Check Substructure (Bias Check)
                self.analyze_substructure(res_df, node)
            else:
                print(f"  [{node}] Clustering skipped (degenerate/unstable).")
            
        self.save_metadata(initial=False, stats=[stats])

    # --- PHASE B: GLOBAL ---
    def run_phase_b(self, nodes):
        all_retained = []
        all_stats = []
        
        for node in nodes:
            print(f"  Processing {node}...")
            raw, ind = self.process_room_data(node)
            stats, flags, ret = self.assess_inclusion(node, raw, ind)
            stats["Room"] = node
            all_stats.append(stats)
            print(f"Stats for {node}: {stats}")
            
            if not ret.empty:
                ret = ret.copy()
                ret["NodeId"] = node
                ret = ret.set_index("NodeId", append=True).swaplevel(0,1) # Node, Time
                all_retained.append(ret)
                
        # Global Inclusion
        sdf = pd.DataFrame(all_stats).set_index("Room")
        sdf[["Total","Raw_Avail","Ind_Valid","Retained"]].plot(kind='bar', figsize=(10,6))
        plt.title("Inclusion Impact (All Rooms)")
        plt.savefig(os.path.join(self.run_dir, "inclusion_impact_all_rooms.png"))
        plt.close()
        
        # Global Cluster
        if all_retained:
            full_df = pd.concat(all_retained)
            res_df, k = self.perform_clustering(full_df, "global")
            if res_df is not None:
                names = self._get_state_names(res_df)
                self.generate_usage_plots(res_df, "global", state_names=names)
                
                # Check Substructure
                self.analyze_substructure(res_df, "global")

                # --- ADVANCED ANALYTICS ---
                # 1. Active State Detection
                # Heuristic: Max LightMean or SoundVar
                try:
                    stats = res_df.groupby("StateID").mean()
                    if "LightMean" in stats.columns:
                        active_id = stats["LightMean"].idxmax()
                    elif "SoundVar" in stats.columns:
                        active_id = stats["SoundVar"].idxmax()
                    else:
                        active_id = stats.index[-1]
                except:
                    active_id = 0
                    
                print(f"  [Global Analysis] Identified State {active_id} as 'Active'.")
                
                # 2. Global Plots
                self.plot_weekly_occupancy(res_df.copy(), active_id, "global")
                self.plot_comfort_zone(res_df.copy(), active_id, "global")
                self.detect_anomalies(res_df.copy(), "global")
                
                # Setup Map for Comparison
                active_state_map = {}
                
                # 3. Per-Room Plots (Slicing Global Result)
                # Index is (Node, Time)
                if isinstance(res_df.index, pd.MultiIndex):
                    nodes = res_df.index.get_level_values(0).unique()
                    for n in nodes:
                        print(f"    Generating insights for {n}...")
                        # xs returns Time index
                        sub = res_df.xs(n, level=0)
                        
                        # Determine Local Active State for this room slice
                        # If the global active_id is not present here, fallback to local max
                        local_active_id = active_id
                        present_states = sub["StateID"].unique()
                        
                        if active_id not in present_states:
                            # Recalculate 'active' relative to this room's behavior
                            try:
                                stats = sub.groupby("StateID").mean()
                                candidate_id = None
                                
                                if "LightMean" in stats.columns:
                                    candidate_id = stats["LightMean"].idxmax()
                                elif "SoundVar" in stats.columns:
                                    candidate_id = stats["SoundVar"].idxmax()
                                else:
                                    candidate_id = stats.index[-1]
                                
                                # Validate thresholds to avoid selecting "Weekend/Night" states
                                # Thresholds: Light > 60 Lux OR SoundVar > 0.6
                                if candidate_id is not None:
                                    c_light = stats.loc[candidate_id, "LightMean"] if "LightMean" in stats.columns else 0
                                    c_sound = stats.loc[candidate_id, "SoundVar"] if "SoundVar" in stats.columns else 0
                                    
                                    if c_light > 60 or c_sound > 0.6:
                                        local_active_id = candidate_id
                                    else:
                                        # Room is effectively inactive.
                                        pass
                            except:
                                local_active_id = active_id
                        
                        # Store in map
                        active_state_map[str(n)] = local_active_id
                        
                        self.plot_weekly_occupancy(sub.copy(), local_active_id, n, state_names=names)
                        self.plot_comfort_zone(sub.copy(), local_active_id, n, state_names=names)
                        self.detect_anomalies(sub.copy(), n, state_names=names)
                        self.plot_parameter_metrics(sub.copy(), n)
                        self.plot_combined_parameters(sub.copy(), n)
                        self.plot_correlations(sub.copy(), n)
                
                # 4. Activity Comparison (All Rooms)
                self.plot_activity_comparison(res_df.copy(), active_state_map, "global", state_names=names)

            else:
                print("  [Global] Clustering skipped (degenerate).")
            
        self.save_metadata(initial=False, stats=all_stats)

    def save_metadata(self, initial=False, stats=None):
        out = self.cfg.copy()
        for k, v in out.items():
            if isinstance(v, (pd.Timestamp, datetime)): out[k] = str(v)
            
        if stats: out["stats"] = stats
        out["meta"] = self.meta
        
        def enc(o): return o.item() if isinstance(o, np.generic) else str(o)
        
        # with open(os.path.join(self.run_dir, "run_config.json"), 'w') as f:
        #     json.dump(out, f, indent=4, default=enc)
        print("\n--- Run Config ---")
        print(json.dumps(out, indent=4, default=enc))
            
        if not initial:
            # pd.DataFrame(self.manifest).to_csv(os.path.join(self.run_dir, "manifest.csv"), index=False)
            pass
