import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
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

plt.style.use('ggplot')

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
        self.signals_map = {
            "Sound": "Sound_ave",
            "Light": "Light_ave"
        }
        
        self.df_clean = None
        self.master_index = None

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
        if "AirQuality_ave" in lbls: self.signals_map["AQ"] = "AirQuality_ave"
        
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
        if s: res["SoundVar"] = s[1]; raw_plots["Sound"] = s[0]
        if l: res["LightChg"] = l[1]; raw_plots["Light"] = l[0]
        
        if "Temp" in self.signals_map:
             t = _resample("Temp", "Delta", 15)
             if t: res["TempDelta"] = t[1]
             
        if "AQ" in self.signals_map:
             aq = _resample("AQ", "Delta", 15)
             if aq: res["AQDelta"] = aq[1]
             
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
            sdf.to_csv(os.path.join(self.run_dir, f"{name_prefix}_k_sweep.csv"), index=False)
            self.log_manifest(f"{name_prefix}_k_sweep.csv", "Sweep Data", name_prefix)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8,5))
            val = sdf[sdf["valid"]]; inv = sdf[~sdf["valid"]]
            if not val.empty: ax.errorbar(val["k"], val["sil_mean"], yerr=val["sil_std"], fmt='bo-', label='Valid')
            if not inv.empty: ax.errorbar(inv["k"], inv["sil_mean"], yerr=inv["sil_std"], fmt='rx', label='Rejected')
            
            if best_k:
                sel = sdf[sdf["k"]==best_k].iloc[0]
                ax.scatter(sel["k"], sel["sil_mean"], c='gold', marker='*', s=300, zorder=10, label=f"Sel K={best_k}")
                
            plt.title(f"K Selection ({name_prefix})\nAdjusted for Stability & Duration")
            plt.legend()
            plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_k_selection.png"))
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
                df.groupby("StateID").mean().to_csv(os.path.join(self.run_dir, f"{name_prefix}_centroids_raw.csv"))
                
                return df, best_k
            
        return None, 0

    def generate_usage_plots(self, tagged_df, name_prefix):
        # Share
        cts = tagged_df["StateID"].value_counts(normalize=True).sort_index()
        cts.plot(kind='bar', figsize=(6,4))
        plt.title(f"State Share ({name_prefix})")
        plt.ylabel("Freq")
        plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_state_share.png"))
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
             fig, ax = plt.subplots()
             im = ax.imshow(mat, cmap='Blues')
             ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns)
             ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index)
             plt.colorbar(im)
             plt.title(f"Transitions ({name_prefix})")
             plt.savefig(os.path.join(self.run_dir, f"{name_prefix}_transition.png"))
             plt.close()

    # --- PHASE A: SINGLE ROOM ---
    def run_phase_a(self, node):
        # 1. Prep
        raw, ind = self.process_room_data(node)
        
        # 2. Accounting
        stats, flags, retained_df = self.assess_inclusion(node, raw, ind)
        
        # 3. Plots
        # 3a. Inclusion Impact
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Total", "Raw", "Ind", "Retained"], 
               [stats["Total"], stats["Raw_Avail"], stats["Ind_Valid"], stats["Retained"]],
               color=['gray', 'orange', 'blue', 'green'])
        plt.title(f"Inclusion Impact: {node}")
        plt.savefig(os.path.join(self.run_dir, f"{node}_inclusion_impact.png"))
        plt.close()
        
        # 3b. Diagnostic
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax2 = ax1.twinx()
        if "Sound" in raw: ax1.plot(raw.index, raw["Sound"], 'b-', alpha=0.3, label="Sound")
        if "SoundVar" in ind: ax2.plot(ind.index, ind["SoundVar"], 'b--', label="SoundVar")
        if "Light" in raw: ax1.plot(raw.index, raw["Light"], 'g-', alpha=0.3, label="Light")
        if "LightChg" in ind: ax2.plot(ind.index, ind["LightChg"], 'g--', label="LightChg")
        plt.title(f"Diagnostic Timeline: {node}")
        plt.savefig(os.path.join(self.run_dir, f"{node}_diagnostic_timeline.png"))
        plt.close()
        
        # 3c. Completeness
        # 0=Miss, 1=Raw, 2=Ind, 3=Ret
        # Simple map: Retained=Green, Else=Grey/Red
        viz = pd.Series(0, index=flags.index)
        viz[flags["has_raw"]] = 1
        viz[flags["has_ind"]] = 2 # Overwrites raw
        viz[flags["is_retained"]] = 3
        
        # Plot strip
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.imshow(viz.values.reshape(1, -1), aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
        ax.set_yticks([]); ax.set_title(f"Completeness Strip: {node}")
        plt.savefig(os.path.join(self.run_dir, f"{node}_completeness.png"))
        plt.close()
        
        # 4. Cluster
        if not retained_df.empty:
            res_df, k = self.perform_clustering(retained_df, node)
            if res_df is not None:
                self.generate_usage_plots(res_df, node)
                res_df.to_csv(os.path.join(self.run_dir, f"{node}_labelled_retained.csv"))
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
            
            if not ret.empty:
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
                self.generate_usage_plots(res_df, "global")
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
        
        with open(os.path.join(self.run_dir, "run_config.json"), 'w') as f:
            json.dump(out, f, indent=4, default=enc)
            
        if not initial:
            pd.DataFrame(self.manifest).to_csv(os.path.join(self.run_dir, "manifest.csv"), index=False)
