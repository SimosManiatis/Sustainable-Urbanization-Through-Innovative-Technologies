import pandas as pd
import numpy as np
from .utils import print_info, print_warning, print_error, print_step, print_success

# Gap policy parameters
GAP_HOURS = 2
GAP_THRESHOLD = pd.Timedelta(hours=GAP_HOURS)

# Expected headers as per requirement
EXPECTED_COLUMNS = [
    "Id", "SendDate", "OrganizationId", "NodeId", "SensorType", 
    "Value", "SensorId", "MaterialId", "UnitId", 
    "IsCorrectValue", "CorrectValue", "AdditionalProps", 
    "GeneratedDate", "ReceivedDate", "TransmittedDate"
]

TARGET_NODES = ["TRP1", "TRP2", "TRP3", "TRP4", "TRP5"]

def process_dataset_globally(df, id_to_label, locations, target_dates=None):
    """
    Implements the strict data processing pipeline.
    target_dates: Optional list of datetime.date objects or strings (YYYY-MM-DD) to filter by.
    """
    total_steps = 15
    current_step = 1

    print_step(current_step, total_steps, "Enforcing headers and filtering columns")
    # Columns to check
    needed_cols = ["NodeId", "SensorId", "Value", "SendDate"]
    # If other cols exist, keep them?
    # We need SendDate, NodeId, SensorId, Value
    # Also correcting value cols
    if "CorrectValue" in df.columns: needed_cols.append("CorrectValue")
    if "IsCorrectValue" in df.columns: needed_cols.append("IsCorrectValue")
    
    # Ensure they exist
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].copy()
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows: {len(df)}\n")

    print_step(current_step, total_steps, "Creating Canonical Value (ValueUsed)")
    if "CorrectValue" in df.columns:
        # If CorrectValue is not null (and presumably valid), use it. Else Value.
        # Assuming NaN means "no correction"
        df["ValueUsed"] = df["CorrectValue"].fillna(df["Value"])
    else:
        df["ValueUsed"] = df["Value"]
    
    # Force numeric, coerce errors
    df["ValueUsed"] = pd.to_numeric(df["ValueUsed"], errors='coerce')
    
    # Drop raw value cols to save memory
    df = df.drop(columns=["Value", "CorrectValue", "IsCorrectValue"], errors="ignore")
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows: {len(df)}\n")

    print_step(current_step, total_steps, "Parsing and standardizing timestamps")
    # Parse SendDate
    df["SendDate"] = pd.to_datetime(df["SendDate"], utc=True, errors='coerce')
    # Drop invalid dates
    initial_len = len(df)
    df = df.dropna(subset=["SendDate"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print_info("Date Parsing", f"Dropped {dropped} rows with invalid SendDate")
        
    # Apply Target Date Filter if provided
    if target_dates:
        print_step(current_step, total_steps, f"Filtering to selected dates: {target_dates}")
        # Ensure target_dates are comparable (pd.Timestamp or date string)
        # Convert SendDate to YYYY-MM-DD string or use .dt.date
        # .dt.date returns objects, which are comparable to datetime.date
        # Let's handle the check robustly
        
        # Create mask
        # Assuming target_dates contains strings or date objects
        # We normalize everything to string YYYY-MM-DD for safety
        normalized_targets = set(str(d) for d in target_dates)
        
        # Filter
        df = df[df["SendDate"].dt.date.astype(str).isin(normalized_targets)]
        
        if df.empty:
            print_error("No data found for the selected dates!")
            return None
        else:
            print_info("Date Filter", f"Kept {len(df)} rows matching selected dates.")
            
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows: {len(df)}\n")

    print_step(current_step, total_steps, "Filtering to target dataset")
    # Filter NodeId
    df = df[df["NodeId"].isin(TARGET_NODES)]
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows after Node filter: {len(df)}\n")

    print_step(current_step, total_steps, "Mapping IDs to measurement labels")
    # Map SensorId -> Label
    # id_to_label uses string keys, ensure SensorId is string
    df["SensorId"] = df["SensorId"].astype(str)
    df["Measurement"] = df["SensorId"].map(id_to_label)
    
    # Filter out rows that didn't map to a known/wanted measurement
    before_map_len = len(df)
    df = df.dropna(subset=["Measurement"])
    if before_map_len - len(df) > 0:
        print_info("Measurement Mapping", f"Dropped {before_map_len - len(df)} rows with unknown SensorIds")
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows after Measurement map: {len(df)}\n")

    print_step(current_step, total_steps, "De-duplicating")
    # Drop duplicates per (NodeId, Measurement, SendDate)
    # Keep last (arbitrary choice, but standard)
    df = df.sort_values("SendDate") # Ensure sorted for keep='last' logic if needed
    df = df.drop_duplicates(subset=["NodeId", "Measurement", "SendDate"], keep='last')
    current_step += 1

    print_step(current_step, total_steps, "Sorting")
    df = df.sort_values(by=["NodeId", "Measurement", "SendDate"])
    current_step += 1

    print_step(current_step, total_steps, "Detecting and excluding 'bad days'")
    # Add Date column for grouping
    df["Date"] = df["SendDate"].dt.date

    # Function to find bad days for a group
    def get_bad_days(group):
        # Time diff
        deltas = group["SendDate"].diff()
        if deltas.max() > GAP_THRESHOLD:
            return group["Date"].unique() # Mark all days involved? 
            # Requirement: "mark the corresponding calendar day as bad"
            # If gap spans across days, it might mark strictly the day where gap STARTs or Ends?
            # Simpler: If a group has a big gap, see which days are covered by that gap?
            # Actually, standard approach: If ANY gap > 2h occurs in a day boundaries or effectively affects the day's integrity.
            # Let's iterate: compute mask of rows where delta > 2h.
            # Then identify which dates those rows fall into.
            
            # Vectorized approach:
            # Shifted dates
            # If SendDate[i] - SendDate[i-1] > 2h
            # Then Date[i] is suspect? Or all dates in the series? 
            # Prompt: "mark the corresponding calendar day as bad"
            # We'll take the set of dates present in the series where gaps happened.
            # But wait, if input is sorted, deltas align.
            
            # Simple heuristic: If max gap > cutoff, exclude the Day of the record where gap *ends*?
            # Or the whole node-measurement combination for that day is invalid?
            # "If any gap > GAP_HOURS occurs, mark the corresponding calendar day as bad"
            pass
        return []

    # Identifying bad days (NodeId, Measurement, Date)
    # We need to compute deltas per (NodeId, Measurement).
    # Then map back to Date.
    
    # Let's do it per group:
    # 1. Calculate TimeDiff per group
    df["TimeDiff"] = df.groupby(["NodeId", "Measurement"])["SendDate"].diff()
    
    # 2. Identify rows where TimeDiff > GAP_THRESHOLD
    bad_gap_mask = df["TimeDiff"] > GAP_THRESHOLD
    
    # 3. Get the dates corresponding to these gaps (and maybe the previous row's date too?)
    # Usually a gap means missing data between T1 and T2. 
    # If T2 - T1 > 2h. 
    # We should flag T1.Date and T2.Date as "Bad" because they contain a big hole.
    # Let's conservatively flag both the day of the gap start and gap end.
    
    bad_rows = df[bad_gap_mask]
    bad_nodes_dates = set()
    
    for idx, row in bad_rows.iterrows():
        # The gap ends at this row.
        # It started at 'prev_time' = row.SendDate - row.TimeDiff
        end_date = row["SendDate"].date()
        start_date = (row["SendDate"] - row["TimeDiff"]).date()
        
        # Add (NodeId, Date) to bad set
        # Prompt option: "per-node (drop that day only for that node) or global"
        # I pick: Global per node (Drop that day for that Node, for ALL measurements of that node? 
        # or just that measurement? Prompt: "convert to exclusion set... per-node (drop that day only for that node)"
        # implies for that Node (all sensors). "so cross-room comparisons stay fair" implies carefulness.
        # "drop that day for all nodes" is 'global'.
        # "drop that day only for that node" is 'per-node'.
        # I will choose "Per-Node Global" -> If TRP1 has a gap in Temp, drop TRP1 data for that day entirely?
        # Re-reading: "For each (NodeId, measurement_label) time-series ... Gap occurs -> mark ... as bad"
        # "Convert to exclusion set: Either per-node... or global"
        # I will choose PER-NODE exclusion. If TRP1 has a gap in ANY sensor, drop TRP1 for that day.
        # This is safer for multi-variate analysis of the room.
        
        bad_nodes_dates.add((row["NodeId"], end_date))
        bad_nodes_dates.add((row["NodeId"], start_date))
        # Handle range if gap > 24h?
        # If gap is huge (days), all intermediate days are missing data anyway (no rows).
        # So removing rows is trivial (they don't exist).
        # But we must remove the partial days at start/end.
    
    # Filter out bad (Node, Date) pairs
    # Create a column "Flag"
    df["DayKey"] = list(zip(df["NodeId"], df["Date"]))
    df = df[~df["DayKey"].isin(bad_nodes_dates)]
    
    df = df.drop(columns=["TimeDiff", "Date", "DayKey"])
    current_step += 1

    print_step(current_step, total_steps, "Resampling to 1-minute resolution")
    # Resample per (NodeId, Measurement)
    # We need to set index to SendDate
    df = df.set_index("SendDate")
    
    # This creates a MultiIndex (NodeId, Measurement, SendDate)
    
    def resample_group(g):
        # Resample to 1T, taking mean if multiple points fall in same minute
        # (Since we deduplicated, likely 0 or 1 point per minute usually)
        if g.empty: return g
        
        # We want a continuous range from min to max
        # resample() by default creates bins.
        # Select numeric columns only for mean
        numeric_cols = ["ValueUsed"]
        # Force float
        g_numeric = g[numeric_cols].astype(float)
        
        g_res = g_numeric.resample('1T').mean()
        
        # Forward fill / Interpolate small gaps
        if "ValueUsed" in g_res.columns:
            g_res["ValueUsed"] = g_res["ValueUsed"].ffill(limit=5)
            # Use appropriate interpolation method for time series
            # interpolate() on DataFrame with DatetimeIndex defaults to 'linear' usually, 'time' is better
            g_res["ValueUsed"] = g_res["ValueUsed"].interpolate(method='time', limit=15)
        
        return g_res

    grouped = df.groupby(["NodeId", "Measurement"])
    resampled_df = grouped.apply(resample_group)
    
    # Drop where ValueUsed is still NaN (long gaps)
    resampled_df = resampled_df.dropna(subset=["ValueUsed"])
    
    # Reset index to flatten
    resampled_df = resampled_df.reset_index()
    # If groupby keys were added to index, handle that.
    # Groupby apply usually adds keys. NodeId, Measurement are in cols now too.
    # The index might be (NodeId, Measurement, SendDate).
    if "level_0" in resampled_df.columns: resampled_df = resampled_df.drop(columns=["level_0", "level_1"], errors='ignore')
    # Actually check index names
    current_step += 1
    
    print_step(current_step, total_steps, "Feature engineering on 1-minute series")
    # Set index again for rolling
    resampled_df = resampled_df.set_index("SendDate").sort_index()
    
    # Group again
    grouped_1min = resampled_df.groupby(["NodeId", "Measurement"])
    
    resampled_df["std_5min"] = grouped_1min["ValueUsed"].transform(lambda x: x.rolling('5T').std())
    resampled_df["delta_15min"] = grouped_1min["ValueUsed"].transform(lambda x: x.diff(periods=15)) # 15 samples = 15 mins
    
    resampled_df = resampled_df.reset_index()
    current_step += 1
    
    print_step(current_step, total_steps, "Aggregating to 15-minute bins")
    # Resample to 15T
    # We want: Mean, Min, Max, Coverage, Aggregated Features
    
    # Define aggregation dict
    agg_funcs = {
        "ValueUsed": ["mean", "min", "max", "count"],
        "std_5min": "mean",
        "delta_15min": "mean"
    }
    
    # We aggregate by (NodeId, Measurement) and Time Grouper 15T
    # Need to group by columns + resample
    # But resample is a time-grouper.
    
    # Reset index to use 'on=' or just level
    
    agg_df = resampled_df.groupby([
        "NodeId", 
        "Measurement", 
        pd.Grouper(key="SendDate", freq='15T')
    ]).agg(agg_funcs)
    
    # Flatten columns
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Calculate coverage
    agg_df["coverage"] = agg_df["ValueUsed_count"] / 15.0
    
    # Reset index
    agg_df = agg_df.reset_index()
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows after Aggregation: {len(agg_df)}\n")
    
    print_step(current_step, total_steps, "Dropping low-quality bins")
    # Drop < 80% coverage
    agg_df = agg_df[agg_df["coverage"] >= 0.8]
    # Drop missing mean
    agg_df = agg_df.dropna(subset=["ValueUsed_mean"])
    current_step += 1
    with open("debug_log.txt", "a") as f: f.write(f"DEBUG: Rows after Quality Gate: {len(agg_df)}\n")
    
    print_step(current_step, total_steps, "Adding time features")
    agg_df["hour"] = agg_df["SendDate"].dt.hour
    agg_df["day_of_week"] = agg_df["SendDate"].dt.dayofweek
    agg_df["is_weekend"] = agg_df["day_of_week"] >= 5
    current_step += 1
    
    print_step(current_step, total_steps, "Normalizing (Z-scores per measurement)")
    # Normalize 'ValueUsed_mean' and engineered features per MEASUREMENT (across all nodes? Prompt: "per measurement_label (and optionally per NodeId) — not globally across all sensor types.")
    # Usually "per measurement type" is correct (Temperature is comparable across rooms).
    
    target_cols_to_norm = ["ValueUsed_mean", "std_5min_mean", "delta_15min_mean"]
    
    for col in target_cols_to_norm:
        if col in agg_df.columns:
            # Z-score: (x - mean) / std
            # Group by Measurement
            agg_df[f"norm_{col}"] = agg_df.groupby("Measurement")[col].transform(lambda x: (x - x.mean()) / x.std())
    
    current_step += 1
    
    print_step(current_step, total_steps, "Finalizing Output Shape (Wide Format)")
    # Pivot: One row per (NodeId, SendDate) -> Cols = Measurement_Feature
    
    # We need to combine Measurement + FeatureName
    # Pivot columns: Measurement
    # Pivot index: NodeId, SendDate
    # Pivot values: all the data columns (ValueUsed_mean, norm_*, etc)
    
    # It's cleaner to melt or manually construct columns.
    # Pivot table handles this multiple values.
    
    value_vars = [c for c in agg_df.columns if c not in ["NodeId", "Measurement", "SendDate", "hour", "day_of_week", "is_weekend"]]
    
    wide_df = agg_df.pivot_table(
        index=["NodeId", "SendDate", "hour", "day_of_week", "is_weekend"],
        columns="Measurement",
        values=value_vars
    )
    
    # Flatten MultiIndex columns
    # New cols: {Measurement}_{Feature}
    wide_df.columns = [f"{m}_{f}" for f, m in wide_df.columns]
    
    wide_df = wide_df.reset_index()
    
    print_success("Pipeline Processing Complete")
    return wide_df
