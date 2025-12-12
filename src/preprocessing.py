import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from .utils import print_header, print_step, print_info, print_success, print_warning, Colors

# Try importing sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def process_dataset_globally(df, id_to_label, locations_map):
    """
    Executes the 9-step Data Preparation and Preprocessing Plan GLOBALLY.
    """
    print_header(f"GLOBAL DATA PREPROCESSING PIPELINE ({datetime.now().strftime('%H:%M:%S')})")

    # --- Pre-Processing Stats ---
    rows_before = len(df)
    cols_before = len(df.columns)

    # --- Step 1: Consolidate Data ---
    print_step(1, 9, "Consolidating Data...")
    
    keep_cols = ['Id', 'SendDate', 'NodeId', 'SensorType', 'Value', 'SensorId', 'MaterialId', 'UnitId', 'GeneratedDate', 'ReceivedDate', 'TransmittedDate']
    actual_cols = [c for c in keep_cols if c in df.columns]
    
    processed = df[actual_cols].copy()
    if 'NodeId' in processed.columns:
         processed['Location'] = processed['NodeId'].map(locations_map).fillna('Unknown')
    else:
         processed['Location'] = 'Unknown'
         
    print_info("Retained Columns", len(processed.columns))
    print_info("Initial Rows", len(processed))
    
    unique_nodes = processed['NodeId'].unique() if 'NodeId' in processed.columns else 'N/A'
    print_info("Unique Nodes", unique_nodes)

    # --- Step 2: Sorting and Time Alignment ---
    print_step(2, 9, "Sorting and Time Alignment...")
    
    processed['SendDate'] = pd.to_datetime(processed['SendDate'], errors='coerce')
    processed = processed.dropna(subset=['SendDate'])
    processed = processed.sort_values('SendDate').reset_index(drop=True)
    
    if not processed.empty:
        print_info("Time Range", f"{processed['SendDate'].min()} to {processed['SendDate'].max()}")
        
    print_success("Global Sort completed.")
    
    processed['Value'] = pd.to_numeric(processed['Value'], errors='coerce')

    # --- Step 2.5: Gap Detection (Exclude Days with >2h Gaps) ---
    print_step("2.5", 9, "Gap Detection (Filtering days with >2h gaps)...")
    
    # Ensure Sort for Gap Calculation
    gap_df = processed.sort_values(['SensorId', 'SendDate'])
    
    # Calculate Gaps
    # Group by Sensor to avoid cross-sensor gap calculation
    gap_df['TimeDiff'] = gap_df.groupby('SensorId')['SendDate'].diff().dt.total_seconds() / 60.0 # in minutes
    
    # Threshold: 2 hours = 120 minutes
    large_gaps = gap_df[gap_df['TimeDiff'] > 120]
    
    if not large_gaps.empty:
        print_warning(f"Found {len(large_gaps)} large gaps (>120min).")
        
        # --- Strict Logic: Flag large_gap_day column ---
        large_gaps['DateStr'] = large_gaps['SendDate'].dt.date.astype(str)
        bad_day_keys = set(zip(large_gaps['SensorId'], large_gaps['DateStr']))
        
        # Create key on main df
        processed['DateStr'] = processed['SendDate'].dt.date.astype(str)
        processed['SensorDateKey'] = list(zip(processed['SensorId'], processed['DateStr']))
        
        # Flag
        processed['large_gap_day'] = processed['SensorDateKey'].apply(lambda k: 1 if k in bad_day_keys else 0)
        
        # Exclude
        initial_len = len(processed)
        processed = processed[processed['large_gap_day'] == 0]
        dropped_count = initial_len - len(processed)
        
        print_info("Flagged Days (large_gap_day=1)", len(bad_day_keys))
        
        # Cleanup
        processed = processed.drop(columns=['DateStr', 'SensorDateKey', 'large_gap_day'])
        
        print_warning(f"Removed {dropped_count} rows where large_gap_day = 1.")
    else:
        print_success("No large gaps found (>2h). Data is continuous.")

    # --- Step 3 & 8: Resampling & Filling ---
    print_step("3 & 8", 9, "Handling Time Alignment (1-min) and Aggregation (15-min)...")
    print_info("Action", "Resampling to 1-min resolution per sensor...")
    
    processed.set_index('SendDate', inplace=True)
    
    group_cols = ['SensorId', 'NodeId', 'Location', 'SensorType']
    group_cols = [c for c in group_cols if c in processed.columns]
    
    resampled_1min = processed.groupby(group_cols)['Value'].resample('1min').mean()
    df_1min = resampled_1min.to_frame()
    
    print_info("Action", "Imputing missing values (FFill limit=30, Interpolate limit=60)...")
    df_1min = df_1min.reset_index()
    
    def fill_group(g):
        g = g.set_index('SendDate').sort_index()
        g['Value'] = g['Value'].ffill(limit=30)
        g['Value'] = g['Value'].interpolate(method='time', limit=60)
        return g

    # Explicit column selection to avoid FutureWarning
    df_1min = df_1min.groupby(group_cols)[['SendDate', 'Value']].apply(fill_group).reset_index()
    
    print_info("1-min Data Points", len(df_1min))
    
    # --- Step 5: Feature Engineering ---
    print_step(5, 9, "Feature Engineering (Rolling Stats)...")
    
    df_1min = df_1min.set_index('SendDate').sort_index()
    
    def calculate_features(g):
        g['std_1min'] = g['Value'].rolling('1min').std()
        g['std_5min'] = g['Value'].rolling('5min').std()
        g['delta_2min'] = g['Value'].diff(periods=2)
        g['delta_15min'] = g['Value'].diff(periods=15)
        return g

    df_features = df_1min.groupby(group_cols)[['Value']].apply(calculate_features)
    df_features = df_features.reset_index()
    
    # --- Step 4: Time Features ---
    print_step(4, 9, "Extracting Time Features...")
    df_features['Hour'] = df_features['SendDate'].dt.hour
    df_features['DayOfWeek'] = df_features['SendDate'].dt.dayofweek
    df_features['IsWeekend'] = df_features['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    feat_list = [c for c in df_features.columns if 'std' in c or 'delta' in c]
    print_info("Generated Features", f"{feat_list}")
    
    # --- Step 8: Aggregation to 15-min ---
    print_step(8, 9, "Downsampling to 15-minute intervals...")
    print_info("Input Shape", df_features.shape)
    
    df_features = df_features.reset_index()
    
    agg_funcs = {
        'Value': ['mean', 'min', 'max'],
        'std_1min': 'mean',
        'std_5min': 'mean',
        'delta_2min': 'mean',
        'delta_15min': 'mean',
        'Hour': 'first',
        'DayOfWeek': 'first',
        'IsWeekend': 'first'
    }
    
    grouper = group_cols + [pd.Grouper(key='SendDate', freq='15min')]
    df_15min = df_features.groupby(grouper).agg(agg_funcs)
    
    df_15min.columns = ['_'.join(col).strip() for col in df_15min.columns.values]
    df_15min = df_15min.reset_index()
    df_15min.rename(columns={'Value_mean': 'Value'}, inplace=True)
    
    print_info("Aggregated Shape", df_15min.shape)
    print_info("Sample Head", "\n" + df_15min[['SendDate', 'Value']].head(3).to_string(index=False))

    # --- Step 6: Normalization ---
    print_step(6, 9, "Normalization (Z-Score)...")
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        numeric_cols = ['Value', 'Value_min', 'Value_max', 'std_1min_mean', 'std_5min_mean', 'delta_2min_mean', 'delta_15min_mean']
        numeric_cols = [c for c in numeric_cols if c in df_15min.columns]
        
        print_info("Selected Cols", numeric_cols)
        
        fill_cols = [c for c in numeric_cols if c != 'Value']
        initial_nans = df_15min[fill_cols].isna().sum().sum()
        df_15min[fill_cols] = df_15min[fill_cols].fillna(0)
        
        if initial_nans > 0:
            print_info("NaNs Filled (0)", initial_nans)
        
        pre_drop_len = len(df_15min)
        df_15min = df_15min.dropna(subset=['Value'])
        dropped_rows = pre_drop_len - len(df_15min)
        
        if dropped_rows > 0:
            print_warning(f"Dropped {dropped_rows} rows with missing Value.")
        
        if not df_15min.empty:
            print_info("Pre-Norm Stats", f"Mean={df_15min['Value'].mean():.2f}, Std={df_15min['Value'].std():.2f}")
            
            scaled = scaler.fit_transform(df_15min[numeric_cols])
            for i, col in enumerate(numeric_cols):
                 df_15min[f"norm_{col}"] = scaled[:, i]
            
            print_success("Added 'norm_*' columns.")
            print_info("Sample Norm", "\n" + df_15min[['SendDate', 'Value', 'norm_Value']].head(3).to_string(index=False))
        else:
            print_warning("Data empty, skipping normalization.")
    
    print_header("PREPROCESSING COMPLETE")
    
    # --- Report ---
    print(f"\n{Colors.HEADER}--- Processing Report ---{Colors.ENDC}")
    print(f"Original Data: {rows_before} rows, {cols_before} columns")
    print(f"Processed Data: {len(df_15min)} rows, {len(df_15min.columns)} columns")
    print(f"Resolution: 15 Minutes")
    print(f"Time Span: {df_15min['SendDate'].min()} -> {df_15min['SendDate'].max()}")
    print("-" * 30 + "\n")
    
    return df_15min
