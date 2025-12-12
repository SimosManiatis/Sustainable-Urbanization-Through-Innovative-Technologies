
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Try importing sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Advanced analysis disabled.")

RAW_DATA_DIR = r'D:\T.U\raw_data'
MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensor_mapping.json')

def load_mapping():
    """Loads the sensor mapping from JSON.
       Returns:
         measurements_map: dict of {Label: UUID}
         uuid_to_label: dict of {UUID: Label}
         locations_map: dict of {SensorName: LocationName}
    """
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                data = json.load(f)
                
                # Handle new structure
                if "measurements" in data:
                    mapping = data.get("measurements", {})
                    locations = data.get("locations", {})
                else:
                    # Legacy structure (fallback)
                    mapping = data
                    locations = {}
                    
                id_to_label = {v: k for k, v in mapping.items()}
                return mapping, id_to_label, locations
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            return {}, {}, {}
    return {}, {}, {}

def find_latest_csv(directory):
    """Finds the latest CSV file in the given directory."""
    list_of_files = glob.glob(os.path.join(directory, '*.csv'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def get_unique_sensors(df, sensor_name_col):
    """Returns sorted list of unique sensor names."""
    if sensor_name_col and sensor_name_col in df.columns:
        return sorted(df[sensor_name_col].astype(str).unique())
    return []

def filter_by_time(df, time_option):
    """Filters dataframe based on time option."""
    if time_option == 'all':
        return df
    
    # Ensure SendDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['SendDate']):
        df['SendDate'] = pd.to_datetime(df['SendDate'], errors='coerce')
        
    max_date = df['SendDate'].max()
    if pd.isna(max_date):
        return df
        
    start_date = None
    
    if time_option == '24h':
        start_date = max_date - timedelta(hours=24)
    elif time_option == '7d':
        start_date = max_date - timedelta(days=7)
    elif time_option == 'custom':
        print(f"\nData range available: {df['SendDate'].min().date()} to {max_date.date()}")
        print("Please enter dates in YYYY-MM-DD format.")
        
        while True:
            start_str = input("Start Date (e.g. 2025-12-01): ").strip()
            if not start_str:
                print("Start date is required.")
                continue
            try:
                start_date = pd.to_datetime(start_str)
                break
            except ValueError:
                print("Invalid format. Try again.")
                
        end_str = input("End Date (Leave empty for latest): ").strip()
        if end_str:
            try:
                end_date_val = pd.to_datetime(end_str)
                end_date_val = end_date_val + timedelta(days=1) - timedelta(microseconds=1)
                return df[(df['SendDate'] >= start_date) & (df['SendDate'] <= end_date_val)]
            except ValueError:
                print("Invalid end date format. Using max date.")
                
        return df[df['SendDate'] >= start_date]

    if start_date:
        return df[df['SendDate'] >= start_date]
        
    return df


def perform_clustering(df, sensor_name, sensor_name_col, id_to_label, time_option, n_clusters=4, location_name=""):
    """Performs K-Means clustering on the sensor data with Advanced Feature Engineering."""
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn is required for this feature.")
        return

    print(f"\nPreparing Clustering Analysis for '{sensor_name}'...")
    
    # Filter by Sensor Name
    mask = (df[sensor_name_col] == sensor_name)
    subset = df[mask].copy()
    
    if subset.empty:
        print("No data found for this sensor.")
        return

    # Filter Time
    subset = filter_by_time(subset, time_option)
    if subset.empty:
        print("No data in selected time range.")
        return

    print("Structuring data (Pivoting to 1-min frequency for Feature Engineering)...")
    subset['SendDate'] = pd.to_datetime(subset['SendDate'])
    subset['Value'] = pd.to_numeric(subset['Value'], errors='coerce')
    
    # Pivot: Index=Time, Columns=SensorId, Values=Value
    pivoted = subset.pivot_table(index='SendDate', columns='SensorId', values='Value', aggfunc='mean')
    
    # Rename columns using mapping
    new_cols = []
    for uid in pivoted.columns:
        label = id_to_label.get(str(uid), f"Unknown_{uid}")
        new_cols.append(label)
    pivoted.columns = new_cols
    
    # Resample to 1-minute to support high-res rolling features
    pivoted = pivoted.resample('1min').mean()
    # Forward fill limited to fill small gaps, but don't fill giant gaps yet
    pivoted = pivoted.ffill(limit=30) 

    # --- Feature Engineering ---
    print("Calculating Advanced Features (Rolling STD, Temporal Deltas)...")
    
    features_df = pivoted.copy()
    
    # Identification of parameter types
    cols = pivoted.columns
    sound_cols = [c for c in cols if "Sound" in c]
    env_cols = [c for c in cols if any(x in c for x in ["Air", "Hum", "Temp", "Light", "Press"])]
    
    new_features = []

    # 1. Sound Variability (Activity State)
    # 2min window min_periods=2 to get simple variance if 1min is too tight
    for c in sound_cols:
        f_1min = features_df[c].rolling('2min', min_periods=2).std()
        f_2_5min = features_df[c].rolling('3min', min_periods=2).std()
        f_5min = features_df[c].rolling('5min', min_periods=3).std()
        
        features_df[f"{c}_std_1min"] = f_1min
        features_df[f"{c}_std_2.5min"] = f_2_5min
        features_df[f"{c}_std_5min"] = f_5min
        new_features.extend([f"{c}_std_1min", f"{c}_std_2.5min", f"{c}_std_5min"])
    
    # 2. Environmental Deltas (Changes)
    for c in env_cols:
        d_2min = features_df[c].diff(periods=2)
        d_15min = features_df[c].diff(periods=15)
        
        features_df[f"{c}_delta_2min"] = d_2min
        features_df[f"{c}_delta_15min"] = d_15min
        new_features.extend([f"{c}_delta_2min", f"{c}_delta_15min"])
        
    # --- Clustering Prep ---
    # Drop NaNs created by rolling/shifting
    final_df = features_df.dropna()
    
    if final_df.empty:
        print("Not enough data points after feature engineering (need > 15 mins contiguous).")
        return

    print(f"Clustering on {len(final_df)} samples using features...")
    # Select columns to cluster on: All numeric
    X = final_df.select_dtypes(include='number')
    
    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    
    # KMeans
    print(f"Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    final_df['Cluster'] = clusters
    

    # --- Visualization ---
    
    centers = kmeans.cluster_centers_
    feature_names = X.columns
    n_features = len(feature_names)
    
    # Check if index is datetime
    if not isinstance(final_df.index, pd.DatetimeIndex):
         final_df.index = pd.to_datetime(final_df.index)

    start_date_str = final_df.index.min().strftime('%Y-%m-%d')
    end_date_str = final_df.index.max().strftime('%Y-%m-%d')
    date_range_str = f"{start_date_str} / {end_date_str}"
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
    

    # Plot 1: Cluster Signatures (Heatmap)
    ax1 = fig.add_subplot(gs[0])
    
    im = ax1.imshow(centers, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax1.set_yticks(range(n_clusters))
    ax1.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
    
    # Minimal Style: No X-Axis Labels
    ax1.set_xticks([]) 
    
    # Styling Plot 1
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_title('') 
    plt.colorbar(im, ax=ax1, label='Z-Score (Standardized)')
    
    # Custom Title Block for Plot 1
    ax1.text(0.0, 1.15, f"{sensor_name} - Cluster Signatures", transform=ax1.transAxes, fontsize=14, fontweight='bold', color='black', ha='left')
    ax1.text(0.0, 1.08, "Feature Z-Scores per Activity State", transform=ax1.transAxes, fontsize=10, color='gray', ha='left')

    # Plot 2: Time Series
    ax2 = fig.add_subplot(gs[1])
    
    # Scatter plot
    sc = ax2.scatter(final_df.index, final_df['Cluster'], c=final_df['Cluster'], cmap='viridis', s=50, alpha=0.8)
    ax2.set_yticks(range(n_clusters))
    ax2.set_ylabel('Activity State (Cluster)')
    
    # Styling Plot 2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('#E0E0E0')
    ax2.grid(False) 
    ax2.yaxis.grid(True, color='#F0F0F0', linestyle='--')
    ax2.set_axisbelow(True)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, color='gray')
    ax2.tick_params(axis='x', colors='gray')
    ax2.tick_params(axis='y', colors='gray')

    # Custom Title Block for Plot 2
    full_title = f"{sensor_name} - {location_name}" if location_name else sensor_name
    
    ax2.text(0.0, 1.15, full_title, transform=ax2.transAxes, fontsize=16, fontweight='bold', color='black', ha='left')
    ax2.text(0.0, 1.08, "Derived Activity States Over Time", transform=ax2.transAxes, fontsize=12, color='gray', ha='left')
    ax2.text(0.0, 1.02, date_range_str, transform=ax2.transAxes, fontsize=10, color='gray', ha='left')
    
    plt.tight_layout()
    plt.show()
    print("Analysis complete.")

def plot_sensor_data(df, sensor_name, sensor_name_col, target_label, target_id, time_option, period_option, location_name=""):
    """Plots Value vs SendDate for the specific sensor and data type."""
    print(f"\nPreparing to plot '{target_label}' for '{sensor_name}'...")
    
    # Filter by Sensor Name
    mask = (df[sensor_name_col] == sensor_name)
    
    # Filter by Sensor ID
    mask &= (df['SensorId'].astype(str) == str(target_id))
    
    subset = df[mask].copy()
    
    if subset.empty:
        print("No data found for this combination.")
        return

    # Process Dates
    print("Processing dates...")
    subset['SendDate'] = pd.to_datetime(subset['SendDate'], errors='coerce')
    subset = subset.dropna(subset=['SendDate'])
    
    # STRICT SORTING
    subset = subset.sort_values('SendDate')
    
    # Filter by Time
    subset = filter_by_time(subset, time_option)
    
    if subset.empty:
        print("No data found for the selected time range.")
        return

    # RESAMPLING / PERIOD
    if period_option and period_option != 'none':
        print(f"Applying period resampling: {period_option}...")
        
        subset = subset.set_index('SendDate')
        
        # Ensure Value is numeric
        subset['Value'] = pd.to_numeric(subset['Value'], errors='coerce')
        
        # Resample ONLY the Value column
        resampled = subset['Value'].resample(period_option).mean()
        
        # IMPORTANT: Do NOT dropna() here if we want gaps to appear as breaks.
        subset = resampled.reset_index()

    # Visualize
    print(f"Plotting {len(subset)} data points...")
    
    # Styling Constants
    color_line = '#000080'
    color_fill = '#ADD8E6'

    # Check if empty (all NaNs?)
    if subset['Value'].dropna().empty:
        print("No valid data points to plot after processing.")
        return

    start_date_str = subset['SendDate'].min().strftime('%Y-%m-%d')
    end_date_str = subset['SendDate'].max().strftime('%Y-%m-%d')
    date_range_str = f"{start_date_str} / {end_date_str}"

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Main Plot
    ax.plot(subset['SendDate'], subset['Value'], color=color_line, linewidth=2)
    ax.fill_between(subset['SendDate'], subset['Value'], color=color_fill, alpha=0.3)
    
    # 2. Remove Axes Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#E0E0E0') 
    
    # 3. Y-Axis Cleanup
    ax.yaxis.set_visible(False)
    ax.grid(False)

    # 4. Custom Title Block (Top Left)
    title_text = f"{sensor_name} - {location_name}" if location_name else sensor_name
    ax.text(0.0, 1.10, title_text, transform=ax.transAxes, fontsize=16, fontweight='bold', color='black', ha='left')
    ax.text(0.0, 1.05, target_label, transform=ax.transAxes, fontsize=12, color='gray', ha='left')
    ax.text(0.0, 1.00, date_range_str, transform=ax.transAxes, fontsize=10, color='gray', ha='left')
    
    # 5. X-Axis Cleanup
    ax.tick_params(axis='x', colors='gray')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    
    plt.show()
    print("Plot display request sent.")

def process_dataset_globally(df, id_to_label, locations_map):
    """
    Executes the 9-step Data Preparation and Preprocessing Plan GLOBALLY on the entire dataset.
    Returns the processed DataFrame (15-min aggregated) suitable for analysis.
    """
    print("="*60)
    print(f"STARTING GLOBAL DATA PREPROCESSING PIPELINE ({datetime.now().strftime('%H:%M:%S')})")
    print("="*60)

    # --- Step 1: Consolidate Data ---
    print("[Step 1/9] Consolidating Data...")
    
    # Columns to keep
    keep_cols = ['Id', 'SendDate', 'NodeId', 'SensorType', 'Value', 'SensorId', 'MaterialId', 'UnitId', 'GeneratedDate', 'ReceivedDate', 'TransmittedDate']
    # Filter columns that exist
    actual_cols = [c for c in keep_cols if c in df.columns]
    
    # Keep extra metadata like 'Room' if it exists or map it now?
    # Mapping Sensors to Locations: Add a 'Location' column for convenience
    # NodeId is the Label usually (e.g. TRP1)
    # Check if we have NodeId
    
    processed = df[actual_cols].copy()
    if 'NodeId' in processed.columns:
         processed['Location'] = processed['NodeId'].map(locations_map).fillna('Unknown')
    else:
         processed['Location'] = 'Unknown'
         
    print(f"  -> Retained {len(processed.columns)} relevant columns.")
    print(f"  -> Initial Rows: {len(processed)}")
    print(f"  -> Unique Nodes found: {processed['NodeId'].unique() if 'NodeId' in processed.columns else 'N/A'}")

    # Remove Non-Relevant Columns
    drop_candidates = ['IsCorrectValue', 'CorrectValue', 'AdditionalProps']
    # They are already dropped by whitelist above.

    # --- Step 2: Sorting and Time Alignment ---
    print("[Step 2/9] Sorting and Time Alignment...")
    
    # Ensure Chronological Order
    processed['SendDate'] = pd.to_datetime(processed['SendDate'], errors='coerce')
    processed = processed.dropna(subset=['SendDate'])
    processed = processed.sort_values('SendDate').reset_index(drop=True)
    if not processed.empty:
        print(f"  -> Time Range: {processed['SendDate'].min()} to {processed['SendDate'].max()}")
    print("  -> Global Sort by SendDate completed.")
    
    # Ensure Value is numeric
    processed['Value'] = pd.to_numeric(processed['Value'], errors='coerce')

    # --- Step 3: Handling Missing Data & Step 8: Aggregation (Combined for Efficiency) ---
    print("[Step 3 & 8] Handling Time Alignment (1-min) and Aggregation (15-min)...")
    
    # We need to process PER SENSOR to strictly align time without mixing data
    # Strategy: GroupBy SensorId -> Resample -> Forward Fill -> Feature Engineering -> Resample to 15min
    
    # 1. Resample to 1-min (Regularize)
    print("  -> Resampling to 1-min resolution per sensor (Handling irregularities)...")
    
    # Define aggregation rules for 1-min resampling
    # For Value, we take mean. For metadata (NodeId, Location), we take 'first'.
    # We can use a trick: Set index to SendDate, then GroupBy SensorId, then Resample
    
    processed.set_index('SendDate', inplace=True)
    
    # Define aggregations
    # We primarily care about 'Value'. Metadata should be preserved.
    # Grouping by SensorId AND other metadata to keep them
    agg_dict = {'Value': 'mean'}
    
    # Group by SensorId. We assume SensorType, NodeId, Location are constant per SensorId.
    # If not, we might lose them or need to grouping by them too.
    # Let's group by [SensorId, NodeId, Location, SensorType] to be safe.
    group_cols = ['SensorId', 'NodeId', 'Location', 'SensorType']
    # Filter to only cols that exist
    group_cols = [c for c in group_cols if c in processed.columns]
    
    # High-Freq Resample (1min)
    resampled_1min = processed.groupby(group_cols)['Value'].resample('1min').mean()
    
    # Unstack or keep as Series? 
    # resampled_1min is a Series with MultiIndex (SensorId... , SendDate)
    # Convert to DataFrame to add features
    df_1min = resampled_1min.to_frame()
    
    # Forward Fill (Step 3) - fill gaps up to 30 mins *within each sensor group*
    # Since we have a MultiIndex with SensorId levels, we can groupby level inputs or simple ffill if sorted?
    # Resample creates a sorted index per group. 
    # But strictly, we should apply ffill per sensor group to avoid bleeding between sensors if we flattened.
    # Here the index is MultiIndex. ffill() on DataFrame usually fills generally.
    # We should iterate or use transform. 
    # Actually, grouped resample result preserves separation.
    
    print("  -> Imputing missing values (FFill limit=30, Interpolate limit=60)...")
    # Apply ffill per group level 0 (SensorId combined)
    # This is complex on MultiIndex. 
    # Simplification: Reset index, then groupby transform.
    df_1min = df_1min.reset_index()
    
    # Define filler function
    def fill_group(g):
        g = g.set_index('SendDate').sort_index()
        g['Value'] = g['Value'].ffill(limit=30)
        g['Value'] = g['Value'].interpolate(method='time', limit=60)
        return g

    # Apply filling
    # Group by the entity columns
    # Fix: Select columns to avoid FutureWarnings and duplication
    # We select 'SendDate' (needed for index) and 'Value' (needed for filling)
    df_1min = df_1min.groupby(group_cols)[['SendDate', 'Value']].apply(fill_group).reset_index()
    # Note: apply might duplicate grouping keys in index, reset_index normally handles it.
    # Check simple reset:
    # After apply, structure might be: OuterIndex... -> SendDate, Value
    # Let's inspect columns after reset.
    # Usually: level_0, level_1... SendDate, Value.
    # We'll clean up columns later.
    
    print(f"  -> Data points after 1-min resampling: {len(df_1min)}")
    
    # --- Step 5: Feature Engineering (on 1-min data) ---
    print("[Step 5/9] Feature Engineering (Rolling Stats)...")
    
    # We need to distinguish Sound vs Environmental for specific features
    # But here we are iterating generally.
    # We can detect type from 'SensorType' or 'NodeId' (e.g. TRP1)
    # But 'SensorType' might be 'Sound' or 'Temperature'.
    
    # Let's assume we proceed with generic features for ALL sensors, 
    # OR we check SensorType column if available.
    
    # We need to set index to SendDate for rolling
    df_1min = df_1min.set_index('SendDate').sort_index()
    
    # Define Rolling Function
    def calculate_features(g):
        # Rolling STD (Sound variability proxy, but valid for any fluctuating signal)
        g['std_1min'] = g['Value'].rolling('1min').std()
        g['std_5min'] = g['Value'].rolling('5min').std()
        
        # Deltas
        g['delta_2min'] = g['Value'].diff(periods=2)
        g['delta_15min'] = g['Value'].diff(periods=15)
        
        return g

    # Apply per group
    # Note: df_1min index is SendDate. We groupby columns.
    # We must explicitly select columns or use include_groups=False (future pandas)
    # But calculate_features needs 'Value'. 
    
    # Apply per group
    # Note: df_1min index is SendDate. We groupby columns.
    # Fix: Select columns to avoid FutureWarnings and duplication or filtering issues
    # We select 'Value' (needed for features). Index 'SendDate' is preserved.
    df_features = df_1min.groupby(group_cols)[['Value']].apply(calculate_features)
    
    # The result has a MultiIndex (group_cols + SendDate)
    # With include_groups=False, grouping columns are NOT in the body, so no need to drop.
    
    # We need to flatten it to access 'SendDate' as a column for feature extraction
    df_features = df_features.reset_index()
    
    # --- Step 4: Time Features ---
    print("[Step 4/9] Extracting Time Features...")
    df_features['Hour'] = df_features['SendDate'].dt.hour
    df_features['DayOfWeek'] = df_features['SendDate'].dt.dayofweek
    df_features['IsWeekend'] = df_features['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    print(f"  -> Generated features: {[c for c in df_features.columns if 'std' in c or 'delta' in c]}")
    
    
    # --- Step 8: Aggregation to 15-min ---
    print("[Step 8/9] Downsampling to 15-minute intervals...")
    print(f"  -> Input shape before aggregation: {df_features.shape}")
    
    # We want 15min bars.
    # Group by Metadata + Resampler
    # For features: Mean is good.
    # For Value: Max, Min, Mean.
    
    # Reset index to make SendDate a column for grouping
    df_features = df_features.reset_index()
    
    # Define Aggregations
    # Metadata columns are grouping keys, so they don't need agg.
    # SendDate -> Resample key.
    
    # columns to agg: Value, std_*, delta_*, Hour...
    target_cols = ['Value', 'std_1min', 'std_5min', 'delta_2min', 'delta_15min', 'Hour', 'DayOfWeek', 'IsWeekend']
    
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
    
    # Group by IDs + resample
    # syntax: groupby([ids, pd.Grouper(key='SendDate', freq='15min')])
    
    grouper = group_cols + [pd.Grouper(key='SendDate', freq='15min')]
    
    df_15min = df_features.groupby(grouper).agg(agg_funcs)
    
    # Flatten Columns (MultiIndex to single)
    # e.g. Value_mean, Value_min...
    df_15min.columns = ['_'.join(col).strip() for col in df_15min.columns.values]
    df_15min = df_15min.reset_index()
    
    # Rename Value_mean to Value for compatibility with existing plotting which expects 'Value'
    df_15min.rename(columns={'Value_mean': 'Value'}, inplace=True)
    
    print(f"  -> Aggregated Shape: {df_15min.shape}")
    print(f"  -> Resolution: 15 Minutes. Columns: {list(df_15min.columns)}")
    print(f"  -> Sample Aggregated Data (Head):")
    print(df_15min[['SendDate', 'Value']].head(3).to_string(index=False))

    # --- Step 6: Normalization (Global) ---
    print("[Step 6/9] Normalization (Z-Score)...")
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        # Scale 'Value' and engineering cols
        # Should we scale 'Value'? Yes, the user said "Normalize the Features... (sensor measurements come in different units)".
        # BUT if we overwrite 'Value', the plots will show Z-scores, not e.g. Degrees Celsius. 
        # This might confuse the user if they want to see "22°C" and see "0.5". 
        # Plan: Add 'Value_Normalized' and keep 'Value' absolute? 
        # User prompt: "Normalize the Features... so that all features contribute equally to the analysis" (implies for clustering).
        # But step 6 is "Normalization". 
        # I will scale and store as NEW columns for clustering, keeping 'Value' original for plotting? 
        # User: "Normalize the Features...". 
        # I will normalize ALL numeric keys into new columns prefixed 'norm_'.
        
        numeric_cols = ['Value', 'Value_min', 'Value_max', 'std_1min_mean', 'std_5min_mean', 'delta_2min_mean', 'delta_15min_mean']
        # Filter existing
        numeric_cols = [c for c in numeric_cols if c in df_15min.columns]
        
        print(f"  -> Columns selected for normalization: {numeric_cols}")
        
        # Check for NaNs before scaling
        # Fill engineered features with 0 (assuming no variance/change if undefined)
        # But 'Value' should be valid.
        fill_cols = [c for c in numeric_cols if c != 'Value']
        initial_nans = df_15min[fill_cols].isna().sum().sum()
        df_15min[fill_cols] = df_15min[fill_cols].fillna(0)
        
        if initial_nans > 0:
            print(f"  -> Filled {initial_nans} NaNs in feature columns with 0.")
        
        # Drop rows where Value is still NaN (should be rare due to Step 3)
        pre_drop_len = len(df_15min)
        df_15min = df_15min.dropna(subset=['Value'])
        dropped_rows = pre_drop_len - len(df_15min)
        
        if dropped_rows > 0:
            print(f"  -> Dropped {dropped_rows} rows with missing 'Value' (after imputation).")
        
        if not df_15min.empty:
            # Stats before
            print(f"  -> Pre-normalization stats (Value): Mean={df_15min['Value'].mean():.2f}, Std={df_15min['Value'].std():.2f}")
            
            scaled = scaler.fit_transform(df_15min[numeric_cols])
            
            # Create new cols
            for i, col in enumerate(numeric_cols):
                 df_15min[f"norm_{col}"] = scaled[:, i]
            
            print(f"  -> Added 'norm_*' columns (Z-Score) for: {numeric_cols}")
            print(f"  -> Sample Normalization (norm_Value head):")
            print(df_15min[['SendDate', 'Value', 'norm_Value']].head(3).to_string(index=False))
        else:
            print("  -> Data empty after cleaning, skipping normalization.")
    
    print("="*60)
    print("GLOBAL PREPROCESSING COMPLETE. DATA READY.")
    print("="*60)
    
    return df_15min



def interactive_menu(df, label_to_id, locations_map):
    """Runs the interactive selection menu."""
    
    sensor_name_col = None
    
    # Robust Sensor Identification
    if 'NodeId' in df.columns:
        sensor_name_col = 'NodeId'
    elif 'SensorId' in df.columns:
        sensor_name_col = 'SensorId'
    else:
        # Fallback to scanning logic
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            sample_vals = df[col].dropna().head(1000).unique()
            if any('TRP' in str(v) for v in sample_vals):
                sensor_name_col = col
                break
            
    if not sensor_name_col:
        print("Could not identify Sensor Name column.")
        print(f"Available columns: {list(df.columns)}")
        return
        
    # Invert mapping for cluster column renaming
    id_to_label = {v: k for k, v in label_to_id.items()}

    while True:
        print("\n" + "="*50)
        print("INTERACTIVE PLOTTING MENU")
        print("="*50)
        
        sensors = get_unique_sensors(df, sensor_name_col)
        print("\nAvailable Sensors:")
        for i, s in enumerate(sensors):
            loc = locations_map.get(s, "")
            display_name = f"{s} ({loc})" if loc else s
            print(f"{i+1}. {display_name}")
        print("x. Exit")
        
        choice = input("\nSelect Sensor (number): ")
        if choice.lower() == 'x':
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sensors):
                selected_sensor = sensors[idx]
            else:
                print("Invalid selection.")
                continue
        except ValueError:
            print("Invalid input.")
            continue
            
        # Menu Options
        print("\nOptions:")
        print("1. Plot Specific Metric")
        print("7. Advanced Analysis (Clustering)")
        print("8. Run Data Preprocessing Pipeline")
        print("b. Back")
        
        mode_choice = input("\nSelect Option: ")
        
        if mode_choice.lower() == 'b':
            continue
            
        if mode_choice == '8':
            # Run Preprocessing Pipeline
            try:
                processed_result = preprocess_pipeline(df, selected_sensor, sensor_name_col, id_to_label, locations_map)
                if processed_result is not None:
                     print("\nPreprocessing finished. Data is ready for downstream tasks.")
                     print(processed_result.head())
                     print(processed_result.describe())
                     
                     # Optional: Offer to save?
                     save_choice = input("Save processed data to CSV? (y/n): ")
                     if save_choice.lower() == 'y':
                         fname = f"{selected_sensor}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                         processed_result.to_csv(fname)
                         print(f"Saved to {fname}")
            except Exception as e:
                print(f"Error in pipeline: {e}")
                import traceback
                traceback.print_exc()
            continue

        if mode_choice == '7':
            # Clustering Mode
            print("\n--- Advanced Clustering Analysis ---")
            
            # Select Time Span
            print("\nSelect Time Span:")
            print("1. All Time")
            print("2. Last 24 Hours")
            print("3. Last 7 Days")
            print("4. Custom Range")
            time_choice = input("Select Time Option: ")
            
            time_option = 'all'
            if time_choice == '2': time_option = '24h'
            if time_choice == '3': time_option = '7d'
            if time_choice == '4': time_option = 'custom'
            
            try:
                k_str = input("Number of Activity States (Clusters) [Default 4]: ")
                k = int(k_str) if k_str.strip() else 4
            except ValueError:
                k = 4
                
            loc_name = locations_map.get(selected_sensor, "")
            perform_clustering(df, selected_sensor, sensor_name_col, id_to_label, time_option, n_clusters=k, location_name=loc_name)
            continue
            
        # Default Plotting Mode logic...
        
        # Select Data Type
        sensor_data = df[df[sensor_name_col] == selected_sensor]
        present_ids = set(sensor_data['SensorId'].astype(str).unique())
        available_labels = [label for label, sid in label_to_id.items() if sid in present_ids]
        available_labels.sort()
        
        print(f"\nAvailable Data Types for {selected_sensor}:")
        for i, label in enumerate(available_labels):
            print(f"{i+1}. {label}")
        print("b. Back")
        
        choice = input("\nSelect Data Type (number): ")
        if choice.lower() == 'b':
            continue
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_labels):
                selected_label = available_labels[idx]
                selected_id = label_to_id[selected_label]
            else:
                print("Invalid selection.")
                continue
        except ValueError:
            print("Invalid input.")
            continue

        # Select Time Span (Duplicated for Plot Mode, can be refactored but keeping linear for now)
        print("\nSelect Time Span:")
        print("1. All Time")
        print("2. Last 24 Hours")
        print("3. Last 7 Days")
        print("4. Custom Range")
        
        time_choice = input("\nSelect Time Option (number): ")
        time_option = 'all'
        
        if time_choice == '2':
            time_option = '24h'
        elif time_choice == '3':
            time_option = '7d'
        elif time_choice == '4':
            time_option = 'custom'
            
        # Select Period
        print("\nSelect Period (Resampling):")
        print("1. None (Raw Data)")
        print("2. Quarter Hour (15 min)")
        print("3. Half Hour (30 min)")
        print("4. Hour")
        print("5. Day")
        print("6. Month")
        print("7. Year")
        
        period_choice = input("\nSelect Period Option (number): ")
        period_option = 'none'
        
        aliases = {
            '2': '15min',
            '3': '30min',
            '4': '1h',
            '5': '1D',
            '6': '1ME',
            '7': '1YE'
        }
        
        if period_choice in aliases:
            period_option = aliases[period_choice]
        
        loc_name = locations_map.get(selected_sensor, "")
        plot_sensor_data(df, selected_sensor, sensor_name_col, selected_label, selected_id, time_option, period_option, location_name=loc_name)

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Directory not found: {RAW_DATA_DIR}")
        return

    mapping_data, id_to_label, locations = load_mapping()
    if not mapping_data:
        print("Mapping file missing or empty.")
        return

    latest_csv = find_latest_csv(RAW_DATA_DIR)
    
    if latest_csv:
        print(f"Reading file: {os.path.basename(latest_csv)}...")
        try:
             df = pd.read_csv(latest_csv, low_memory=False)
             
             # Execute Global Preprocessing Pipeline
             df = process_dataset_globally(df, id_to_label, locations)
             
             if df is not None:
                interactive_menu(df, mapping_data, locations)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No CSV files found.")

if __name__ == "__main__":
    main()