import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime
from .utils import print_header, print_info, print_error, print_success, Colors
from .data_loader import load_raw_data

def get_user_choice(prompt, options, return_key=True):
    """Helper to get user choice from a list or dict."""
    print(f"\n{Colors.HEADER}{prompt}{Colors.ENDC}")
    
    if isinstance(options, list):
        for i, opt in enumerate(options):
            print(f"{i+1}. {opt}")
        
        while True:
            try:
                val = input(f"{Colors.CYAN}Select (1-{len(options)}): {Colors.ENDC}")
                idx = int(val) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                print_error("Invalid selection.")
            except ValueError:
                print_error("Please enter a number.")
                
    elif isinstance(options, dict):
        keys = list(options.keys())
        for i, key in enumerate(keys):
            print(f"{i+1}. {key} ({options[key]})")
            
        while True:
            try:
                val = input(f"{Colors.CYAN}Select (1-{len(keys)}): {Colors.ENDC}")
                idx = int(val) - 1
                if 0 <= idx < len(keys):
                    return keys[idx] if return_key else options[keys[idx]]
                print_error("Invalid selection.")
            except ValueError:
                print_error("Please enter a number.")

def parse_date_input(prompt):
    """Parses date string to datetime."""
    while True:
        val = input(f"{Colors.CYAN}{prompt} (YYYY-MM-DD [HH:MM]): {Colors.ENDC}").strip()
        if not val:
            return None
        
        # Try formats
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                dt = datetime.strptime(val, fmt)
                # Assume naive input is local -> convert to UTC if needed?
                # Data is UTC. User likely thinks in local.
                # For simplicity, we assume user input is 'comparable' to data timezone or matching.
                # Ideally we localize. Let's return the naive object and assume caller handles comparison or data is naive-compatible.
                return pd.Timestamp(dt).tz_localize('UTC') # Assume input is intended as UTC for strict comparison
            except ValueError:
                continue
        print_error("Invalid format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM")

def run_sensor_trend_explorer():
    print_header("Sensor Trend Explorer")
    
    # 1. Load Data (Just metadata first? No, we need full data to filter)
    # We load everything.
    print_info("Setup", "Loading dataset...")
    df, id_to_label, locations = load_raw_data()
    
    if df is None or df.empty:
        print_error("No data available.")
        return

    # Ensure SendDate is datetime
    df["SendDate"] = pd.to_datetime(df["SendDate"], utc=True, errors='coerce')
    df = df.dropna(subset=["SendDate"])

    # 2. Prompt for Node
    available_nodes = sorted(df["NodeId"].astype(str).unique())
    node_map = {n: locations.get(n, "Unknown Location") for n in available_nodes}
    
    selected_node = get_user_choice("Select Node:", node_map)
    print_success(f"Selected Node: {selected_node}")
    
    # Filter by Node
    node_df = df[df["NodeId"] == selected_node].copy()
    
    # 3. Prompt for Sensor
    # We want labels like "Light_ave".
    # Map SensorId -> Label using id_to_label
    # Filter rows where SensorId is in id_to_label
    node_df["Label"] = node_df["SensorId"].map(id_to_label)
    
    # Available labels for this node
    available_labels = sorted(node_df["Label"].dropna().unique())
    
    if not available_labels:
        print_error("No identifiable sensors for this node.")
        return

    selected_label = get_user_choice("Select Sensor:", available_labels)
    print_success(f"Selected Sensor: {selected_label}")
    
    # Filter by Sensor
    sensor_df = node_df[node_df["Label"] == selected_label].copy()

    # 4. Prompt for Date Range
    print("\nDate Range Selection:")
    min_date = sensor_df["SendDate"].min()
    max_date = sensor_df["SendDate"].max()
    print(f"Data available from {min_date} to {max_date}")
    
    start_date = parse_date_input("Start Date")
    end_date = parse_date_input("End Date")
    
    if start_date:
        sensor_df = sensor_df[sensor_df["SendDate"] >= start_date]
    if end_date:
        sensor_df = sensor_df[sensor_df["SendDate"] <= end_date]
        
    if sensor_df.empty:
        print_error("No data in selected date range.")
        return

    # 5. Prompt for Period
    # Map friendly name to pandas offset alias
    periods = {
        "Quarter Hour": "15min",
        "Half Hour": "30min",
        "Hour": "1h",
        "Day": "1D",
        "Week": "1W",
        "Month": "1ME" 
    }
    selected_period_name = get_user_choice("Select Aggregation Period:", periods)
    selected_rule = periods[selected_period_name]
    
    # 6. Prompt for Data Type (Aggregation)
    agg_types = {
        "Average": "mean",
        "Minimum": "min",
        "Maximum": "max",
        "Total": "sum",
        # "First": "first", # Not standard Agg func string always? 'first' works in pandas
        # "Last": "last"
    }
    selected_agg_name = get_user_choice("Select Aggregation Type:", agg_types)
    selected_agg_func = agg_types[selected_agg_name]
    
    # 7. Processing
    print_info("Processing", "Aggregating data...")
    
    # Determine ValueUsed
    if "CorrectValue" in sensor_df.columns:
        sensor_df["ValueUsed"] = sensor_df["CorrectValue"].fillna(sensor_df["Value"])
    else:
        sensor_df["ValueUsed"] = sensor_df["Value"]
        
    sensor_df["ValueUsed"] = pd.to_numeric(sensor_df["ValueUsed"], errors='coerce')
    sensor_df = sensor_df.dropna(subset=["ValueUsed"])
    
    # Sort and Set Index
    sensor_df = sensor_df.sort_values("SendDate")
    # De-duplicate: If multiple points exist for same timestamp/sensor/node?
    # We just sort. Resample handles list of values in bin.
    
    # Resample
    ts_series = sensor_df.set_index("SendDate")["ValueUsed"]
    aggregated_data = ts_series.resample(selected_rule).agg(selected_agg_func)
    
    # Remove NaNs (empty bins)
    aggregated_data = aggregated_data.dropna()
    
    if aggregated_data.empty:
        print_error("Aggregation resulted in empty dataset.")
        return
        
    # Summary
    print("\nSummary:")
    print(f"  Count: {len(aggregated_data)} buckets")
    print(f"  Min:   {aggregated_data.min():.2f}")
    print(f"  Mean:  {aggregated_data.mean():.2f}")
    print(f"  Max:   {aggregated_data.max():.2f}")
    
    # 8. Plotting
    try:
        # Create output dir
        if not os.path.exists("reports"):
            os.makedirs("reports")
            
        plot_filename = f"trend_{selected_node}_{selected_label}_{selected_agg_name}_{selected_period_name.replace(' ', '')}.png"
        plot_path = os.path.join("reports", plot_filename)
        
        plt.figure(figsize=(12, 6))
        
        # Style: Light Blue line with fill
        color_line = "#03A9F4" # Light Blue
        color_fill = "#03A9F4"
        alpha_fill = 0.2
        
        # Plot
        plt.plot(aggregated_data.index, aggregated_data.values, color=color_line, linewidth=2, label=selected_label)
        plt.fill_between(aggregated_data.index, aggregated_data.values, color=color_fill, alpha=alpha_fill)
        
        # Formatting
        title_str = f"{selected_node}\n{selected_label} ({selected_agg_name} / {selected_period_name})"
        plt.title(title_str, loc='left', fontsize=14, fontweight='bold')
        
        # Minimalist aesthetic
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False) # Maybe keep Y?
        
        # X-axis formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(plot_path, dpi=300)
        print_success(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print_error(f"Plotting failed: {e}")
