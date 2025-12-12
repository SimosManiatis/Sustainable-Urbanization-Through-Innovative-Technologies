import pandas as pd
from datetime import datetime, timedelta
from .utils import print_header, print_info, Colors
from .analysis import perform_clustering, find_optimal_k, save_clustering_results
from .visualization import plot_sensor_data, plot_optimization_metrics, plot_cluster_analysis, plot_feature_profiles

def get_unique_sensors(df, sensor_name_col):
    if sensor_name_col and sensor_name_col in df.columns:
        return sorted(df[sensor_name_col].astype(str).unique())
    return []

def select_date_drilldown(df):
    """
    Interactive Year -> Month -> Day selection.
    Returns (start_date, end_date)
    """
    if not pd.api.types.is_datetime64_any_dtype(df['SendDate']):
        df['SendDate'] = pd.to_datetime(df['SendDate'], errors='coerce')
        
    valid_dates = df['SendDate'].dropna()
    if valid_dates.empty:
        return None, None
        
    print_header("Date Selection")
    
    # Year
    years = sorted(valid_dates.dt.year.unique())
    print("Available Years:")
    for i, y in enumerate(years):
        print(f"{i+1}. {y}")
        
    try:
        y_idx = int(input("Select Year: ")) - 1
        selected_year = years[y_idx]
    except (ValueError, IndexError):
        return None, None
        
    print(f"\nSelected Year: {Colors.CYAN}{selected_year}{Colors.ENDC}")
    
    # Month
    mask_year = (valid_dates.dt.year == selected_year)
    months = sorted(valid_dates[mask_year].dt.month.unique())
    print("\nAvailable Months:")
    for i, m in enumerate(months):
        print(f"{i+1}. {m}")
        
    try:
        m_idx = int(input("Select Month: ")) - 1
        selected_month = months[m_idx]
    except (ValueError, IndexError):
        return None, None
        
    print(f"Selected Month: {Colors.CYAN}{selected_month}{Colors.ENDC}")
    
    # Day
    mask_month = mask_year & (valid_dates.dt.month == selected_month)
    days = sorted(valid_dates[mask_month].dt.day.unique())
    print("\nAvailable Days:")
    for i, d in enumerate(days):
        print(f"{i+1}. {d}")
    
    try:
        d_idx = int(input("Select Start Day: ")) - 1
        start_day = days[d_idx]
    except (ValueError, IndexError):
        return None, None
        
    # Construct Start Date
    start_date = datetime(selected_year, selected_month, start_day)
    
    # Ask for End Date or Duration?
    print(f"\nStart Date set to: {Colors.GREEN}{start_date.date()}{Colors.ENDC}")
    print("Select End Day (or Enter to select just one day):")
    
    # Filter days >= start_day
    future_days = [d for d in days if d >= start_day]
    for i, d in enumerate(future_days):
        print(f"{i+1}. {d}")
        
    end_input = input("Select End Day (Enter for same day): ")
    if not end_input.strip():
        end_date = start_date + timedelta(days=1) - timedelta(seconds=1)
    else:
        try:
            ed_idx = int(end_input) - 1
            end_day = future_days[ed_idx]
            end_date = datetime(selected_year, selected_month, end_day) + timedelta(days=1) - timedelta(seconds=1)
        except (ValueError, IndexError):
             end_date = start_date + timedelta(days=1) - timedelta(seconds=1)
             
    print(f"Time Range: {start_date} to {end_date}")
    return start_date, end_date

def interactive_menu(df, label_to_id, locations_map):
    """Runs the interactive selection menu."""
    
    sensor_name_col = None
    if 'NodeId' in df.columns:
        sensor_name_col = 'NodeId'
    elif 'SensorId' in df.columns:
        sensor_name_col = 'SensorId'
            
    if not sensor_name_col:
        print("Could not identify Sensor Name column.")
        return
        
    id_to_label = {v: k for k, v in label_to_id.items()}

    while True:
        print_header("INTERACTIVE PLOTTING MENU")
        
        sensors = get_unique_sensors(df, sensor_name_col)
        print("Available Sensors:")
        for i, s in enumerate(sensors):
            loc = locations_map.get(s, "")
            display_name = f"{s} ({loc})" if loc else s
            print(f"{i+1}. {display_name}")
        print("x. Exit")
        
        choice = input(f"\n{Colors.CYAN}Select Sensor (number): {Colors.ENDC}")
        if choice.lower() == 'x':
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sensors):
                selected_sensor = sensors[idx]
            else:
                continue
        except ValueError:
            continue
            
        # Menu Options
        print("\nOptions:")
        print("1. Plot Specific Metric")
        print("7. Advanced Analysis (Clustering)")
        print(f"{Colors.FAIL}8. [Removed] Preprocessing (Already Done){Colors.ENDC}")
        print("b. Back")
        
        mode_choice = input(f"\n{Colors.CYAN}Select Option: {Colors.ENDC}")
        
        if mode_choice.lower() == 'b':
            continue
            
        if mode_choice == '7':
             # Clustering Mode
            print(f"\n{Colors.CYAN}--- Advanced Clustering Analysis ---{Colors.ENDC}")
            print("1. Determine Optimal Clusters (Elbow Method)")
            print("2. Run Clustering & Visualization")
            print("b. Back")
            
            sub_choice = input("Select Action: ")
            if sub_choice.lower() == 'b': continue

            # Time Logic
            print("\nTime Range:")
            print("1. All Time")
            print("2. Custom Range")
            t_choice = input("Select Time: ")
            
            time_option = 'all'
            s_date, e_date = None, None
            
            if t_choice == '2':
                time_option = 'custom'
                s_date, e_date = select_date_drilldown(df)
                if not s_date:
                    time_option = 'all'

            loc_name = locations_map.get(selected_sensor, "")

            if sub_choice == '1':
                # Elbow Method
                print("\nRunning Optimization...")
                k_range, inertia, silhouette = find_optimal_k(df, selected_sensor, sensor_name_col, time_option, s_date, e_date)
                if k_range:
                    plot_optimization_metrics(k_range, inertia, silhouette)
            
            elif sub_choice == '2':
                # Run Clustering
                try:
                    k = int(input("Number of Clusters (default 4): ") or 4)
                except:
                    k = 4
                    
                results, centroids, label_map = perform_clustering(df, selected_sensor, sensor_name_col, id_to_label, time_option, n_clusters=k, location_name=loc_name, start_date=s_date, end_date=e_date)
                
                
                if results is not None:
                     # 1. Dashboard
                     plot_cluster_analysis(results, centroids, label_map, selected_sensor)
                     
                     # 2. Feature Profiles
                     print("\nDisplaying Feature Profiles (Centroid Bar Chart)...")
                     plot_feature_profiles(centroids, label_map, selected_sensor)
                     
                     # 3. Save?
                     save_choice = input(f"\n{Colors.CYAN}Save results to CSV? (y/n): {Colors.ENDC}")
                     if save_choice.lower() == 'y':
                         save_clustering_results(results, label_map, selected_sensor)
            
            continue
            
        # Plot Mode
        sensor_data = df[df[sensor_name_col] == selected_sensor]
        present_ids = set(sensor_data['SensorId'].astype(str).unique())
        available_labels = [label for label, sid in label_to_id.items() if sid in present_ids]
        available_labels.sort()
        
        print(f"\nAvailable Data Types:")
        for i, label in enumerate(available_labels):
            print(f"{i+1}. {label}")
        print("b. Back")
        
        choice = input("\nSelect Data Type: ")
        if choice.lower() == 'b': continue
        
        try:
            idx = int(choice) - 1
            selected_label = available_labels[idx]
            selected_id = label_to_id[selected_label]
        except: continue
        
        # Time
        print("\nSelect Time Span:")
        print("1. All Time")
        print("2. Last 24 Hours")
        print("3. Last 7 Days")
        print("4. Custom Range (Drill-down)")
        
        t_choice = input(f"\n{Colors.CYAN}Select Time Option: {Colors.ENDC}")
        time_option = 'all'
        s_date, e_date = None, None
        
        if t_choice == '2': time_option = '24h'
        elif t_choice == '3': time_option = '7d'
        elif t_choice == '4':
             time_option = 'custom'
             s_date, e_date = select_date_drilldown(df)
             
        # Period
        print("\nSelect Period:")
        print("1. None")
        print("2. 15 min")
        print("3. 30 min")
        print("4. Hour")
        
        p_choice = input("\nSelect Period: ")
        aliases = {'2': '15min', '3': '30min', '4': '1h'}
        period_option = aliases.get(p_choice, 'none')
        
        loc_name = locations_map.get(selected_sensor, "")
        plot_sensor_data(df, selected_sensor, sensor_name_col, selected_label, selected_id, time_option, period_option, location_name=loc_name, start_date=s_date, end_date=e_date)
