import sys
import os

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_data
from src.preprocessing import process_dataset_globally
from src.analysis import perform_global_clustering, find_optimal_k_global
from src.reporting import generate_textual_report, export_deliverables
from src.visualization import plot_cluster_analysis
from src.ui import interactive_menu
from src.utils import print_header, Colors

def main():
    # Mode Selection
    print_header("Sustainable Urbanization Pipeline")
    print("1. Standard Pipeline (Full Process)")
    print("2. Sensor Trend Explorer (Interactive)")
    
    mode = input(f"{Colors.CYAN}Select Mode (1-2): {Colors.ENDC}").strip()
    
    if mode == "2":
        from src.explorer import run_sensor_trend_explorer
        run_sensor_trend_explorer()
        return
        
    # Standard Pipeline Execution
    print_header("Standard Pipeline Mode")
    
    # 1. Load Data
    df, id_to_label, locations = load_raw_data()
    
    if df is not None:
        # Prompt for dates
        print("\nDate Filter (Optional)")
        print("Enter dates to analyze (YYYY-MM-DD), separated by commas.")
        print("Press Enter to analyze all data.")
        user_input = input("Dates: ").strip()
        
        target_dates = None
        if user_input:
            try:
                # Parse inputs to list of strings
                target_dates = [d.strip() for d in user_input.split(',')]
                print(f"Selected dates: {target_dates}")
            except Exception as e:
                print(f"Error parsing dates: {e}. proceeding with all data.")
                target_dates = None

        # 2. Preprocess
        df = process_dataset_globally(df, id_to_label, locations, target_dates=target_dates)
        
        if df is None:
            print("ERROR: Preprocessing returned None.")
        elif df.empty:
            print("ERROR: Preprocessed dataframe is empty.")
        else:
             print(f"DEBUG: Processing successful. Data shape: {df.shape}")

        # 3. Analysis & Clustering (Automated)
        if df is not None and not df.empty:
            # Optimal K? 
            # For automation, we can just run perform_global_clustering with default K=4
            # or run find_optimal_k_global(df) and pick best.
            # Let's stick to K=4 as a safe default for "Inactive, Low, Med, High"
            
            clustered_df, centroids, label_map = perform_global_clustering(df, n_clusters=4)
            
            # 4. Reporting
            generate_textual_report(clustered_df, centroids, label_map)
            export_deliverables(clustered_df, centroids)
            
            # 5. Visualization (Saved to reports)
            plot_cluster_analysis(clustered_df, centroids, label_map, sensor_name="Global_Activity")
            
            print("Pipeline completed successfully. Check 'reports/' folder.")
        
        # UI is disabled as the data structure (Wide) is not compatible with the existing UI (Long)
        # interactive_menu(df, id_to_label, locations)

if __name__ == "__main__":
    main()
