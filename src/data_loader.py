import os
import glob
import json
import pandas as pd
from .constants import RAW_DATA_DIR, MAPPING_FILE
from .utils import print_info, print_warning, print_error, Colors

def load_mapping():
    """Loads sensor mapping from JSON file."""
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                mapping_data = json.load(f)
            
            id_to_label = {}
            locations = {}

            # Handle "measurements": {"Label": "ID"} -> Need ID: Label
            if "measurements" in mapping_data and isinstance(mapping_data["measurements"], dict):
                for label, sid in mapping_data["measurements"].items():
                    id_to_label[str(sid)] = label
            
            # Handle "locations": {"NodeId": "LocationName"}
            if "locations" in mapping_data and isinstance(mapping_data["locations"], dict):
                locations = mapping_data["locations"]
            
            return mapping_data, id_to_label, locations

        except json.JSONDecodeError:
            print_error("Mapping file is invalid JSON.")
            return {}, {}, {}
    else:
        print_warning(f"Mapping file not found at {MAPPING_FILE}")
        return {}, {}, {}

def find_latest_csv(directory):
    """Finds the latest CSV file in the directory."""
    try:
        list_of_files = glob.glob(os.path.join(directory, '*.csv'))
        if not list_of_files:
            return None
        return max(list_of_files, key=os.path.getctime)
    except Exception as e:
        print_error(f"Error searching for CSV: {e}")
        return None

def load_raw_data():
    """Orchestrates the loading of the latest CSV."""
    if not os.path.exists(RAW_DATA_DIR):
        print_error(f"Directory not found: {RAW_DATA_DIR}")
        return None, None, None

    mapping_data, id_to_label, locations = load_mapping()
    
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV files. Reading and concatenating...")
        dfs = []
        for file in csv_files:
            try:
                # Read only necessary columns to save memory if possible, but user asked to filter LATER.
                # However, for speed, we might want to read all and filter immediately in preprocessing.
                # Use low_memory=False to avoid dtypes warning
                df_chunk = pd.read_csv(file, low_memory=False)
                dfs.append(df_chunk)
            except Exception as e:
                print_error(f"Error reading {os.path.basename(file)}: {e}")
        
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            return full_df, id_to_label, locations
        else:
             print_error("No valid data loaded from CSVs.")
             return None, None, None

    else:
        print_error("No CSV files found.")
        return None, None, None
