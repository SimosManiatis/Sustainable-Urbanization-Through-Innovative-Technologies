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
            
            # Robust JSON handling
            if isinstance(mapping_data, dict):
                # Try to extract list if wrapped
                if "sensors" in mapping_data and isinstance(mapping_data["sensors"], list):
                     mapping_data = mapping_data["sensors"]
                elif all(isinstance(v, dict) for v in mapping_data.values()):
                     print_info("Mapping Format", "Dictionary detected, converting to list.")
                     mapping_data = list(mapping_data.values())
                else:
                     # Check values for a list
                     found = False
                     for k, v in mapping_data.items():
                         if isinstance(v, list):
                             mapping_data = v
                             found = True
                             break
                     if not found:
                         print_warning("Unknown JSON structure. Expected list or dict of sensors.")

            # Create helper maps
            id_to_label = {}
            locations = {}
            
            # Ensure we iterate a list
            if not isinstance(mapping_data, list):
                print_warning("Mapping data is not a list after parsing.")
                mapping_data = []

            for item in mapping_data:
                # Items should be dicts
                if not isinstance(item, dict):
                    continue
                
                sid = item.get('SensorId')
                label = item.get('Label')
                loc = item.get('Location')
                
                if sid is not None and label:
                    id_to_label[str(sid)] = label
                if label and loc:
                     locations[label] = loc
            
            return mapping_data, id_to_label, locations
        except json.JSONDecodeError:
            print_error("Mapping file is invalid JSON.")
            return [], {}, {}
    else:
        print_warning(f"Mapping file not found at {MAPPING_FILE}")
        return [], {}, {}

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
    
    latest_csv = find_latest_csv(RAW_DATA_DIR)
    
    if latest_csv:
        print(f"Reading file: {Colors.BOLD}{os.path.basename(latest_csv)}{Colors.ENDC}...")
        try:
             df = pd.read_csv(latest_csv, low_memory=False)
             return df, id_to_label, locations
        except Exception as e:
            print_error(f"Error reading CSV: {e}")
            return None, None, None
    else:
        print_error("No CSV files found.")
        return None, None, None
