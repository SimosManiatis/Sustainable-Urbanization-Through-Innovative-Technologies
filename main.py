import sys
import os

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_data
from src.preprocessing import process_dataset_globally
from src.ui import interactive_menu

def main():
    # Load Data
    df, id_to_label, locations = load_raw_data()
    
    if df is not None:
        # Preprocess
        df = process_dataset_globally(df, id_to_label, locations)
        
        # UI
        if df is not None:
            interactive_menu(df, id_to_label, locations)

if __name__ == "__main__":
    main()
