import sys
import os

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.explorer import run_sensor_trend_explorer
from src.utils import print_header

def main():
    print_header("Sensor Trend Explorer")
    run_sensor_trend_explorer()

if __name__ == "__main__":
    main()
