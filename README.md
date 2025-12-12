# Sustainable Urbanization Through Innovative Technologies

A tool to analyze and visualize sensor data. It helps us track environmental trends and understand how spaces are being used to support sustainable urban planning.

## Structure

*   **`main.py`**: The entry point for the application.
*   **`src/`**: Contains the core logic modules (data loading, preprocessing, analysis, visualization, UI).
*   **`sensor_mapping.json`**: Configuration file mapping sensor IDs to locations and labels.
*   **`archive/`**: Contains legacy scripts (e.g., `csv loader.py`) and backup files.
*   **`raw_data/`**: (External) Directory containing the raw CSV sensor data.

## Setup

1.  **Prerequisites**: Python 3.8+
2.  **Environment**:
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate virtual environment
    # Windows:
    .venv\Scriptsctivate
    # Mac/Linux:
    source .venv/bin/activate
    ```
3.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application expects the raw data to be located at `D:\T.Uaw_data`.
If you need to change this, update `src/constants.py`:

```python
RAW_DATA_DIR = r'path	o\youraw_data'
```

## Usage

Run the main application:

```bash
python main.py
```
