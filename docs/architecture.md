# System Architecture & algorithmic Details

## 1. Introduction
The **Sensor Trend Explorer & Analysis Tool** is a hierarchical data processing system designed to transform raw environmental time-series data into actionable insights through statistical analysis and unsupervised learning.

## 2. Directory Structure -> See `d:/T.U/.../docs/architecture.md` (Filesystem)

## 3. Data Processing Pipeline

The pipeline transforms raw CSV dump data into a structured feature matrix suitable for machine learning.

### 3.1 Data Ingestion (`data_loader.py`)
-   **CSV Parsing**: Loads raw CSV files containing `SensorId`, `Value`, `SendDate`, and `CorrectValue`.
-   **Mapping**: Joins raw data with `sensor_mapping.json` to attach high-level metadata:
    -   `SensorId` $\rightarrow$ `Label` (e.g., "Temperature_ave")
    -   `NodeId` $\rightarrow$ `Location` (e.g., "TRP4 - Meeting Room")
-   **Sanitization**:
    -   Converts `SendDate` to UTC timestamps.
    -   Coerces `Value` to numeric, handling non-numeric errors.
    -   **Prioritizes** `CorrectValue` if present (calibrated data), falling back to `Value`.

### 3.2 Feature Engineering (`analysis.py` / `explorer.py`)
Features are calculated from the raw 1-minute time series before being aggregated to the analysis period (e.g., 15-min).

#### A. Pre-calculation Resampling
Raw data is first resampled to a **1-minute** grid to ensure temporal consistency before feature extraction.
$$ X_{1min} = \text{mean}(X_{raw}) \text{ per minute} $$

#### B. Computed Features
1.  **Rolling Standard Deviation (Volatility)**
    -   Captures rapid fluctuations (e.g., sound variations indicating speech).
    -   Formula: $\sigma_w = \sqrt{\frac{1}{w-1} \sum_{i=0}^{w-1} (x_{t-i} - \bar{x})^2}$
    -   Windows ($w$): 1 min, 2.5 min, 5 min.
    -   *Implementation*: `series.rolling(window=w).std()`

2.  **Delta (Change Rate)**
    -   Captures the net change over a window (e.g., Temperature rise).
    -   Formula: $\Delta_w = x_t - x_{t-w}$
    -   Windows: 2 min, 15 min.
    -   *Implementation*: `series.diff(periods=w)`

### 3.3 Aggregation & Matrix Construction
To analyze long-term trends, high-frequency features are aggregated into larger "bins" (e.g., 15 minutes).

1.  **Resampling**:
    -   **Levels** (Temp, CO2): Aggregated using `mean()`.
    -   **Activity Features** (Sound Std, Light Delta): Aggregated using `mean()` (or `max()` for peak detectors).
2.  **Master Timeline Alignment**:
    -   A dense time grid is created from $T_{start}$ to $T_{end}$ with frequency $f$.
    -   $\text{Index} = \{t_0, t_0+f, ..., t_{end}\}$
    -   All sensor streams are reindexed to this master grid. Missing bins result in `NaN`.
3.  **Gating (Researcher Mode)**:
    -   **Strict Validity**: Rows (bins) containing *any* `NaN` values in critical features are dropped.
    -   **Threshold**: If $>90\%$ of data is missing, the dataset may be rejected.
4.  **Winsorization**:
    -   To prevent clustering distortion by extreme outliers, features are clipped to the **1st and 99th percentiles**.

## 4. Unsupervised Learning (Clustering)

The core analysis engine uses K-Means to discover latent "Activity States" from the multidimensional feature matrix.

### 4.1 Preprocessing
-   **Standardization**:
    -   Features are scaled to Zero Mean and Unit Variance to ensure equal weighting.
    -   $z = \frac{x - \mu}{\sigma}$

### 4.2 K-Means Clustering Strategy
-   **Algorithm**: Lloyd's algorithm with K-Means++ initialization.
-   **Parameters**:
    -   `n_init=10`: Run 10 times with different seeds, keep best inertia.
    -   `random_state=42`: Fixed seed for reproducibility.

### 4.3 Automatic K-Selection (Optimization)
The tool automatically determines the optimal number of clusters ($K$) using a "Sweep" approach:
1.  Iterate $K$ from 2 to 10.
2.  **Calculate Metrics**:
    -   **Silhouette Score**: Measures cluster separation ($-1 \text{ to } +1$).
    -   **Inertia (SSE)**: Sum of Squared Errors (distance to centroid).
3.  **Rejection Heuristics**:
    -   **Minimum Size Rule**: Reject $K$ if any resulting cluster contains $<3\%$ of total data. This prevents "outlier clusters" consisting of just a few noise points.
4.  **Selection**: Choose the valid $K$ with the highest Silhouette Score.

## 5. Clean CSV Export Logic

When exporting data for publication, specific transformation rules are applied:

1.  **Filtering**: Data is sliced by user-defined Date Range and Node selection.
2.  **Column Selection**:
    -   User selects specific parameters (e.g., `Sound_ave`).
    -   Technical columns (`IP`, `Mac`, `Iamalive`, `Labels`) are strictly excluded.
3.  **Resampling**:
    -   Raw data is averaged into the user-selected interval (e.g., 1 Hour).
    -   $\bar{x}_{bin} = \frac{1}{N} \sum x_i$
4.  **Empty Value Handling options**:
    -   **Keep All**: Retains the Master Timeline index. Missing data $\rightarrow$ empty cells.
    -   **Drop Empty**: Removes rows where the selected parameters have no data (`dropna(how='all')`).
    -   **Forward Fill**: Imputes missing values using the last known valid observation (`ffill()`).
5.  **Precision**:
    -   All floating-point output is rounded to **3 decimal places**.

## 6. Output Artifacts

-   **`reports/analysis_validation/`**:
    -   **feature_profiles.png**: Bar charts showing the Z-score signature of each cluster (e.g., "High Sound + Low Light").
    -   **hourly_heatmap.png**: 24h x 7d grid showing when specific clusters are active.
    -   **transition_matrix.png**: Probabilities of switching from State A $\rightarrow$ State B.
-   **`reports/clean_export/`**:
    -   **export_*.csv**: The cleaner, resampled, and user-configured dataset.
