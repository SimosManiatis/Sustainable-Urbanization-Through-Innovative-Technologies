import os
import pandas as pd
import matplotlib.pyplot as plt
from .utils import print_header, print_info, print_success, Colors

def generate_textual_report(df, centroids, label_map, output_dir="reports"):
    """
    Generates a "Methods + Results" textual report.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    report_path = os.path.join(output_dir, "analysis_report.txt")
    
    with open(report_path, "w") as f:
        # Title
        f.write("Sustainable Urbanization - Sensor Data Analysis Report\n")
        f.write("====================================================\n\n")
        
        # Methods
        f.write("1. METHODS\n")
        f.write("----------\n")
        f.write("Data Preprocessing:\n")
        f.write("- Time Alignment: Timestamps parsed to UTC.\n")
        f.write("- Gap Detection: Days with data gaps > 2 hours were excluded per node.\n")
        f.write("- Resampling: Data resampled to 1-minute resolution (interpolated max 15 mins).\n")
        f.write("- Feature Engineering: Rolling volatility (std_5min) and temporal deltas (delta_15min).\n")
        f.write("- Aggregation: 15-minute non-overlapping bins. Bins with <80% coverage dropped.\n")
        f.write("- Normalization: Z-score standardization per measurement type.\n\n")
        
        f.write("Clustering:\n")
        f.write("- Scope: Global clustering (all rooms combined) to ensure comparable states.\n")
        if centroids is not None:
            f.write(f"- Algorithm: K-Means (K={len(centroids)}).\n")
            f.write("- Features: Normalized variability and mean value features.\n")
            f.write("- Labeling: Clusters labeled by centroid intensity (Inactive -> High).\n\n")
        else:
            f.write("- Clustering: Skipped or Failed.\n\n")
        
        # Results
        f.write("2. RESULTS\n")
        f.write("----------\n")
        
        if centroids is not None and label_map:
            # Cluster Profiles
            f.write("Cluster Profiles:\n")
            for cid, label in label_map.items():
                f.write(f"- Cluster {cid} ({label}):\n")
                # Write top features for this cluster?
                # Just mean Z-scores of key features
                centroid = centroids.iloc[cid]
                top_features = centroid.abs().sort_values(ascending=False).head(3)
                for feat, val in top_features.items():
                        f.write(f"    {feat}: {val:.2f}\n")
                f.write("\n")
            
            # Utilization
            f.write("Room Utilization (% Time in State):\n")
            utilization = df.groupby(['NodeId', 'ActivityLabel']).size().unstack(fill_value=0)
            # Convert to percentages
            utilization_pct = utilization.div(utilization.sum(axis=1), axis=0) * 100
            
            f.write(utilization_pct.round(1).to_string())
            f.write("\n\n")
            
            # Peak Activity
            f.write("Peak Activity Windows:\n")
            # For each room, find hour with most "High Activity"
            high_labels = [l for l in label_map.values() if "High" in l]
            
            if high_labels:
                high_df = df[df['ActivityLabel'].isin(high_labels)]
                if not high_df.empty:
                    peak_hours = high_df.groupby('NodeId')['hour'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
                    f.write(peak_hours.to_string())
                else:
                        f.write("No high activity detected.")
            else:
                    f.write("No 'High' activity labels assigned.")
        else:
            f.write("Clustering results not available.\n")
        
        f.write("\n\n(End of Report)\n")
        
    print_success(f"Report generated: {report_path}")


def export_deliverables(df, centroids, output_dir="reports"):
    """
    Exports clean data, clustered data, and summaries.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Full Clustered Data
    df.to_csv(os.path.join(output_dir, "clustered_data_full.csv"), index=False)
    
    # 2. Daily Summary
    # Ensure Date column exists
    if "Date" not in df.columns and "SendDate" in df.columns:
        df["Date"] = df["SendDate"].dt.date
        
    daily = df.groupby(['NodeId', 'Date', 'ActivityLabel']).size().unstack(fill_value=0)
    daily.to_csv(os.path.join(output_dir, "daily_activity_counts.csv"))
    
    # 3. Centroids
    if centroids is not None:
        centroids.to_csv(os.path.join(output_dir, "cluster_centroids.csv"))
    
    print_success(f"Data exported to {output_dir}/")
