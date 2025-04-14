# visualizations/dbscan.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import uuid

def generate_dbscan_visuals():
    # Step 1: Load and engineer features
    df = pd.read_csv("EPL.csv")
    df["TotalGoals"] = df["FullTimeHomeTeamGoals"] + df["FullTimeAwayTeamGoals"]
    df["GoalDifference"] = df["FullTimeHomeTeamGoals"] - df["FullTimeAwayTeamGoals"]
    df["WinRate"] = df["HomeTeamPoints"] / (df["HomeTeamPoints"] + df["AwayTeamPoints"])

    features = df[["TotalGoals", "GoalDifference", "WinRate"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 2: K-distance graph to estimate epsilon
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(scaled_features)
    distances, indices = nbrs.kneighbors(scaled_features)
    distances = np.sort(distances[:, k - 1])  # 10th neighbor distance

    # Create folder to store plots
    os.makedirs('static/plots', exist_ok=True)

    # K-distance graph
    fig_kd, ax_kd = plt.subplots(figsize=(10, 6))
    ax_kd.plot(distances)
    ax_kd.set_xlabel('Data Points Sorted by Distance')
    ax_kd.set_ylabel(f'{k}th Nearest Neighbor Distance')
    ax_kd.set_title('K-Distance Graph to Determine Epsilon')

    kd_path = f"static/plots/dbscan_k_distance_{uuid.uuid4()}.png"
    fig_kd.savefig(kd_path, bbox_inches='tight')
    plt.close(fig_kd)

    # Step 3: Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    df["Cluster"] = dbscan.fit_predict(scaled_features)

    # Step 4: Scatter plot for Clustering
    silhouette_avg = None
    if len(set(df["Cluster"])) > 1 and -1 not in set(df["Cluster"]):
        silhouette_avg = silhouette_score(scaled_features, df["Cluster"])

    # Scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sc = ax_scatter.scatter(df["TotalGoals"], df["WinRate"], c=df["Cluster"], cmap="rainbow", edgecolors='k')
    ax_scatter.set_xlabel("Total Goals")
    ax_scatter.set_ylabel("Win Rate")
    ax_scatter.set_title("DBSCAN Clustering of EPL Teams")
    fig_scatter.colorbar(sc, ax=ax_scatter, label="Cluster")

    scatter_path = f"static/plots/dbscan_scatter_{uuid.uuid4()}.png"
    fig_scatter.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig_scatter)

    return {
        'kd_graph': kd_path,
        'scatter_plot': scatter_path,
        'silhouette_score': silhouette_avg
    }
