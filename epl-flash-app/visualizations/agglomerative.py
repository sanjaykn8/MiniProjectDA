# visualizations/agglomerative.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import os
import uuid

def generate_agglomerative_visuals():
    # Step 1: Load and process data
    df = pd.read_csv("EPL.csv")
    df["TotalGoals"] = df["FullTimeHomeTeamGoals"] + df["FullTimeAwayTeamGoals"]
    df["GoalDifference"] = df["FullTimeHomeTeamGoals"] - df["FullTimeAwayTeamGoals"]
    df["WinRate"] = df["HomeTeamPoints"] / (df["HomeTeamPoints"] + df["AwayTeamPoints"])

    features = df[["TotalGoals", "GoalDifference", "WinRate"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 2: Create folder to store plots
    os.makedirs('static/plots', exist_ok=True)
    
    # Dendrogram
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 6))
    sch.dendrogram(sch.linkage(scaled_features, method='ward'), ax=ax_dendro)
    ax_dendro.set_title('Dendrogram for Agglomerative Clustering')
    ax_dendro.set_xlabel('Teams')
    ax_dendro.set_ylabel('Euclidean Distance')

    dendro_path = f"static/plots/agglomerative_dendrogram_{uuid.uuid4()}.png"
    fig_dendro.savefig(dendro_path, bbox_inches='tight')
    plt.close(fig_dendro)

    # Step 3: Apply Agglomerative Clustering
    optimal_k = 6
    agglo_cluster = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete', metric='euclidean')
    df["Cluster"] = agglo_cluster.fit_predict(scaled_features)

    # Step 4: Scatter plot for Clustering
    silhouette_avg = None
    if len(set(df["Cluster"])) > 1:
        silhouette_avg = silhouette_score(scaled_features, df["Cluster"])

    # Scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sc = ax_scatter.scatter(df["TotalGoals"], df["WinRate"], c=df["Cluster"], cmap="rainbow", edgecolors='k')
    ax_scatter.set_xlabel("Total Goals")
    ax_scatter.set_ylabel("Win Rate")
    ax_scatter.set_title("Agglomerative Clustering of EPL Teams")
    fig_scatter.colorbar(sc, ax=ax_scatter, label="Cluster")

    scatter_path = f"static/plots/agglomerative_scatter_{uuid.uuid4()}.png"
    fig_scatter.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig_scatter)

    return {
        'dendrogram': dendro_path,
        'scatter_plot': scatter_path,
        'silhouette_score': silhouette_avg
    }
