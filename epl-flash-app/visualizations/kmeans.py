# visualizations/kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

def generate_kmeans_visuals():
    # Load data and fill missing numeric values
    df = pd.read_csv("EPL.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Create new features
    df["TotalGoals"] = df["FullTimeHomeTeamGoals"] + df["FullTimeAwayTeamGoals"]
    df["GoalDifference"] = df["FullTimeHomeTeamGoals"] - df["FullTimeAwayTeamGoals"]
    df["WinRate"] = df["HomeTeamPoints"] / (df["HomeTeamPoints"] + df["AwayTeamPoints"])

    # Scale features for clustering
    features = df[["TotalGoals", "GoalDifference", "WinRate"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Compute inertia for a range of k values (Elbow Method)
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Create folder to store plots
    os.makedirs('static/plots', exist_ok=True)
    
    # Elbow method plot
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 6))
    ax_elbow.plot(K_range, inertia, marker='o')
    ax_elbow.set_xlabel('Number of Clusters (K)')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('Elbow Method for Optimal K')
    
    elbow_path = f"static/plots/kmeans_elbow_{uuid.uuid4()}.png"
    fig_elbow.savefig(elbow_path, bbox_inches='tight')
    plt.close(fig_elbow)
    
    # K-Means Clustering with optimal_k (use k=10 for example)
    optimal_k = 10
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_features)
    
    # Compute silhouette score if there is more than one cluster
    silhouette_avg = None
    if len(set(df["Cluster"])) > 1:
        silhouette_avg = silhouette_score(scaled_features, df["Cluster"])
    
    # Scatter plot of the clustering
    fig_scatter, ax_scatter = plt.subplots(figsize=(9, 6))
    sc = ax_scatter.scatter(df["TotalGoals"], df["WinRate"], c=df["Cluster"], cmap="viridis", edgecolors='k')
    ax_scatter.set_xlabel("Total Goals")
    ax_scatter.set_ylabel("Win Rate")
    ax_scatter.set_title("K-Means Clustering of EPL Teams")
    fig_scatter.colorbar(sc, ax=ax_scatter, label="Cluster")
    
    scatter_path = f"static/plots/kmeans_scatter_{uuid.uuid4()}.png"
    fig_scatter.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig_scatter)
    
    return {
        'elbow_plot': elbow_path,
        'scatter_plot': scatter_path,
        'silhouette_score': silhouette_avg
    }
