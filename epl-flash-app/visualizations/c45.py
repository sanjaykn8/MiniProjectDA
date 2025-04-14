# visualizations/c45.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import uuid

def entropy(data):
    total = len(data)
    counts = Counter(data)
    return -sum((count/total) * log2(count/total) for count in counts.values())

def generate_c45_visuals():
    df = pd.read_csv('EPL.csv')
    df = df[['HomeTeam', 'AwayTeam', 'FullTimeResult', 'HalfTimeResult']]
    df = df.apply(lambda x: pd.factorize(x)[0])
    
    # Prepare training data
    X = df.drop(columns=['FullTimeResult'])
    y = df['FullTimeResult']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Initialize and fit the C4.5 model (using entropy)
    c45_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, 
                                      min_samples_split=10, min_samples_leaf=5)
    c45_tree.fit(X_train, y_train)
    y_pred = c45_tree.predict(X_test)

    # Metrics
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
    }

    # Create folder to store plots
    os.makedirs('static/plots', exist_ok=True)

    # Plot 1: Decision Tree
    fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
    plot_tree(c45_tree, feature_names=X.columns, class_names=['H', 'D', 'A'], filled=True, ax=ax_tree)
    tree_path = f"static/plots/c45_tree_{uuid.uuid4()}.png"
    fig_tree.savefig(tree_path, bbox_inches='tight')
    plt.close(fig_tree)

    # Plot 2: Correlation Heatmap
    corr_matrix = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Feature Correlation Heatmap")
    heatmap_path = f"static/plots/c45_corr_{uuid.uuid4()}.png"
    fig_corr.savefig(heatmap_path, bbox_inches='tight')
    plt.close(fig_corr)

    return {
        'tree_plot': tree_path,
        'heatmap': heatmap_path,
        'metrics': metrics
    }
