# visualizations/knn.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import uuid

def generate_knn_visuals():
    # Load data and define features/target
    df = pd.read_csv("EPL.csv") 
    features = [
        "HomeTeamShots", "AwayTeamShots", "HomeTeamShotsOnTarget", "AwayTeamShotsOnTarget",
        "HomeTeamCorners", "AwayTeamCorners", "HomeTeamFouls", "AwayTeamFouls",
        "HomeTeamYellowCards", "AwayTeamYellowCards", "B365HomeTeam", "B365Draw", "B365AwayTeam"
    ]
    target = "FullTimeResult"
    
    # Encode target labels
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    df.dropna(subset=features + [target], inplace=True)

    # Prepare features and labels
    X = df[features]
    y = df[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train KNN with k=100
    k = 100
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Compute metrics and report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create folder to store plots
    os.makedirs('static/plots', exist_ok=True)
    
    # Classification report file
    report_path = f"static/plots/knn_report_{uuid.uuid4()}.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n{report}")
    
    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    
    cm_path = f"static/plots/knn_cm_{uuid.uuid4()}.png"
    fig_cm.savefig(cm_path, bbox_inches='tight')
    plt.close(fig_cm)
    
    return {
        'accuracy': round(accuracy, 4),
        'classification_report': report_path,
        'confusion_matrix': cm_path
    }
