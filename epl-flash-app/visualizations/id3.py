import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from math import log2
import io, base64
import seaborn as sns

def entropy(data):
    total = len(data)
    counts = Counter(data)
    return -sum((count / total) * log2(count / total) for count in counts.values())

def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = sum(
        (len(df[df[feature] == v]) / len(df)) * entropy(df[df[feature] == v][target])
        for v in values
    )
    return total_entropy - weighted_entropy

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def generate_id3_visuals():
    df = pd.read_csv("EPL.csv")
    df = df[['HomeTeam', 'AwayTeam', 'FullTimeResult', 'HalfTimeResult']]
    df = df.apply(lambda x: pd.factorize(x)[0])

    info_gains = {feature: information_gain(df, feature, 'FullTimeResult') for feature in df.columns[:-1]}
    best_feature = max(info_gains, key=info_gains.get)

    X = df.drop(columns=['FullTimeResult'])
    y = df['FullTimeResult']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=3,
                                        min_samples_split=10, min_samples_leaf=5)
    tree_model.fit(X_train, y_train)

    # Plot the tree
    fig_tree = plt.figure(figsize=(10, 6))
    plot_tree(tree_model, feature_names=X.columns, class_names=['H', 'D', 'A'], filled=True)
    tree_img = fig_to_base64(fig_tree)

    # Confusion matrix
    y_pred = tree_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - ID3")
    cm_img = fig_to_base64(fig_cm)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
        "f1": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "best_feature": best_feature,
        "info_gains": info_gains,
        "tree_img": tree_img,
        "cm_img": cm_img
    }

    return metrics
