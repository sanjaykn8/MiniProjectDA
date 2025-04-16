import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend

def generate_apriori_fp_plots():
    dataset_path = "EPL.csv"
    
    # Read CSV into list of transactions (skip header)
    transactions = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            transactions.append([item.strip() for item in line.strip().split(',') if item.strip() != ""])
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    def get_rules(df, algo="apriori"):
        if algo == "apriori":
            frequent = apriori(df, min_support=0.1, use_colnames=True)
        else:
            frequent = fpgrowth(df, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=2)
        rules = rules[
            (rules['confidence'] >= 0.1) &
            (rules['antecedents'].apply(lambda x: len(x) >= 1))
        ]
        return rules

    # Ensure directory exists
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Generate scatter plots for Apriori and FP-Growth
    for algo in ["apriori", "fp"]:
        rules = get_rules(df, algo)
        if rules.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='coolwarm')
        plt.colorbar(scatter, ax=ax)
        ax.set_title(f"{algo.upper()} Scatter Plot")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        path = os.path.join(output_dir, f"{algo}_scatter.png")
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)

    # Generate network graph for Apriori rules
    rules = get_rules(df, "apriori").sort_values(by='lift', ascending=False).head(10)
    if not rules.empty:
        G = nx.DiGraph()
        for _, rule in rules.iterrows():
            for a in rule['antecedents']:
                for c in rule['consequents']:
                    G.add_edge(a, c, weight=rule['lift'])
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(6, 4))
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
        labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
        ax.set_title("Association Rules Network")
        path = os.path.join(output_dir, "network_graph.png")
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)

    return [path.replace("static/", "").replace("\\", "/") for path in plot_paths]

