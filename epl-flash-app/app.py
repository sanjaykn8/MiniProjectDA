from flask import Flask, render_template, request
from visualizations.apriori_fp import generate_apriori_fp_plots
from visualizations.bayes import generate_bayes_visuals
from visualizations.id3 import generate_id3_visuals  
from visualizations.cart import generate_cart_visuals
from visualizations.c45 import generate_c45_visuals
from visualizations.knn import generate_knn_visuals
from visualizations.kmeans import generate_kmeans_visuals
from visualizations.agglomerative import generate_agglomerative_visuals
from visualizations.dbscan import generate_dbscan_visuals
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

@app.route('/')
def index():
    algorithms = [
        "Apriori & FP Growth",
        "Naïve Bayes",
        "ID3 Decision Tree",
        "CART",
        "C4.5",
        "KNN",
        "KMeans",
        "Agglomerative",
        "DBScan",
        "Predict Match"
    ]
    return render_template('index.html', algorithms=algorithms)

@app.route('/visualize', methods=['POST'])
def visualize():
    selected = request.form.get('algo')
    
    if selected == "Apriori & FP Growth":
        plots = generate_apriori_fp_plots()
        return render_template('apr.html', plots=plots, algo=selected)
    
    elif selected == "Naïve Bayes":
        results = generate_bayes_visuals()
        return render_template('bayes.html', metrics=results, algo=selected)
    
    elif selected == "ID3 Decision Tree":
        results = generate_id3_visuals()
        return render_template('id3.html', metrics=results, algo=selected)
    
    elif selected == "CART":
        results = generate_cart_visuals()
        return render_template('cart.html', metrics=results['metrics'], tree=results['tree_plot'], heatmap=results['heatmap'], algo=selected)
    
    elif selected == "C4.5":
        results = generate_c45_visuals()
        return render_template('c45.html', metrics=results['metrics'], tree=results['tree_plot'], heatmap=results['heatmap'], algo=selected)
    
    elif selected == "KNN":
        knn_data = generate_knn_visuals()
        with open(knn_data["classification_report"], "r") as f:
            report_text = f.read()
        return render_template("knn_result.html",
                               accuracy=knn_data["accuracy"],
                               report=report_text,
                               cm_image=knn_data["confusion_matrix"],
                               algo=selected)
        
    elif selected == "KMeans":
        results = generate_kmeans_visuals()
        return render_template('kmeans.html', 
                            elbow_plot=results['elbow_plot'], 
                            scatter_plot=results['scatter_plot'], 
                            silhouette_score=results['silhouette_score'], 
                            algo=selected)
        
    elif selected == "Agglomerative": 
        results = generate_agglomerative_visuals()
        return render_template('agglomerative.html', 
                            dendrogram=results['dendrogram'], 
                            scatter_plot=results['scatter_plot'], 
                            silhouette_score=results['silhouette_score'], 
                            algo=selected)
    
    elif selected == "DBScan":
        results = generate_dbscan_visuals()
        return render_template('dbscan.html', 
                            kd_graph=results['kd_graph'], 
                            scatter_plot=results['scatter_plot'], 
                            silhouette_score=results['silhouette_score'], 
                            algo=selected)
    
    elif selected == "Predict Match":
        return render_template('predict_match.html')

    return "Algorithm not implemented yet", 400

@app.route('/predict_match', methods=['POST'])
def predict_match():
    df = pd.read_csv('EPL.csv')
    def compute_team_stats(df):
        team_stats = {}

        for team in pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K')):
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]

            home_wins = (home_matches['FullTimeResult'] == 'H').sum()
            away_wins = (away_matches['FullTimeResult'] == 'A').sum()
            home_games = len(home_matches)
            away_games = len(away_matches)

            home_win_pct = home_wins / home_games if home_games > 0 else 0
            away_win_pct = away_wins / away_games if away_games > 0 else 0

            home_avg_goals = home_matches['FullTimeHomeTeamGoals'].mean() if home_games > 0 else 0
            away_avg_goals = away_matches['FullTimeAwayTeamGoals'].mean() if away_games > 0 else 0
            home_avg_shots = home_matches['HomeTeamShots'].mean() if home_games > 0 else 0
            away_avg_shots = away_matches['AwayTeamShots'].mean() if away_games > 0 else 0

            home_avg_shots_on_target = home_matches['HomeTeamShotsOnTarget'].mean() if home_games > 0 else 0
            away_avg_shots_on_target = away_matches['AwayTeamShotsOnTarget'].mean() if away_games > 0 else 0
            home_avg_fouls = home_matches['HomeTeamFouls'].mean() if home_games > 0 else 0
            away_avg_fouls = away_matches['AwayTeamFouls'].mean() if away_games > 0 else 0

            team_stats[team] = {
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'home_avg_goals': home_avg_goals,
                'away_avg_goals': away_avg_goals,
                'home_avg_shots': home_avg_shots,
                'away_avg_shots': away_avg_shots,
                'home_avg_shots_on_target': home_avg_shots_on_target,
                'away_avg_shots_on_target': away_avg_shots_on_target,
                'home_avg_fouls': home_avg_fouls,
                'away_avg_fouls': away_avg_fouls
            }
        return team_stats
    team_stats = compute_team_stats(df)

    features = []
    goal_labels = []
    shot_labels = []
    shot_on_target_labels = []
    foul_labels = []

    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team in team_stats and away_team in team_stats:
            home = team_stats[home_team]
            away = team_stats[away_team]

            features.append([
                home['home_win_pct'], away['away_win_pct'],
                home['home_avg_goals'], away['away_avg_goals'],
                home['home_avg_shots'], away['away_avg_shots'],
                home['home_avg_shots_on_target'], away['away_avg_shots_on_target'],
                home['home_avg_fouls'], away['away_avg_fouls']
            ])

            goal_labels.append([row['FullTimeHomeTeamGoals'], row['FullTimeAwayTeamGoals']])  
            shot_labels.append([row['HomeTeamShots'], row['AwayTeamShots']])
            shot_on_target_labels.append([row['HomeTeamShotsOnTarget'], row['AwayTeamShotsOnTarget']])
            foul_labels.append([row['HomeTeamFouls'], row['AwayTeamFouls']])
                
    X = np.array(features)
    y_goals = np.array(goal_labels)
    y_shots = np.array(shot_labels)
    y_shots_on_target = np.array(shot_on_target_labels)
    y_fouls = np.array(foul_labels)

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y_goals, test_size=0.1, random_state=42)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_shots, test_size=0.1, random_state=42)
    X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(X, y_shots_on_target, test_size=0.1, random_state=42)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fouls, test_size=0.1, random_state=42)

    goal_regressor = DecisionTreeRegressor(random_state=42)
    goal_regressor.fit(X_train_g, y_train_g)

    shot_regressor = DecisionTreeRegressor(random_state=42)
    shot_regressor.fit(X_train_s, y_train_s)

    shot_on_target_regressor = DecisionTreeRegressor(random_state=42)
    shot_on_target_regressor.fit(X_train_st, y_train_st)

    foul_regressor = DecisionTreeRegressor(random_state=42)
    foul_regressor.fit(X_train_f, y_train_f)
        
    def predict_match_result(home_team, away_team):
        if home_team not in team_stats or away_team not in team_stats:
            return "Invalid teams!"

        home = team_stats[home_team]
        away = team_stats[away_team]

        input_features = np.array([[
            home['home_win_pct'], away['away_win_pct'],
            home['home_avg_goals'], away['away_avg_goals'],
            home['home_avg_shots'], away['away_avg_shots'],
            home['home_avg_shots_on_target'], away['away_avg_shots_on_target'],
            home['home_avg_fouls'], away['away_avg_fouls']
        ]]).reshape(1, -1)

        goal_pred = goal_regressor.predict(input_features)[0]  
        shot_pred = shot_regressor.predict(input_features)[0]
        shot_on_target_pred = shot_on_target_regressor.predict(input_features)[0]
        foul_pred = foul_regressor.predict(input_features)[0]

        home_goals, away_goals = int(round(goal_pred[0])), int(round(goal_pred[1]))
        home_shots, away_shots = int(round(shot_pred[0])), int(round(shot_pred[1]))
        home_shots_on_target, away_shots_on_target = int(round(shot_on_target_pred[0])), int(round(shot_on_target_pred[1]))
        home_fouls, away_fouls = int(round(foul_pred[0])), int(round(foul_pred[1]))

        if home_goals > away_goals:
             result_pred = "H"  
        elif away_goals > home_goals:
            result_pred = "A"  
        else:
            result_pred = "D"  

        result =  f"""
            Predicted Outcome: {result_pred}
            Expected Full-Time Goals: {home_team} {home_goals} - {away_goals} {away_team}
            Expected Shots: {home_team} {home_shots} - {away_shots} {away_team}
            Expected Shots on Target: {home_team} {home_shots_on_target} - {away_shots_on_target} {away_team}
            Expected Fouls: {home_team} {home_fouls} - {away_fouls} {away_team}
            """
        return result

    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')

    result = predict_match_result(home_team, away_team)
    return render_template('prediction_result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
