from flask import Flask, render_template, request
from visualizations.id3 import generate_id3_visuals  
from visualizations.cart import generate_cart_visuals
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import random

app = Flask(__name__)

@app.route('/')
def index():
    algorithms = [
        "ID3 Decision Tree",
        "CART",
        "Predict Match"
    ]
    return render_template('index.html', algorithms=algorithms)

@app.route('/visualize', methods=['POST'])
def visualize():
    selected = request.form.get('algo')
    if selected == "ID3 Decision Tree":
        results = generate_id3_visuals()
        return render_template('id3.html', metrics=results, algo=selected)
    
    elif selected == "CART":
        results = generate_cart_visuals()
        return render_template('cart.html', metrics=results['metrics'], tree=results['tree_plot'], heatmap=results['heatmap'], algo=selected)
    
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
        
    def tournament():
        teams = request.form.getlist('team[]')
        
        team_stats = compute_team_stats(df)

        # Prepare training data
        features, goal_labels, shot_labels, shot_on_target_labels, foul_labels = [], [], [], [], []
        for _, row in df.iterrows():
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            if home_team in team_stats and away_team in team_stats:
                home, away = team_stats[home_team], team_stats[away_team]
                features.append([home['home_win_pct'], away['away_win_pct'],
                                 home['home_avg_goals'], away['away_avg_goals'],
                                 home['home_avg_shots'], away['away_avg_shots'],
                                 home['home_avg_shots_on_target'], away['away_avg_shots_on_target'],
                                 home['home_avg_fouls'], away['away_avg_fouls']])
                goal_labels.append([row['FullTimeHomeTeamGoals'], row['FullTimeAwayTeamGoals']])

        X = np.array(features)
        y_goals = np.array(goal_labels)

        X_train_g, _, y_train_g, _ = train_test_split(X, y_goals, test_size=0.1, random_state=42)
        goal_regressor = DecisionTreeRegressor(random_state=42)
        goal_regressor.fit(X_train_g, y_train_g)

        def predict_match_result(home_team, away_team):
            if home_team not in team_stats or away_team not in team_stats:
                print(f"Warning: Missing team stats for {home_team} or {away_team}. Returning default score.")
                return 0, 0
            home, away = team_stats[home_team], team_stats[away_team] 
            input_features = np.array([[
                home['home_win_pct'], away['away_win_pct'],
                home['home_avg_goals'], away['away_avg_goals'],
                home['home_avg_shots'], away['away_avg_shots'],
                home['home_avg_shots_on_target'], away['away_avg_shots_on_target'],
                home['home_avg_fouls'], away['away_avg_fouls']
            ]])
            goal_pred = goal_regressor.predict(input_features)[0]
            return int(round(goal_pred[0])), int(round(goal_pred[1]))

        def simulate_round(team_list):
            stats = {
                team: {
                    'played': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'points': 0,
                    'goals_scored': 0,
                    'goals_conceded': 0
                } for team in team_list
            }
            results = []

            for i in range(len(team_list)):
                for j in range(len(team_list)):
                    if i != j:
                        home, away = team_list[i], team_list[j]
                        hg, ag = predict_match_result(home, away)
                        stats[home]['played'] += 1
                        stats[away]['played'] += 1
                        stats[home]['goals_scored'] += hg
                        stats[home]['goals_conceded'] += ag
                        stats[away]['goals_scored'] += ag
                        stats[away]['goals_conceded'] += hg

                        if hg > ag:
                            stats[home]['points'] += 3
                            stats[home]['wins'] += 1
                            stats[away]['losses'] += 1
                        elif hg < ag:
                            stats[away]['points'] += 3
                            stats[away]['wins'] += 1
                            stats[home]['losses'] += 1
                        else:
                            stats[home]['points'] += 1
                            stats[away]['points'] += 1
                            stats[home]['draws'] += 1
                            stats[away]['draws'] += 1

                        results.append((home, hg, away, ag))

            return stats, results

        round1_stats, group_results = simulate_round(teams)
        sorted_teams = sorted(teams, key=lambda x: (round1_stats[x]['points'], round1_stats[x]['goals_scored'] - round1_stats[x]['goals_conceded']), reverse=True)
        semis = sorted_teams[:4]

        semi_stats, semi_results = simulate_round(semis)
        sorted_semis = sorted(semis, key=lambda x: (semi_stats[x]['points'], semi_stats[x]['goals_scored'] - semi_stats[x]['goals_conceded']), reverse=True)
        finalists = sorted_semis[:2]

        # Finals
        final_home1, final_away1 = finalists[0], finalists[1]
        fg1, fg2 = predict_match_result(final_home1, final_away1)
        fg3, fg4 = predict_match_result(final_away1, final_home1)
        final_score1 = fg1 + fg4
        final_score2 = fg2 + fg3
        winner = final_home1 if final_score1 > final_score2 else final_away1 if final_score2 > final_score1 else "Draw"
        if final_score1 == final_score2:
            winner = random.choice([final_home1, final_away1])  # or simulate penalties

        return render_template(
            'prediction_result.html',
            results=group_results,
            points_table=round1_stats,  # updated
            semi_finals=semis,
            semi_results=semi_results,
            finalists=finalists,
            final_match=f"{final_home1} ({fg1} + {fg4}) vs {final_away1} ({fg2} + {fg3})",
            winner=winner
        )

    return tournament()

if __name__ == "__main__":
    app.run(debug=True)
