import pandas as pd
import plotly.express as px

def generate_team_cluster_plot(selected_team):
    df = pd.read_csv("EPL.csv")

    # Aggregate team stats
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    team_data = []

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]

        stats = {
            'Team': team,
            'Avg Goals Scored': (home['FullTimeHomeTeamGoals'].mean() + away['FullTimeAwayTeamGoals'].mean()) / 2,
            'Avg Goals Conceded': (home['FullTimeAwayTeamGoals'].mean() + away['FullTimeHomeTeamGoals'].mean()) / 2,
            'Avg Shots': (home['HomeTeamShots'].mean() + away['AwayTeamShots'].mean()) / 2,
            'Avg Fouls': (home['HomeTeamFouls'].mean() + away['AwayTeamFouls'].mean()) / 2
        }
        team_data.append(stats)

    df_team_stats = pd.DataFrame(team_data)

    fig = px.scatter(df_team_stats,
                     x='Avg Goals Scored',
                     y='Avg Goals Conceded',
                     size='Avg Shots',
                     color='Avg Fouls',
                     hover_name='Team',
                     title=f"Visualization for {selected_team}")

    fig.add_scatter(x=[df_team_stats[df_team_stats['Team'] == selected_team]['Avg Goals Scored'].values[0]],
                    y=[df_team_stats[df_team_stats['Team'] == selected_team]['Avg Goals Conceded'].values[0]],
                    mode='markers+text',
                    marker=dict(size=20, color='red'),
                    text=[selected_team],
                    textposition="top center",
                    name=selected_team)

    return fig
