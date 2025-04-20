import plotly.graph_objects as go
import pandas as pd

def generate_dashboard_data(teams, goals_scored, goals_conceded, points):
    bar_plot = go.Figure(data=[go.Bar(
        x=teams,
        y=points,
        hovertemplate="Team: %{x}<br>Points: %{y}",
    )])

    bar_plot.update_layout(
        title="All Teams by Points",
        xaxis_title="Team",
        yaxis_title="Points",
        template="plotly_white"
    )

    # Create the Goals Scored vs Goals Conceded Scatter Plot
    scatter_plot = go.Figure(data=[go.Scatter(
        x=goals_scored,
        y=goals_conceded,
        mode='markers+text',
        text=teams,
        hovertemplate="Goals Scored: %{x}<br>Goals Conceded: %{y}<br>Team: %{text}<br>Points: %{customdata}",
        customdata=points,  # Adding points as customdata
        marker=dict(size=12, color=points, colorscale='Viridis', showscale=True)
    )])

    scatter_plot.update_layout(
        title="Goals Scored vs Goals Conceded",
        xaxis_title="Goals Scored",
        yaxis_title="Goals Conceded",
        template="plotly_white"
    )

    return bar_plot, scatter_plot
