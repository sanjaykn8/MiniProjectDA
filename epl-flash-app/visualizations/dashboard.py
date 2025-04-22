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
    
    violin_plot_scored = go.Figure(data=go.Violin(
        y=goals_scored,
        box_visible=True,
        line_color='green',
        fillcolor='rgba(0, 250, 0, 0.5)',
        hovertemplate="Goals Scored: %{y}<br>",
    ))

    violin_plot_scored.update_layout(
        title="Distribution of Goals Scored",
        yaxis_title="Goals Scored",
        template="plotly_white"
    )

    # Violin Plot for Goals Conceded
    violin_plot_conceded = go.Figure(data=go.Violin(
        y=goals_conceded,
        box_visible=True,
        line_color='red',
        fillcolor='rgba(250, 0, 0, 0.5)',
        hovertemplate="Goals Conceded: %{y}<br>",
    ))

    violin_plot_conceded.update_layout(
        title="Distribution of Goals Conceded",
        yaxis_title="Goals Conceded",
        template="plotly_white"
    )

    return bar_plot, scatter_plot, violin_plot_conceded, violin_plot_scored
