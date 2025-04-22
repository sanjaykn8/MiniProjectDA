import plotly.graph_objects as go

def generate_dashboard_data(teams, goals_scored, goals_conceded, points):
    # Bar Plot for Points
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

    # Scatter Plot: Goals Scored vs Goals Conceded
    scatter_plot = go.Figure(data=[go.Scatter(
        x=goals_scored,
        y=goals_conceded,
        mode='markers+text',
        text=teams,
        hovertemplate="Goals Scored: %{x}<br>Goals Conceded: %{y}<br>Team: %{text}<br>Points: %{customdata}",
        customdata=points,
        marker=dict(size=12, color=points, colorscale='Viridis', showscale=True)
    )])

    scatter_plot.update_layout(
        title="Goals Scored vs Goals Conceded",
        xaxis_title="Goals Scored",
        yaxis_title="Goals Conceded",
        template="plotly_white"
    )

    pie_chart = go.Figure(data=[go.Pie(
        labels=teams,
        values=points,
        hole=0.3,
        hovertemplate="Team: %{label}<br>Points: %{value} (%{percent})"
    )])

    pie_chart.update_layout(
        title="Points Distribution Among Teams",
        template="plotly_white"
    )

    # Spider (Radar) Chart: Goals Scored, Conceded, and Points
    radar_chart = go.Figure()
    for i, team in enumerate(teams):
        radar_chart.add_trace(go.Scatterpolar(
            r=[goals_scored[i], goals_conceded[i], points[i]],
            theta=["Goals Scored", "Goals Conceded", "Points"],
            fill='toself',
            name=team
        ))

    radar_chart.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title="Team Stats Overview (Radar Chart)",
        showlegend=True,
        template="plotly_white"
    )

    # Stacked Bar Chart: Goals Scored vs Conceded
    stacked_bar = go.Figure(data=[
        go.Bar(name='Goals Scored', x=teams, y=goals_scored, marker_color='green'),
        go.Bar(name='Goals Conceded', x=teams, y=goals_conceded, marker_color='red')
    ])

    stacked_bar.update_layout(
        barmode='stack',
        title="Goals Scored vs Conceded (Stacked Bar)",
        xaxis_title="Team",
        yaxis_title="Goals",
        template="plotly_white"
    )

    # Histogram: Distribution of Points
    histogram = go.Figure(data=[go.Histogram(
        x=points,
        nbinsx=5,
        marker_color='blue',
        hovertemplate="Points: %{x}<br>Count: %{y}"
    )])

    histogram.update_layout(
        title="Distribution of Points",
        xaxis_title="Points",
        yaxis_title="Number of Teams",
        template="plotly_white"
    )

    return bar_plot, scatter_plot, pie_chart, stacked_bar, radar_chart
