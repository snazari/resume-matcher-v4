import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json

# Load your matching results
with open('../output/matches-large.json', 'r') as f:
    data = json.load(f)

# Convert to a format suitable for visualization
matches_list = []
for resume_path, jobs in data.get('job_matches', {}).items():
    for job in jobs:
        matches_list.append({
            'resume_path': resume_path,
            'candidate_name': job.get('candidate_name', 'Unknown'),
            'job_id': job.get('job_id', ''),
            'job_title': job.get('job_title', 'Unknown'),
            'job_company': job.get('job_company', 'Unknown'),
            'match_score': job.get('score', 0) * 100  # Convert to percentage
        })

df = pd.DataFrame(matches_list)

# Create a Dash application
app = dash.Dash(__name__, title="Resume-Job Matcher Dashboard")

app.layout = html.Div([
    html.H1("Resume-Job Matching Results Dashboard"),

    html.Div([
        html.Div([
            html.H3("Filters"),
            html.Label("Minimum Match Score:"),
            dcc.Slider(
                id='score-slider',
                min=0,
                max=100,
                step=5,
                value=40,
                marks={i: f'{i}%' for i in range(0, 101, 10)}
            ),
            html.Label("Job Title:"),
            dcc.Dropdown(
                id='job-dropdown',
                options=[{'label': title, 'value': title} for title in df['job_title'].unique()],
                multi=True
            ),
            html.Label("Company:"),
            dcc.Dropdown(
                id='company-dropdown',
                options=[{'label': company, 'value': company} for company in df['job_company'].unique()],
                multi=True
            ),
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px'}),

        html.Div([
            html.H3("Top Candidates by Job"),
            dcc.Graph(id='heatmap-graph')
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'})
    ]),

    html.Div([
        html.H3("Detailed Match Results"),
        dash_table.DataTable(
            id='results-table',
            columns=[
                {'name': 'Candidate', 'id': 'candidate_name'},
                {'name': 'Job Title', 'id': 'job_title'},
                {'name': 'Company', 'id': 'job_company'},
                {'name': 'Match Score', 'id': 'match_score', 'type': 'numeric', 'format': {'specifier': '.1f'}}
            ],
            sort_action='native',
            filter_action='native',
            page_size=10,
            style_cell={
                'textAlign': 'left',
                'padding': '10px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'match_score', 'filter_query': '{match_score} >= 80'},
                    'backgroundColor': '#baffba',
                    'color': 'black'
                },
                {
                    'if': {'column_id': 'match_score', 'filter_query': '{match_score} < 60'},
                    'backgroundColor': '#ffbaba',
                    'color': 'black'
                }
            ]
        )
    ], style={'padding': '20px'})
])


@app.callback(
    [Output('heatmap-graph', 'figure'),
     Output('results-table', 'data')],
    [Input('score-slider', 'value'),
     Input('job-dropdown', 'value'),
     Input('company-dropdown', 'value')]
)
def update_visualizations(min_score, selected_jobs, selected_companies):
    # Filter data based on user selections
    filtered_df = df[df['match_score'] >= min_score]

    if selected_jobs and len(selected_jobs) > 0:
        filtered_df = filtered_df[filtered_df['job_title'].isin(selected_jobs)]

    if selected_companies and len(selected_companies) > 0:
        filtered_df = filtered_df[filtered_df['job_company'].isin(selected_companies)]

    # Create heatmap
    pivot_df = filtered_df.pivot_table(
        values='match_score',
        index='candidate_name',
        columns='job_title',
        aggfunc='mean'
    ).round(1)

    heatmap = px.imshow(
        pivot_df,
        color_continuous_scale='blues',
        labels=dict(x="Job Title", y="Candidate", color="Match Score (%)"),
        text_auto='.1f'
    )

    heatmap.update_layout(
        title="Candidate-Job Match Heatmap",
        xaxis_title="Job Title",
        yaxis_title="Candidate",
    )

    # Table data
    table_data = filtered_df.to_dict('records')

    return heatmap, table_data


if __name__ == '__main__':
    app.run_server(debug=True)