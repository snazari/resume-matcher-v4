import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np

# Load your matching results
with open('../output/matched_results.json', 'r') as f:
    data = json.load(f)

# Collect candidate rankings for each job
job_rankings = {}
candidate_names = set()

for resume_path, jobs in data.get('job_matches', {}).items():
    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        candidate_name = job.get('candidate_name', 'Unknown')
        match_score = job.get('score', 0)

        candidate_names.add(candidate_name)

        if job_title not in job_rankings:
            job_rankings[job_title] = []

        job_rankings[job_title].append({
            'candidate': candidate_name,
            'score': match_score
        })

# Sort candidates by score for each job and assign ranks
for job_title in job_rankings:
    job_rankings[job_title] = sorted(
        job_rankings[job_title],
        key=lambda x: x['score'],
        reverse=True
    )

    # Assign ranks
    for i, entry in enumerate(job_rankings[job_title]):
        entry['rank'] = i + 1

# Create a dataframe for the bump chart
bump_data = []
for job_title, rankings in job_rankings.items():
    for entry in rankings:
        if entry['rank'] <= 10:  # Only include top 10 for clarity
            bump_data.append({
                'Job Title': job_title,
                'Candidate': entry['candidate'],
                'Rank': entry['rank'],
                'Score': entry['score'] * 100  # Convert to percentage
            })

bump_df = pd.DataFrame(bump_data)

# Create the bump chart
fig = go.Figure()

# Get unique job titles and candidates
job_titles = sorted(bump_df['Job Title'].unique())
candidates = sorted(candidate_names)

# Add a line for each candidate
for candidate in candidates:
    candidate_data = bump_df[bump_df['Candidate'] == candidate]

    if not candidate_data.empty:
        # Only include candidates who appear in the rankings
        fig.add_trace(go.Scatter(
            x=candidate_data['Job Title'],
            y=candidate_data['Rank'],
            mode='lines+markers',
            name=candidate,
            line=dict(width=2),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>' +
                          'Candidate: ' + candidate + '<br>' +
                          'Rank: %{y}<br>' +
                          'Score: %{text:.1f}%',
            text=candidate_data['Score']
        ))

# Customize layout
fig.update_layout(
    title="Candidate Rankings Across Job Positions",
    xaxis_title="Job Position",
    yaxis_title="Rank",
    yaxis=dict(
        autorange="reversed",  # Invert y-axis so rank 1 is at the top
        tickmode='linear',
        tick0=1,
        dtick=1
    ),
    hovermode="closest",
    legend_title="Candidates",
    height=800,
    width=1200
)

fig.write_html("candidate_rankings_bump_chart.html")