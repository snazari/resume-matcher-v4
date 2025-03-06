import networkx as nx
import plotly.graph_objects as go
import numpy as np
import json

# Load your matching results
with open('matched_results.json', 'r') as f:
    data = json.load(f)

# Create a graph
G = nx.Graph()

# Add nodes and edges
for resume_path, jobs in data.get('job_matches', {}).items():
    candidate_name = jobs[0].get('candidate_name', 'Unknown')

    # Add candidate node
    G.add_node(candidate_name, type='candidate')

    # Add job nodes and edges
    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        job_id = job.get('job_id', '')
        node_id = f"{job_title}_{job_id}"

        # Only add edges for matches above a threshold
        match_score = job.get('score', 0)
        if match_score >= 0.6:  # 60% match threshold
            G.add_node(node_id, type='job', title=job_title)
            G.add_edge(candidate_name, node_id, weight=match_score)

# Create position layout
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Create edge trace
edge_x = []
edge_y = []
edge_weights = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_weights.append(edge[2]['weight'])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines')

# Create node traces for candidates and jobs
candidate_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'candidate']
job_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'job']

candidate_trace = go.Scatter(
    x=[pos[node][0] for node in candidate_nodes],
    y=[pos[node][1] for node in candidate_nodes],
    text=[node for node in candidate_nodes],
    mode='markers+text',
    marker=dict(
        size=15,
        color='lightblue',
        line=dict(width=2)
    ),
    textposition="top center",
    hoverinfo='text'
)

job_trace = go.Scatter(
    x=[pos[node][0] for node in job_nodes],
    y=[pos[node][1] for node in job_nodes],
    text=[G.nodes[node]['title'] for node in job_nodes],
    mode='markers+text',
    marker=dict(
        size=15,
        color='lightgreen',
        symbol='square',
        line=dict(width=2)
    ),
    textposition="bottom center",
    hoverinfo='text'
)

# Create the figure
fig = go.Figure(data=[edge_trace, candidate_trace, job_trace],
                layout=go.Layout(
                    title="Candidate-Job Match Network",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

fig.show()