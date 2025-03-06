import plotly.express as px
import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA

# Load your matching results
with open('../output/matches-large.json', 'r') as f:
    data = json.load(f)

# Get the resume data and job matches
resume_data = data.get('results', [])
job_matches = data.get('job_matches', {})

# Create a features dictionary for each resume
# In a real scenario, these would be actual features extracted from resumes
# Here, we'll simulate some features
resume_features = {}

for result in resume_data:
    if 'error' not in result:
        file_path = result.get('file_path', '')
        years_exp = result.get('years_experience', 0)

        # Count degrees by type
        degrees = result.get('degrees', [])
        bachelors_count = sum(1 for d in degrees if 'bachelor' in d.lower() or 'bs' in d.lower() or 'ba' in d.lower())
        masters_count = sum(1 for d in degrees if 'master' in d.lower() or 'ms' in d.lower() or 'ma' in d.lower())
        phd_count = sum(1 for d in degrees if 'phd' in d.lower() or 'doctor' in d.lower())

        # Create feature vector (in a real app, this would include skills, etc.)
        resume_features[file_path] = {
            'years_experience': years_exp,
            'education_level': bachelors_count + masters_count * 2 + phd_count * 3,
            'num_degrees': len(degrees),
            # Simulate some additional features
            'technical_score': np.random.uniform(0.5, 1.0) * years_exp / 10 + masters_count * 0.2,
            'management_score': np.random.uniform(0.3, 1.0) * years_exp / 15 + phd_count * 0.3,
            'communication_score': np.random.uniform(0.7, 1.0)
        }

# Create a dataframe with all resume features
features_df = pd.DataFrame.from_dict(resume_features, orient='index')
features_df['file_path'] = features_df.index
features_df.reset_index(drop=True, inplace=True)

# Add candidate names from job matches
features_df['candidate_name'] = 'Unknown'
for file_path, matches in job_matches.items():
    if matches:  # If there are matches
        candidate_name = matches[0].get('candidate_name', 'Unknown')
        features_df.loc[features_df['file_path'] == file_path, 'candidate_name'] = candidate_name

# Add best match info
features_df['best_match_job'] = ''
features_df['best_match_score'] = 0.0

for file_path, matches in job_matches.items():
    if matches:
        best_match = max(matches, key=lambda x: x.get('score', 0))
        best_job = best_match.get('job_title', 'Unknown')
        best_score = best_match.get('score', 0)

        features_df.loc[features_df['file_path'] == file_path, 'best_match_job'] = best_job
        features_df.loc[features_df['file_path'] == file_path, 'best_match_score'] = best_score * 100

# Use PCA to reduce to 3 dimensions for visualization
feature_columns = ['years_experience', 'education_level', 'num_degrees',
                   'technical_score', 'management_score', 'communication_score']

# Normalize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df[feature_columns])

# Apply PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_features)

# Add PCA results to dataframe
features_df['pca_1'] = pca_result[:, 0]
features_df['pca_2'] = pca_result[:, 1]
features_df['pca_3'] = pca_result[:, 2]

# Create 3D scatter plot
fig = px.scatter_3d(
    features_df,
    x='pca_1',
    y='pca_2',
    z='pca_3',
    color='best_match_job',
    size='best_match_score',
    hover_name='candidate_name',
    hover_data={
        'pca_1': False,
        'pca_2': False,
        'pca_3': False,
        'years_experience': True,
        'education_level': True,
        'best_match_score': True,
        'best_match_job': True
    },
    opacity=0.7,
    title='3D Visualization of Candidates by Feature Space'
)

# Update marker size reference
fig.update_traces(marker=dict(sizeref=0.05))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Feature Dimension 1',
        yaxis_title='Feature Dimension 2',
        zaxis_title='Feature Dimension 3',
    ),
    legend_title="Best Matching Job",
    width=1000,
    height=800
)

# Save as interactive HTML
fig.write_html("candidate_3d_visualization.html")