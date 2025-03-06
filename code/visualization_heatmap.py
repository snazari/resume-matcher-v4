import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns

# Load your matching results
with open('../output/matched_results.json', 'r') as f:
    data = json.load(f)

# Define job requirement categories
# In a real application, these would be extracted from job descriptions
job_requirements = {
    'Software Engineer': {
        'Technical Skills': 0.5,
        'Education': 0.2,
        'Experience': 0.2,
        'Communication': 0.1
    },
    'Data Scientist': {
        'Technical Skills': 0.4,
        'Education': 0.3,
        'Experience': 0.2,
        'Communication': 0.1
    },
    'Project Manager': {
        'Technical Skills': 0.2,
        'Education': 0.1,
        'Experience': 0.4,
        'Communication': 0.3
    }
}


# Simulate candidate scores in each category
# In a real application, you would extract these from resume analysis
def simulate_candidate_scores(resume_data, match_score):
    """Simulate category scores based on resume data and match score"""
    # Extract real data where available
    years_exp = resume_data.get('years_experience', 0)
    degrees = resume_data.get('degrees', [])
    education_level = 0
    for degree in degrees:
        if 'phd' in degree.lower() or 'doctor' in degree.lower():
            education_level = 3
            break
        elif 'master' in degree.lower() or 'ms' in degree.lower() or 'ma' in degree.lower():
            education_level = max(education_level, 2)
        elif 'bachelor' in degree.lower() or 'bs' in degree.lower() or 'ba' in degree.lower():
            education_level = max(education_level, 1)

    # Normalize to 0-1 range
    education_score = min(1.0, education_level / 3)
    experience_score = min(1.0, years_exp / 10)

    # Simulate technical and communication scores with some randomness
    # but weighted by the match score to simulate relevance
    base_technical = 0.3 + 0.7 * match_score
    technical_score = base_technical * (0.8 + 0.2 * np.random.random())

    base_communication = 0.5 + 0.5 * match_score
    communication_score = base_communication * (0.8 + 0.2 * np.random.random())

    return {
        'Technical Skills': technical_score,
        'Education': education_score,
        'Experience': experience_score,
        'Communication': communication_score
    }


# Collect alignment data for visualization
alignment_data = []

# Process each resume-job match
for resume_path, jobs in data.get('job_matches', {}).items():
    # Get resume data
    resume_info = next((r for r in data.get('results', []) if r.get('file_path') == resume_path), {})

    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        candidate_name = job.get('candidate_name', 'Unknown')
        match_score = job.get('score', 0)

        # Skip if job title is not in our predefined list
        if job_title not in job_requirements:
            continue

        # Get job requirement weights
        req_weights = job_requirements[job_title]

        # Get candidate category scores
        candidate_scores = simulate_candidate_scores(resume_info, match_score)

        # Calculate weighted alignment for each category
        for category, weight in req_weights.items():
            candidate_score = candidate_scores.get(category, 0)
            weighted_alignment = candidate_score * weight

            alignment_data.append({
                'Candidate': candidate_name,
                'Job Title': job_title,
                'Category': category,
                'Weight': weight,
                'Score': candidate_score,
                'Weighted Alignment': weighted_alignment,
                'Match Score': match_score * 100
            })

# Convert to dataframe
alignment_df = pd.DataFrame(alignment_data)

# Create a pivot table for the heatmap
pivot_df = alignment_df.pivot_table(
    values='Score',
    index=['Candidate', 'Job Title'],
    columns='Category',
    aggfunc='mean'
)

# Create a multi-index heatmap
plt.figure(figsize=(14, len(pivot_df) * 0.5 + 2))
sns.heatmap(
    pivot_df,
    annot=True,
    cmap='Blues',
    linewidths=0.5,
    vmin=0,
    vmax=1,
    fmt='.2f',
    cbar_kws={'label': 'Score (0-1)'}
)
plt.title('Candidate-Job Requirement Alignment')
plt.tight_layout()
plt.savefig('requirement_alignment_heatmap.png', dpi=300)
plt.close()

# Create a second visualization showing the weighted alignment
weighted_pivot = alignment_df.pivot_table(
    values='Weighted Alignment',
    index=['Candidate', 'Job Title'],
    columns='Category',
    aggfunc='mean'
)

# Add a total column
weighted_pivot['Total Alignment'] = weighted_pivot.sum(axis=1)

# Sort by total alignment
weighted_pivot = weighted_pivot.sort_values('Total Alignment', ascending=False)

# Create the heatmap
plt.figure(figsize=(16, len(weighted_pivot) * 0.5 + 2))
sns.heatmap(
    weighted_pivot,
    annot=True,
    cmap='Greens',
    linewidths=0.5,
    vmin=0,
    vmax=weighted_pivot.drop('Total Alignment', axis=1).values.max(),
    fmt='.2f',
    cbar_kws={'label': 'Weighted Alignment'}
)
plt.title('Weighted Candidate-Job Requirement Alignment')
plt.tight_layout()
plt.savefig('weighted_alignment_heatmap.png', dpi=300)
plt.close()