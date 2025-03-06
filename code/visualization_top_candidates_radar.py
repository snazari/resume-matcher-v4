import matplotlib.pyplot as plt
import numpy as np
import json
from math import pi

# Load matching results and extract top candidates for each job
with open('../output/matches-large.json', 'r') as f:
    data = json.load(f)

# Extract top candidates for each job
top_candidates = {}
for resume_path, jobs in data.get('job_matches', {}).items():
    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        candidate_name = job.get('candidate_name', 'Unknown')
        match_score = job.get('score', 0)

        if job_title not in top_candidates:
            top_candidates[job_title] = []

        top_candidates[job_title].append({
            'name': candidate_name,
            'match_score': match_score,
            'resume_path': resume_path
        })

# Sort candidates by match score and take top 5 for each job
for job_title in top_candidates:
    top_candidates[job_title] = sorted(
        top_candidates[job_title],
        key=lambda x: x['match_score'],
        reverse=True
    )[:5]  # Take top 5

# Simulate specific skill ratings for each candidate
# In a real application, you would extract these from the resume data
skills = ['Technical Skills', 'Education', 'Experience', 'Communication', 'Leadership']
for job_title, candidates in top_candidates.items():
    for candidate in candidates:
        # Generate simulated skill ratings based on match score
        # In reality, you would use actual data from your resume extraction
        candidate['skill_ratings'] = {
            skill: min(1.0, candidate['match_score'] * (0.8 + np.random.random() * 0.4))
            for skill in skills
        }

# Create radar charts
for job_title, candidates in top_candidates.items():
    # Number of skills (dimensions)
    N = len(skills)

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Set figure size
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], skills, size=12)

    # Set y-axis limits
    ax.set_ylim(0, 1)

    # Draw the candidates' skill ratings
    for i, candidate in enumerate(candidates):
        # Extract skill ratings in the same order as skills list
        values = [candidate['skill_ratings'][skill] for skill in skills]
        values += values[:1]  # Close the polygon

        # Plot skill ratings
        ax.plot(angles, values, linewidth=2, label=f"{candidate['name']} ({candidate['match_score'] * 100:.1f}%)")
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Add title
    plt.title(f"Top Candidates for {job_title}", size=20, y=1.1)

    plt.tight_layout()
    plt.savefig(f'radar_chart_{job_title.replace(" ", "_")}.png', dpi=300)
    plt.close()