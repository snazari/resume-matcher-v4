import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np

# Load your matching results
with open('../output/matched_results.json', 'r') as f:
    data = json.load(f)

# Extract match scores for each job title
job_scores = {}
for resume_path, jobs in data.get('job_matches', {}).items():
    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        if job_title not in job_scores:
            job_scores[job_title] = []
        job_scores[job_title].append(job.get('score', 0) * 100)  # Convert to percentage

# Set up the plot
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")

# Create multiple histograms
for i, (job_title, scores) in enumerate(job_scores.items()):
    plt.subplot(len(job_scores), 1, i + 1)
    sns.histplot(scores, bins=20, kde=True)
    plt.title(f"Match Score Distribution: {job_title}")
    plt.xlabel("Match Score (%)")
    plt.ylabel("Count")

    # Add vertical lines for key statistics
    plt.axvline(np.median(scores), color='r', linestyle='--', label=f'Median: {np.median(scores):.1f}%')
    plt.axvline(np.percentile(scores, 75), color='g', linestyle='--',
                label=f'75th Percentile: {np.percentile(scores, 75):.1f}%')
    plt.legend()

plt.tight_layout()
plt.savefig('match_score_distributions.png', dpi=300)
plt.show()