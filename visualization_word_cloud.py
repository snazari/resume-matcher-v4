from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os
import re

# Load your matching results
with open('matched_results-updated.json', 'r') as f:
    data = json.load(f)

# Extract results data
result_data = data.get('results', [])

# Create a mapping of resume paths to extracted skills
# For this example, we'll simulate skills extraction
# In a real scenario, you'd use the actual skills extracted from resumes
resume_skills = {}
for result in result_data:
    if 'error' not in result:
        # Simulate extracting skills based on degrees and experience
        # In a real application, you'd have actual skills extracted
        skills = []

        # Add skills based on degrees
        for degree in result.get('degrees', []):
            degree_lower = degree.lower()
            if 'computer' in degree_lower or 'software' in degree_lower:
                skills.extend(['Python', 'Java', 'SQL', 'Machine Learning', 'Data Structures'])
            elif 'data' in degree_lower:
                skills.extend(['Python', 'R', 'SQL', 'Data Analysis', 'Statistics'])
            elif 'engineering' in degree_lower:
                skills.extend(['CAD', 'Project Management', 'Mathematics', 'Problem Solving'])
            elif 'business' in degree_lower or 'mba' in degree_lower:
                skills.extend(['Leadership', 'Management', 'Strategy', 'Finance', 'Marketing'])
            elif 'science' in degree_lower:
                skills.extend(['Research', 'Analysis', 'Technical Writing', 'Laboratory Skills'])

        # Add generic skills based on years of experience
        experience = result.get('years_experience', 0)
        if experience > 8:
            skills.extend(['Leadership', 'Team Management', 'Strategy', 'Mentoring'])
        elif experience > 5:
            skills.extend(['Project Management', 'Team Collaboration', 'Communication'])
        elif experience > 2:
            skills.extend(['Problem Solving', 'Time Management', 'Attention to Detail'])
        else:
            skills.extend(['Adaptability', 'Learning', 'Basic Communication'])

        # Store skills for this resume
        resume_skills[result.get('file_path', '')] = skills

# Collect skills from top matches
threshold = 0.7  # Only consider matches with score above 70%
skills_by_job = {}

for resume_path, jobs in data.get('job_matches', {}).items():
    for job in jobs:
        job_title = job.get('job_title', 'Unknown')
        match_score = job.get('score', 0)

        if match_score >= threshold:
            if job_title not in skills_by_job:
                skills_by_job[job_title] = []

            # Add skills for this resume-job match
            if resume_path in resume_skills:
                skills_by_job[job_title].extend(resume_skills[resume_path])

# Create word clouds for each job
for job_title, skills in skills_by_job.items():
    if skills:
        # Join skills into a single text
        text = ' '.join(skills)

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            collocations=False,
            max_words=100
        ).generate(text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Common Skills for {job_title} Matches")
        plt.tight_layout()
        plt.show()

        # Save the word cloud
        safe_title = re.sub(r'[^\w\s]', '', job_title).replace(' ', '_')
        plt.savefig(f'skills_wordcloud_{safe_title}.png', dpi=300)
        plt.close()