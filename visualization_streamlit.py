import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os


# Load your matching results
@st.cache_data
def load_data():
    with open('matched_results.json', 'r') as f:
        return json.load(f)


data = load_data()

# Extract useful data
candidates = {}
for resume_path, jobs in data.get('job_matches', {}).items():
    if jobs:  # If there are any matches
        candidate_name = jobs[0].get('candidate_name', os.path.basename(resume_path))
        candidates[candidate_name] = {
            'resume_path': resume_path,
            'matches': jobs
        }

# App title
st.title("Resume-Job Match Explorer")

# Sidebar with filters
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Match Score", 0.0, 1.0, 0.5, 0.05)

# Get unique job titles for filtering
all_job_titles = set()
for candidate_info in candidates.values():
    for match in candidate_info['matches']:
        all_job_titles.add(match.get('job_title', 'Unknown'))

selected_jobs = st.sidebar.multiselect(
    "Filter by Job Title",
    options=sorted(list(all_job_titles)),
    default=[]
)

# Main area
st.header("Candidate Overview")

# Create a dataframe for overview
overview_data = []
for candidate_name, info in candidates.items():
    best_match = max(info['matches'], key=lambda x: x.get('score', 0))
    best_score = best_match.get('score', 0)
    best_job = best_match.get('job_title', 'Unknown')

    overview_data.append({
        'Candidate': candidate_name,
        'Best Match Job': best_job,
        'Best Match Score': best_score * 100,  # Convert to percentage
        'Number of Good Matches': sum(1 for m in info['matches'] if m.get('score', 0) >= min_score)
    })

overview_df = pd.DataFrame(overview_data)

# Apply job title filter if any selected
if selected_jobs:
    filtered_candidates = []
    for candidate_name, info in candidates.items():
        if any(match.get('job_title', 'Unknown') in selected_jobs for match in info['matches']):
            filtered_candidates.append(candidate_name)

    overview_df = overview_df[overview_df['Candidate'].isin(filtered_candidates)]

# Show the overview
st.dataframe(overview_df.sort_values('Best Match Score', ascending=False), hide_index=True)

# Bar chart of best match scores
fig = px.bar(
    overview_df.sort_values('Best Match Score'),
    x='Best Match Score',
    y='Candidate',
    color='Best Match Job',
    title='Candidates by Best Match Score',
    labels={'Best Match Score': 'Match Score (%)'},
    orientation='h'
)
st.plotly_chart(fig)

# Detailed view for a selected candidate
st.header("Candidate Detail View")
selected_candidate = st.selectbox("Select a candidate", list(candidates.keys()))

if selected_candidate:
    st.subheader(f"Job Matches for {selected_candidate}")

    # Get candidate's matches
    matches = candidates[selected_candidate]['matches']

    # Filter by minimum score
    matches = [m for m in matches if m.get('score', 0) >= min_score]

    # Filter by selected jobs if any
    if selected_jobs:
        matches = [m for m in matches if m.get('job_title', 'Unknown') in selected_jobs]

    # Convert to dataframe for display
    matches_df = pd.DataFrame([
        {
            'Job Title': m.get('job_title', 'Unknown'),
            'Company': m.get('job_company', 'Unknown'),
            'Match Score': m.get('score', 0) * 100  # Convert to percentage
        }
        for m in matches
    ])

    # Display matches in descending order of score
    if not matches_df.empty:
        st.dataframe(matches_df.sort_values('Match Score', ascending=False), hide_index=True)

        # Bar chart of match scores
        match_fig = px.bar(
            matches_df.sort_values('Match Score', ascending=True),
            x='Match Score',
            y='Job Title',
            color='Company',
            title=f'Job Matches for {selected_candidate}',
            labels={'Match Score': 'Match Score (%)'},
            orientation='h'
        )
        st.plotly_chart(match_fig)
    else:
        st.write("No matches meet the current filtering criteria.")

    # Resume path information
    st.text(f"Resume file: {candidates[selected_candidate]['resume_path']}")