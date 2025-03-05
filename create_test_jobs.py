import json

# Sample job data
sample_jobs = [
    {
        "id": "job001",
        "title": "Senior Software Engineer",
        "company": "Tech Innovations Inc.",
        "description": "Developing enterprise-level applications using Python and cloud technologies.",
        "required_skills": ["Python", "AWS", "Docker", "Kubernetes", "Machine Learning"],
        "required_experience": "5+ years",
        "required_education": ["Bachelor's in Computer Science", "Master's preferred"]
    },
    {
        "id": "job002",
        "title": "Data Scientist",
        "company": "Analytics Partners",
        "description": "Build predictive models and extract insights from large datasets.",
        "required_skills": ["Python", "R", "SQL", "Machine Learning", "Statistics"],
        "required_experience": "3+ years",
        "required_education": ["Master's in Data Science", "PhD preferred"]
    },
    {
        "id": "job003",
        "title": "Project Manager",
        "company": "Global Systems Ltd.",
        "description": "Manage technology projects from inception to deployment.",
        "required_skills": ["Project Management", "Agile", "Scrum", "Stakeholder Management"],
        "required_experience": "7+ years",
        "required_education": ["Bachelor's degree", "PMP Certification"]
    }
]

# Save to JSON file
with open("sample_jobs.json", "w") as f:
    json.dump(sample_jobs, f, indent=2)

print("Sample jobs file created: sample_jobs.json")