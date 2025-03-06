#!/usr/bin/env python3
"""
Resume Information Extractor with Job Matching Capabilities

This script extracts structured information from resumes and optionally
matches them against job openings using semantic search.

Usage:
    python main.py --file path/to/resume.pdf
    python main.py --directory path/to/resumes/folder
    python main.py --file path/to/resume.pdf --match --job-file jobs.json
    python main.py --directory path/to/resumes/ --match --job-file jobs.json --output results.json
"""

import pandas as pd
import re
import uuid
import os
import sys
import json
import argparse
import requests
from typing import Dict, List, Union, Optional, Any
from datetime import datetime

# Check if required packages are installed
try:
    import requests
except ImportError:
    print("Error: The 'requests' package is required. Install it with: pip install requests")
    sys.exit(1)

# Try to import sentence_transformers (only needed for matching)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import semantic_search

    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False


# Define LLM Backend classes
class LLMBackend:
    """Abstract base class for different LLM backends"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")


class OllamaBackend(LLMBackend):
    """Backend for using Ollama local LLM"""

    def __init__(self, model_name: str = "llama3.1:latest", endpoint: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.endpoint = endpoint

        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")

            # Check if the model is available
            available_models = [model["name"] for model in response.json().get("models", [])]
            if self.model_name not in available_models:
                print(
                    f"Warning: Model '{self.model_name}' not found in Ollama. Available models: {', '.join(available_models)}")
                print(f"You may need to pull the model first: ollama pull {self.model_name}")
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to Ollama. Please ensure Ollama is installed and running.")
            print("You can install Ollama from https://ollama.ai")
            print("Run 'ollama serve' to start the Ollama server.")
            sys.exit(1)

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Keep deterministic for consistent extraction
                "num_predict": 1024  # Adjust based on expected output size
            }
        }

        try:
            response = requests.post(self.endpoint, json=payload)

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} {response.text}")

            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""


class ResumeInfoExtractor:
    """Extract structured information from resumes using few-shot learning with LLMs"""

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def extract_resume_information(self, resume_text: str) -> Dict:
        """Extract name, years of experience, and degrees from resume text using few-shot learning."""
        # Create a prompt with examples
        prompt = """Extract structured information from resumes. For each resume, identify:
1. The person's full name
2. Total years of professional experience (as a number)
3. A list of all academic degrees

Examples:

Resume:
JANE DOE
jdoe@email.com | (123) 456-7890
EXPERIENCE
Software Engineer, ABC Tech (2015-Present)
Junior Developer, XYZ Solutions (2012-2015)
EDUCATION
Master of Computer Science, Stanford University (2012)
B.S. Computer Science, UC Berkeley (2010)

Expected Output:
{
  "name": "Jane Doe",
  "years_experience": 11,
  "degrees": ["Master of Computer Science", "B.S. Computer Science"]
}

Resume:
Robert Johnson
robertj@email.com
Professional Experience:
Marketing Manager with 7+ years driving brand growth
Work History:
Head of Digital Marketing, Global Brands Inc., Jan 2020-Present
Marketing Specialist, Regional Products, 2016-2019
Education Background:
MBA, Harvard Business School, 2016
Bachelor of Arts in Communications, Yale University, 2013

Expected Output:
{
  "name": "Robert Johnson",
  "years_experience": 7,
  "degrees": ["MBA", "Bachelor of Arts in Communications"]
}

Now extract information from this resume:
"""

        # Add the target resume text
        full_prompt = prompt + resume_text + "\n\nOutput in JSON format:"

        # Generate response from the LLM
        response = self.backend.generate(full_prompt)

        # Extract JSON from the response
        extracted_info = self._extract_json_from_response(response)

        # Validate and correct the extracted information
        validated_info, issues = self._validate_extraction(extracted_info)

        if issues:
            print(f"Warning: Found {len(issues)} issues in extraction: {issues}")

        return validated_info

    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from the LLM response."""
        try:
            # Look for JSON within the response - find the first { and last }
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)

            # If no JSON found but response is already valid JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # No valid JSON found, try to extract fields manually
            print(f"Warning: Could not parse response as JSON: {response}")

            result = {}

            # Extract name
            if "name" in response.lower():
                name_start = response.lower().find("name")
                name_end = response.find("\n", name_start)
                if name_end == -1:
                    name_end = len(response)
                name_line = response[name_start:name_end]
                name_parts = name_line.split(":")
                if len(name_parts) > 1:
                    result["name"] = name_parts[1].strip(' "\'')

            # Extract years of experience
            if "experience" in response.lower():
                exp_start = response.lower().find("experience")
                exp_end = response.find("\n", exp_start)
                if exp_end == -1:
                    exp_end = len(response)
                exp_line = response[exp_start:exp_end]
                exp_parts = exp_line.split(":")
                if len(exp_parts) > 1:
                    try:
                        result["years_experience"] = int(exp_parts[1].strip(' "\''))
                    except ValueError:
                        # Try to extract just the number
                        import re
                        numbers = re.findall(r'\d+', exp_parts[1])
                        if numbers:
                            result["years_experience"] = int(numbers[0])

            # Extract degrees
            if "degrees" in response.lower():
                deg_start = response.lower().find("degrees")
                deg_end = response.find("\n", deg_start)
                if deg_end == -1:
                    deg_end = len(response)
                deg_line = response[deg_start:deg_end]
                deg_parts = deg_line.split(":")
                if len(deg_parts) > 1:
                    degrees_text = deg_parts[1].strip()
                    # Handle different formats (list, comma-separated, etc.)
                    if degrees_text.startswith("[") and degrees_text.endswith("]"):
                        # Process as list
                        degrees_list = degrees_text.strip("[]").split(",")
                        result["degrees"] = [d.strip(' "\'') for d in degrees_list]
                    else:
                        # Process as comma-separated text
                        result["degrees"] = [d.strip(' "\'') for d in degrees_text.split(",")]

            return result

    def _validate_extraction(self, extracted_info: Dict) -> tuple:
        """Validate and correct extracted information."""
        issues = []

        # Make a copy to avoid modifying the input
        validated = extracted_info.copy()

        # Check name format
        if not validated.get('name'):
            issues.append("Name missing")
        elif len(validated['name'].split()) < 2:
            issues.append("Name may be incomplete (less than 2 words)")

        # Check experience value
        experience = validated.get('years_experience')
        if experience is None:
            issues.append("Years of experience not found")
        elif isinstance(experience, str):
            # Try to convert string to number
            try:
                validated['years_experience'] = int(experience.split()[0])
            except:
                issues.append("Years of experience not in expected format")

        # Check degrees
        degrees = validated.get('degrees', [])
        if not degrees:
            issues.append("No degrees found")
        elif isinstance(degrees, str):
            # Convert string representation to actual list
            if '[' in degrees and ']' in degrees:
                # Handle JSON-like string
                degrees = degrees.strip('[]').split(',')
                validated['degrees'] = [d.strip(' "\'') for d in degrees]
            else:
                # Handle plain text
                validated['degrees'] = [degrees.strip()]

        return validated, issues


# Job Matching Functions
def enrich_resume_data(resume_data):
    """Transform raw extracted resume data into richer text suitable for embedding."""
    enriched_data = {}

    # Basic identity information
    name = resume_data.get('name', '')
    years_exp = resume_data.get('years_experience', 0)
    degrees = resume_data.get('degrees', [])

    # Create a consistent text representation
    profile_text = f"Professional with {years_exp} years of experience. "
    profile_text += f"Education: {', '.join(degrees)}. "

    enriched_data['id'] = resume_data.get('file_path', '')
    enriched_data['name'] = name
    enriched_data['profile_text'] = profile_text
    enriched_data['raw_data'] = resume_data  # Keep original data

    return enriched_data

# 'all-MiniLM-L6-v2'
#
def create_resume_embeddings(resume_collection, model_name='deepseek-r1:14b'):
    """Generate embeddings for a collection of resume data."""
    # Load an appropriate model
    model = SentenceTransformer(model_name)

    resume_texts = []
    resume_ids = []

    for resume in resume_collection:
        resume_texts.append(resume['profile_text'])
        resume_ids.append(resume['id'])

    # Generate embeddings
    print(f"Generating embeddings for {len(resume_texts)} resumes...")
    embeddings = model.encode(resume_texts, show_progress_bar=True)

    # Create a structured collection with embeddings
    resume_embeddings = []
    for i, embedding in enumerate(embeddings):
        resume_embeddings.append({
            'id': resume_ids[i],
            'embedding': embedding,
            'raw_data': resume_collection[i]['raw_data']
        })

    return resume_embeddings, model


def load_job_openings(job_file_path):
    """Load job opening data from a JSON file."""
    try:
        with open(job_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading job openings: {e}")
        return []


def load_jobs_from_excel(excel_file, sheet_name=0):
    """
    Load job listings from an Excel file and convert to the format required
    by the resume matcher.

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing job listings
    sheet_name : str or int, default 0
        Name or index of the sheet containing the job data

    Returns:
    --------
    list
        A list of job dictionaries in the format required by the resume matcher
    """
    print(f"Reading job listings from: {excel_file}")

    # Read the Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"Successfully read {len(df)} job listings")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

    # Print column names to help with debugging
    print("Columns found in Excel file:")
    for col in df.columns:
        print(f"  - {col}")

    # Map Excel columns to job data fields
    # Adjust these mappings based on your actual Excel column names
    column_mappings = {
        "id": ["id", "job id", "job_id", "jobid", "reference", "ref"],
        "title": ["title", "job title", "job_title", "position", "role"],
        "company": ["company", "company name", "employer", "organization"],
        "description": ["description", "job description", "details", "summary"],
        "required_skills": ["skills", "required skills", "skill requirements", "technical skills"],
        "required_experience": ["experience", "required experience", "years of experience", "exp"],
        "required_education": ["education", "required education", "educational requirements", "degree"]
    }

    # Find the best match for each expected field
    field_to_column = {}
    for field, possible_columns in column_mappings.items():
        found = False
        for col_name in possible_columns:
            matches = [c for c in df.columns if c.lower() == col_name.lower()]
            if matches:
                field_to_column[field] = matches[0]
                found = True
                break

        if not found:
            print(f"Warning: No column found for '{field}'. Using a default value.")

    # Convert dataframe to list of job dictionaries
    jobs = []
    for _, row in df.iterrows():
        job = {}

        # Process ID (generate one if not present)
        if "id" in field_to_column:
            job["id"] = str(row[field_to_column["id"]])
        else:
            job["id"] = f"job{len(jobs) + 1:03d}"

        # Process basic text fields
        for field in ["title", "company", "description"]:
            if field in field_to_column:
                job[field] = str(row[field_to_column[field]])
            else:
                job[field] = ""

        # Process required_skills (convert to list if needed)
        if "required_skills" in field_to_column:
            skills_value = row[field_to_column["required_skills"]]
            if pd.isna(skills_value):
                job["required_skills"] = []
            elif isinstance(skills_value, str):
                # Split by commas, semicolons, or newlines
                skills = re.split(r'[,;\n]+', skills_value)
                job["required_skills"] = [s.strip() for s in skills if s.strip()]
            else:
                job["required_skills"] = [str(skills_value)]
        else:
            job["required_skills"] = []

        # Process required_experience
        if "required_experience" in field_to_column:
            job["required_experience"] = str(row[field_to_column["required_experience"]])
        else:
            job["required_experience"] = ""

        # Process required_education (convert to list if needed)
        if "required_education" in field_to_column:
            education_value = row[field_to_column["required_education"]]
            if pd.isna(education_value):
                job["required_education"] = []
            elif isinstance(education_value, str):
                # Split by commas, semicolons, or newlines
                education = re.split(r'[,;\n]+', education_value)
                job["required_education"] = [e.strip() for e in education if e.strip()]
            else:
                job["required_education"] = [str(education_value)]
        else:
            job["required_education"] = []

        jobs.append(job)

    print(f"Successfully processed {len(jobs)} job listings")
    return jobs

def prepare_jobs_for_resume_matcher(excel_file, output_json_file, sheet_name=0):
    """
    Read job listings from Excel and save them in the format required by the resume matcher.

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing job listings
    output_json_file : str
        Path where the JSON job data should be saved
    sheet_name : str or int, default 0
        Name or index of the sheet containing the job data

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Load jobs from Excel
    jobs = load_jobs_from_excel(excel_file, sheet_name)

    if not jobs:
        print("No jobs found or error processing the Excel file.")
        return False

    # Save to JSON file
    try:
        with open(output_json_file, 'w') as f:
            json.dump(jobs, f, indent=2)

        print(f"Successfully saved {len(jobs)} job listings to {output_json_file}")
        print("Sample job listing:")
        print(json.dumps(jobs[0], indent=2))
        return True
    except Exception as e:
        print(f"Error saving job data to JSON: {e}")
        return False

def process_job_openings(job_descriptions, model):
    """Process job descriptions for semantic matching."""
    job_texts = []
    job_ids = []

    for job in job_descriptions:
        # Create a consistent text representation of the job
        job_text = f"{job.get('title', '')}. {job.get('description', '')}. "

        # Add required skills if available
        if 'required_skills' in job:
            if isinstance(job['required_skills'], list):
                job_text += f"Required skills: {', '.join(job['required_skills'])}. "
            else:
                job_text += f"Required skills: {job['required_skills']}. "

        # Add required experience if available
        if 'required_experience' in job:
            job_text += f"Required experience: {job['required_experience']}. "

        # Add required education if available
        if 'required_education' in job:
            if isinstance(job['required_education'], list):
                job_text += f"Required education: {', '.join(job['required_education'])}. "
            else:
                job_text += f"Required education: {job['required_education']}. "

        job_texts.append(job_text)
        job_ids.append(job.get('id', str(len(job_ids))))

    # Generate embeddings for jobs
    print(f"Generating embeddings for {len(job_texts)} jobs...")
    job_embeddings = model.encode(job_texts, show_progress_bar=True)

    # Create a structured collection with embeddings
    job_data = []
    for i, embedding in enumerate(job_embeddings):
        job_data.append({
            'id': job_ids[i],
            'embedding': embedding,
            'text': job_texts[i],
            'raw_data': job_descriptions[i]
        })

    return job_data


def find_matching_jobs_for_resume(resume_embedding, job_embeddings, top_k=5, resume_data=None):
    """Find the best matching jobs for a given resume embedding."""
    # Convert embeddings to the format required by semantic_search
    query_embedding = np.array([resume_embedding])
    corpus_embeddings = np.array([job['embedding'] for job in job_embeddings])

    # Perform semantic search
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

    # Format results
    matches = []
    for hit in hits:
        job_index = hit['corpus_id']
        job = job_embeddings[job_index]

        # Create the match entry dictionary
        match_entry = {
            'job_id': job['id'],
            'score': float(hit['score']),  # Convert to float for JSON serialization
            'job_title': job['raw_data'].get('title', 'Untitled'),
            'job_company': job['raw_data'].get('company', 'Unknown'),
            'similarity_score': float(hit['score'])
        }

        # Add candidate name if resume data is provided
        if resume_data:
            match_entry['candidate_name'] = resume_data.get('name', 'Unknown')

        matches.append(match_entry)

    return matches

# File processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import pdfplumber
    except ImportError:
        print("Error: The 'pdfplumber' package is required for processing PDF files.")
        print("Install it with: pip install pdfplumber")
        sys.exit(1)

    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        import docx
    except ImportError:
        print("Error: The 'python-docx' package is required for processing DOCX files.")
        print("Install it with: pip install python-docx")
        sys.exit(1)

    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


def process_resume(file_path: str, backend: LLMBackend) -> Dict:
    """Process a single resume file"""
    print(f"Processing: {file_path}")

    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract text based on file type
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only PDF and DOCX are supported.")

    if not text.strip():
        raise ValueError(f"Could not extract text from file: {file_path}")

    # Create extractor with the specified backend
    extractor = ResumeInfoExtractor(backend)

    # Extract information
    info = extractor.extract_resume_information(text)
    info['file_path'] = file_path

    return info


def batch_process_resumes(file_paths: List[str], backend: LLMBackend,
                          batch_size: int = 5) -> List[Dict]:
    """Process multiple resume files in batches"""
    results = []

    total_batches = (len(file_paths) + batch_size - 1) // batch_size

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{total_batches} ({len(batch)} files)")

        batch_results = []
        for file_path in batch:
            try:
                info = process_resume(file_path, backend)
                batch_results.append(info)
                print(f"✓ Successfully processed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(file_path)}: {e}")
                batch_results.append({
                    'file_path': file_path,
                    'error': str(e)
                })

        results.extend(batch_results)

    return results


def main():
    """Main function to handle command-line arguments and run the program"""
    parser = argparse.ArgumentParser(
        description="Extract information from resumes using Ollama LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', help='Path to a single resume file (PDF or DOCX)')
    input_group.add_argument('--directory', help='Directory containing multiple resume files')

    parser.add_argument('--model', default='llama3.1:latest', help='Ollama model name to use')
    parser.add_argument('--output', help='Output JSON file path (defaults to stdout)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of files to process in each batch')

    # Job matching options
    parser.add_argument('--match', action='store_true', help='Enable resume-job matching')
    parser.add_argument('--job-file', help='JSON file containing job openings for matching')
    parser.add_argument('--embedding-model', default='paraphrase-mpnet-base-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top matches to return for each resume')
    parser.add_argument('--excel-jobs',
                        help='Path to Excel file containing job listings')
    parser.add_argument('--excel-sheet', default=0,
                        help='Sheet name or index in the Excel file (default: 0)')

    args = parser.parse_args()

    # Initialize the LLM backend
    print(f"Initializing Ollama with model: {args.model}")
    backend = OllamaBackend(model_name=args.model)

    # In the main function
    if args.excel_jobs:
        print(f"Converting Excel job listings from {args.excel_jobs}")

        # Generate a temporary JSON file for the jobs
        temp_json_path = "jobs_from_excel.json"

        if prepare_jobs_for_resume_matcher(args.excel_jobs, temp_json_path, args.excel_sheet):
            # Use this JSON file for job matching
            args.job_file = temp_json_path
            print(f"Excel jobs converted and saved to {temp_json_path}")
        else:
            print("Error converting Excel jobs. Continuing with original job file if specified.")

    # Process resume(s)
    if args.file:
        try:
            result = process_resume(args.file, backend)
            results = [result]
        except Exception as e:
            print(f"Error processing file: {e}")
            return 1
    else:  # args.directory
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            return 1

        # Get all PDF and DOCX files in the directory
        resume_files = []
        for file in os.listdir(args.directory):
            if file.lower().endswith(('.pdf', '.docx', '.doc')):
                resume_files.append(os.path.join(args.directory, file))

        if not resume_files:
            print(f"No PDF or DOCX files found in directory: {args.directory}")
            return 1

        print(f"Found {len(resume_files)} resume files to process")
        results = batch_process_resumes(resume_files, backend, args.batch_size)

    # Prepare output data
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "num_files": len(results),
            "success_count": sum(1 for r in results if "error" not in r),
            "error_count": sum(1 for r in results if "error" in r)
        },
        "results": results
    }

    # Perform resume-job matching if requested
    if args.match:
        if not HAVE_SENTENCE_TRANSFORMERS:
            print("Error: The 'sentence-transformers' package is required for matching.")
            print("Install it with: pip install sentence-transformers")
            return 1

        if not args.job_file:
            print("Error: Job file (--job-file) is required for matching")
            return 1

        print("\nPerforming resume-job matching...")

        # Load job openings
        job_openings = load_job_openings(args.job_file)
        if not job_openings:
            print("No job openings found or error loading file")
            return 1

        print(f"Loaded {len(job_openings)} job openings")

        # Process successfully extracted resumes
        valid_resumes = [r for r in results if "error" not in r]
        print(f"Processing {len(valid_resumes)} valid resumes")

        if not valid_resumes:
            print("No valid resumes to match")
            return 1

        # Enrich resume data
        enriched_resumes = [enrich_resume_data(resume) for resume in valid_resumes]

        # Create resume embeddings
        resume_embeddings, model = create_resume_embeddings(enriched_resumes, args.embedding_model)

        # Process job openings
        job_embeddings = process_job_openings(job_openings, model)

        # Find matches for each resume
        print("Finding job matches for each resume...")
        all_matches = {}

        for resume in resume_embeddings:
            resume_id = resume['id']
            resume_name = resume['raw_data'].get('name', 'Unknown')
            print(f"Finding matches for {resume_name}...")

            matches = find_matching_jobs_for_resume(
                resume['embedding'],
                job_embeddings,
                top_k=args.top_k,
                resume_data=resume['raw_data']
            )
            all_matches[resume_id] = matches

        # Add matches to output data
        output_data['job_matches'] = all_matches

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            if args.match:
                # Handle numpy arrays for serialization
                json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            else:
                json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nExtracted Information:")
        if args.match:
            # Only print basic results if we have job matches to avoid overwhelming output
            for result in results:
                if "error" not in result:
                    print(f"- {result.get('name', 'Unknown')}: {result.get('years_experience', 0)} years, " +
                          f"{len(result.get('degrees', []))} degrees")

            print("\nJob Matches (Top 3 for each resume):")
            for resume_id, matches in all_matches.items():
                resume_name = next((r['name'] for r in results if r['file_path'] == resume_id), "Unknown")
                print(f"\n{resume_name}:")
                for i, match in enumerate(matches[:3]):
                    print(f"  {i + 1}. {match['job_title']} ({match['job_company']}) - Score: {match['score']:.4f}")
        else:
            # Print full results if no job matching
            print(json.dumps(output_data, indent=2))

    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {len(results)}")
    print(f"Successfully processed: {output_data['metadata']['success_count']}")
    print(f"Failed to process: {output_data['metadata']['error_count']}")

    if args.match:
        print(f"Job matches generated: {len(all_matches)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())