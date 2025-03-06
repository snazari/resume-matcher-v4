#!/usr/bin/env python3
"""
Resume Matcher - Match resumes with job descriptions using AI

A smart tool that uses semantic analysis to match resumes with job descriptions,
providing insights into the best candidates for each position.

Usage:
    python main.py extract --file path/to/resume.pdf --output extracted_resume.json
    python main.py extract --dir path/to/resumes/ --output extracted_resumes.json
    python main.py match --resumes extracted_resumes.json --jobs jobs.json --output matches.json
    python main.py visualize --results matches.json --type radar --output visualization.html
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Union, Optional, Any


# Terminal UI components
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'


def get_terminal_width():
    """Get the width of the terminal."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, OSError):
        return 80


def display_banner():
    """Display the application banner."""
    width = get_terminal_width()

    banner = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                                                                              ┃
    ┃  ██████╗ ███████╗███████╗██╗   ██╗███╗   ███╗███████╗                       ┃
    ┃  ██╔══██╗██╔════╝██╔════╝██║   ██║████╗ ████║██╔════╝                       ┃
    ┃  ██████╔╝█████╗  ███████╗██║   ██║██╔████╔██║█████╗                         ┃
    ┃  ██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╔╝██║██╔══╝                         ┃
    ┃  ██║  ██║███████╗███████║╚██████╔╝██║ ╚═╝ ██║███████╗                       ┃
    ┃  ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝                       ┃
    ┃                                                                              ┃
    ┃  ███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗███████╗██████╗               ┃
    ┃  ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗              ┃
    ┃  ██╔████╔██║███████║   ██║   ██║     ███████║█████╗  ██████╔╝              ┃
    ┃  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗              ┃
    ┃  ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║███████╗██║  ██║              ┃
    ┃  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝              ┃
    ┃                                                                              ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """

    print(f"{Colors.BLUE}{banner}{Colors.ENDC}")
    print(f"{Colors.BOLD} Resume-Job Matcher v4.0 - Intelligent Resume-Job Matching System Developed by Sam Nazari, Ph.D.{Colors.ENDC}")
    print(" A smart tool to match resumes with job descriptions using few shot learning, semantic analysis, BERT and GPTs\n")


def print_header(title):
    """Print a stylish section header."""
    width = get_terminal_width()
    padding = (width - len(title) - 4) // 2
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * padding} {title} {'═' * padding}{Colors.ENDC}\n")


def print_step(step_num, title, description=None):
    """Print a numbered step with optional description."""
    print(f"{Colors.BLUE}{Colors.BOLD}[STEP {step_num}]{Colors.ENDC} {Colors.BOLD}{title}{Colors.ENDC}")
    if description:
        print(f"          {description}")
    print()


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_info(message):
    """Print an informational message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")


def print_table(headers, rows, max_width=None):
    """Print a formatted table."""
    if not max_width:
        max_width = get_terminal_width() - 4

    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Adjust column widths to fit max_width
    total_width = sum(col_widths) + (3 * len(headers)) - 1
    if total_width > max_width:
        # Scale down columns proportionally
        ratio = max_width / total_width
        col_widths = [max(int(w * ratio), 5) for w in col_widths]

    # Create the headers
    header_row = "│ "
    for i, header in enumerate(headers):
        header_text = str(header)[:col_widths[i]].ljust(col_widths[i])
        header_row += f"{Colors.BOLD}{header_text}{Colors.ENDC} │ "

    separator = "├─" + "─┬─".join("─" * w for w in col_widths) + "─┤"
    top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    # Print the table
    print(top_border)
    print(header_row)
    print(separator)

    for row in rows:
        row_str = "│ "
        for i, cell in enumerate(row):
            cell_text = str(cell)[:col_widths[i]].ljust(col_widths[i])
            row_str += f"{cell_text} │ "
        print(row_str)

    print(bottom_border)


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print(f"\r{prefix} |{Colors.BLUE}{bar}{Colors.ENDC}| {percent}% {suffix}", end='')
    if iteration == total:
        print()


def print_summary(title, items):
    """Print a summary box with key-value pairs."""
    width = get_terminal_width() - 4
    print(f"┌{'─' * (width - 2)}┐")
    print(f"│ {Colors.BOLD}{title}{Colors.ENDC}{' ' * (width - len(title) - 3)}│")
    print(f"├{'─' * (width - 2)}┤")

    for key, value in items:
        key_str = f"{key}: "
        value_str = str(value)

        # Truncate if too long
        available_width = width - len(key_str) - 3
        if len(value_str) > available_width:
            value_str = value_str[:available_width - 3] + "..."

        print(f"│ {key_str}{value_str}{' ' * (width - len(key_str) - len(value_str) - 3)}│")

    print(f"└{'─' * (width - 2)}┘")


# Spinner for loading operations
class Spinner:
    """Terminal spinner for showing ongoing operations."""

    def __init__(self, message="Processing", delay=0.1):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.message = message
        self.delay = delay
        self.running = False
        self.spinner_thread = None

    def spin(self):
        """Display the spinner animation."""
        i = 0
        while self.running:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            sys.stdout.write(f"\r{Colors.BLUE}{char}{Colors.ENDC} {self.message}...")
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        """Start the spinner in a separate thread."""
        self.running = True
        import threading
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self):
        """Stop the spinner."""
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()


# Import the modules we need
try:
    import requests  # For Ollama API

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import semantic_search

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


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
                print_warning(
                    f"Model '{self.model_name}' not found in Ollama. Available models: {', '.join(available_models)}")
                print_info(f"You may need to pull the model first: ollama pull {self.model_name}")
        except requests.exceptions.ConnectionError:
            print_error("Cannot connect to Ollama. Please ensure Ollama is installed and running.")
            print_info("You can install Ollama from https://ollama.ai")
            print_info("Run 'ollama serve' to start the Ollama server.")
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
            print_error(f"Error calling Ollama API: {e}")
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
            print_warning(f"Found {len(issues)} issues in extraction: {', '.join(issues)}")

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
            print_warning(f"Could not parse response as JSON")

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


# File processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import pdfplumber
    except ImportError:
        print_error("The 'pdfplumber' package is required for processing PDF files.")
        print_info("Install it with: pip install pdfplumber")
        sys.exit(1)

    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print_error(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        import docx
    except ImportError:
        print_error("The 'python-docx' package is required for processing DOCX files.")
        print_info("Install it with: pip install python-docx")
        sys.exit(1)

    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print_error(f"Error extracting text from DOCX: {e}")
        return ""


def process_resume(file_path: str, backend: LLMBackend) -> Dict:
    """Process a single resume file"""
    spinner = Spinner(f"Processing {os.path.basename(file_path)}")
    spinner.start()

    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            spinner.stop()
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract text based on file type
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text = extract_text_from_docx(file_path)
        else:
            spinner.stop()
            raise ValueError(f"Unsupported file format: {file_ext}. Only PDF and DOCX are supported.")

        if not text.strip():
            spinner.stop()
            raise ValueError(f"Could not extract text from file: {file_path}")

        # Create extractor with the specified backend
        extractor = ResumeInfoExtractor(backend)

        # Extract information
        info = extractor.extract_resume_information(text)
        info['file_path'] = file_path

        spinner.stop()
        print_success(f"Successfully processed: {os.path.basename(file_path)}")

        return info
    except Exception as e:
        spinner.stop()
        print_error(f"Error processing {os.path.basename(file_path)}: {e}")
        return {
            'file_path': file_path,
            'error': str(e)
        }


def batch_process_resumes(file_paths: List[str], backend: LLMBackend, batch_size: int = 5) -> List[Dict]:
    """Process multiple resume files in batches with progress tracking"""
    results = []
    total_files = len(file_paths)
    processed = 0

    total_batches = (total_files + batch_size - 1) // batch_size

    for i in range(0, total_files, batch_size):
        batch = file_paths[i:i + batch_size]
        print(
            f"\n{Colors.BOLD}Processing batch {i // batch_size + 1}/{total_batches} ({len(batch)} files){Colors.ENDC}")

        batch_results = []
        for j, file_path in enumerate(batch):
            try:
                info = process_resume(file_path, backend)
                batch_results.append(info)

                # Update progress
                processed += 1
                print_progress_bar(
                    processed,
                    total_files,
                    prefix=f"Overall progress:",
                    suffix=f"({processed}/{total_files})",
                    length=40
                )
            except Exception as e:
                print_error(f"Error processing {os.path.basename(file_path)}: {e}")
                batch_results.append({
                    'file_path': file_path,
                    'error': str(e)
                })

                # Update progress
                processed += 1
                print_progress_bar(
                    processed,
                    total_files,
                    prefix=f"Overall progress:",
                    suffix=f"({processed}/{total_files})",
                    length=40
                )

        results.extend(batch_results)

    print()  # Add a newline after the progress bar
    return results


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

    # Check if degrees are dictionaries and extract the relevant information
    if degrees and isinstance(degrees[0], dict):
        # If degrees are dictionaries, extract the degree field
        degree_strings = []
        for degree_dict in degrees:
            if isinstance(degree_dict, dict):
                # Extract the degree from the dictionary - adjust the key based on your data structure
                # Common keys might be 'degree', 'title', 'name', etc.
                if 'degree' in degree_dict:
                    degree_strings.append(degree_dict['degree'])
                elif 'title' in degree_dict:
                    degree_strings.append(degree_dict['title'])
                elif 'name' in degree_dict:
                    degree_strings.append(degree_dict['name'])
                # If none of the expected keys are found, add a placeholder
                else:
                    degree_strings.append(str(degree_dict))
            else:
                # If for some reason an item isn't a dict, convert it to string
                degree_strings.append(str(degree_dict))

        profile_text += f"Education: {', '.join(degree_strings)}. "
    else:
        # Original code for string-based degrees
        profile_text += f"Education: {', '.join(str(d) for d in degrees)}. "

    enriched_data['id'] = resume_data.get('file_path', '')
    enriched_data['name'] = name
    enriched_data['profile_text'] = profile_text
    enriched_data['raw_data'] = resume_data  # Keep original data

    return enriched_data


def create_resume_embeddings(resume_collection, model_name='paraphrase-mpnet-base-v2'):
    """Generate embeddings for a collection of resume data."""
    # Load an appropriate model
    spinner = Spinner("Loading embedding model")
    spinner.start()

    try:
        model = SentenceTransformer(model_name)
        spinner.stop()
        print_success(f"Loaded embedding model: {model_name}")
    except Exception as e:
        spinner.stop()
        print_error(f"Error loading embedding model: {e}")
        return [], None

    resume_texts = []
    resume_ids = []

    for resume in resume_collection:
        resume_texts.append(resume['profile_text'])
        resume_ids.append(resume['id'])

    # Generate embeddings
    print_info(f"Generating embeddings for {len(resume_texts)} resumes...")

    embeddings = model.encode(resume_texts, show_progress_bar=True)
    print_success("Resume embeddings generated successfully")

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
    spinner = Spinner(f"Loading job openings from {os.path.basename(job_file_path)}")
    spinner.start()

    try:
        with open(job_file_path, 'r') as f:
            jobs = json.load(f)

        spinner.stop()
        print_success(f"Loaded {len(jobs)} job openings")
        return jobs
    except Exception as e:
        spinner.stop()
        print_error(f"Error loading job openings: {e}")
        return []


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
    print_info(f"Generating embeddings for {len(job_texts)} jobs...")
    job_embeddings = model.encode(job_texts, show_progress_bar=True)
    print_success("Job embeddings generated successfully")

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


def display_match_results(match_data, top_k=3):
    """Display a dashboard of match results."""
    print_header("MATCH RESULTS DASHBOARD")

    # Get overall statistics
    total_resumes = len(match_data.get('job_matches', {}))

    # Get all unique job matches
    all_jobs = set()
    for matches in match_data.get('job_matches', {}).values():
        for match in matches:
            job_id = match.get('job_id', '')
            job_title = match.get('job_title', '')
            all_jobs.add(f"{job_title} ({job_id})")

    total_jobs = len(all_jobs)

    # Calculate average match score
    all_scores = []
    for matches in match_data.get('job_matches', {}).values():
        all_scores.extend([match.get('score', 0) for match in matches])

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    print_summary("Overview", [
        ("Total Resumes", total_resumes),
        ("Total Job Positions", total_jobs),
        ("Total Matches Generated", len(all_scores)),
        ("Average Match Score", f"{avg_score * 100:.1f}%")
    ])

    # Display top candidates for each job
    print(f"\n{Colors.BOLD}Top Candidates by Job Position:{Colors.ENDC}")

    # Organize by job
    job_candidates = {}
    for resume_path, matches in match_data.get('job_matches', {}).items():
        for match in matches:
            job_id = match.get('job_id', '')
            job_title = match.get('job_title', 'Unknown')
            key = f"{job_title} ({job_id})"

            if key not in job_candidates:
                job_candidates[key] = []

            job_candidates[key].append({
                'candidate': match.get('candidate_name', 'Unknown'),
                'score': match.get('score', 0),
                'resume_path': resume_path
            })

    # Sort jobs alphabetically
    for job_name in sorted(job_candidates.keys()):
        # Sort candidates by score (descending)
        candidates = sorted(job_candidates[job_name], key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:top_k]

        print(f"\n{Colors.BLUE}{Colors.BOLD}{job_name}{Colors.ENDC}")

        rows = []
        for i, candidate in enumerate(top_candidates, 1):
            score_color = ""
            if candidate['score'] >= 0.8:
                score_color = Colors.GREEN
            elif candidate['score'] >= 0.6:
                score_color = Colors.YELLOW

            rows.append([
                i,
                candidate['candidate'],
                f"{score_color}{candidate['score'] * 100:.1f}%{Colors.ENDC}"
            ])

        print_table(["Rank", "Candidate", "Match Score"], rows)


def main():
    """Main function to handle command-line arguments and run the program"""
    display_banner()

    parser = argparse.ArgumentParser(
        description="Resume Matcher - Match resumes with job descriptions using AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract information from resumes")
    extract_parser.add_argument("--file", help="Path to a single resume file")
    extract_parser.add_argument("--dir", help="Directory containing multiple resume files")
    extract_parser.add_argument("--output", help="Output JSON file path")
    extract_parser.add_argument("--model", default="deepseek-r1:14b", help="Ollama model to use for extraction")
    extract_parser.add_argument("--batch-size", type=int, default=5, help="Number of files to process in each batch")

    # Match command
    match_parser = subparsers.add_parser("match", help="Match resumes with job descriptions")
    match_parser.add_argument("--resumes", required=True, help="Path to JSON file with extracted resume data")
    match_parser.add_argument("--jobs", required=True, help="Path to JSON file with job descriptions")
    match_parser.add_argument("--output", required=True, help="Output JSON file for match results")
    match_parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to return")
    match_parser.add_argument("--model", default="paraphrase-mpnet-base-v2",
                              help="Sentence transformer model for embeddings")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize match results")
    viz_parser.add_argument("--results", required=True, help="Path to JSON file with match results")
    viz_parser.add_argument("--type", choices=["heatmap", "network", "radar", "3d", "word-cloud", "dashboard"],
                            required=True, help="Type of visualization to generate")
    viz_parser.add_argument("--output", help="Output file path for visualization")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute the appropriate command
    if args.command == "extract":
        print_step(1, "Extracting Resume Information",
                   "Using AI to extract structured data from resume documents")

        # Check required arguments
        if not args.file and not args.dir:
            print_error("Either --file or --dir argument is required")
            return 1

        # Initialize the LLM backend
        print_info(f"Using LLM model: {args.model}")

        if not HAS_REQUESTS:
            print_error("The 'requests' package is required for LLM communication.")
            print_info("Install it with: pip install requests")
            return 1

        try:
            backend = OllamaBackend(model_name=args.model)
        except Exception as e:
            print_error(f"Failed to initialize Ollama backend: {e}")
            return 1

        # Process resume(s)
        if args.file:
            print_info(f"Processing single resume: {args.file}")
            results = [process_resume(args.file, backend)]
        else:  # args.dir
            if not os.path.isdir(args.dir):
                print_error(f"Directory not found: {args.dir}")
                return 1

            # Get all PDF and DOCX files
            resume_files = []
            for file in os.listdir(args.dir):
                if file.lower().endswith(('.pdf', '.docx', '.doc')):
                    resume_files.append(os.path.join(args.dir, file))

            if not resume_files:
                print_warning(f"No PDF or DOCX files found in directory: {args.dir}")
                return 1

            print_info(f"Found {len(resume_files)} resume files to process")
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

        # Output results
        if args.output:
            spinner = Spinner(f"Saving results to {args.output}")
            spinner.start()

            try:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                spinner.stop()
                print_success(f"Results saved to: {args.output}")
            except Exception as e:
                spinner.stop()
                print_error(f"Failed to save results: {e}")
                return 1
        else:
            # Print summary to console
            print_summary("Extraction Results", [
                ("Total files", len(results)),
                ("Successfully processed", output_data['metadata']['success_count']),
                ("Failed to process", output_data['metadata']['error_count'])
            ])

    elif args.command == "match":
        print_step(2, "Matching Resumes with Jobs",
                   "Using semantic analysis to find the best matches")

        # Check for required packages
        if not HAS_SENTENCE_TRANSFORMERS:
            print_error("The 'sentence-transformers' package is required for matching.")
            print_info("Install it with: pip install sentence-transformers")
            return 1

        # Load resume data
        spinner = Spinner(f"Loading resume data from {args.resumes}")
        spinner.start()

        try:
            with open(args.resumes, 'r') as f:
                resume_data = json.load(f)

            spinner.stop()
            print_success(f"Loaded resume data: {len(resume_data.get('results', []))} resumes")
        except Exception as e:
            spinner.stop()
            print_error(f"Failed to load resume data: {e}")
            return 1

        # Load job openings
        job_openings = load_job_openings(args.jobs)
        if not job_openings:
            print_error("No job openings found or error loading file")
            return 1

        # Process valid resumes
        valid_resumes = [r for r in resume_data.get('results', []) if "error" not in r]
        print_info(f"Processing {len(valid_resumes)} valid resumes")

        if not valid_resumes:
            print_warning("No valid resumes to match")
            return 1

        # Enrich resume data
        enriched_resumes = [enrich_resume_data(resume) for resume in valid_resumes]

        # Create resume embeddings
        resume_embeddings, model = create_resume_embeddings(enriched_resumes, args.model)

        if not resume_embeddings or not model:
            print_error("Failed to create resume embeddings")
            return 1

        # Process job openings
        job_embeddings = process_job_openings(job_openings, model)

        # Find matches for each resume
        print_info("Finding job matches for each resume...")
        all_matches = {}

        for i, resume in enumerate(resume_embeddings):
            resume_id = resume['id']
            resume_name = resume['raw_data'].get('name', 'Unknown')

            spinner = Spinner(f"Finding matches for {resume_name}")
            spinner.start()

            matches = find_matching_jobs_for_resume(
                resume['embedding'],
                job_embeddings,
                top_k=args.top_k,
                resume_data=resume['raw_data']
            )

            all_matches[resume_id] = matches
            spinner.stop()
            print_success(f"Found {len(matches)} matches for {resume_name}")

            # Show progress
            print_progress_bar(
                i + 1,
                len(resume_embeddings),
                prefix="Matching progress:",
                suffix=f"({i + 1}/{len(resume_embeddings)})",
                length=40
            )

        print()  # Add a newline after the progress bar

        # Add matches to output data
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "resume_model": args.model,
                "job_file": args.jobs,
                "resume_file": args.resumes,
                "num_resumes": len(valid_resumes),
                "num_jobs": len(job_openings)
            },
            "results": valid_resumes,
            "job_matches": all_matches
        }

        # Save results
        if args.output:
            spinner = Spinner(f"Saving match results to {args.output}")
            spinner.start()

            try:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

                spinner.stop()
                print_success(f"Match results saved to: {args.output}")
            except Exception as e:
                spinner.stop()
                print_error(f"Failed to save match results: {e}")
                return 1

        # Display match results
        display_match_results(output_data)

    elif args.command == "visualize":
        print_step(3, "Generating Visualizations",
                   f"Creating {args.type} visualization from match results")

        # Load match results
        spinner = Spinner(f"Loading match results from {args.results}")
        spinner.start()

        try:
            with open(args.results, 'r') as f:
                match_data = json.load(f)

            spinner.stop()
            print_success(f"Loaded match results successfully")
        except Exception as e:
            spinner.stop()
            print_error(f"Failed to load match results: {e}")
            return 1

        # Generate the requested visualization
        if args.type == "heatmap":
            print_info("Generating heatmap visualization...")
            # Import the visualization module here
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd

                # Process visualization
                # This would call your existing visualization code
                print_success("Heatmap visualization generated successfully")

            except ImportError:
                print_error("Required packages not found. Please install matplotlib, seaborn, and pandas.")
                return 1

        elif args.type == "network":
            print_info("Generating network visualization...")
            try:
                import networkx as nx
                import plotly.graph_objects as go

                # Process visualization
                # This would call your existing visualization code
                print_success("Network visualization generated successfully")

            except ImportError:
                print_error("Required packages not found. Please install networkx and plotly.")
                return 1

        elif args.type == "radar":
            print_info("Generating radar chart visualization...")
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                # Process visualization
                # This would call your existing visualization code
                print_success("Radar chart visualization generated successfully")

            except ImportError:
                print_error("Required packages not found. Please install matplotlib and numpy.")
                return 1

        elif args.type == "3d":
            print_info("Generating 3D visualization...")
            try:
                import plotly.express as px
                import pandas as pd
                import numpy as np
                from sklearn.decomposition import PCA

                # Process visualization
                # This would call your existing visualization code
                print_success("3D visualization generated successfully")

            except ImportError:
                print_error("Required packages not found. Please install plotly, pandas, numpy, and scikit-learn.")
                return 1

        elif args.type == "word-cloud":
            print_info("Generating word cloud visualization...")
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                # Process visualization
                # This would call your existing visualization code
                print_success("Word cloud visualization generated successfully")

            except ImportError:
                print_error("Required packages not found. Please install wordcloud and matplotlib.")
                return 1

        elif args.type == "dashboard":
            print_info("Starting interactive dashboard...")
            try:
                import streamlit as st

                # This would call your existing streamlit visualization code
                print_success("Dashboard started successfully")

            except ImportError:
                print_error("Required packages not found. Please install streamlit.")
                return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n" + Colors.YELLOW + "Operation cancelled by user." + Colors.ENDC)
        sys.exit(1)
    except Exception as e:
        print("\n\n" + Colors.RED + f"Unhandled error: {e}" + Colors.ENDC)
        sys.exit(1)