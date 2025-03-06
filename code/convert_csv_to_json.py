import argparse
import sys
import os
import pandas as pd
import json
import re


# ANSI color codes for styled output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(title):
    """Print a stylish section header."""
    width = 60
    padding = (width - len(title) - 2) // 2
    print("\n" + "═" * width)
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * padding} {title} {'═' * padding}{Colors.ENDC}")
    print("═" * width + "\n")


def load_jobs_from_arpa_format(file_path):
    """
    Load job listings from file based on file extension.
    Supports Excel (.xls, .xlsx), CSV (.csv), and tab-delimited (.txt, .tsv, .tab) formats.

    Parameters:
    -----------
    file_path : str
        Path to the job listings file

    Returns:
    --------
    list
        A list of job dictionaries in the format required by the resume matcher
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.xls', '.xlsx', '.xlsm']:
        return load_jobs_from_excel(file_path)
    elif file_ext == '.csv':
        return load_jobs_from_csv(file_path)
    elif file_ext in ['.txt', '.tsv', '.tab']:
        return load_jobs_from_tab_delimited(file_path)
    else:
        print(
            f"{Colors.RED}Unsupported file format: {file_ext}. Please provide an Excel, CSV, or tab-delimited file.{Colors.ENDC}")
        return []


def load_jobs_from_excel(excel_file, sheet_name=0):
    """Load job listings from Excel file."""
    print(f"{Colors.BLUE}Reading job listings from Excel: {excel_file}{Colors.ENDC}")

    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"{Colors.GREEN}Successfully read {len(df)} job listings{Colors.ENDC}")
        return process_job_dataframe(df)
    except Exception as e:
        print(f"{Colors.RED}Error reading Excel file: {e}{Colors.ENDC}")
        return []


def load_jobs_from_csv(csv_file):
    """Load job listings from CSV file."""
    print(f"{Colors.BLUE}Reading job listings from CSV: {csv_file}{Colors.ENDC}")

    try:
        # First attempt to read with standard CSV settings
        df = pd.read_csv(csv_file)
        print(f"{Colors.GREEN}Successfully read {len(df)} job listings{Colors.ENDC}")

        # Check if we might have a malformed CSV (too few columns)
        if len(df.columns) <= 2:
            print(
                f"{Colors.YELLOW}CSV might have non-standard delimiters. Attempting to detect delimiter...{Colors.ENDC}")

            # Try to detect the delimiter by reading the first few lines
            with open(csv_file, 'r') as f:
                sample = f.read(2048)  # Read a sample of the file

            # Count potential delimiters
            delimiters = {
                ',': sample.count(','),
                ';': sample.count(';'),
                '\t': sample.count('\t'),
                '|': sample.count('|')
            }

            # Find the most common delimiter
            best_delimiter = max(delimiters, key=delimiters.get)

            if best_delimiter != ',' and delimiters[best_delimiter] > 0:
                print(f"{Colors.YELLOW}Detected possible delimiter: '{best_delimiter}'. Retrying...{Colors.ENDC}")
                df = pd.read_csv(csv_file, sep=best_delimiter)
                print(f"{Colors.GREEN}Successfully read {len(df)} job listings with custom delimiter{Colors.ENDC}")

        return process_job_dataframe(df)
    except Exception as e:
        print(f"{Colors.RED}Error reading CSV file: {e}{Colors.ENDC}")
        return []


def load_jobs_from_tab_delimited(tab_file):
    """Load job listings from tab-delimited text file."""
    print(f"{Colors.BLUE}Reading job listings from tab-delimited file: {tab_file}{Colors.ENDC}")

    try:
        df = pd.read_csv(tab_file, sep='\t')
        print(f"{Colors.GREEN}Successfully read {len(df)} job listings{Colors.ENDC}")
        return process_job_dataframe(df)
    except Exception as e:
        print(f"{Colors.RED}Error reading tab-delimited file: {e}{Colors.ENDC}")
        return []


def process_job_dataframe(df):
    """
    Process a DataFrame containing job listings and convert to the format
    needed for the resume matcher.

    This function handles the core logic of mapping columns and extracting
    structured data regardless of the original file format.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing job listing data

    Returns:
    --------
    list
        A list of job dictionaries in the resume matcher format
    """
    # Process column names and map them
    columns = df.columns.tolist()
    print(f"{Colors.YELLOW}Columns found: {', '.join(columns)}{Colors.ENDC}")

    # Define mappings with flexible matching
    column_mappings = {
        "title": ["Role", "Title", "Position", "Job Title"],
        "description": ["Description", "Job Description", "Summary", "Details"],
        "education": ["Degree", "Education", "Required Education", "Qualification"],
        "experience": ["Years of Exp", "Experience", "Required Experience", "Yrs"],
        "skills": ["Expanded Experience", "Skills", "Required Skills", "Qualifications"]
    }

    # Find the actual column names in the file
    field_to_column = {}
    for field, possible_names in column_mappings.items():
        found = False
        for name in possible_names:
            # Look for exact matches first
            exact_matches = [col for col in columns if col == name]
            if exact_matches:
                field_to_column[field] = exact_matches[0]
                found = True
                break

            # If no exact match, look for partial matches
            if not found:
                partial_matches = [col for col in columns if name.lower() in col.lower()]
                if partial_matches:
                    field_to_column[field] = partial_matches[0]
                    found = True
                    break

        if not found:
            print(f"{Colors.YELLOW}Warning: No column found for '{field}'{Colors.ENDC}")

    # Convert to the format needed for resume matcher
    jobs = []
    for idx, row in df.iterrows():
        job = {"id": f"job{idx + 1:03d}"}

        # Process title
        if "title" in field_to_column:
            title_col = field_to_column["title"]
            job["title"] = str(row[title_col]) if not pd.isna(row[title_col]) else f"Position {idx + 1}"
        else:
            job["title"] = f"Position {idx + 1}"

        # Process description
        if "description" in field_to_column:
            desc_col = field_to_column["description"]
            job["description"] = str(row[desc_col]) if not pd.isna(row[desc_col]) else ""
        else:
            job["description"] = ""

        # Set company - this could be customized based on your data
        job["company"] = "ARPA-H"

        # Process education requirements
        if "education" in field_to_column:
            edu_col = field_to_column["education"]

            if pd.isna(row[edu_col]):
                job["required_education"] = []
            else:
                edu_text = str(row[edu_col])
                # Split on commas, 'or', and similar separators
                education_parts = re.split(r'[,;/]|\s+or\s+', edu_text)
                job["required_education"] = [part.strip() for part in education_parts if part.strip()]
        else:
            job["required_education"] = []

        # Process experience requirements
        if "experience" in field_to_column:
            exp_col = field_to_column["experience"]

            if pd.isna(row[exp_col]):
                job["required_experience"] = ""
            else:
                # Try to handle various formats
                exp_value = row[exp_col]
                # If it's already a number, format it as years
                if isinstance(exp_value, (int, float)):
                    job["required_experience"] = f"{exp_value} years"
                else:
                    # Try to extract a number if it's text
                    exp_text = str(exp_value)
                    # Look for patterns like "5+ years" or "at least 3 years"
                    matches = re.search(r'(\d+)(?:\+|\s*\+)?(?:\s*years|\s*yrs)', exp_text, re.IGNORECASE)
                    if matches:
                        job["required_experience"] = f"{matches.group(1)} years"
                    else:
                        # Just use the text as is
                        job["required_experience"] = exp_text
        else:
            job["required_experience"] = ""

        # Process skills from expanded experience
        if "skills" in field_to_column:
            skills_col = field_to_column["skills"]

            if pd.isna(row[skills_col]):
                job["required_skills"] = []
            else:
                skills_text = str(row[skills_col])

                # First try to extract specific skills
                skills = extract_skills_from_text(skills_text)
                job["required_skills"] = skills
        else:
            job["required_skills"] = []

        jobs.append(job)

    print(f"{Colors.GREEN}Successfully processed {len(jobs)} job listings{Colors.ENDC}")
    return jobs


def extract_skills_from_text(text):
    """
    Extract skills from a text description.
    Uses multiple approaches to identify skill phrases in the text.

    Parameters:
    -----------
    text : str
        Text containing skills information

    Returns:
    --------
    list
        A list of extracted skills
    """
    skills = []

    # Method 1: Look for "Experience in/with X" patterns
    experience_patterns = re.findall(r'Experience\s+(?:in|with)\s+([^.]+?)(?:required|preferred|\.)', text)

    for pattern in experience_patterns:
        # Split by "and" and commas for individual skills
        parts = re.split(r'(?:and|,)', pattern)
        for part in parts:
            clean_part = part.strip()
            if clean_part and clean_part not in skills:
                skills.append(clean_part)

    # Method 2: If no skills were found, try sentence-by-sentence extraction
    if not skills:
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sentence in sentences:
            # Look for skill-related keywords
            if any(keyword in sentence.lower() for keyword in
                   ["experience", "knowledge", "proficiency", "skill", "ability"]):
                # Try to extract the skill part
                if ":" in sentence:
                    # Handle colon-separated format like "Skills: Python, Java"
                    skill_part = sentence.split(":", 1)[1].strip()
                    parts = re.split(r'(?:and|,|;)', skill_part)
                    for part in parts:
                        clean_part = part.strip()
                        if clean_part and clean_part not in skills:
                            skills.append(clean_part)
                else:
                    # Just use the sentence as a skill
                    if sentence not in skills:
                        skills.append(sentence)

    # Method 3: If still no skills, split the text into bullet-point-like items
    if not skills:
        # Split by common delimiters that might indicate separate skills
        parts = re.split(r'(?:;\s*|\.\s*|\n\s*)', text)
        for part in parts:
            clean_part = part.strip()
            if clean_part and clean_part not in skills:
                skills.append(clean_part)

    # Method 4: If all else fails, just use the entire text as one skill
    if not skills and text.strip():
        skills = [text.strip()]

    return skills


def save_jobs_to_json(jobs, output_file):
    """Save processed jobs to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(jobs, f, indent=2)

        print(f"{Colors.GREEN}Successfully saved {len(jobs)} job listings to {output_file}{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"{Colors.RED}Error saving job data to JSON: {e}{Colors.ENDC}")
        return False


def convert_arpa_jobs(input_file, output_file):
    """Convert job listings to resume matcher format."""
    print_header("JOB LISTING CONVERTER")

    # Load jobs from input file
    jobs = load_jobs_from_arpa_format(input_file)

    if not jobs:
        print(f"{Colors.RED}No job listings were processed. Check the input file.{Colors.ENDC}")
        return False

    # Save jobs to output JSON file
    success = save_jobs_to_json(jobs, output_file)

    if success:
        print_job_summary(jobs)

    return success


def print_job_summary(jobs):
    """Print a summary of the processed jobs."""
    print_header("JOB LISTINGS SUMMARY")

    # Print a table of jobs
    print(f"{Colors.BOLD}{'ID':<8} {'Title':<35} {'Experience':<15} {'Education':<35}{Colors.ENDC}")
    print("-" * 93)

    for job in jobs:
        job_id = job.get("id", "N/A")
        title = job.get("title", "Unknown")
        if len(title) > 35:
            title = title[:32] + "..."

        experience = job.get("required_experience", "N/A")

        education = job.get("required_education", [])
        if education:
            edu_text = ", ".join(education)
            if len(edu_text) > 35:
                edu_text = edu_text[:32] + "..."
        else:
            edu_text = "N/A"

        print(f"{job_id:<8} {title:<35} {experience:<15} {edu_text:<35}")

    print("\n" + "-" * 93)
    print(f"{Colors.GREEN}Total Jobs: {len(jobs)}{Colors.ENDC}")

    # Show a sample of the first job in detail
    if jobs:
        print_header("SAMPLE JOB DETAILS")
        sample_job = jobs[0]

        print(f"{Colors.BOLD}Title:{Colors.ENDC} {sample_job.get('title', 'N/A')}")
        print(f"{Colors.BOLD}ID:{Colors.ENDC} {sample_job.get('id', 'N/A')}")
        print(f"{Colors.BOLD}Company:{Colors.ENDC} {sample_job.get('company', 'N/A')}")
        print(f"{Colors.BOLD}Experience:{Colors.ENDC} {sample_job.get('required_experience', 'N/A')}")

        print(f"{Colors.BOLD}Education:{Colors.ENDC}")
        for edu in sample_job.get("required_education", []):
            print(f"  - {edu}")

        print(f"{Colors.BOLD}Skills:{Colors.ENDC}")
        for skill in sample_job.get("required_skills", []):
            print(f"  - {skill}")

        print(f"{Colors.BOLD}Description:{Colors.ENDC}")
        description = sample_job.get("description", "")
        if len(description) > 200:
            print(f"  {description[:200]}...")
        else:
            print(f"  {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert job listings to the format required by the resume matcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', required=True,
                        help='Path to the input file (Excel, CSV, or tab-delimited)')
    parser.add_argument('--output', default='jobs.json',
                        help='Path to the output JSON file')
    parser.add_argument('--company', default='ARPA-H',
                        help='Company name to use for all job listings')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"{Colors.RED}Error: Input file not found: {args.input}{Colors.ENDC}")
        return 1

    success = convert_arpa_jobs(args.input, args.output)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())