import json
import requests
from typing import Dict, List, Union, Optional, Literal


class LLMBackend:
    """Abstract base class for different LLM backends"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")


class OllamaBackend(LLMBackend):
    """Backend for using Ollama local LLM"""

    def __init__(self, model_name: str = "llama3", endpoint: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.endpoint = endpoint

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Keep deterministic
                "num_predict": 1024  # Adjust based on expected output size
            }
        }

        response = requests.post(self.endpoint, json=payload)

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} {response.text}")

        return response.json().get("response", "")


class HuggingFaceEndpointBackend(LLMBackend):
    """Backend for using Hugging Face Inference API"""

    def __init__(self, model_id: str, api_token: str,
                 api_url: str = "https://api-inference.huggingface.co/models/"):
        self.model_id = model_id
        self.api_url = f"{api_url.rstrip('/')}/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def generate(self, prompt: str) -> str:
        payload = {"inputs": prompt, "parameters": {"temperature": 0.01, "max_new_tokens": 1024}}

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"HuggingFace API error: {response.status_code} {response.text}")

        # Handle different response formats from different models
        result = response.json()

        # Format might be [{"generated_text": "..."}] or {"generated_text": "..."}
        if isinstance(result, list):
            return result[0].get("generated_text", "").replace(prompt, "")
        elif isinstance(result, dict):
            return result.get("generated_text", "").replace(prompt, "")
        else:
            return str(result).replace(prompt, "")


class HuggingFaceLocalBackend(LLMBackend):
    """Backend for using locally cached Hugging Face models"""

    def __init__(self, model_id: str):
        # Import inside the class to make it optional
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with appropriate device placement
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")

        # Use BitsAndBytes for 4-bit quantization if available
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print("Model loaded with 4-bit quantization")
        except ImportError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            print("Model loaded without quantization")

    def generate(self, prompt: str) -> str:
        # Import inside the method to make torch optional
        import torch

        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)

        # Generate with the model
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=0.01,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the output, remove the prompt
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Return only the new text (remove the prompt)
        return generated_text[len(prompt):]


class ResumeInfoExtractor:
    """Extract structured information from resumes using few-shot learning with LLMs"""

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def extract_resume_information(self, resume_text: str) -> Dict:
        """Extract name, years of experience, and degrees from resume text."""
        # Create a prompt with examples of different resume styles
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
        # Try to find JSON within the response text
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


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        raise ImportError("Please install pdfplumber: pip install pdfplumber")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        raise ImportError("Please install python-docx: pip install python-docx")


def process_resume(file_path: str, backend: LLMBackend) -> Dict:
    """Process a single resume file"""
    # Extract text based on file type
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Create extractor with the specified backend
    extractor = ResumeInfoExtractor(backend)

    # Extract information
    info = extractor.extract_resume_information(text)
    info['file_path'] = file_path

    return info


def batch_process_resumes(file_paths: List[str], backend: LLMBackend,
                          batch_size: int = 10) -> List[Dict]:
    """Process multiple resume files in batches"""
    results = []

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        batch_results = []

        for file_path in batch:
            try:
                info = process_resume(file_path, backend)
                batch_results.append(info)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                batch_results.append({
                    'file_path': file_path,
                    'error': str(e)
                })

        results.extend(batch_results)
        print(f"Processed batch {i // batch_size + 1}/{(len(file_paths) + batch_size - 1) // batch_size}")

    return results