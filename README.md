# Resume Matcher v4

An intelligent tool that uses semantic analysis to match resumes with job descriptions, providing insights into the best candidates for each position. Developed by Sam Nazari, Ph.D.

## Overview

Resume Matcher v4 is an advanced application that leverages AI and natural language processing to help recruiters and hiring managers efficiently match candidate resumes with job descriptions. The system extracts structured information from resumes, generates semantic embeddings for both resumes and job descriptions, and calculates similarity scores to identify the best matches. Multiple visualizations provide insights into the matching results.

## Key Features

- **Resume Information Extraction**: Automatically extracts structured data (name, experience, education) from PDF and DOCX resumes using LLMs
- **Semantic Matching**: Uses sentence transformers to generate embeddings and calculate semantic similarity between resumes and job descriptions
- **Local LLM Integration**: Built-in support for Ollama and other LLM backends
- **Rich Visualizations**: Multiple visualization options including radar charts, heatmaps, 3D scatter plots, and network graphs
- **Batch Processing**: Efficiently processes multiple resumes in parallel
- **Command Line Interface**: Easy-to-use CLI with various commands for extraction, matching, and visualization

## Project Structure

```
resume-matcher-v4/
│
├── code/                 # Application source code
│   ├── main.py           # Main application script
│   ├── main_ollama_semantic_search.py  # Alternative implementation with Ollama
│   ├── resume-extractor.py             # Resume text extraction utilities
│   ├── convert_csv_to_json.py          # Data conversion utilities
│   ├── create_test_jobs.py             # Utility to create test job listings
│   └── visualization_*.py              # Various visualization scripts
│
├── data/                 # Input data directory
│   ├── candidates/       # Resume files (PDF, DOCX)
│   └── jobs/             # Job description files (JSON, Excel, CSV)
│
└── output/               # Output directory for results and visualizations
```

## Technical Architecture

### Core Classes

#### LLM Backend Classes

These classes provide an abstraction layer for language model interactions:

##### 1. `LLMBackend` (Abstract Base Class)
- Provides the interface for different language model backends
- Key method: `generate(prompt: str)` - Must be implemented by subclasses

##### 2. `OllamaBackend` (Implementation)
- Uses Ollama local LLM for text generation
- Initializes with a model name (default: "llama3.1:latest") and endpoint
- Verifies Ollama is running and the requested model is available
- Generates text using the Ollama API with controlled parameters (temperature=0.0)

##### 3. `HuggingFaceEndpointBackend` (Implementation)
- Connects to Hugging Face Inference API
- Requires a model ID and API token
- Handles different response formats from various models

##### 4. `HuggingFaceLocalBackend` (Implementation)
- Uses locally cached Hugging Face models
- Support for optimizations like 4-bit quantization with BitsAndBytes
- Automatically selects CUDA or CPU based on availability

#### Resume Processing

##### `ResumeInfoExtractor`
- Extracts structured information from resumes using few-shot learning
- Constructor accepts a LLMBackend instance for text generation
- Key methods:
  - `extract_resume_information(resume_text: str)`: Uses prompt engineering with examples to extract name, experience, and education
  - `_extract_json_from_response(response: str)`: Parses JSON from LLM response
  - `_validate_extraction(extracted_info: Dict)`: Validates and corrects the extracted information

#### UI Components

The main.py file contains several UI-related classes and functions:
- `Colors`: ANSI color codes for terminal output
- `Spinner`: Animated terminal spinner for showing ongoing operations
- Various printing functions for displaying formatted output (banners, tables, progress bars)

### Core Functions

#### Resume Processing

1. **Text Extraction**
   - `extract_text_from_pdf(file_path: str)`: Extracts text from PDF resumes
   - `extract_text_from_docx(file_path: str)`: Extracts text from DOCX resumes

2. **Resume Processing**
   - `process_resume(file_path: str, backend: LLMBackend)`: Processes a single resume file
   - `batch_process_resumes(file_paths: List[str], backend: LLMBackend, batch_size: int)`: Processes multiple resumes in batches

#### Job Matching

1. **Data Enhancement**
   - `enrich_resume_data(resume_data)`: Transforms raw resume data into richer text for embedding
   - `create_resume_embeddings(resume_collection, model_name)`: Generates semantic embeddings for resumes
   - `load_job_openings(job_file_path)`: Loads job opening data from a JSON file
   - `process_job_openings(job_descriptions, model)`: Processes job descriptions for semantic matching

2. **Matching Logic**
   - `find_matching_jobs_for_resume(resume_embedding, job_embeddings, top_k, resume_data)`: Finds best matching jobs for a resume
   - `display_match_results(match_data, top_k)`: Displays a dashboard of match results

#### Data Conversion & Loading

- `load_jobs_from_excel(excel_file, sheet_name)`: Loads job listings from Excel
- `prepare_jobs_for_resume_matcher(excel_file, output_json_file, sheet_name)`: Converts Excel jobs to the required JSON format

### Visualization Components

The application provides multiple visualization options:

1. **Radar Charts (`visualization_top_candidates_radar.py`)**
   - Visualizes multiple skill dimensions for top candidates per job
   - Creates a radar/spider chart comparing candidates across different skill categories
   - Generates PNG files with radar charts for each job position

2. **Heatmaps (`visualization_heatmap.py`)**
   - Creates heatmaps to visualize the alignment between candidates and job requirements
   - Two types of visualization:
     - Score-based heatmap showing raw scores by category
     - Weighted alignment heatmap accounting for requirement importance
   - Uses seaborn and matplotlib for visualization

3. **3D Visualization (`visualization_3d_plotly.py`)**
   - Creates an interactive 3D scatter plot using Plotly
   - Uses PCA to reduce resume features to 3 dimensions
   - Color-codes candidates by their best matching job
   - Size of points indicates match score
   - Includes hover information with candidate details

4. **Other Visualizations**
   - Network graphs (`visualization_networkx.py`)
   - Quality matrices (`visualization_quality_matrix.py`) 
   - Ranking bump charts (`visualization_ranking_bump_charts.py`)
   - Score distribution histograms (`visualization_score_distribution_histogram.py`)
   - Streamlit dashboard (`visualization_streamlit.py`)
   - Word clouds (`visualization_word_cloud.py`)

## Detailed Class & Method Breakdown

### LLM Backend Classes

#### `LLMBackend` (Abstract Base Class)
- **Purpose**: Serves as an interface for various language model implementations
- **Methods**:
  - `generate(prompt: str) -> str`: Abstract method that must be implemented by subclasses to generate text based on a prompt

#### `OllamaBackend` (Implementation)
- **Purpose**: Provides integration with Ollama local LLM service
- **Constructor Parameters**:
  - `model_name: str = "llama3.1:latest"`: The name of the Ollama model to use
  - `endpoint: str = "http://localhost:11434/api/generate"`: The Ollama API endpoint
- **Methods**:
  - `__init__(model_name: str, endpoint: str)`: Initializes the backend, checks if Ollama is running and if the model is available
  - `generate(prompt: str) -> str`: Sends a request to the Ollama API and returns the generated text
    - Uses temperature=0.0 for deterministic outputs
    - Sets num_predict=1024 to control output length

#### `HuggingFaceEndpointBackend` (Implementation)
- **Purpose**: Connects to Hugging Face Inference API for model access
- **Constructor Parameters**:
  - `model_id: str`: The ID of the model on Hugging Face
  - `api_token: str`: Authentication token for Hugging Face API
  - `api_url: str = "https://api-inference.huggingface.co/models/"`: Base URL for the API
- **Methods**:
  - `__init__(model_id: str, api_token: str, api_url: str)`: Sets up the connection with proper authentication
  - `generate(prompt: str) -> str`: Sends a request to the Hugging Face API and handles different response formats

#### `HuggingFaceLocalBackend` (Implementation)
- **Purpose**: Uses locally cached Hugging Face models for offline inference
- **Constructor Parameters**:
  - `model_id: str`: The ID of the model to load locally
- **Methods**:
  - `__init__(model_id: str)`: Loads the model with appropriate optimizations (4-bit quantization if available)
  - `generate(prompt: str) -> str`: Tokenizes input, generates text using the local model, and formats the response

### Resume Processing Classes

#### `ResumeInfoExtractor`
- **Purpose**: Extracts structured information from resumes using few-shot learning with LLMs
- **Constructor Parameters**:
  - `backend: LLMBackend`: An instance of an LLM backend for text generation
- **Methods**:
  - `__init__(backend: LLMBackend)`: Initializes with the specified LLM backend
  - `extract_resume_information(resume_text: str) -> Dict`: Uses few-shot learning to extract name, experience, and education from a resume
    - Creates a prompt with examples of different resume formats
    - Sends the prompt to the LLM backend
    - Extracts and validates the structured information
  - `_extract_json_from_response(response: str) -> Dict`: Helper method to parse JSON from the LLM response
    - Handles edge cases where the JSON might be embedded in other text
    - Attempts to extract valid JSON or falls back to manual extraction
  - `_validate_extraction(extracted_info: Dict) -> Tuple[Dict, List[str]]`: Validates and corrects extracted information
    - Ensures required fields are present
    - Converts years of experience to a numeric value
    - Validates the format of degrees
    - Returns the validated information and a list of issues found

### UI Components (in main.py)

#### `Colors`
- **Purpose**: Provides ANSI color codes for terminal output formatting
- **Constants**:
  - `HEADER`, `BLUE`, `GREEN`, `YELLOW`, `RED`, `ENDC`, `BOLD`, `UNDERLINE`, `BG_BLACK`, `BG_BLUE`, `BG_GREEN`: Various color and formatting codes

#### `Spinner`
- **Purpose**: Creates an animated terminal spinner for showing ongoing operations
- **Constructor Parameters**:
  - `message: str = "Processing"`: The message to display next to the spinner
  - `delay: float = 0.1`: The delay between spinner animation frames
- **Methods**:
  - `__init__(message: str, delay: float)`: Sets up the spinner configuration
  - `spin()`: Displays the spinner animation by cycling through characters
  - `start()`: Starts the spinner in a separate thread
  - `stop()`: Stops the spinner animation and cleans up the display

#### Terminal UI Functions

- `get_terminal_width() -> int`: Determines the width of the terminal
- `display_banner()`: Shows the application banner with ASCII art
- `print_header(title: str)`: Prints a stylish section header
- `print_step(step_num: int, title: str, description: str = None)`: Prints a numbered step with optional description
- `print_success(message: str)`: Prints a success message with green checkmark
- `print_error(message: str)`: Prints an error message with red X
- `print_warning(message: str)`: Prints a warning message with yellow warning symbol
- `print_info(message: str)`: Prints an informational message with blue info symbol
- `print_table(headers: List[str], rows: List[List[Any]], max_width: int = None)`: Prints a formatted table with proper alignment
- `print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50, fill: str = '█')`: Displays a progress bar for operations
- `print_summary(title: str, items: List[Tuple[str, Any]])`: Prints a summary box with key-value pairs

### File Processing Functions

#### Text Extraction Functions
- `extract_text_from_pdf(file_path: str) -> str`: 
  - Extracts text content from PDF files using PyPDF2
  - Handles potential errors and returns the extracted text

- `extract_text_from_docx(file_path: str) -> str`: 
  - Extracts text content from DOCX files using python-docx
  - Preserves paragraphs and returns the extracted text

#### Resume Processing Functions
- `process_resume(file_path: str, backend: LLMBackend) -> Dict`:
  - Determines file type and extracts text using appropriate function
  - Creates a ResumeInfoExtractor instance to process the resume
  - Returns structured information with metadata

- `batch_process_resumes(file_paths: List[str], backend: LLMBackend, batch_size: int = 5) -> List[Dict]`:
  - Processes multiple resume files in batches
  - Shows progress with a progress bar
  - Returns a list of processed resume data

### Job Matching Functions

#### Data Enhancement Functions
- `enrich_resume_data(resume_data: Dict) -> str`:
  - Transforms raw resume data into a richer text representation
  - Formats experience, education, and skills information
  - Returns a string suitable for embedding

- `create_resume_embeddings(resume_collection: List[Dict], model_name: str = 'paraphrase-mpnet-base-v2') -> Dict`:
  - Generates semantic embeddings for a collection of resume data
  - Uses SentenceTransformer with the specified model
  - Returns a dictionary mapping resume paths to their embeddings

- `load_job_openings(job_file_path: str) -> List[Dict]`:
  - Loads job opening data from a JSON file
  - Validates the structure of job descriptions
  - Returns a list of job opening dictionaries

- `load_jobs_from_excel(excel_file: str, sheet_name: str = 0) -> List[Dict]`:
  - Loads job listings from an Excel file
  - Converts spreadsheet data to the format required by the resume matcher
  - Returns a list of job dictionaries

- `prepare_jobs_for_resume_matcher(excel_file: str, output_json_file: str, sheet_name: str = 0) -> bool`:
  - Reads job listings from Excel and saves them in JSON format
  - Converts the data structure for compatibility
  - Returns True if successful, False otherwise

- `process_job_openings(job_descriptions: List[Dict], model) -> Dict`:
  - Processes job descriptions for semantic matching
  - Generates embeddings for each job description
  - Returns a dictionary with job data and embeddings

#### Matching Logic Functions
- `find_matching_jobs_for_resume(resume_embedding, job_embeddings: Dict, top_k: int = 5, resume_data: Dict = None) -> List[Dict]`:
  - Finds the best matching jobs for a given resume embedding
  - Calculates semantic similarity scores
  - Returns a list of top-k matching jobs with scores

- `display_match_results(match_data: Dict, top_k: int = 3) -> None`:
  - Displays a dashboard of match results in the terminal
  - Shows top matches for each resume
  - Includes scores and match highlights

### Visualization Classes & Functions

#### Radar Chart Visualization (visualization_top_candidates_radar.py)
- **Purpose**: Creates radar charts showing top candidates for each job
- **Functions**:
  - Extracts top candidates by job title
  - Simulates skill ratings for candidates
  - Creates radar charts using matplotlib
  - Saves visualizations as PNG files

#### Heatmap Visualization (visualization_heatmap.py)
- **Purpose**: Creates heatmaps showing alignment between candidates and job requirements
- **Functions**:
  - `simulate_candidate_scores(resume_data, match_score)`: Generates category scores based on resume data
  - Creates pivot tables for visualization
  - Generates two types of heatmaps (raw scores and weighted alignment)
  - Uses seaborn for visualization

#### 3D Visualization (visualization_3d_plotly.py)
- **Purpose**: Creates an interactive 3D scatter plot of candidates
- **Functions**:
  - Extracts and simulates features from resume data
  - Uses PCA to reduce dimensions to 3D
  - Creates interactive visualization with plotly
  - Color-codes candidates by best matching job

#### Other Visualization Components
- **Network Graph (visualization_networkx.py)**:
  - Creates a network graph showing relationships between resumes and jobs
  - Uses NetworkX for graph creation and visualization

- **Quality Matrix (visualization_quality_matrix.py)**:
  - Creates a matrix visualization of match quality
  - Highlights strengths and weaknesses across dimensions

- **Ranking Bump Charts (visualization_ranking_bump_charts.py)**:
  - Shows how candidate rankings change across different criteria
  - Visualizes movement in rankings

- **Score Distribution (visualization_score_distribution_histogram.py)**:
  - Creates histograms of match scores
  - Helps identify score distributions and outliers

- **Streamlit Dashboard (visualization_streamlit.py)**:
  - Creates an interactive web dashboard using Streamlit
  - Allows filtering and exploration of match results

- **Word Cloud (visualization_word_cloud.py)**:
  - Generates word clouds from resume and job description text
  - Highlights frequently occurring terms

## Usage

### Resume Information Extraction

Extract structured information from a single resume:
```
python code/main.py extract --file path/to/resume.pdf --output extracted_resume.json
```

Process a directory of resumes:
```
python code/main.py extract --dir path/to/resumes/ --output extracted_resumes.json
```

### Job Matching

Match resumes against job descriptions:
```
python code/main.py match --resumes extracted_resumes.json --jobs jobs.json --output matches.json
```

### Visualization

Generate visualizations:
```
python code/main.py visualize --results matches.json --type radar --output visualization.html
```

### Alternative Implementation with Ollama

Use the semantic search implementation with Ollama:
```
python code/main_ollama_semantic_search.py --file path/to/resume.pdf --match --job-file jobs.json
```

## Workflow

The typical workflow for using this application is:

1. **Extract Information from Resumes**:
   - The system processes resume documents (PDF/DOCX) to extract structured information
   - LLM (via Ollama or other backends) is used for advanced information extraction
   - Results are saved as JSON for further processing

2. **Process Job Descriptions**:
   - Job descriptions are loaded from JSON or Excel files
   - Key requirements and responsibilities are identified
   - Job details are prepared for semantic matching

3. **Semantic Matching**:
   - Resume and job descriptions are converted into semantic embeddings
   - Similarity scores are calculated between resumes and jobs
   - Best matches are identified based on semantic similarity

4. **Results Analysis**:
   - Various visualizations help analyze and interpret the matching results
   - Radar charts compare top candidates for each position
   - Heatmaps show alignment between candidates and job requirements
   - 3D visualizations and network graphs provide additional perspectives

## Technologies Used

- **Python**: Core programming language
- **Sentence Transformers**: For generating semantic embeddings
- **Ollama**: Local LLM for text generation and analysis
- **PyPDF2/docx**: For document parsing
- **Matplotlib/Plotly/Seaborn/NetworkX**: For data visualization
- **Pandas**: For data manipulation and analysis
- **scikit-learn**: For PCA and data preprocessing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/resume-matcher-v4.git
   cd resume-matcher-v4
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install [Ollama](https://ollama.ai) for local LLM processing.

4. Pull the required model (default is llama3.1):
   ```
   ollama pull llama3.1
   ```

## Current Limitations

### Embedding Generation Process
The application recalculates embeddings for all candidates every time it is run. *There is no persistent storage mechanism for embeddings in the current implementation.* 

Here's the specific workflow:

1. **Per-Run Processing:** When you run the application with the match command, it:
   - Loads previously extracted resume data from a JSON file (--resumes parameter)
   - Loads job descriptions from a JSON file (--jobs parameter)
   - Generates embeddings for all resumes and jobs during that specific run
2. **No Embedding Persistence:** The code in create_resume_embeddings() and process_job_openings() functions create embeddings in-memory using the Sentence Transformer model. These embeddings are not stored persistently between runs and exist only for the duration of the program execution.

### Matching Process Flow
Load resume data → Enrich resume data → Create embeddings → 
Load job openings → Process job openings (create job embeddings) → 
Find matches → Save results

### Performance Implications
For large datasets, this approach means embedding generation might be a computational bottleneck. The application uses sentence-transformers library with models like *paraphrase-mpnet-base-v2* to generate embeddings. There's a progress bar (show_progress_bar=True) when generating embeddings to indicate this might be a time-consuming operation

### Potential Enhancement
One of the planned enhancements in the feature roadmap is "Add support for vector databases (FAISS)" which would address this exact limitation. A vector database like FAISS would allow:

1. Persistent storage of embeddings
2. Incremental updates (only calculate embeddings for new candidates)
3. Faster similarity searches
4. Better scalability for large numbers of resumes and job descriptions

Currently, if you have 100 resumes and run the matching process twice, the application will calculate those 100 embeddings twice. With a vector database implementation, you could calculate each embedding once and reuse it across multiple runs.

## Feature Roadmap

### Future Enhancements

- Add support for vector databases(FAISS)
- Add support for multimodel embeddings
- Add support for web scraping (crawl4AI)
- Add support for more LLM backends
- Add support for more document formats
- Add support for more visualization types
- Add support for more matching algorithms
- Add support for more matching criteria
- Add support for more matching metrics
- Add support for more matching thresholds
- Add support for more matching filters
- Add support for more matching options
- Add support for more matching parameters
- Add support for more matching results
- Add support for more matching outputs
- Add support for more matching visualizations
- Add support for more matching analytics
- Add support for more matching reporting
