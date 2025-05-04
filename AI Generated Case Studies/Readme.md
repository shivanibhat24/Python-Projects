# AI Case Study Generator

A user-friendly Python tool for generating AI-powered case studies using LLaMA 3, LangChain, ChromaDB, and pdfkit. This tool allows users to create custom case studies based on knowledge bases and specific parameters.

## Features

- Generate case studies on any topic with customizable parameters
- Build and query knowledge bases from PDF, TXT, CSV, and Markdown files
- Export case studies to PDF, HTML, Markdown, or JSON formats
- Customize case study difficulty, format, length, and more
- Use LLaMA 3 for high-quality case study generation
- Integrate with your own custom prompts

## Installation

### Prerequisites

- Python 3.8 or higher
- [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) (for PDF export functionality)

### Setup

1. Clone this repository or download the script:

```bash
git clone https://github.com/yourusername/ai-case-study-generator.git
cd ai-case-study-generator
```

2. Install the required dependencies:

```bash
pip install langchain llamacpp chromadb pdfkit markdown sentence-transformers
```

3. Download a LLaMA 3 model in GGUF format from [Hugging Face](https://huggingface.co/). Place it in your preferred directory.

## Usage

### Command Line Interface

The tool provides several commands through its command-line interface:

#### Create a Knowledge Base

```bash
python case_study_generator.py create-kb --llama-model /path/to/llama3-model.gguf --files document1.pdf document2.txt folder/*.pdf
```

#### Clear an Existing Knowledge Base

```bash
python case_study_generator.py clear-kb --llama-model /path/to/llama3-model.gguf
```

#### Generate a Case Study

```bash
python case_study_generator.py generate --llama-model /path/to/llama3-model.gguf --topic "Machine Learning Ethics" --industry "Healthcare" --difficulty advanced --format problem-solution --length long --output case_study.pdf
```

### Full Command Options

```
usage: case_study_generator.py [-h] --llama-model LLAMA_MODEL [--embeddings-model EMBEDDINGS_MODEL] [--db-directory DB_DIRECTORY]
                             [--n-gpu-layers N_GPU_LAYERS] [--n-batch N_BATCH] [--n-ctx N_CTX] [--temperature TEMPERATURE] [--verbose]
                             {create-kb,clear-kb,generate} ...

AI Case Study Generator

positional arguments:
  {create-kb,clear-kb,generate}
                        Commands
    create-kb           Create a knowledge base from documents
    clear-kb            Clear the existing knowledge base
    generate            Generate a case study

optional arguments:
  -h, --help            show this help message and exit
  --llama-model LLAMA_MODEL
                        Path to LLaMA 3 model
  --embeddings-model EMBEDDINGS_MODEL
                        HuggingFace embeddings model
  --db-directory DB_DIRECTORY
                        Directory for ChromaDB
  --n-gpu-layers N_GPU_LAYERS
                        Number of GPU layers
  --n-batch N_BATCH     Batch size
  --n-ctx N_CTX         Context size
  --temperature TEMPERATURE
                        LLM temperature
  --verbose             Enable verbose output
```

### Create Knowledge Base Options

```
usage: case_study_generator.py create-kb [-h] --files FILES [FILES ...] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]

optional arguments:
  -h, --help            show this help message and exit
  --files FILES [FILES ...]
                        Paths to documents
  --chunk-size CHUNK_SIZE
                        Size of text chunks
  --chunk-overlap CHUNK_OVERLAP
                        Overlap between chunks
```

### Generate Case Study Options

```
usage: case_study_generator.py generate [-h] --topic TOPIC [--industry INDUSTRY] [--difficulty {beginner,intermediate,advanced}]
                                      [--format {standard,scenario,problem-solution}] [--length {short,medium,long}]
                                      [--no-questions] [--custom-prompt CUSTOM_PROMPT] [--output OUTPUT]
                                      [--format-output {pdf,html,markdown,json}] [--k-docs K_DOCS]

optional arguments:
  -h, --help            show this help message and exit
  --topic TOPIC         Main topic for the case study
  --industry INDUSTRY   Specific industry to focus on
  --difficulty {beginner,intermediate,advanced}
                        Difficulty level
  --format {standard,scenario,problem-solution}
                        Format type
  --length {short,medium,long}
                        Length of case study
  --no-questions        Exclude discussion questions
  --custom-prompt CUSTOM_PROMPT
                        Path to custom prompt file
  --output OUTPUT       Output file path
  --format-output {pdf,html,markdown,json}
                        Output format
  --k-docs K_DOCS       Number of documents to retrieve from knowledge base
```

## Using Custom Prompts

You can create custom prompts in a text file and use them with the `--custom-prompt` option:

```bash
python case_study_generator.py generate --llama-model /path/to/llama3-model.gguf --topic "Sustainable Energy" --custom-prompt my_prompt.txt
```

Example custom prompt:

```
You are an expert case study writer specializing in sustainability.

Create a detailed case study on "{topic}" that presents a realistic scenario of a company implementing sustainable practices.

Structure your case study with the following sections:
1. Overview of the company and its industry
2. Initial sustainability challenges
3. Implementation strategy
4. Results and metrics
5. Lessons learned
6. Future directions

Include specific, realistic data points and focus on practical applications rather than theory.
```

## Example Workflow

1. Create a knowledge base with your reference documents:

```bash
python case_study_generator.py create-kb --llama-model models/llama3-8b.gguf --files resources/*.pdf resources/*.txt
```

2. Generate a case study on your desired topic:

```bash
python case_study_generator.py generate --llama-model models/llama3-8b.gguf --topic "Ethical AI Implementation" --industry "Finance" --difficulty advanced --format problem-solution --length medium --output case_studies/ethical_ai_finance.pdf
```

3. View the generated PDF case study in your preferred PDF reader.

## Python API Usage

You can also use the CaseStudyGenerator class directly in your Python code:

```python
from case_study_generator import CaseStudyGenerator

# Initialize the generator
generator = CaseStudyGenerator(
    llama_model_path="models/llama3-8b.gguf",
    n_gpu_layers=1  # Set to higher number if GPU available
)

# Create knowledge base
documents = generator.load_documents(["document1.pdf", "document2.txt"])
generator.create_knowledge_base(documents)

# Generate case study
case_study = generator.generate_case_study(
    topic="Digital Transformation",
    industry="Retail",
    difficulty="intermediate",
    format_type="scenario",
    length="medium"
)

# Export to PDF
generator.export_to_pdf(case_study, "digital_transformation_retail.pdf")
