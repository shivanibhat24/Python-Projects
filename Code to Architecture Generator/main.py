"""
Code-to-Architecture Generator
------------------------------
A system that converts source code into architecture diagrams using:
- StarCoder for code understanding
- LangChain for orchestration
- PlantUML for diagram generation
- FastAPI for web interface
"""

import os
import tempfile
from typing import List, Dict, Optional
import subprocess
import base64
from pathlib import Path
import glob

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize FastAPI
app = FastAPI(title="Code-to-Architecture Generator")

# Directory setup
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
STATIC_DIR = Path("static")

for directory in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# StarCoder configuration (assuming a local or remote StarCoder instance)
STARCODER_API_URL = os.getenv("STARCODER_API_URL", "http://localhost:8080/generate")
STARCODER_MAX_TOKENS = 2048
STARCODER_TEMPERATURE = 0.1

# PlantUML setup - using web service for rendering or local installation
PLANTUML_PATH = os.getenv("PLANTUML_PATH", "plantuml")  # Path to PlantUML jar or executable
USE_PLANTUML_WEB = os.getenv("USE_PLANTUML_WEB", "false").lower() == "true"
PLANTUML_WEB_URL = "https://www.plantuml.com/plantuml/png/"

# Initialize StarCoder via LangChain
llm = HuggingFaceTextGenInference(
    inference_server_url=STARCODER_API_URL,
    max_new_tokens=STARCODER_MAX_TOKENS,
    temperature=STARCODER_TEMPERATURE,
)

# Data models
class ArchitectureRequest(BaseModel):
    code_content: str
    diagram_type: str = "class"  # class, sequence, component, etc.
    include_details: bool = True

class ArchitectureResponse(BaseModel):
    plantuml_code: str
    diagram_url: str
    analysis: str

# Prompt templates for code analysis
CLASS_DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
    Analyze the following source code and generate a PlantUML class diagram:
    
    ```
    {code}
    ```
    
    Generate ONLY PlantUML code for a comprehensive class diagram, showing all classes, 
    their attributes, methods, and relationships (inheritance, composition, association).
    Format your response as valid PlantUML that can be directly used for rendering.
    
    @startuml
    
    """
)

COMPONENT_DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
    Analyze the following source code and generate a PlantUML component diagram:
    
    ```
    {code}
    ```
    
    Generate ONLY PlantUML code for a component diagram showing the high-level architecture, 
    modules, services, and their interactions. Focus on system components and their interfaces.
    Format your response as valid PlantUML that can be directly used for rendering.
    
    @startuml
    
    """
)

SEQUENCE_DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
    Analyze the following source code and generate a PlantUML sequence diagram:
    
    ```
    {code}
    ```
    
    Generate ONLY PlantUML code for a sequence diagram showing the key interactions between 
    components. Focus on important function calls, API requests, and data flow.
    Format your response as valid PlantUML that can be directly used for rendering.
    
    @startuml
    
    """
)

CODE_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["code"],
    template="""
    Analyze the following source code and provide a detailed architectural analysis:
    
    ```
    {code}
    ```
    
    Your analysis should include:
    1. Main components and their responsibilities
    2. Key design patterns identified
    3. Dependencies and data flow
    4. Potential architectural improvements
    
    Format your analysis as a clear, structured report.
    """
)

# Chain initialization
diagram_chains = {
    "class": LLMChain(llm=llm, prompt=CLASS_DIAGRAM_PROMPT),
    "component": LLMChain(llm=llm, prompt=COMPONENT_DIAGRAM_PROMPT),
    "sequence": LLMChain(llm=llm, prompt=SEQUENCE_DIAGRAM_PROMPT),
}
analysis_chain = LLMChain(llm=llm, prompt=CODE_ANALYSIS_PROMPT)

def ensure_valid_plantuml(plantuml_code: str) -> str:
    """Ensure PlantUML code has proper start and end tags."""
    code = plantuml_code.strip()
    
    # Add @startuml if missing
    if not code.startswith("@startuml"):
        code = "@startuml\n" + code
    
    # Add @enduml if missing
    if not code.endswith("@enduml"):
        code = code + "\n@enduml"
    
    return code

def generate_diagram_image(plantuml_code: str) -> str:
    """Generate diagram image from PlantUML code and return the file path or URL."""
    plantuml_code = ensure_valid_plantuml(plantuml_code)
    
    if USE_PLANTUML_WEB:
        # Use web service
        encoded = base64.b64encode(plantuml_code.encode("utf-8"))
        url = f"{PLANTUML_WEB_URL}{encoded.decode('utf-8')}"
        return url
    else:
        # Use local PlantUML
        with tempfile.NamedTemporaryFile(suffix=".puml", delete=False) as tmp:
            tmp.write(plantuml_code.encode())
            tmp_path = tmp.name
        
        output_path = str(OUTPUT_DIR / f"{Path(tmp_path).stem}.png")
        subprocess.run([PLANTUML_PATH, "-tpng", tmp_path, "-o", str(OUTPUT_DIR)])
        
        os.unlink(tmp_path)  # Remove temporary file
        return output_path

def process_code(code_content: str, diagram_type: str) -> Dict:
    """Process code content and generate architecture diagram and analysis."""
    # Generate PlantUML code
    if diagram_type not in diagram_chains:
        diagram_type = "class"  # Default to class diagram
    
    chain = diagram_chains[diagram_type]
    plantuml_result = chain.run(code=code_content)
    
    # Ensure it's complete plantuml code
    if "@enduml" not in plantuml_result:
        plantuml_result += "\n@enduml"
    
    # Generate analysis
    analysis_result = analysis_chain.run(code=code_content)
    
    # Generate diagram image
    diagram_url = generate_diagram_image(plantuml_result)
    
    return {
        "plantuml_code": plantuml_result,
        "diagram_url": diagram_url,
        "analysis": analysis_result
    }

def process_directory(directory_path: str) -> Dict:
    """Process all code files in a directory."""
    all_code = ""
    for extension in ['*.py', '*.js', '*.java', '*.cpp', '*.ts', '*.go']:
        files = glob.glob(os.path.join(directory_path, '**', extension), recursive=True)
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_code += f"\n# File: {file_path}\n{f.read()}\n\n"
    
    return process_code(all_code, "component")  # Default to component diagram for directories

# API Endpoints
@app.post("/api/generate-diagram", response_model=ArchitectureResponse)
async def generate_diagram(request: ArchitectureRequest):
    """Generate architecture diagram from provided code."""
    try:
        result = process_code(request.code_content, request.diagram_type)
        return ArchitectureResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing code: {str(e)}")

@app.post("/api/upload-code", response_model=ArchitectureResponse)
async def upload_code(
    file: UploadFile = File(...),
    diagram_type: str = Form("class")
):
    """Upload code file and generate architecture diagram."""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Read code from file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code_content = f.read()
        
        # Process code
        result = process_code(code_content, diagram_type)
        return ArchitectureResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/api/upload-directory")
async def upload_directory(
    files: List[UploadFile] = File(...),
    base_dir: str = Form("project")
):
    """Upload multiple files representing a directory structure."""
    try:
        # Create base directory
        base_path = UPLOAD_DIR / base_dir
        base_path.mkdir(exist_ok=True)
        
        # Save all files
        for file in files:
            # Extract relative path
            rel_path = file.filename
            full_path = base_path / rel_path
            
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(full_path, "wb") as f:
                f.write(await file.read())
        
        # Process directory
        result = process_directory(str(base_path))
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the web UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code-to-Architecture Generator</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            .container { max-width: 1200px; }
            #code-editor { height: 400px; width: 100%; border: 1px solid #ddd; }
            #diagram-container img { max-width: 100%; }
            .nav-tabs { margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Code-to-Architecture Generator</h1>
            
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="code-tab" data-bs-toggle="tab" data-bs-target="#code-panel" type="button" role="tab">Paste Code</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload-panel" type="button" role="tab">Upload File</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="directory-tab" data-bs-toggle="tab" data-bs-target="#directory-panel" type="button" role="tab">Upload Directory</button>
                </li>
            </ul>
            
            <div class="tab-content" id="myTabContent">
                <!-- Code Input Tab -->
                <div class="tab-pane fade show active" id="code-panel" role="tabpanel">
                    <form id="code-form" class="mb-4">
                        <div class="mb-3">
                            <label for="code-editor" class="form-label">Source Code</label>
                            <textarea id="code-editor" class="form-control" rows="10"></textarea>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <label for="diagram-type" class="form-label">Diagram Type</label>
                                <select id="diagram-type" class="form-select">
                                    <option value="class">Class Diagram</option>
                                    <option value="component">Component Diagram</option>
                                    <option value="sequence">Sequence Diagram</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary">Generate Diagram</button>
                    </form>
                </div>
                
                <!-- File Upload Tab -->
                <div class="tab-pane fade" id="upload-panel" role="tabpanel">
                    <form id="file-upload-form" class="mb-4">
                        <div class="mb-3">
                            <label for="file-upload" class="form-label">Upload Code File</label>
                            <input type="file" id="file-upload" class="form-control" accept=".py,.js,.java,.cpp,.ts,.go">
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <label for="file-diagram-type" class="form-label">Diagram Type</label>
                                <select id="file-diagram-type" class="form-select">
                                    <option value="class">Class Diagram</option>
                                    <option value="component">Component Diagram</option>
                                    <option value="sequence">Sequence Diagram</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary">Generate Diagram</button>
                    </form>
                </div>
                
                <!-- Directory Upload Tab -->
                <div class="tab-pane fade" id="directory-panel" role="tabpanel">
                    <form id="directory-upload-form" class="mb-4">
                        <div class="mb-3">
                            <label for="directory-upload" class="form-label">Upload Multiple Files</label>
                            <input type="file" id="directory-upload" class="form-control" multiple accept=".py,.js,.java,.cpp,.ts,.go">
                        </div>
                        <div class="mb-3">
                            <label for="base-dir" class="form-label">Project Name</label>
                            <input type="text" id="base-dir" class="form-control" value="project">
                        </div>
                        <button type="submit" class="btn btn-primary">Generate Diagram</button>
                    </form>
                </div>
            </div>
            
            <hr>
            
            <!-- Results Section -->
            <div id="results-section" class="mt-4" style="display: none;">
                <h2>Generated Architecture</h2>
                
                <ul class="nav nav-tabs" id="resultTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="diagram-tab" data-bs-toggle="tab" data-bs-target="#diagram-tab-pane" type="button" role="tab">Diagram</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="plantuml-tab" data-bs-toggle="tab" data-bs-target="#plantuml-tab-pane" type="button" role="tab">PlantUML Code</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="analysis-tab" data-bs-toggle="tab" data-bs-target="#analysis-tab-pane" type="button" role="tab">Analysis</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="resultTabContent">
                    <div class="tab-pane fade show active" id="diagram-tab-pane" role="tabpanel" tabindex="0">
                        <div id="diagram-container" class="mt-3 p-3 border rounded">
                            <img id="diagram-image" src="" alt="Architecture Diagram">
                        </div>
                    </div>
                    <div class="tab-pane fade" id="plantuml-tab-pane" role="tabpanel" tabindex="0">
                        <pre id="plantuml-code" class="mt-3 p-3 border rounded bg-light"></pre>
                    </div>
                    <div class="tab-pane fade" id="analysis-tab-pane" role="tabpanel" tabindex="0">
                        <div id="analysis-content" class="mt-3 p-3 border rounded"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            document.getElementById('code-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                const codeContent = document.getElementById('code-editor').value;
                const diagramType = document.getElementById('diagram-type').value;
                
                await generateDiagram({
                    code_content: codeContent,
                    diagram_type: diagramType
                });
            });
            
            document.getElementById('file-upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('file-upload');
                const diagramType = document.getElementById('file-diagram-type').value;
                
                if (fileInput.files.length === 0) {
                    alert('Please select a file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('diagram_type', diagramType);
                
                try {
                    const response = await fetch('/api/upload-code', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Upload failed');
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
            
            document.getElementById('directory-upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('directory-upload');
                const baseDir = document.getElementById('base-dir').value;
                
                if (fileInput.files.length === 0) {
                    alert('Please select files');
                    return;
                }
                
                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append('files', fileInput.files[i]);
                }
                formData.append('base_dir', baseDir);
                
                try {
                    const response = await fetch('/api/upload-directory', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Upload failed');
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
            
            async function generateDiagram(data) {
                try {
                    const response = await fetch('/api/generate-diagram', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) throw new Error('Generation failed');
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function displayResults(result) {
                document.getElementById('results-section').style.display = 'block';
                document.getElementById('diagram-image').src = result.diagram_url;
                document.getElementById('plantuml-code').textContent = result.plantuml_code;
                document.getElementById('analysis-content').innerHTML = result.analysis.replace(/\n/g, '<br>');
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
