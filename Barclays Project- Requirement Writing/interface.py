import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional

# Import local modules
from requirements_extractor import RequirementsManager
from advanced_nlp_classifier import RequirementClassifier
from config_management import config
from logging_module import logger_manager

class RequirementInput(BaseModel):
    text: str

class WebInterface:
    def __init__(self):
        # FastAPI app
        self.app = FastAPI(
            title="Requirements AI System",
            description="AI-powered requirements gathering and analysis",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.requirements_manager = RequirementsManager()
        self.requirement_classifier = RequirementClassifier()
        self.logger = logger_manager.get_logger('WebInterface')
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.post("/classify-requirement")
        async def classify_requirement(req: RequirementInput):
            """
            Classify a single requirement
            """
            try:
                result = self.requirement_classifier.classify_requirement(req.text)
                return result
            except Exception as e:
                self.logger.error(f"Classification error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/upload-requirements")
        async def upload_requirements(files: List[UploadFile] = File(...)):
            """
            Process multiple requirement documents
            """
            results = []
            for file in files:
                # Save uploaded file temporarily
                temp_path = os.path.join(
                    config.get('paths.temp_dir', './temp'), 
                    file.filename
                )
                
                try:
                    # Save file
                    with open(temp_path, 'wb') as buffer:
                        buffer.write(await file.read())
                    
                    # Process document
                    result = self.requirements_manager.process_document(temp_path)
                    results.append(result)
                
                except Exception as e:
                    self.logger.error(f"Error processing {file.filename}: {e}")
                    results.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            return results
        
        @self.app.get("/download/{document_type}/{filename}")
        async def download_document(document_type: str, filename: str):
            """
            Download generated documents
            """
            output_dir = config.get('paths.output_dir', './outputs')
            file_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")
            
            return FileResponse(
                path=file_path, 
                media_type='application/octet-stream',
                filename=filename
            )
    
    def run(self, host='0.0.0.0', port=8000):
        """
        Run the web interface
        """
        uvicorn.run(
            self.app, 
            host=host, 
            port=port
        )

def main():
    web_interface = WebInterface()
    web_interface.run()

if __name__ == "__main__":
    main()
