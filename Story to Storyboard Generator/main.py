"""
Story-to-Storyboard Generator
----------------------------
A Python application that converts narrative text into visual storyboards using:
- LLaMA 3 for text understanding and scene extraction
- Stable Diffusion for AI image generation
- LangChain for processing and orchestration
- ChromaDB for vectorizing and storing scene information
"""

import os
import sys
import json
import argparse
import textwrap
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import re

# Third-party imports
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

class StoryboardGenerator:
    """Main class for the Story-to-Storyboard Generator application"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the StoryboardGenerator with configuration settings
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get("output_directory", "storyboards"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self._init_llm()
        self._init_image_generator()
        self._init_vector_db()
        self._init_chains()
        
        print("✅ Story-to-Storyboard Generator initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Dictionary containing configuration settings
        """
        try:
            if not os.path.exists(config_path):
                # Create default config if not exists
                default_config = {
                    "llm": {
                        "model_path": "llama-3-8b-instruct.gguf",  # Path to your LLaMA 3 model
                        "n_ctx": 4096,
                        "temperature": 0.75
                    },
                    "image_generator": {
                        "model_id": "stabilityai/stable-diffusion-2-1",
                        "image_size": [512, 512],
                        "num_inference_steps": 30
                    },
                    "embeddings": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                    },
                    "output_directory": "storyboards",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
                
                os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                print(f"Created default configuration at {config_path}")
                return default_config
            
            with open(config_path, "r") as f:
                config = json.load(f)
                return config
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _init_llm(self):
        """Initialize the LLaMA 3 model"""
        llm_config = self.config.get("llm", {})
        model_path = llm_config.get("model_path", "llama-3-8b-instruct.gguf")
        
        if not os.path.exists(model_path):
            print(f"⚠️ LLaMA 3 model not found at {model_path}")
            print("Please download the model or update the configuration")
            print("Continuing in demonstration mode...")
        
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=llm_config.get("n_ctx", 4096),
                temperature=llm_config.get("temperature", 0.75),
                verbose=True
            )
            print("🧠 LLaMA 3 model loaded successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("Continuing with reduced functionality...")
            self.llm = None
    
    def _init_image_generator(self):
        """Initialize the Stable Diffusion image generator"""
        img_config = self.config.get("image_generator", {})
        model_id = img_config.get("model_id", "stabilityai/stable-diffusion-2-1")
        
        try:
            self.image_generator = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.image_generator = self.image_generator.to("cuda")
                print("🖼️ Stable Diffusion loaded on GPU")
            else:
                print("🖼️ Stable Diffusion loaded on CPU (slower performance)")
                
        except Exception as e:
            print(f"Error initializing image generator: {e}")
            print("Continuing with reduced functionality...")
            self.image_generator = None
    
    def _init_vector_db(self):
        """Initialize the vector database for storing and retrieving scene embeddings"""
        try:
            embeddings_config = self.config.get("embeddings", {})
            model_name = embeddings_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.persist_directory = "chroma_db"
            
            # Will create if it doesn't exist
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("🔍 Vector database initialized")
            
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            print("Continuing with reduced functionality...")
            self.embeddings = None
            self.vectordb = None
    
    def _init_chains(self):
        """Initialize LangChain processing chains"""
        # Scene extraction prompt
        scene_extraction_template = """
        You are a professional story analyst and screenwriter.
        Extract distinct scenes from the following story text. For each scene:
        1. Identify the setting, characters, and main action
        2. Create a brief summary (2-3 sentences)
        3. Craft a detailed visual description for an AI art generator (focus on visuals, style, mood)
        4. Identify the key emotion or theme of the scene
        
        STORY TEXT:
        {text}
        
        FORMAT YOUR RESPONSE AS JSON:
        {{
            "scenes": [
                {{
                    "scene_number": 1,
                    "summary": "Brief scene summary",
                    "visual_description": "Detailed description for image generation",
                    "emotion": "Primary emotion/theme of scene"
                }},
                ...
            ]
        }}
        """
        
        self.scene_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template=scene_extraction_template
        )
        
        if self.llm:
            self.scene_extraction_chain = LLMChain(
                llm=self.llm,
                prompt=self.scene_extraction_prompt
            )
        
    def process_story(self, story_text: str, title: str = "My Story") -> str:
        """Process a story text into a storyboard
        
        Args:
            story_text: The full story text to process
            title: Title of the story
            
        Returns:
            Path to the generated storyboard
        """
        print(f"📝 Processing story: {title}")
        
        # Create a unique ID for this storyboard
        storyboard_id = str(uuid.uuid4())[:8]
        output_folder = self.output_dir / f"{title.replace(' ', '_')}_{storyboard_id}"
        output_folder.mkdir(exist_ok=True)
        
        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(story_text)
        
        # Process chunks to extract scenes
        all_scenes = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            scenes = self._extract_scenes(chunk)
            if scenes:
                # Add scene numbering across chunks
                for j, scene in enumerate(scenes):
                    scene["scene_number"] = len(all_scenes) + j + 1
                all_scenes.extend(scenes)
        
        # Generate images for each scene
        print(f"Generating images for {len(all_scenes)} scenes...")
        for scene in all_scenes:
            image_path = self._generate_scene_image(
                scene["visual_description"],
                output_folder / f"scene_{scene['scene_number']}.png"
            )
            scene["image_path"] = str(image_path)
        
        # Save scene data
        scenes_json_path = output_folder / "scenes.json"
        with open(scenes_json_path, "w") as f:
            json.dump({"title": title, "scenes": all_scenes}, f, indent=2)
        
        # Generate final storyboard
        storyboard_path = self._create_storyboard_html(title, all_scenes, output_folder)
        
        print(f"✨ Storyboard created successfully at {storyboard_path}")
        return str(storyboard_path)
    
    def _extract_scenes(self, text_chunk: str) -> List[Dict[str, Any]]:
        """Extract scenes from a text chunk using LLaMA 3
        
        Args:
            text_chunk: A chunk of the story text
            
        Returns:
            List of scene dictionaries
        """
        if not self.llm:
            # Return placeholder data in demo mode
            return [
                {
                    "scene_number": 1,
                    "summary": "This is a placeholder scene summary since LLaMA 3 is not available.",
                    "visual_description": "A placeholder visual description for image generation.",
                    "emotion": "Demonstration"
                }
            ]
        
        try:
            response = self.scene_extraction_chain.run(text_chunk)
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                response_json = json.loads(json_match.group(1))
                return response_json.get("scenes", [])
            else:
                print("⚠️ Failed to parse JSON from LLM response")
                return []
                
        except Exception as e:
            print(f"Error extracting scenes: {e}")
            return []
    
    def _generate_scene_image(self, description: str, output_path: Path) -> Path:
        """Generate an image for a scene using Stable Diffusion
        
        Args:
            description: Visual description to generate an image from
            output_path: Where to save the generated image
            
        Returns:
            Path to the generated image
        """
        if not self.image_generator:
            # Create a placeholder image in demo mode
            img = Image.new('RGB', (512, 512), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            
            # Try to load a font, or use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
                
            wrapped_text = textwrap.fill(description, width=40)
            d.text((10, 10), "PLACEHOLDER IMAGE\n" + wrapped_text, fill=(255, 255, 255), font=font)
            
            img.save(output_path)
            return output_path
        
        try:
            # Enhanced prompt for better results
            enhanced_prompt = f"High quality digital art, detailed, professional illustration: {description}"
            
            # Generate the image
            image = self.image_generator(
                prompt=enhanced_prompt,
                num_inference_steps=self.config.get("image_generator", {}).get("num_inference_steps", 30)
            ).images[0]
            
            # Save the image
            image.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error generating image: {e}")
            # Create a fallback image
            img = Image.new('RGB', (512, 512), color=(200, 0, 0))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"Error generating image: {str(e)}", fill=(255, 255, 255))
            img.save(output_path)
            return output_path
    
    def _create_storyboard_html(self, title: str, scenes: List[Dict[str, Any]], output_folder: Path) -> Path:
        """Create an HTML storyboard from the processed scenes
        
        Args:
            title: Title of the story
            scenes: List of scene dictionaries with images
            output_folder: Directory to save the HTML file
            
        Returns:
            Path to the HTML storyboard file
        """
        html_path = output_folder / "storyboard.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title} - Storyboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                h1 {{
                    color: #2c3e50;
                    font-size: 36px;
                }}
                .storyboard {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                    gap: 30px;
                }}
                .scene {{
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }}
                .scene:hover {{
                    transform: translateY(-5px);
                }}
                .scene img {{
                    width: 100%;
                    height: 300px;
                    object-fit: cover;
                }}
                .scene-content {{
                    padding: 20px;
                }}
                .scene-number {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 20px;
                    display: inline-block;
                    margin-bottom: 10px;
                }}
                .scene-summary {{
                    color: #2c3e50;
                    font-size: 18px;
                    margin-bottom: 15px;
                    line-height: 1.4;
                }}
                .scene-emotion {{
                    color: #7f8c8d;
                    font-style: italic;
                }}
                footer {{
                    text-align: center;
                    margin-top: 40px;
                    color: #7f8c8d;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>{title}</h1>
                <p>Generated Storyboard with AI</p>
            </header>
            
            <div class="storyboard">
        """
        
        # Add each scene
        for scene in scenes:
            relative_img_path = os.path.basename(scene["image_path"])
            
            html_content += f"""
                <div class="scene">
                    <img src="{relative_img_path}" alt="Scene {scene['scene_number']}">
                    <div class="scene-content">
                        <div class="scene-number">Scene {scene['scene_number']}</div>
                        <p class="scene-summary">{scene['summary']}</p>
                        <p class="scene-emotion">Mood: {scene['emotion']}</p>
                    </div>
                </div>
            """
        
        # Close HTML document
        html_content += """
            </div>
            
            <footer>
                <p>Created with Story-to-Storyboard Generator</p>
            </footer>
        </body>
        </html>
        """
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path

    def store_story_in_vectordb(self, story_text: str, metadata: Dict[str, Any]) -> str:
        """Store a story in the vector database for later retrieval and similarity search
        
        Args:
            story_text: The story text to store
            metadata: Additional information about the story
            
        Returns:
            ID of the stored story
        """
        if not self.vectordb:
            print("⚠️ Vector database not available")
            return ""
            
        try:
            # Split text into chunks for storage
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("chunk_size", 1000),
                chunk_overlap=self.config.get("chunk_overlap", 200)
            )
            
            chunks = text_splitter.split_text(story_text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = i
                doc_metadata["chunk_total"] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # Store in vector database
            ids = self.vectordb.add_documents(documents)
            self.vectordb.persist()
            
            print(f"✅ Story stored in vector database with {len(ids)} chunks")
            return metadata.get("id", "")
            
        except Exception as e:
            print(f"Error storing story in vector database: {e}")
            return ""
    
    def find_similar_scenes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find scenes similar to a query using vector similarity search
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of similar scenes with metadata
        """
        if not self.vectordb:
            print("⚠️ Vector database not available")
            return []
            
        try:
            results = self.vectordb.similarity_search(query, k=limit)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Error searching for similar scenes: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Story-to-Storyboard Generator")
    parser.add_argument("--input", "-i", type=str, help="Path to input story text file")
    parser.add_argument("--title", "-t", type=str, default="My Story", help="Title of the story")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = StoryboardGenerator(config_path=args.config)
    
    # Process the story
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                story_text = f.read()
            
            storyboard_path = generator.process_story(story_text, args.title)
            print(f"Storyboard generated at: {storyboard_path}")
            
            # Store in vector database for later retrieval
            generator.store_story_in_vectordb(story_text, {
                "id": f"story_{str(uuid.uuid4())[:8]}",
                "title": args.title,
                "source": args.input,
                "timestamp": time.time()
            })
            
        except Exception as e:
            print(f"Error processing story: {e}")
    else:
        print("No input file specified. Use --input to provide a story text file.")
        print("Example: python storyboard_generator.py --input my_story.txt --title 'Adventures of AI'")


if __name__ == "__main__":
    main()
