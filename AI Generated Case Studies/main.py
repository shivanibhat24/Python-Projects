"""
AI Case Study Generator

A user-friendly tool for creating custom case studies using LLaMA 3, LangChain, ChromaDB, and pdfkit.
This tool allows users to generate case studies based on knowledge bases and custom prompts.
"""

import os
import argparse
import time
import json
import tempfile
import pdfkit
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

# LangChain imports
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredMarkdownLoader
)
from langchain.vectorstores import Chroma

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CaseStudyGenerator:
    """Main class for generating AI case studies."""
    
    def __init__(
        self, 
        llama_model_path: str,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_directory: str = "./chroma_db",
        n_gpu_layers: int = 1,
        n_batch: int = 512,
        n_ctx: int = 4096,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the Case Study Generator.
        
        Args:
            llama_model_path: Path to the LLaMA 3 model
            embeddings_model: HuggingFace embeddings model to use
            db_directory: Directory to store ChromaDB
            n_gpu_layers: Number of GPU layers to use
            n_batch: Batch size for LLaMA model
            n_ctx: Context window size
            temperature: LLM temperature parameter
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.db_directory = db_directory
        
        # Initialize embeddings model
        logger.info(f"Initializing embeddings model: {embeddings_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        
        # Initialize LLaMA model
        logger.info(f"Loading LLaMA 3 model from: {llama_model_path}")
        self.llm = LlamaCpp(
            model_path=llama_model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            verbose=verbose
        )
        
        # Initialize vector store
        self.vector_store = None
        
        logger.info("CaseStudyGenerator initialized successfully")
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """
        Load documents from various file formats.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading {len(file_paths)} document(s)")
        documents = []
        
        for file_path in file_paths:
            file_ext = os.path.splitext(file_path)[1].lower()
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_ext == '.txt':
                    loader = TextLoader(file_path)
                elif file_ext == '.csv':
                    loader = CSVLoader(file_path)
                elif file_ext in ['.md', '.markdown']:
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_ext}. Skipping {file_path}")
                    continue
                
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Loaded document: {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} document(s) in total")
        return documents
    
    def create_knowledge_base(self, documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Create a knowledge base from documents.
        
        Args:
            documents: List of documents to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not documents:
            logger.warning("No documents provided to create knowledge base")
            return
        
        logger.info(f"Creating knowledge base with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        
        # Create or update vector store
        if os.path.exists(self.db_directory):
            logger.info(f"Loading existing vector store from {self.db_directory}")
            self.vector_store = Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"Adding {len(splits)} new document chunks to existing vector store")
            self.vector_store.add_documents(splits)
        else:
            logger.info(f"Creating new vector store in {self.db_directory}")
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.db_directory
            )
        
        # Persist vector store
        self.vector_store.persist()
        logger.info("Knowledge base created and persisted successfully")
    
    def clear_knowledge_base(self) -> None:
        """Clear the existing knowledge base."""
        if os.path.exists(self.db_directory):
            import shutil
            logger.info(f"Clearing knowledge base at {self.db_directory}")
            shutil.rmtree(self.db_directory)
            logger.info("Knowledge base cleared successfully")
        else:
            logger.info("No knowledge base found to clear")
        
        self.vector_store = None
    
    def retrieve_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The query to search for
            k: Number of documents to retrieve
            
        Returns:
            String containing relevant context
        """
        if not self.vector_store:
            logger.warning("No knowledge base available for retrieval")
            return ""
        
        logger.info(f"Retrieving {k} relevant documents for query: '{query}'")
        docs = self.vector_store.similarity_search(query, k=k)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Retrieved {len(docs)} documents totaling {len(context)} characters")
        
        return context
    
    def generate_case_study(
        self,
        topic: str,
        industry: Optional[str] = None,
        difficulty: str = "intermediate",
        format_type: str = "standard",
        length: str = "medium",
        include_questions: bool = True,
        custom_prompt: Optional[str] = None,
        k_docs: int = 5
    ) -> Dict[str, str]:
        """
        Generate a case study based on the given parameters.
        
        Args:
            topic: The main topic for the case study
            industry: Specific industry to focus on (optional)
            difficulty: Difficulty level (beginner, intermediate, advanced)
            format_type: Format type (standard, scenario, problem-solution)
            length: Length of case study (short, medium, long)
            include_questions: Whether to include discussion questions
            custom_prompt: Custom prompt override (optional)
            k_docs: Number of documents to retrieve from knowledge base
            
        Returns:
            Dictionary containing case study title and content
        """
        # Get context from knowledge base if available
        relevant_context = ""
        if self.vector_store:
            relevant_context = self.retrieve_relevant_context(topic, k=k_docs)
        
        # Define length parameters
        length_params = {
            "short": "approximately 500 words",
            "medium": "approximately 1000-1500 words",
            "long": "approximately 2500-3000 words"
        }
        
        length_instruction = length_params.get(length.lower(), length_params["medium"])
        
        # Create the system prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            industry_text = f"in the {industry} industry" if industry else ""
            
            system_prompt = f"""
            You are an expert case study writer specialized in creating educational content {industry_text}.
            
            Create a comprehensive case study on "{topic}" that is {length_instruction}.
            
            The case study should be at a {difficulty} level and follow a {format_type} format.
            
            {f'Include 3-5 discussion questions at the end.' if include_questions else ''}
            
            The case study should be well-structured with clear sections including:
            1. An engaging title
            2. Executive summary or introduction
            3. Background information
            4. Key challenges or problems
            5. Analysis and insights
            6. Solutions or recommendations
            7. Conclusion with key takeaways
            {f'8. Discussion questions' if include_questions else ''}
            
            Use concrete examples, data points, and realistic scenarios to make the case study engaging and practical.
            
            Use the following relevant information as reference (but don't limit yourself to just this information):
            {relevant_context}
            """
        
        logger.info(f"Generating case study on topic: '{topic}'")
        logger.info(f"Parameters: difficulty={difficulty}, format={format_type}, length={length}")
        
        # Create prompt template and chain
        prompt = PromptTemplate(
            input_variables=["system_prompt"],
            template="{system_prompt}"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Generate the case study
        start_time = time.time()
        result = chain.run(system_prompt=system_prompt)
        end_time = time.time()
        
        logger.info(f"Case study generated in {end_time - start_time:.2f} seconds")
        
        # Extract title from the case study
        lines = result.strip().split('\n')
        title = lines[0].replace('#', '').strip()
        if not title:
            title = f"Case Study on {topic}"
        
        return {
            "title": title,
            "content": result
        }
    
    def export_to_html(self, case_study: Dict[str, str], output_path: Optional[str] = None) -> str:
        """
        Export case study to HTML format.
        
        Args:
            case_study: Case study dictionary with title and content
            output_path: Path to save the HTML file (optional)
            
        Returns:
            Path to the saved HTML file
        """
        import markdown
        
        title = case_study["title"]
        content = case_study["content"]
        
        # Convert markdown to HTML
        html_content = markdown.markdown(content)
        
        # Create full HTML document
        html = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                blockquote {{
                    border-left: 4px solid #ccc;
                    margin-left: 0;
                    padding-left: 15px;
                    color: #555;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                .footer {{
                    margin-top: 50px;
                    border-top: 1px solid #eee;
                    padding-top: 20px;
                    font-size: 0.9em;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {html_content}
                <div class="footer">
                    <p>Generated with AI Case Study Generator</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Determine output path
        if not output_path:
            safe_title = "".join([c if c.isalnum() else "_" for c in title])
            output_path = f"{safe_title}.html"
        
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"Case study exported to HTML: {output_path}")
        return output_path
    
    def export_to_pdf(self, case_study: Dict[str, str], output_path: Optional[str] = None) -> str:
        """
        Export case study to PDF format using pdfkit.
        
        Args:
            case_study: Case study dictionary with title and content
            output_path: Path to save the PDF file (optional)
            
        Returns:
            Path to the saved PDF file
        """
        title = case_study["title"]
        
        # First export to HTML
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            html_path = tmp.name
        
        self.export_to_html(case_study, html_path)
        
        # Determine output path
        if not output_path:
            safe_title = "".join([c if c.isalnum() else "_" for c in title])
            output_path = f"{safe_title}.pdf"
        
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert HTML to PDF
        try:
            logger.info(f"Converting HTML to PDF: {output_path}")
            pdfkit.from_file(html_path, output_path)
            logger.info(f"Case study exported to PDF: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            logger.info("Make sure wkhtmltopdf is installed. On Ubuntu: sudo apt-get install wkhtmltopdf")
            logger.info("On MacOS: brew install wkhtmltopdf")
            logger.info("On Windows: Download and install from https://wkhtmltopdf.org/downloads.html")
        
        # Clean up temporary HTML file
        try:
            os.unlink(html_path)
        except:
            pass
        
        return output_path


def main():
    """Main function to run the case study generator from command line."""
    parser = argparse.ArgumentParser(description="AI Case Study Generator")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create knowledge base parser
    kb_parser = subparsers.add_parser("create-kb", help="Create a knowledge base from documents")
    kb_parser.add_argument("--files", nargs="+", required=True, help="Paths to documents")
    kb_parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    kb_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    # Clear knowledge base parser
    clear_kb_parser = subparsers.add_parser("clear-kb", help="Clear the existing knowledge base")
    
    # Generate case study parser
    gen_parser = subparsers.add_parser("generate", help="Generate a case study")
    gen_parser.add_argument("--topic", required=True, help="Main topic for the case study")
    gen_parser.add_argument("--industry", help="Specific industry to focus on")
    gen_parser.add_argument("--difficulty", default="intermediate", 
                           choices=["beginner", "intermediate", "advanced"],
                           help="Difficulty level")
    gen_parser.add_argument("--format", default="standard", 
                           choices=["standard", "scenario", "problem-solution"],
                           help="Format type")
    gen_parser.add_argument("--length", default="medium", 
                           choices=["short", "medium", "long"],
                           help="Length of case study")
    gen_parser.add_argument("--no-questions", action="store_true", 
                           help="Exclude discussion questions")
    gen_parser.add_argument("--custom-prompt", help="Path to custom prompt file")
    gen_parser.add_argument("--output", help="Output file path")
    gen_parser.add_argument("--format-output", default="pdf", 
                           choices=["pdf", "html", "markdown", "json"],
                           help="Output format")
    gen_parser.add_argument("--k-docs", type=int, default=5,
                           help="Number of documents to retrieve from knowledge base")
    
    # Global arguments
    parser.add_argument("--llama-model", required=True, help="Path to LLaMA 3 model")
    parser.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="HuggingFace embeddings model")
    parser.add_argument("--db-directory", default="./chroma_db", help="Directory for ChromaDB")
    parser.add_argument("--n-gpu-layers", type=int, default=1, help="Number of GPU layers")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize the case study generator
    generator = CaseStudyGenerator(
        llama_model_path=args.llama_model,
        embeddings_model=args.embeddings_model,
        db_directory=args.db_directory,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Execute the requested command
    if args.command == "create-kb":
        documents = generator.load_documents(args.files)
        generator.create_knowledge_base(documents, args.chunk_size, args.chunk_overlap)
    
    elif args.command == "clear-kb":
        generator.clear_knowledge_base()
    
    elif args.command == "generate":
        # Load custom prompt if provided
        custom_prompt = None
        if args.custom_prompt:
            try:
                with open(args.custom_prompt, "r", encoding="utf-8") as f:
                    custom_prompt = f.read()
            except Exception as e:
                logger.error(f"Error loading custom prompt: {str(e)}")
        
        # Generate case study
        case_study = generator.generate_case_study(
            topic=args.topic,
            industry=args.industry,
            difficulty=args.difficulty,
            format_type=args.format,
            length=args.length,
            include_questions=not args.no_questions,
            custom_prompt=custom_prompt,
            k_docs=args.k_docs
        )
        
        # Export case study
        output_path = args.output
        if args.format_output == "pdf":
            output_path = generator.export_to_pdf(case_study, output_path)
        elif args.format_output == "html":
            output_path = generator.export_to_html(case_study, output_path)
        elif args.format_output == "markdown":
            if not output_path:
                safe_title = "".join([c if c.isalnum() else "_" for c in case_study["title"]])
                output_path = f"{safe_title}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(case_study["content"])
            logger.info(f"Case study exported to Markdown: {output_path}")
        elif args.format_output == "json":
            if not output_path:
                safe_title = "".join([c if c.isalnum() else "_" for c in case_study["title"]])
                output_path = f"{safe_title}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(case_study, f, indent=2)
            logger.info(f"Case study exported to JSON: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
