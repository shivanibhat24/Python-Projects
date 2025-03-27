import os
import sys
import argparse
from typing import List

# Import local modules
from config_management import config
from logging_module import logger_manager
from requirements_extractor import RequirementsManager
from integration_module import integration_manager

class RequirementsAISystem:
    def __init__(self):
        self.logger = logger_manager.get_logger('RequirementsAISystem')
        self.requirements_manager = RequirementsManager()
    
    def process_documents(self, document_paths: List[str]):
        """
        Process multiple requirement documents
        """
        results = []
        for doc_path in document_paths:
            try:
                self.logger.info(f"Processing document: {doc_path}")
                result = self.requirements_manager.process_document(doc_path)
                
                # Optionally sync with external systems
                integration_manager.sync_requirements(result['requirements'])
                
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {doc_path}: {e}")
        
        return results

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='AI-Powered Requirements Gathering System')
    parser.add_argument('documents', nargs='+', help='Path to requirement documents')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set debug mode if specified
    if args.debug:
        config.update('system.debug_mode', True)
        config.update('system.log_level', 'DEBUG')
    
    # Initialize system
    system = RequirementsAISystem()
    
    try:
        # Process documents
        results = system.process_documents(args.documents)
        
        # Output results
        print("Requirements Processing Complete:")
        for result in results:
            print(f"Version ID: {result['version_id']}")
            print(f"Word Document: {result['word_document']}")
            print(f"Excel Backlog: {result['excel_backlog']}")
    
    except Exception as e:
        logger_manager.log_exception("Critical error in requirements processing")
        sys.exit(1)

if __name__ == "__main__":
    main()
