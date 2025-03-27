import requests
from typing import Dict, Any, Optional
from config_management import config
from logging_module import logger_manager

class JiraIntegration:
    """
    Integration with Jira for backlog management
    """
    def __init__(self):
        self.logger = logger_manager.get_logger('JiraIntegration')
        self.base_url = config.get('integrations.jira.url')
        self.token = config.get('integrations.jira.token')
        
        if not self.base_url or not self.token:
            self.logger.warning("Jira integration not fully configured")
    
    def is_enabled(self) -> bool:
        """
        Check if Jira integration is enabled and configured
        """
        return config.get('integrations.jira.enabled', False) and \
               self.base_url and self.token
    
    def create_issue(self, 
                     project_key: str, 
                     summary: str, 
                     description: str, 
                     issue_type: str = 'Story') -> Optional[Dict[str, Any]]:
        """
        Create a new issue in Jira
        """
        if not self.is_enabled():
            self.logger.warning("Jira integration is not enabled")
            return None
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.token}'
            }
            
            payload = {
                'fields': {
                    'project': {'key': project_key},
                    'summary': summary,
                    'description': description,
                    'issuetype': {'name': issue_type}
                }
            }
            
            response = requests.post(
                f'{self.base_url}/rest/api/3/issue', 
                json=payload, 
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Jira API Error: {e}")
            return None

class ConfluenceIntegration:
    """
    Integration with Confluence for documentation
    """
    def __init__(self):
        self.logger = logger_manager.get_logger('ConfluenceIntegration')
        self.base_url = config.get('integrations.confluence.url')
        self.token = config.get('integrations.confluence.token')
        
        if not self.base_url or not self.token:
            self.logger.warning("Confluence integration not fully configured")
    
    def is_enabled(self) -> bool:
        """
        Check if Confluence integration is enabled and configured
        """
        return config.get('integrations.confluence.enabled', False) and \
               self.base_url and self.token
    
    def create_page(self, 
                    space_key: str, 
                    title: str, 
                    content: str) -> Optional[Dict[str, Any]]:
        """
        Create a new page in Confluence
        """
        if not self.is_enabled():
            self.logger.warning("Confluence integration is not enabled")
            return None
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.token}'
            }
            
            payload = {
                'type': 'page',
                'title': title,
                'space': {'key': space_key},
                'body': {
                    'storage': {
                        'value': content,
                        'representation': 'storage'
                    }
                }
            }
            
            response = requests.post(
                f'{self.base_url}/wiki/rest/api/content', 
                json=payload, 
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Confluence API Error: {e}")
            return None

class IntegrationManager:
    """
    Centralized manager for external system integrations
    """
    def __init__(self):
        self.jira = JiraIntegration()
        self.confluence = ConfluenceIntegration()
    
    def sync_requirements(self, requirements: Dict[str, Any]):
        """
        Synchronize requirements across integrated systems
        """
        # Jira synchronization
        if self.jira.is_enabled():
            for req in requirements.get('functional_requirements', []):
                self.jira.create_issue(
                    project_key='REQ',
                    summary=req[:250],  # Truncate for Jira
                    description=req
                )
        
        # Confluence documentation
        if self.confluence.is_enabled():
            # Generate comprehensive requirements document
            content = self._generate_confluence_content(requirements)
            self.confluence.create_page(
                space_key='REQUIREMENTS',
                title='Requirements Specification',
                content=content
            )
    
    def _generate_confluence_content(self, requirements: Dict[str, Any]) -> str:
        """
        Generate Confluence-compatible storage format content
        """
        content = "<h1>Requirements Specification</h1>"
        
        content += "<h2>Functional Requirements</h2>"
        content += "<ul>"
        for req in requirements.get('functional_requirements', []):
            content += f"<li>{req}</li>"
        content += "</ul>"
        
        content += "<h2>Non-Functional Requirements</h2>"
        content += "<ul>"
        for req in requirements.get('non_functional_requirements', []):
            content += f"<li>{req}</li>"
        content += "</ul>"
        
        return content

# Singleton instance
integration_manager = IntegrationManager()
