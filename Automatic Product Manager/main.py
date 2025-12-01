"""
PM Intelligence System - Transform Chaos into Structured Product Outputs
Converts messy customer feedback, interviews, and requests into actionable PM deliverables
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import anthropic

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Priority(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class EffortSize(Enum):
    XS = "XS (1-2 days)"
    S = "S (3-5 days)"
    M = "M (1-2 weeks)"
    L = "L (2-4 weeks)"
    XL = "XL (1-2 months)"

class InputType(Enum):
    INTERVIEW = "customer_interview"
    FEATURE_REQUEST = "feature_request"
    FEEDBACK = "customer_feedback"
    SUPPORT_TICKET = "support_ticket"
    BRAINDUMP = "ceo_braindump"

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UserStory:
    title: str
    description: str
    as_a: str
    i_want: str
    so_that: str
    acceptance_criteria: List[str]
    priority: str
    effort_estimate: str
    technical_notes: Optional[str] = None
    
@dataclass
class Feature:
    name: str
    user_stories: List[UserStory]
    architecture_suggestions: List[str]
    dependencies: List[str]
    risks: List[str]
    total_effort: str

@dataclass
class RoadmapItem:
    feature_name: str
    sprint: str
    priority: str
    effort: str
    stakeholders: List[str]

@dataclass
class JiraTicket:
    ticket_type: str  # Epic, Story, Task, Bug
    title: str
    description: str
    acceptance_criteria: List[str]
    story_points: int
    priority: str
    labels: List[str]
    linked_tickets: List[str]

# ============================================================================
# LLM ORCHESTRATION ENGINE
# ============================================================================

class PMIntelligenceEngine:
    """Core engine that orchestrates LLM calls to transform inputs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-20250514"
    
    def process_input(self, raw_input: str, input_type: InputType) -> Dict[str, Any]:
        """Main entry point: processes any type of input"""
        
        # Step 1: Extract insights and structure
        insights = self._extract_insights(raw_input, input_type)
        
        # Step 2: Generate user stories
        user_stories = self._generate_user_stories(insights)
        
        # Step 3: Create architecture suggestions
        architecture = self._suggest_architecture(insights, user_stories)
        
        # Step 4: Estimate effort
        effort_estimates = self._estimate_effort(user_stories, architecture)
        
        # Step 5: Generate roadmap
        roadmap = self._create_roadmap(user_stories, effort_estimates)
        
        # Step 6: Create Jira tickets
        jira_tickets = self._generate_jira_tickets(user_stories, roadmap)
        
        return {
            "insights": insights,
            "user_stories": user_stories,
            "architecture": architecture,
            "effort_estimates": effort_estimates,
            "roadmap": roadmap,
            "jira_tickets": jira_tickets,
            "summary": self._generate_executive_summary(insights, roadmap)
        }
    
    def _extract_insights(self, raw_input: str, input_type: InputType) -> Dict[str, Any]:
        """Extract structured insights from messy input"""
        
        prompt = f"""You are a senior product manager analyzing {input_type.value}.

Extract the following from this input:
1. Core problems mentioned
2. User needs and pain points
3. Desired outcomes
4. Any technical constraints mentioned
5. Stakeholders involved
6. Business impact (if mentioned)

Input:
{raw_input}

Respond with a JSON object containing these insights."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"raw_analysis": content}
    
    def _generate_user_stories(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate user stories from insights"""
        
        prompt = f"""Based on these product insights, create detailed user stories.

Insights:
{json.dumps(insights, indent=2)}

For each user story, provide:
- Title
- Description
- As a [user type]
- I want [functionality]
- So that [benefit/value]
- Acceptance criteria (specific, testable conditions)
- Priority (Critical/High/Medium/Low)
- Effort estimate (XS/S/M/L/XL)
- Technical notes (if applicable)

Create 3-7 user stories. Respond with a JSON array."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []
    
    def _suggest_architecture(self, insights: Dict[str, Any], 
                            user_stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level architecture suggestions"""
        
        prompt = f"""As a technical architect, suggest a high-level implementation approach.

Product Insights:
{json.dumps(insights, indent=2)}

User Stories:
{json.dumps(user_stories, indent=2)}

Provide:
1. System components needed
2. Data models/entities
3. API endpoints (if applicable)
4. Third-party integrations
5. Technical risks and mitigation strategies
6. Scalability considerations

Respond with a JSON object."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"architecture": content}
    
    def _estimate_effort(self, user_stories: List[Dict[str, Any]], 
                        architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate development effort for each component"""
        
        prompt = f"""As an engineering manager, provide effort estimates.

User Stories:
{json.dumps(user_stories, indent=2)}

Architecture:
{json.dumps(architecture, indent=2)}

For each user story, estimate:
- Story points (1-13 Fibonacci scale)
- Development time bucket (XS/S/M/L/XL)
- Risk level (Low/Medium/High)
- Dependencies on other stories

Also provide total sprint allocation estimate.
Respond with a JSON object."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    
    def _create_roadmap(self, user_stories: List[Dict[str, Any]], 
                       effort_estimates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a sprint-based roadmap"""
        
        prompt = f"""Create a sprint roadmap (assume 2-week sprints).

User Stories:
{json.dumps(user_stories, indent=2)}

Effort Estimates:
{json.dumps(effort_estimates, indent=2)}

Organize into sprints based on:
- Priority
- Dependencies
- Team capacity (assume 8 story points per sprint)
- Risk mitigation (do risky items early)

Provide a JSON array with sprint allocations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []
    
    def _generate_jira_tickets(self, user_stories: List[Dict[str, Any]], 
                              roadmap: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate Jira-ready ticket breakdowns"""
        
        prompt = f"""Create Jira ticket breakdown from these user stories and roadmap.

User Stories:
{json.dumps(user_stories, indent=2)}

Roadmap:
{json.dumps(roadmap, indent=2)}

For each story, create:
1. One Epic ticket (high-level feature)
2. Multiple Story tickets (user-facing functionality)
3. Task tickets (technical implementation)

Each ticket needs:
- Type (Epic/Story/Task/Bug)
- Title
- Description
- Acceptance criteria
- Story points
- Priority
- Labels (frontend, backend, design, etc.)
- Linked tickets (blockers, relates to)

Respond with a JSON array of tickets."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []
    
    def _generate_executive_summary(self, insights: Dict[str, Any], 
                                   roadmap: List[Dict[str, Any]]) -> str:
        """Generate executive summary for stakeholders"""
        
        prompt = f"""Create a 3-paragraph executive summary for leadership.

Insights:
{json.dumps(insights, indent=2)}

Roadmap:
{json.dumps(roadmap, indent=2)}

Cover:
1. What problem we're solving and why it matters
2. What we're building (high-level)
3. Timeline and resource requirements

Make it non-technical and business-focused."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# ============================================================================
# OUTPUT FORMATTERS
# ============================================================================

class OutputFormatter:
    """Formats processed data for different outputs"""
    
    @staticmethod
    def to_markdown(results: Dict[str, Any]) -> str:
        """Convert results to markdown format"""
        md = ["# Product Requirements Document\n"]
        
        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(results['summary'] + "\n")
        
        # Insights
        md.append("\n## Key Insights\n")
        for key, value in results['insights'].items():
            md.append(f"### {key.replace('_', ' ').title()}\n")
            if isinstance(value, list):
                for item in value:
                    md.append(f"- {item}\n")
            else:
                md.append(f"{value}\n")
        
        # User Stories
        md.append("\n## User Stories\n")
        for i, story in enumerate(results['user_stories'], 1):
            md.append(f"\n### Story {i}: {story.get('title', 'Untitled')}\n")
            md.append(f"**As a** {story.get('as_a', 'user')}\n")
            md.append(f"**I want** {story.get('i_want', 'functionality')}\n")
            md.append(f"**So that** {story.get('so_that', 'value')}\n\n")
            md.append(f"**Priority:** {story.get('priority', 'Medium')}\n")
            md.append(f"**Effort:** {story.get('effort_estimate', 'M')}\n\n")
            md.append("**Acceptance Criteria:**\n")
            for criterion in story.get('acceptance_criteria', []):
                md.append(f"- {criterion}\n")
        
        # Architecture
        md.append("\n## Architecture Suggestions\n")
        arch = results['architecture']
        for key, value in arch.items():
            md.append(f"### {key.replace('_', ' ').title()}\n")
            if isinstance(value, list):
                for item in value:
                    md.append(f"- {item}\n")
            else:
                md.append(f"{value}\n")
        
        # Roadmap
        md.append("\n## Roadmap\n")
        for sprint in results['roadmap']:
            sprint_num = sprint.get('sprint', 'N/A')
            md.append(f"\n### {sprint_num}\n")
            md.append(f"- **Feature:** {sprint.get('feature_name', 'N/A')}\n")
            md.append(f"- **Priority:** {sprint.get('priority', 'N/A')}\n")
            md.append(f"- **Effort:** {sprint.get('effort', 'N/A')}\n")
        
        # Jira Tickets
        md.append("\n## Jira Ticket Breakdown\n")
        for ticket in results['jira_tickets']:
            md.append(f"\n### [{ticket.get('ticket_type', 'Story')}] {ticket.get('title', 'Untitled')}\n")
            md.append(f"**Story Points:** {ticket.get('story_points', 'N/A')}\n")
            md.append(f"**Priority:** {ticket.get('priority', 'Medium')}\n")
            md.append(f"**Labels:** {', '.join(ticket.get('labels', []))}\n")
        
        return ''.join(md)
    
    @staticmethod
    def to_jira_import(results: Dict[str, Any]) -> str:
        """Convert to Jira CSV import format"""
        csv = ["Summary,Issue Type,Priority,Description,Acceptance Criteria,Story Points,Labels"]
        
        for ticket in results['jira_tickets']:
            title = ticket.get('title', '').replace(',', ';')
            issue_type = ticket.get('ticket_type', 'Story')
            priority = ticket.get('priority', 'Medium')
            description = ticket.get('description', '').replace(',', ';').replace('\n', ' ')
            ac = ' | '.join(ticket.get('acceptance_criteria', [])).replace(',', ';')
            points = ticket.get('story_points', 3)
            labels = ' '.join(ticket.get('labels', []))
            
            csv.append(f'"{title}",{issue_type},{priority},"{description}","{ac}",{points},{labels}')
        
        return '\n'.join(csv)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Example usage of the PM Intelligence System"""
    
    # Example: CEO brain dump
    ceo_braindump = """
    Spoke with 5 enterprise customers this week. They're all complaining about
    the same thing - onboarding takes too long. One VP said their team spends
    3 weeks just setting up integrations. 
    
    We need to make this dead simple. Think Stripe or Twilio level easy.
    
    Also, the support team is drowning in tickets about webhook failures.
    Users don't know what went wrong. We need better error messages and 
    maybe some kind of debugging dashboard.
    
    Timeline: this is blocking a $500K deal. Need something in 6 weeks.
    
    Team capacity: 2 backend devs, 1 frontend, 1 designer.
    """
    
    print("🚀 PM Intelligence System - Starting Analysis...")
    print("=" * 70)
    
    # Initialize engine
    engine = PMIntelligenceEngine()
    
    # Process input
    print("\n📊 Processing CEO brain dump...\n")
    results = engine.process_input(ceo_braindump, InputType.BRAINDUMP)
    
    # Format output
    formatter = OutputFormatter()
    markdown_output = formatter.to_markdown(results)
    
    print(markdown_output)
    
    # Save outputs
    print("\n💾 Saving outputs...")
    with open('prd_output.md', 'w') as f:
        f.write(markdown_output)
    
    with open('jira_import.csv', 'w') as f:
        f.write(formatter.to_jira_import(results))
    
    with open('raw_output.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Complete! Files saved:")
    print("   - prd_output.md (Full PRD)")
    print("   - jira_import.csv (Jira import file)")
    print("   - raw_output.json (Raw JSON output)")

if __name__ == "__main__":
    main()
