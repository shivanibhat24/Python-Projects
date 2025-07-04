#!/usr/bin/env python3
"""
Legal Document Analyzer & Summarizer with Llama 4 Integration
A comprehensive tool for analyzing legal documents using Llama 4 as the AI engine.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
from pathlib import Path
import PyPDF2
import docx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeyClause:
    """Represents a key clause found in the document"""
    clause_type: str
    content: str
    importance: str  # 'low', 'medium', 'high', 'critical'
    section: str
    page_number: Optional[int] = None
    confidence: float = 0.0

@dataclass
class RiskAssessment:
    """Represents a potential risk identified in the document"""
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    category: str
    description: str
    recommendation: str
    affected_clauses: List[str]
    severity_score: float = 0.0

@dataclass
class DocumentAnalysis:
    """Complete analysis results for a legal document"""
    summary: str
    key_clauses: List[KeyClause]
    risks: List[RiskAssessment]
    document_type: str
    jurisdiction: Optional[str]
    parties_involved: List[str]
    effective_date: Optional[str]
    termination_date: Optional[str]
    analysis_timestamp: datetime
    confidence_score: float

class LlamaLegalAnalyzer:
    """Main analyzer class that integrates with Llama 4"""
    
    def __init__(self, model_path: str = "meta-llama/Llama-4-7b-chat-hf"):
        """Initialize the Llama 4 model for legal analysis"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Legal document patterns
        self.legal_patterns = {
            'contract_types': [
                r'employment\s+agreement', r'service\s+agreement', r'lease\s+agreement',
                r'purchase\s+agreement', r'non-disclosure\s+agreement', r'license\s+agreement',
                r'partnership\s+agreement', r'consulting\s+agreement'
            ],
            'risk_indicators': [
                r'indemnif\w+', r'liability', r'penalty', r'breach', r'default',
                r'termination', r'non-compete', r'confidentiality', r'intellectual\s+property'
            ],
            'date_patterns': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\b\d{1,2}\s+\w+\s+\d{4}\b',
                r'\b\w+\s+\d{1,2},?\s+\d{4}\b'
            ]
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the Llama 4 model and tokenizer"""
        try:
            logger.info(f"Loading Llama 4 model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Llama 4 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Llama 4 model: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def generate_llama_prompt(self, document_text: str, analysis_type: str) -> str:
        """Generate appropriate prompt for Llama 4 based on analysis type"""
        base_prompt = f"""
You are a specialized legal AI assistant with expertise in document analysis. 
Analyze the following legal document with precision and attention to detail.

Document Text:
{document_text[:4000]}...

"""
        
        if analysis_type == "summary":
            return base_prompt + """
Provide a comprehensive summary that includes:
1. Document type and purpose
2. Key parties involved
3. Main obligations and rights
4. Important dates and deadlines
5. Financial terms (if applicable)

Keep the summary clear, concise, and legally accurate."""
        
        elif analysis_type == "key_clauses":
            return base_prompt + """
Identify and extract key clauses from this document. For each clause, provide:
1. Clause type (e.g., Termination, Confidentiality, Payment, etc.)
2. Exact content or paraphrased content
3. Importance level (low/medium/high/critical)
4. Section reference
5. Brief explanation of significance

Focus on clauses that have significant legal or business implications."""
        
        elif analysis_type == "risks":
            return base_prompt + """
Conduct a comprehensive risk assessment of this document. Identify:
1. Potential legal risks and liabilities
2. Unfavorable terms or conditions
3. Ambiguous language that could cause disputes
4. Missing protective clauses
5. Compliance issues

For each risk, provide:
- Risk level (low/medium/high/critical)
- Category (e.g., Financial, Legal, Operational)
- Clear description of the risk
- Specific recommendation to mitigate the risk"""
        
        return base_prompt
    
    def query_llama(self, prompt: str, max_tokens: int = 1024) -> str:
        """Query Llama 4 with the given prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error querying Llama 4: {e}")
            return "Error in generating response"
    
    def parse_key_clauses(self, llama_response: str) -> List[KeyClause]:
        """Parse Llama 4 response into KeyClause objects"""
        clauses = []
        
        # Simple parsing logic - in production, this would be more sophisticated
        sections = re.split(r'\n\s*\n', llama_response)
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # Extract clause information
            clause_type = "General"
            content = ""
            importance = "medium"
            section_ref = "Unknown"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Type:") or line.startswith("Clause:"):
                    clause_type = line.split(":", 1)[1].strip()
                elif line.startswith("Content:"):
                    content = line.split(":", 1)[1].strip()
                elif line.startswith("Importance:"):
                    importance = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Section:"):
                    section_ref = line.split(":", 1)[1].strip()
                elif not any(line.startswith(prefix) for prefix in ["Type:", "Content:", "Importance:", "Section:"]):
                    if not content:
                        content = line
            
            if content:
                clauses.append(KeyClause(
                    clause_type=clause_type,
                    content=content,
                    importance=importance,
                    section=section_ref,
                    confidence=0.85
                ))
        
        return clauses
    
    def parse_risks(self, llama_response: str) -> List[RiskAssessment]:
        """Parse Llama 4 response into RiskAssessment objects"""
        risks = []
        
        sections = re.split(r'\n\s*\n', llama_response)
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if len(lines) < 2:
                continue
                
            risk_level = "medium"
            category = "General"
            description = ""
            recommendation = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Risk Level:") or line.startswith("Level:"):
                    risk_level = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Category:"):
                    category = line.split(":", 1)[1].strip()
                elif line.startswith("Description:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("Recommendation:"):
                    recommendation = line.split(":", 1)[1].strip()
                elif not any(line.startswith(prefix) for prefix in ["Risk Level:", "Level:", "Category:", "Description:", "Recommendation:"]):
                    if not description:
                        description = line
            
            if description:
                risks.append(RiskAssessment(
                    risk_level=risk_level,
                    category=category,
                    description=description,
                    recommendation=recommendation or "Review with legal counsel",
                    affected_clauses=[],
                    severity_score=0.7
                ))
        
        return risks
    
    def analyze_document(self, document_text: str) -> DocumentAnalysis:
        """Perform comprehensive analysis of the legal document"""
        logger.info("Starting document analysis...")
        
        # Generate summary
        summary_prompt = self.generate_llama_prompt(document_text, "summary")
        summary_response = self.query_llama(summary_prompt, max_tokens=512)
        
        # Extract key clauses
        clauses_prompt = self.generate_llama_prompt(document_text, "key_clauses")
        clauses_response = self.query_llama(clauses_prompt, max_tokens=1024)
        key_clauses = self.parse_key_clauses(clauses_response)
        
        # Assess risks
        risks_prompt = self.generate_llama_prompt(document_text, "risks")
        risks_response = self.query_llama(risks_prompt, max_tokens=1024)
        risks = self.parse_risks(risks_response)
        
        # Extract basic metadata
        document_type = self._identify_document_type(document_text)
        parties = self._extract_parties(document_text)
        dates = self._extract_dates(document_text)
        
        analysis = DocumentAnalysis(
            summary=summary_response,
            key_clauses=key_clauses,
            risks=risks,
            document_type=document_type,
            jurisdiction=self._extract_jurisdiction(document_text),
            parties_involved=parties,
            effective_date=dates.get('effective'),
            termination_date=dates.get('termination'),
            analysis_timestamp=datetime.now(),
            confidence_score=0.85
        )
        
        logger.info("Document analysis completed")
        return analysis
    
    def _identify_document_type(self, text: str) -> str:
        """Identify the type of legal document"""
        text_lower = text.lower()
        
        for pattern in self.legal_patterns['contract_types']:
            if re.search(pattern, text_lower):
                return pattern.replace(r'\s+', ' ').replace(r'\w+', '').strip()
        
        return "Legal Document"
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract parties involved in the document"""
        parties = []
        
        # Look for common party indicators
        party_patterns = [
            r'between\s+([^,]+),?\s+and\s+([^,\n]+)',
            r'party\s+of\s+the\s+first\s+part[:\s]+([^,\n]+)',
            r'party\s+of\s+the\s+second\s+part[:\s]+([^,\n]+)'
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    parties.extend([m.strip() for m in match])
                else:
                    parties.append(match.strip())
        
        return list(set(parties))
    
    def _extract_dates(self, text: str) -> Dict[str, str]:
        """Extract important dates from the document"""
        dates = {}
        
        for pattern in self.legal_patterns['date_patterns']:
            matches = re.findall(pattern, text)
            if matches:
                dates['effective'] = matches[0]
                break
        
        return dates
    
    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract governing jurisdiction"""
        jurisdiction_patterns = [
            r'governed\s+by\s+the\s+laws\s+of\s+([^,\.\n]+)',
            r'jurisdiction\s+of\s+([^,\.\n]+)',
            r'courts\s+of\s+([^,\.\n]+)'
        ]
        
        for pattern in jurisdiction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def export_analysis(self, analysis: DocumentAnalysis, format: str = "json") -> str:
        """Export analysis results in various formats"""
        if format == "json":
            return json.dumps(analysis.__dict__, indent=2, default=str)
        elif format == "report":
            return self._generate_report(analysis)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_report(self, analysis: DocumentAnalysis) -> str:
        """Generate a human-readable report"""
        report = f"""
LEGAL DOCUMENT ANALYSIS REPORT
Generated on: {analysis.analysis_timestamp}
Document Type: {analysis.document_type}
Confidence Score: {analysis.confidence_score:.2f}

EXECUTIVE SUMMARY
{analysis.summary}

KEY CLAUSES IDENTIFIED ({len(analysis.key_clauses)})
{'='*50}
"""
        
        for i, clause in enumerate(analysis.key_clauses, 1):
            report += f"""
{i}. {clause.clause_type} ({clause.importance.upper()})
   Section: {clause.section}
   Content: {clause.content}
"""
        
        report += f"""

RISK ASSESSMENT ({len(analysis.risks)} risks identified)
{'='*50}
"""
        
        for i, risk in enumerate(analysis.risks, 1):
            report += f"""
{i}. {risk.category} Risk - {risk.risk_level.upper()}
   Description: {risk.description}
   Recommendation: {risk.recommendation}
"""
        
        return report

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Legal Document Analyzer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Analyzer & Summarizer")
    st.subtitle("Powered by Llama 4 AI Engine")
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading Llama 4 model..."):
            try:
                st.session_state.analyzer = LlamaLegalAnalyzer()
                st.success("Llama 4 model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
    
    # Sidebar
    st.sidebar.title("Options")
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Full Analysis", "Summary Only", "Key Clauses", "Risk Assessment"]
    )
    
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["JSON", "Report", "Both"]
    )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Document Input")
        
        input_method = st.radio(
            "Input Method",
            ["Upload File", "Paste Text"]
        )
        
        document_text = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload legal document",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if uploaded_file is not None:
                try:
                    # Save uploaded file temporarily
                    with open(f"temp_{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    document_text = st.session_state.analyzer.extract_text_from_file(f"temp_{uploaded_file.name}")
                    st.success(f"Document loaded: {len(document_text)} characters")
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:
            document_text = st.text_area(
                "Paste document text",
                height=300,
                placeholder="Paste your legal document text here..."
            )
    
    with col2:
        st.header("Analysis Results")
        
        if document_text and st.button("Analyze Document", type="primary"):
            with st.spinner("Analyzing document with Llama 4..."):
                try:
                    analysis = st.session_state.analyzer.analyze_document(document_text)
                    
                    # Display results
                    st.subheader("📋 Document Summary")
                    st.write(analysis.summary)
                    
                    st.subheader("🔍 Key Clauses")
                    for clause in analysis.key_clauses:
                        importance_color = {
                            'critical': '🔴',
                            'high': '🟡',
                            'medium': '🟢',
                            'low': '⚪'
                        }.get(clause.importance, '⚪')
                        
                        st.write(f"{importance_color} **{clause.clause_type}** ({clause.section})")
                        st.write(f"   {clause.content}")
                        st.write("")
                    
                    st.subheader("⚠️ Risk Assessment")
                    for risk in analysis.risks:
                        risk_color = {
                            'critical': '🔴',
                            'high': '🟡',
                            'medium': '🟢',
                            'low': '⚪'
                        }.get(risk.risk_level, '⚪')
                        
                        with st.expander(f"{risk_color} {risk.category} Risk - {risk.risk_level.upper()}"):
                            st.write(f"**Description:** {risk.description}")
                            st.write(f"**Recommendation:** {risk.recommendation}")
                    
                    # Export options
                    st.subheader("📤 Export Analysis")
                    
                    if export_format in ["JSON", "Both"]:
                        json_export = st.session_state.analyzer.export_analysis(analysis, "json")
                        st.download_button(
                            "Download JSON",
                            json_export,
                            f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                    
                    if export_format in ["Report", "Both"]:
                        report_export = st.session_state.analyzer.export_analysis(analysis, "report")
                        st.download_button(
                            "Download Report",
                            report_export,
                            f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.error(f"Analysis error: {e}")

if __name__ == "__main__":
    main()
