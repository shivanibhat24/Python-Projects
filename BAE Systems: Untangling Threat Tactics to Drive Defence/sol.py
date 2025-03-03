import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import requests
from collections import defaultdict

class ThreatIntelligenceParser:
    def __init__(self, attack_data_path=None):
        """
        Initialize the Threat Intelligence Parser
        
        Parameters:
        attack_data_path (str): Path to local ATT&CK data file. If None, data will be fetched from MITRE.
        """
        # Load pre-trained NLP model
        print("Loading NLP model...")
        self.nlp = spacy.load("en_core_web_md")
        
        # Load ATT&CK framework data
        self.attack_data = self.load_attack_data(attack_data_path)
        
        # Extract technique information
        self.techniques = {technique['technique_id']: technique for technique in self.attack_data['techniques']}
        
        # Create technique descriptions for matching
        self.technique_descriptions = {
            technique_id: f"{technique['name']} {technique['description']}"
            for technique_id, technique in self.techniques.items()
        }
        
        # Prepare vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        self.vectorizer.fit(list(self.technique_descriptions.values()))
        
        # Keywords and regex patterns for identifying TTPs
        self.ttp_patterns = [
            r'T\d{4}(\.\d{3})?',  # ATT&CK technique IDs (e.g., T1566, T1566.001)
            r'tactics?:?\s*([\w\s,]+)',  # Tactics mentioned directly
            r'techniques?:?\s*([\w\s,]+)',  # Techniques mentioned directly
            r'procedures?:?\s*([\w\s,]+)',  # Procedures mentioned directly
            r'leveraged|utilized|employed|executed|launched|performed|initiated|carried out|conducted',  # Action verbs
            r'attack vector|attack surface|entry point|initial access|lateral movement',  # Common threat concepts
        ]
        
        # Keywords for security control identification
        self.control_keywords = [
            "mitigate", "detect", "prevent", "block", "monitor", "filter", "scan",
            "restrict", "limit", "patch", "update", "disable", "enable", "implement",
            "security control", "countermeasure", "defense", "protection", "security measure"
        ]
        
    def load_attack_data(self, local_path=None):
        """
        Load ATT&CK data from local file or fetch from MITRE
        
        Parameters:
        local_path (str): Path to local ATT&CK data file
        
        Returns:
        dict: Processed ATT&CK data
        """
        if local_path:
            with open(local_path, 'r') as f:
                attack_data = json.load(f)
        else:
            # Fetch from MITRE ATT&CK Enterprise STIX data
            print("Fetching ATT&CK data from MITRE...")
            url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
            response = requests.get(url)
            stix_data = response.json()
            
            # Process STIX data into a more usable format
            attack_data = self._process_stix_data(stix_data)
            
        return attack_data
    
    def _process_stix_data(self, stix_data):
        """
        Process STIX data into a more usable format
        
        Parameters:
        stix_data (dict): Raw STIX data
        
        Returns:
        dict: Processed ATT&CK data
        """
        tactics = []
        techniques = []
        mitigations = []
        
        # Extract objects by type
        for obj in stix_data.get('objects', []):
            obj_type = obj.get('type')
            
            if obj_type == 'attack-pattern':
                # Handle technique
                technique_id = None
                for external_ref in obj.get('external_references', []):
                    if external_ref.get('source_name') == 'mitre-attack':
                        technique_id = external_ref.get('external_id')
                
                if technique_id:
                    technique = {
                        'technique_id': technique_id,
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'tactics': [],
                        'mitigations': []
                    }
                    
                    # Extract tactic names
                    for kill_chain_phase in obj.get('kill_chain_phases', []):
                        if kill_chain_phase.get('kill_chain_name') == 'mitre-attack':
                            tactic = kill_chain_phase.get('phase_name')
                            if tactic:
                                technique['tactics'].append(tactic)
                    
                    techniques.append(technique)
            
            elif obj_type == 'course-of-action':
                # Handle mitigation
                mitigation_id = None
                for external_ref in obj.get('external_references', []):
                    if external_ref.get('source_name') == 'mitre-attack':
                        mitigation_id = external_ref.get('external_id')
                
                if mitigation_id:
                    mitigation = {
                        'mitigation_id': mitigation_id,
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'techniques': []  # Will be populated later
                    }
                    mitigations.append(mitigation)
            
            elif obj_type == 'x-mitre-tactic':
                # Handle tactic
                tactic_id = None
                for external_ref in obj.get('external_references', []):
                    if external_ref.get('source_name') == 'mitre-attack':
                        tactic_id = external_ref.get('external_id')
                
                if tactic_id:
                    tactic = {
                        'tactic_id': tactic_id,
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'shortname': obj.get('x_mitre_shortname', '')
                    }
                    tactics.append(tactic)
        
        # Process relationships
        for obj in stix_data.get('objects', []):
            if obj.get('type') == 'relationship':
                source_ref = obj.get('source_ref')
                target_ref = obj.get('target_ref')
                relationship_type = obj.get('relationship_type')
                
                if relationship_type == 'mitigates':
                    # Find the mitigation and technique
                    for mitigation in mitigations:
                        if mitigation.get('mitigation_id') in source_ref:
                            for technique in techniques:
                                if technique.get('technique_id') in target_ref:
                                    mitigation['techniques'].append(technique['technique_id'])
                                    technique['mitigations'].append(mitigation['mitigation_id'])
        
        return {
            'tactics': tactics,
            'techniques': techniques,
            'mitigations': mitigations
        }
    
    def extract_explicit_techniques(self, text):
        """
        Extract explicitly mentioned ATT&CK technique IDs from text
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        list: List of identified technique IDs
        """
        pattern = r'T\d{4}(\.\d{3})?'
        return re.findall(pattern, text)
    
    def extract_potential_ttp_paragraphs(self, text):
        """
        Extract paragraphs likely describing TTPs
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        list: List of paragraphs potentially describing TTPs
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        potential_ttp_paragraphs = []
        
        for p in paragraphs:
            # Check if paragraph matches any of the TTP patterns
            matches_pattern = any(re.search(pattern, p, re.IGNORECASE) for pattern in self.ttp_patterns)
            
            # Check if paragraph contains key terms from ATT&CK
            doc = self.nlp(p)
            entities = [ent.text.lower() for ent in doc.ents]
            contains_attack_terms = any(term in p.lower() for term in [
                "malware", "phishing", "backdoor", "trojan", "ransomware", "exploit",
                "vulnerability", "remote", "access", "credential", "privilege", "lateral",
                "command", "control", "exfiltration", "persistence", "evasion"
            ])
            
            if matches_pattern or contains_attack_terms:
                potential_ttp_paragraphs.append(p)
        
        return potential_ttp_paragraphs
    
    def extract_potential_control_paragraphs(self, text):
        """
        Extract paragraphs likely describing security controls
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        list: List of paragraphs potentially describing security controls
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        potential_control_paragraphs = []
        
        for p in paragraphs:
            # Check if paragraph contains control keywords
            if any(keyword in p.lower() for keyword in self.control_keywords):
                potential_control_paragraphs.append(p)
            
            # Check for sentences that seem like recommendations
            doc = self.nlp(p)
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if (sent_text.startswith("organizations should") or 
                    sent_text.startswith("we recommend") or
                    sent_text.startswith("it is recommended") or
                    "recommend" in sent_text or
                    "advised to" in sent_text):
                    if p not in potential_control_paragraphs:
                        potential_control_paragraphs.append(p)
        
        return potential_control_paragraphs
    
    def map_text_to_techniques(self, text, threshold=0.2):
        """
        Map text to ATT&CK techniques using semantic similarity
        
        Parameters:
        text (str): Text to analyze
        threshold (float): Similarity threshold for matching
        
        Returns:
        list: List of tuples (technique_id, similarity_score)
        """
        # Vectorize the input text
        text_vector = self.vectorizer.transform([text])
        
        # Vectorize all technique descriptions
        technique_vectors = self.vectorizer.transform(list(self.technique_descriptions.values()))
        
        # Calculate similarity
        similarities = cosine_similarity(text_vector, technique_vectors)[0]
        
        # Match with technique IDs
        technique_ids = list(self.technique_descriptions.keys())
        
        # Find techniques above threshold
        matches = [(technique_ids[i], similarities[i]) for i in range(len(technique_ids)) if similarities[i] > threshold]
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def process_report(self, report_text):
        """
        Process a threat intelligence report and map to ATT&CK framework
        
        Parameters:
        report_text (str): Text of the threat intelligence report
        
        Returns:
        dict: Structured analysis of the report
        """
        results = {
            'explicit_techniques': [],
            'inferred_techniques': [],
            'identified_controls': [],
            'mapped_mitigations': []
        }
        
        # Extract explicitly mentioned techniques
        results['explicit_techniques'] = self.extract_explicit_techniques(report_text)
        
        # Extract paragraphs potentially describing TTPs
        ttp_paragraphs = self.extract_potential_ttp_paragraphs(report_text)
        
        # Map paragraphs to techniques
        for paragraph in ttp_paragraphs:
            technique_matches = self.map_text_to_techniques(paragraph)
            for technique_id, score in technique_matches:
                if technique_id not in results['explicit_techniques']:
                    results['inferred_techniques'].append({
                        'technique_id': technique_id,
                        'name': self.techniques[technique_id]['name'],
                        'confidence': float(score),
                        'source_text': paragraph
                    })
        
        # Extract potential security controls
        control_paragraphs = self.extract_potential_control_paragraphs(report_text)
        results['identified_controls'] = control_paragraphs
        
        # Get recommended mitigations for identified techniques
        all_techniques = results['explicit_techniques'] + [t['technique_id'] for t in results['inferred_techniques']]
        mitigations_map = defaultdict(list)
        
        for technique_id in all_techniques:
            if technique_id in self.techniques:
                for mitigation_id in self.techniques[technique_id].get('mitigations', []):
                    # Find the mitigation details
                    for mitigation in self.attack_data['mitigations']:
                        if mitigation['mitigation_id'] == mitigation_id:
                            mitigations_map[mitigation_id].append(technique_id)
                            # Only add if not already in results
                            if not any(m['mitigation_id'] == mitigation_id for m in results['mapped_mitigations']):
                                results['mapped_mitigations'].append({
                                    'mitigation_id': mitigation_id,
                                    'name': mitigation['name'],
                                    'description': mitigation['description'],
                                    'techniques_addressed': []  # Will be populated after processing all techniques
                                })
        
        # Update techniques addressed by each mitigation
        for mitigation in results['mapped_mitigations']:
            mitigation_id = mitigation['mitigation_id']
            mitigation['techniques_addressed'] = [
                {
                    'technique_id': technique_id,
                    'name': self.techniques[technique_id]['name'] if technique_id in self.techniques else 'Unknown'
                }
                for technique_id in mitigations_map[mitigation_id]
            ]
        
        return results
    
    def generate_defense_recommendations(self, analysis_results):
        """
        Generate defense recommendations based on analysis results
        
        Parameters:
        analysis_results (dict): Results from process_report
        
        Returns:
        dict: Structured defense recommendations
        """
        recommendations = {
            'prioritized_mitigations': [],
            'report_controls': [],
            'gaps': []
        }
        
        # Identify controls mentioned in the report
        report_controls = analysis_results.get('identified_controls', [])
        recommendations['report_controls'] = report_controls
        
        # Process mitigations from ATT&CK
        attack_mitigations = analysis_results.get('mapped_mitigations', [])
        
        # Prioritize mitigations by the number of techniques they address
        prioritized = sorted(
            attack_mitigations,
            key=lambda x: len(x.get('techniques_addressed', [])),
            reverse=True
        )
        
        recommendations['prioritized_mitigations'] = prioritized
        
        # Identify potential gaps where techniques don't have explicitly mentioned controls
        all_techniques = analysis_results.get('explicit_techniques', []) + [
            t['technique_id'] for t in analysis_results.get('inferred_techniques', [])
        ]
        
        # Check which techniques might not be addressed by report controls
        # This is a simplistic approach - in reality, we would need NLP to match controls to techniques
        techniques_without_controls = []
        
        for technique_id in all_techniques:
            if technique_id not in self.techniques:
                continue
                
            technique_name = self.techniques[technique_id]['name']
            technique_desc = self.techniques[technique_id]['description']
            
            # Simple check if technique is mentioned in any control paragraph
            found_in_controls = False
            for control in report_controls:
                if (technique_name.lower() in control.lower() or 
                    any(word.lower() in control.lower() for word in technique_name.lower().split() if len(word) > 4)):
                    found_in_controls = True
                    break
            
            if not found_in_controls:
                techniques_without_controls.append({
                    'technique_id': technique_id,
                    'name': technique_name,
                    'description': technique_desc,
                    'recommended_mitigations': [
                        m['name'] for m in prioritized 
                        if any(t['technique_id'] == technique_id for t in m.get('techniques_addressed', []))
                    ]
                })
        
        recommendations['gaps'] = techniques_without_controls
        
        return recommendations
    
    def generate_report(self, report_text):
        """
        Generate a comprehensive analysis of a threat intelligence report
        
        Parameters:
        report_text (str): Text of the threat intelligence report
        
        Returns:
        dict: Comprehensive analysis and recommendations
        """
        # Process the report
        analysis = self.process_report(report_text)
        
        # Generate defense recommendations
        recommendations = self.generate_defense_recommendations(analysis)
        
        # Combine results
        result = {
            'analysis': analysis,
            'recommendations': recommendations,
            'summary': {
                'explicit_techniques_count': len(analysis['explicit_techniques']),
                'inferred_techniques_count': len(analysis['inferred_techniques']),
                'identified_controls_count': len(analysis['identified_controls']),
                'mapped_mitigations_count': len(analysis['mapped_mitigations']),
                'potential_gap_count': len(recommendations['gaps'])
            }
        }
        
        return result
    
    def export_to_markdown(self, results, output_path):
        """
        Export analysis results to a Markdown file
        
        Parameters:
        results (dict): Analysis results from generate_report
        output_path (str): Path to save the Markdown file
        """
        md_content = "# Threat Intelligence Analysis Report\n\n"
        
        # Summary section
        md_content += "## Summary\n\n"
        summary = results['summary']
        md_content += f"- Explicitly mentioned ATT&CK techniques: {summary['explicit_techniques_count']}\n"
        md_content += f"- Inferred techniques: {summary['inferred_techniques_count']}\n"
        md_content += f"- Security controls identified in report: {summary['identified_controls_count']}\n"
        md_content += f"- MITRE ATT&CK mitigations mapped: {summary['mapped_mitigations_count']}\n"
        md_content += f"- Potential defense gaps: {summary['potential_gap_count']}\n\n"
        
        # Techniques section
        md_content += "## Identified Techniques\n\n"
        md_content += "### Explicitly Mentioned Techniques\n\n"
        
        if results['analysis']['explicit_techniques']:
            for technique_id in results['analysis']['explicit_techniques']:
                if technique_id in self.techniques:
                    technique = self.techniques[technique_id]
                    md_content += f"- **{technique_id}: {technique['name']}**\n"
                    md_content += f"  - Tactics: {', '.join(technique['tactics'])}\n"
                    md_content += f"  - Description: {technique['description'][:200]}...\n\n"
                else:
                    md_content += f"- **{technique_id}**: Unknown technique\n\n"
        else:
            md_content += "No explicitly mentioned techniques found.\n\n"
        
        md_content += "### Inferred Techniques\n\n"
        
        if results['analysis']['inferred_techniques']:
            for technique in results['analysis']['inferred_techniques']:
                md_content += f"- **{technique['technique_id']}: {technique['name']}**\n"
                md_content += f"  - Confidence: {technique['confidence']:.2f}\n"
                md_content += f"  - Source text: \"{technique['source_text'][:200]}...\"\n\n"
        else:
            md_content += "No inferred techniques found.\n\n"
        
        # Security Controls section
        md_content += "## Security Controls\n\n"
        md_content += "### Controls Mentioned in Report\n\n"
        
        if results['analysis']['identified_controls']:
            for i, control in enumerate(results['analysis']['identified_controls'], 1):
                md_content += f"{i}. {control}\n\n"
        else:
            md_content += "No security controls explicitly mentioned in the report.\n\n"
        
        # Mitigations section
        md_content += "## MITRE ATT&CK Mitigations\n\n"
        
        if results['recommendations']['prioritized_mitigations']:
            for mitigation in results['recommendations']['prioritized_mitigations']:
                md_content += f"### {mitigation['mitigation_id']}: {mitigation['name']}\n\n"
                md_content += f"{mitigation['description']}\n\n"
                md_content += "**Addresses techniques:**\n\n"
                
                for technique in mitigation['techniques_addressed']:
                    md_content += f"- {technique['technique_id']}: {technique['name']}\n"
                
                md_content += "\n"
        else:
            md_content += "No MITRE ATT&CK mitigations mapped.\n\n"
        
        # Gaps section
        md_content += "## Potential Defense Gaps\n\n"
        
        if results['recommendations']['gaps']:
            for gap in results['recommendations']['gaps']:
                md_content += f"### {gap['technique_id']}: {gap['name']}\n\n"
                md_content += f"{gap['description'][:200]}...\n\n"
                md_content += "**Recommended mitigations:**\n\n"
                
                if gap['recommended_mitigations']:
                    for mitigation in gap['recommended_mitigations']:
                        md_content += f"- {mitigation}\n"
                else:
                    md_content += "No specific mitigations found.\n"
                
                md_content += "\n"
        else:
            md_content += "No potential defense gaps identified.\n\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(md_content)
        
        return md_content

# Example usage
if __name__ == "__main__":
    # Initialize parser
    parser = ThreatIntelligenceParser()
    
    # Example report text (abbreviated)
    sample_report = """
    Threat Actor APT29 Campaign Analysis
    
    Recent observations indicate that APT29 has been targeting organizations in the healthcare sector with a sophisticated phishing campaign (T1566). The attackers use spear-phishing emails with malicious attachments that exploit vulnerabilities in Microsoft Office (T1203).
    
    After gaining initial access, the attackers establish persistence by creating scheduled tasks (T1053.005) and modifying registry run keys (T1547.001). They conduct internal reconnaissance using built-in Windows commands and tools.
    
    For lateral movement, the attackers leverage Pass-the-Hash techniques (T1550.002) and exploit Windows Remote Services (T1021).
    
    Data is exfiltrated using encrypted channels over HTTP (T1071.001), often during regular business hours to blend with normal network traffic.
    
    Organizations should implement multi-factor authentication to prevent unauthorized access. Email filtering and user awareness training are critical to prevent successful phishing attacks. Regular patching of vulnerabilities, especially in Microsoft Office products, is strongly recommended.
    
    Monitor for suspicious scheduled tasks and registry modifications. Implement network segmentation to limit lateral movement capabilities.
    """
    
    # Process the report
    results = parser.generate_report(sample_report)
    
    # Export results to markdown
    md_content = parser.export_to_markdown(results, "threat_analysis.md")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"- Explicitly mentioned techniques: {results['summary']['explicit_techniques_count']}")
    print(f"- Inferred techniques: {results['summary']['inferred_techniques_count']}")
    print(f"- Identified controls: {results['summary']['identified_controls_count']}")
    print(f"- Mapped mitigations: {results['summary']['mapped_mitigations_count']}")
    print(f"- Potential gaps: {results['summary']['potential_gap_count']}")
