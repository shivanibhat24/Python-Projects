import os
import json
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
import spacy

# Download required NLTK data
nltk.download('punkt')

class ScientificHypothesisGenerator:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        """
        Initialize the Scientific Hypothesis Generator
        
        Args:
            model_name (str): The name of the transformer model to use
        """
        print("Initializing Scientific Hypothesis Generator...")
        
        # Load NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()
        
        # Storage for extracted data
        self.papers = []
        self.entities = {}
        self.relationships = []
        self.claims = []
        
        print("Initialization complete.")
    
    def fetch_papers_from_pubmed(self, query, max_results=10):
        """
        Fetch papers from PubMed based on query
        
        Args:
            query (str): Search query for PubMed
            max_results (int): Maximum number of results to fetch
            
        Returns:
            list: List of paper metadata
        """
        print(f"Fetching papers from PubMed with query: {query}")
        
        # Base URL for PubMed API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Search for IDs
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={max_results}"
        response = requests.get(search_url)
        search_results = response.json()
        
        if 'esearchresult' not in search_results or 'idlist' not in search_results['esearchresult']:
            print("No results found")
            return []
        
        id_list = search_results['esearchresult']['idlist']
        
        # Fetch details for each ID
        papers = []
        for paper_id in tqdm(id_list, desc="Fetching paper details"):
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={paper_id}&retmode=xml"
            response = requests.get(fetch_url)
            
            if response.status_code == 200:
                try:
                    root = ET.fromstring(response.content)
                    article = root.find('.//Article')
                    
                    if article is not None:
                        title = article.find('.//ArticleTitle')
                        abstract = article.find('.//Abstract/AbstractText')
                        
                        title_text = title.text if title is not None else "No title"
                        abstract_text = abstract.text if abstract is not None else "No abstract"
                        
                        papers.append({
                            'id': paper_id,
                            'title': title_text,
                            'abstract': abstract_text,
                            'source': 'PubMed'
                        })
                except Exception as e:
                    print(f"Error processing paper ID {paper_id}: {e}")
        
        self.papers.extend(papers)
        print(f"Fetched {len(papers)} papers from PubMed.")
        return papers
    
    def fetch_papers_from_arxiv(self, query, max_results=10):
        """
        Fetch papers from ArXiv based on query
        
        Args:
            query (str): Search query for ArXiv
            max_results (int): Maximum number of results to fetch
            
        Returns:
            list: List of paper metadata
        """
        print(f"Fetching papers from ArXiv with query: {query}")
        
        # Base URL for ArXiv API
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print("Failed to fetch data from ArXiv")
            return []
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Define namespace
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in tqdm(root.findall('.//atom:entry', namespace), desc="Processing ArXiv papers"):
            try:
                paper_id = entry.find('.//atom:id', namespace).text.split('/abs/')[-1]
                title = entry.find('.//atom:title', namespace).text.strip()
                summary = entry.find('.//atom:summary', namespace).text.strip()
                
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'abstract': summary,
                    'source': 'ArXiv'
                })
            except Exception as e:
                print(f"Error processing ArXiv entry: {e}")
        
        self.papers.extend(papers)
        print(f"Fetched {len(papers)} papers from ArXiv.")
        return papers
    
    def process_text(self, text):
        """
        Process text with spaCy to extract entities and relationships
        
        Args:
            text (str): Text to process
            
        Returns:
            tuple: (entities, relationships)
        """
        if not text or text == "No abstract":
            return [], []
        
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entity_id = f"{ent.label_}_{ent.text.lower().replace(' ', '_')}"
            entities.append({
                'id': entity_id,
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
            # Add to entity dictionary
            if entity_id not in self.entities:
                self.entities[entity_id] = {
                    'text': ent.text,
                    'type': ent.label_,
                    'mentions': 0
                }
            self.entities[entity_id]['mentions'] += 1
        
        # Extract relationships
        relationships = []
        for sent in doc.sents:
            sent_doc = self.nlp(sent.text)
            
            # Simple subject-verb-object extraction
            subject = None
            verb = None
            obj = None
            
            for token in sent_doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                elif token.dep_ in ["dobj", "pobj"] and token.head.pos_ == "VERB":
                    obj = token.text
            
            if subject and verb and obj:
                rel = {
                    'subject': subject,
                    'predicate': verb,
                    'object': obj,
                    'sentence': sent.text
                }
                relationships.append(rel)
                self.relationships.append(rel)
        
        return entities, relationships
    
    def extract_claims(self, text, paper_id):
        """
        Extract scientific claims from text
        
        Args:
            text (str): Text to process
            paper_id (str): ID of the paper
            
        Returns:
            list: Extracted claims
        """
        # Identify sentences that likely contain claims
        claim_indicators = [
            "we found", "results show", "demonstrates", "we demonstrate",
            "we observed", "we report", "our findings", "concluded",
            "indicates that", "suggests that", "evidence for", "reveals that",
            "our results", "our study", "we identified", "we discover",
            "our analysis"
        ]
        
        sentences = sent_tokenize(text)
        claims = []
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            
            # Check if sentence contains claim indicators
            contains_indicator = any(indicator in lower_sent for indicator in claim_indicators)
            
            if contains_indicator:
                claim = {
                    'text': sentence,
                    'paper_id': paper_id,
                    'confidence': 0.8,  # Simplified confidence score
                    'entities': []
                }
                
                # Add entities mentioned in the claim
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    claim['entities'].append({
                        'text': ent.text,
                        'type': ent.label_
                    })
                
                claims.append(claim)
                self.claims.append(claim)
        
        return claims
    
    def process_papers(self):
        """
        Process all fetched papers to extract entities, relationships, and claims
        """
        print("Processing papers to extract information...")
        
        for paper in tqdm(self.papers, desc="Processing papers"):
            # Process title and abstract
            title_entities, title_relationships = self.process_text(paper['title'])
            abstract_entities, abstract_relationships = self.process_text(paper['abstract'])
            
            # Extract claims
            paper_claims = self.extract_claims(paper['abstract'], paper['id'])
            
            # Add to paper metadata
            paper['entities'] = title_entities + abstract_entities
            paper['relationships'] = title_relationships + abstract_relationships
            paper['claims'] = paper_claims
        
        print(f"Processed {len(self.papers)} papers.")
        print(f"Extracted {len(self.entities)} unique entities, {len(self.relationships)} relationships, and {len(self.claims)} claims.")
    
    def build_knowledge_graph(self):
        """
        Build a knowledge graph from extracted entities and relationships
        """
        print("Building knowledge graph...")
        
        # Add entities as nodes
        for entity_id, entity_info in self.entities.items():
            self.knowledge_graph.add_node(
                entity_id,
                label=entity_info['text'],
                type=entity_info['type'],
                mentions=entity_info['mentions']
            )
        
        # Add relationships as edges
        for i, rel in enumerate(self.relationships):
            # Try to find entity IDs for subject and object
            subject_candidates = [eid for eid, e in self.entities.items() if e['text'].lower() == rel['subject'].lower()]
            object_candidates = [eid for eid, e in self.entities.items() if e['text'].lower() == rel['object'].lower()]
            
            # If found, add the relationship
            if subject_candidates and object_candidates:
                subject_id = subject_candidates[0]
                object_id = object_candidates[0]
                
                self.knowledge_graph.add_edge(
                    subject_id,
                    object_id,
                    label=rel['predicate'],
                    sentence=rel['sentence'],
                    id=f"rel_{i}"
                )
        
        print(f"Knowledge graph built with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges.")
    
    def visualize_knowledge_graph(self, max_nodes=50):
        """
        Visualize the knowledge graph
        
        Args:
            max_nodes (int): Maximum number of nodes to visualize
        """
        if self.knowledge_graph.number_of_nodes() == 0:
            print("Knowledge graph is empty.")
            return
        
        # Select subset of nodes for visualization
        if self.knowledge_graph.number_of_nodes() > max_nodes:
            # Get top nodes by degree
            top_nodes = sorted(self.knowledge_graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.knowledge_graph.subgraph([node for node, degree in top_nodes])
        else:
            subgraph = self.knowledge_graph
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(subgraph)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        node_labels = {node: subgraph.nodes[node]['label'] for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def generate_embeddings(self):
        """
        Generate embeddings for all claims
        
        Returns:
            tuple: (claim_texts, embeddings)
        """
        claim_texts = [claim['text'] for claim in self.claims]
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(claim_texts)
        
        return claim_texts, embeddings
    
    def find_claim_connections(self, similarity_threshold=0.6):
        """
        Find connections between claims based on semantic similarity
        
        Args:
            similarity_threshold (float): Threshold for similarity score
            
        Returns:
            list: Connected claim pairs with similarity scores
        """
        print("Finding connections between claims...")
        
        claim_texts, embeddings = self.generate_embeddings()
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find connections
        connections = []
        for i in range(len(self.claims)):
            for j in range(i+1, len(self.claims)):
                similarity = similarity_matrix[i][j]
                
                if similarity > similarity_threshold:
                    connection = {
                        'claim1': self.claims[i],
                        'claim2': self.claims[j],
                        'similarity': similarity
                    }
                    connections.append(connection)
        
        print(f"Found {len(connections)} connections between claims.")
        return connections
    
    def generate_hypotheses(self, num_hypotheses=5):
        """
        Generate novel scientific hypotheses based on claim connections
        
        Args:
            num_hypotheses (int): Number of hypotheses to generate
            
        Returns:
            list: Generated hypotheses
        """
        print("Generating scientific hypotheses...")
        
        # Find connections between claims
        connections = self.find_claim_connections()
        
        # Sort connections by similarity score
        sorted_connections = sorted(connections, key=lambda x: x['similarity'], reverse=True)
        
        # Generate hypotheses from top connections
        hypotheses = []
        for i, connection in enumerate(sorted_connections[:num_hypotheses*2]):
            if len(hypotheses) >= num_hypotheses:
                break
                
            claim1 = connection['claim1']
            claim2 = connection['claim2']
            
            # Skip if claims are from the same paper
            if claim1['paper_id'] == claim2['paper_id']:
                continue
            
            # Generate hypothesis
            hypothesis_text = self._formulate_hypothesis(claim1, claim2)
            
            if hypothesis_text:
                hypothesis = {
                    'text': hypothesis_text,
                    'supporting_claims': [claim1, claim2],
                    'similarity_score': connection['similarity'],
                    'novelty_score': 0.5,  # Placeholder for a more sophisticated scoring
                    'plausibility_score': connection['similarity'] * 0.8,  # Simple plausibility score
                    'paper_ids': [claim1['paper_id'], claim2['paper_id']]
                }
                
                hypotheses.append(hypothesis)
        
        # Calculate overall scores
        for hypothesis in hypotheses:
            hypothesis['overall_score'] = (
                hypothesis['similarity_score'] * 0.4 + 
                hypothesis['novelty_score'] * 0.3 + 
                hypothesis['plausibility_score'] * 0.3
            )
        
        # Sort by overall score
        hypotheses = sorted(hypotheses, key=lambda x: x['overall_score'], reverse=True)
        
        print(f"Generated {len(hypotheses)} hypotheses.")
        return hypotheses
    
    def _formulate_hypothesis(self, claim1, claim2):
        """
        Formulate a hypothesis based on two claims
        
        Args:
            claim1 (dict): First claim
            claim2 (dict): Second claim
            
        Returns:
            str: Formulated hypothesis
        """
        # Extract entities from claims
        entities1 = set([e['text'].lower() for e in claim1['entities']])
        entities2 = set([e['text'].lower() for e in claim2['entities']])
        
        # Find common entities
        common_entities = entities1.intersection(entities2)
        
        # If no common entities, find if there are related entities
        if not common_entities and (len(entities1) > 0 and len(entities2) > 0):
            # For simplicity, we'll connect the first entity from each claim
            entity1 = next(iter(entities1))
            entity2 = next(iter(entities2))
            
            hypothesis = f"There may be a relationship between {entity1} and {entity2} that could explain the observations in both studies."
            return hypothesis
        
        # If there are common entities, build a hypothesis around them
        elif common_entities:
            common_entity = next(iter(common_entities))
            
            # Extract unique entities from each claim
            unique_entities1 = entities1 - common_entities
            unique_entities2 = entities2 - common_entities
            
            if unique_entities1 and unique_entities2:
                entity1 = next(iter(unique_entities1))
                entity2 = next(iter(unique_entities2))
                
                hypothesis = f"The role of {common_entity} in {entity1} may be connected to its function in {entity2}, suggesting a novel pathway or mechanism."
                return hypothesis
            
            elif unique_entities1:
                entity1 = next(iter(unique_entities1))
                hypothesis = f"The observed effects of {common_entity} may extend beyond the known context to influence {entity1}."
                return hypothesis
                
            elif unique_entities2:
                entity2 = next(iter(unique_entities2))
                hypothesis = f"The mechanisms of {common_entity} may have unexplored connections to {entity2}."
                return hypothesis
        
        # Fallback hypothesis
        return f"Based on findings across multiple studies, there may be an unexplored connection between the mechanisms described in these research papers."
    
    def explain_hypothesis(self, hypothesis):
        """
        Provide an explanation for a generated hypothesis
        
        Args:
            hypothesis (dict): Hypothesis to explain
            
        Returns:
            str: Explanation
        """
        # Extract supporting claims
        claim1 = hypothesis['supporting_claims'][0]
        claim2 = hypothesis['supporting_claims'][1]
        
        # Get paper IDs
        paper1_id = claim1['paper_id']
        paper2_id = claim2['paper_id']
        
        # Find paper details
        paper1 = next((p for p in self.papers if p['id'] == paper1_id), None)
        paper2 = next((p for p in self.papers if p['id'] == paper2_id), None)
        
        if not paper1 or not paper2:
            return "Could not find paper details."
        
        explanation = f"Hypothesis Explanation:\n\n"
        explanation += f"The hypothesis '{hypothesis['text']}' is derived from connecting findings across two distinct studies:\n\n"
        
        explanation += f"1. From '{paper1['title']}' (Paper ID: {paper1_id}):\n"
        explanation += f"   - Claim: \"{claim1['text']}\"\n\n"
        
        explanation += f"2. From '{paper2['title']}' (Paper ID: {paper2_id}):\n"
        explanation += f"   - Claim: \"{claim2['text']}\"\n\n"
        
        explanation += f"Similarity Score: {hypothesis['similarity_score']:.2f}\n"
        explanation += f"Plausibility Score: {hypothesis['plausibility_score']:.2f}\n"
        explanation += f"Novelty Score: {hypothesis['novelty_score']:.2f}\n"
        explanation += f"Overall Score: {hypothesis['overall_score']:.2f}\n\n"
        
        explanation += "The connection is established through semantic similarity and shared concepts between these claims. "
        explanation += "This hypothesis represents a potentially unexplored research direction that could be worth investigating."
        
        return explanation


# Example usage
if __name__ == "__main__":
    # Initialize the hypothesis generator
    generator = ScientificHypothesisGenerator()
    
    # Fetch papers from PubMed and ArXiv
    generator.fetch_papers_from_pubmed("cancer AND immunotherapy", max_results=5)
    generator.fetch_papers_from_arxiv("cancer immunotherapy", max_results=5)
    
    # Process papers to extract information
    generator.process_papers()
    
    # Build knowledge graph
    generator.build_knowledge_graph()
    
    # Generate hypotheses
    hypotheses = generator.generate_hypotheses(num_hypotheses=3)
    
    # Print hypotheses with explanations
    for i, hypothesis in enumerate(hypotheses):
        print(f"\nHypothesis {i+1}: {hypothesis['text']}")
        print("-" * 80)
        explanation = generator.explain_hypothesis(hypothesis)
        print(explanation)
        print("=" * 80)
