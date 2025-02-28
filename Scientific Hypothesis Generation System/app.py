from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib
matplotlib.use('Agg')
import io
import base64
from model import ScientificHypothesisGenerator
import time
import threading

app = Flask(__name__, static_folder='static')

# Global variables to store state
generator = None
papers = []
hypotheses = []
processing_status = {
    "status": "idle",
    "message": "System ready",
    "progress": 0
}
knowledge_graph_image = None

def background_process(pubmed_query, arxiv_query, max_papers, num_hypotheses):
    """
    Run processing in background thread
    """
    global processing_status, generator, papers, hypotheses, knowledge_graph_image
    
    try:
        # Initialize the generator if not already done
        if generator is None:
            processing_status = {"status": "initializing", "message": "Initializing model...", "progress": 10}
            generator = ScientificHypothesisGenerator()
        
        # Fetch papers
        processing_status = {"status": "fetching", "message": "Fetching papers from PubMed...", "progress": 20}
        if pubmed_query:
            generator.fetch_papers_from_pubmed(pubmed_query, max_results=max_papers//2)
        
        processing_status = {"status": "fetching", "message": "Fetching papers from ArXiv...", "progress": 40}
        if arxiv_query:
            generator.fetch_papers_from_arxiv(arxiv_query, max_results=max_papers//2)
        
        # Process papers
        processing_status = {"status": "processing", "message": "Processing papers...", "progress": 60}
        generator.process_papers()
        
        # Build knowledge graph
        processing_status = {"status": "building_graph", "message": "Building knowledge graph...", "progress": 80}
        generator.build_knowledge_graph()
        
        # Generate knowledge graph visualization
        knowledge_graph_image = generate_graph_image(dark_mode=False)
        
        # Generate hypotheses
        processing_status = {"status": "generating", "message": "Generating hypotheses...", "progress": 90}
        hypotheses = generator.generate_hypotheses(num_hypotheses=num_hypotheses)
        
        # Save papers for display
        papers = generator.papers
        
        processing_status = {"status": "completed", "message": "Processing complete", "progress": 100}
    
    except Exception as e:
        processing_status = {"status": "error", "message": f"Error: {str(e)}", "progress": 0}

def generate_graph_image(dark_mode=False):
    """Generate an image of the knowledge graph"""
    global generator
    
    if generator is None or generator.knowledge_graph.number_of_nodes() == 0:
        return None
    
    # Set up dark mode if requested
    if dark_mode:
        plt.style.use('dark_background')
        node_color = 'white'
        edge_color = 'gray'
        font_color = 'white'
        background = '#121212'
    else:
        plt.style.use('default')
        node_color = '#1f77b4'  # Default blue
        edge_color = '#777777'
        font_color = 'black'
        background = 'white'
    
    # Select subset of nodes for visualization
    max_nodes = 50
    if generator.knowledge_graph.number_of_nodes() > max_nodes:
        # Get top nodes by degree
        top_nodes = sorted(generator.knowledge_graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        subgraph = generator.knowledge_graph.subgraph([node for node, degree in top_nodes])
    else:
        subgraph = generator.knowledge_graph
    
    plt.figure(figsize=(10, 8))
    
    if dark_mode:
        plt.rcParams.update({
            'axes.facecolor': background,
            'figure.facecolor': background,
        })
    
    pos = nx.spring_layout(subgraph, seed=42)  # Consistent layout
    
    # Draw nodes with varying sizes based on mentions
    node_sizes = [300 + (subgraph.nodes[node].get('mentions', 1) * 20) for node in subgraph.nodes()]
    node_colors = [to_hex(plt.cm.tab10(i % 10)) for i in range(len(subgraph.nodes()))]
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, edge_color=edge_color)
    
    # Draw labels
    node_labels = {node: subgraph.nodes[node]['label'][:15] + '...' if len(subgraph.nodes[node]['label']) > 15 
                  else subgraph.nodes[node]['label'] for node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8, font_color=font_color)
    
    plt.title("Knowledge Graph Visualization", color=font_color)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Convert to base64 string
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode('utf-8')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/status')
def status():
    """Return the current processing status"""
    return jsonify(processing_status)

@app.route('/process', methods=['POST'])
def process():
    """Start the processing pipeline"""
    global processing_status
    
    if processing_status["status"] not in ["idle", "completed", "error"]:
        return jsonify({"success": False, "message": "Processing already in progress"})
    
    data = request.json
    pubmed_query = data.get('pubmed_query', '')
    arxiv_query = data.get('arxiv_query', '')
    max_papers = int(data.get('max_papers', 10))
    num_hypotheses = int(data.get('num_hypotheses', 5))
    
    if not pubmed_query and not arxiv_query:
        return jsonify({"success": False, "message": "Please provide at least one search query"})
    
    # Reset status
    processing_status = {"status": "starting", "message": "Starting process...", "progress": 5}
    
    # Start background thread
    thread = threading.Thread(target=background_process, args=(pubmed_query, arxiv_query, max_papers, num_hypotheses))
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "message": "Processing started"})

@app.route('/papers')
def get_papers():
    """Return the list of papers"""
    global papers
    
    # Create a simpler version for the UI
    simple_papers = []
    for paper in papers:
        simple_paper = {
            'id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'][:200] + '...' if len(paper['abstract']) > 200 else paper['abstract'],
            'source': paper['source'],
            'num_entities': len(paper.get('entities', [])),
            'num_claims': len(paper.get('claims', []))
        }
        simple_papers.append(simple_paper)
    
    return jsonify(simple_papers)

@app.route('/hypotheses')
def get_hypotheses():
    """Return the list of hypotheses"""
    global hypotheses
    
    # Create a simpler version for the UI
    simple_hypotheses = []
    for hyp in hypotheses:
        simple_hyp = {
            'text': hyp['text'],
            'similarity_score': round(hyp['similarity_score'], 2),
            'novelty_score': round(hyp['novelty_score'], 2),
            'plausibility_score': round(hyp['plausibility_score'], 2),
            'overall_score': round(hyp['overall_score'], 2),
            'supporting_papers': hyp['paper_ids']
        }
        simple_hypotheses.append(simple_hyp)
    
    return jsonify(simple_hypotheses)

@app.route('/hypothesis/<int:index>')
def get_hypothesis_details(index):
    """Return details for a specific hypothesis"""
    global generator, hypotheses
    
    if index < 0 or index >= len(hypotheses):
        return jsonify({"error": "Hypothesis index out of range"})
    
    hypothesis = hypotheses[index]
    explanation = generator.explain_hypothesis(hypothesis)
    
    return jsonify({
        "hypothesis": hypothesis['text'],
        "explanation": explanation,
        "scores": {
            "similarity": round(hypothesis['similarity_score'], 2),
            "novelty": round(hypothesis['novelty_score'], 2),
            "plausibility": round(hypothesis['plausibility_score'], 2),
            "overall": round(hypothesis['overall_score'], 2)
        }
    })

@app.route('/graph')
def get_graph():
    """Return the knowledge graph visualization"""
    global knowledge_graph_image
    
    dark_mode = request.args.get('dark_mode', 'false').lower() == 'true'
    
    # Generate new image if dark mode changed or no image exists
    if knowledge_graph_image is None or dark_mode != (plt.rcParams['axes.facecolor'] == '#121212'):
        knowledge_graph_image = generate_graph_image(dark_mode)
    
    return jsonify({"image": knowledge_graph_image})

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the system"""
    global processing_status, papers, hypotheses, knowledge_graph_image
    
    if processing_status["status"] not in ["idle", "completed", "error"]:
        return jsonify({"success": False, "message": "Cannot reset while processing"})
    
    # Reset all data
    papers = []
    hypotheses = []
    knowledge_graph_image = None
    processing_status = {"status": "idle", "message": "System reset", "progress": 0}
    
    return jsonify({"success": True, "message": "System reset successfully"})

if __name__ == '__main__':
    app.run(debug=True)
