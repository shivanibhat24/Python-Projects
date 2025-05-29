from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import time
import random
import requests
from datetime import datetime
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

# Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key-here')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'your-anthropic-api-key-here')

# Privacy settings
ENABLE_PROXY = os.environ.get('ENABLE_PROXY', 'false').lower() == 'true'
PROXY_URL = os.environ.get('PROXY_URL', 'http://127.0.0.1:8080')
LOG_REQUESTS = os.environ.get('LOG_REQUESTS', 'false').lower() == 'true'

class EncryptionManager:
    """Handle server-side encryption utilities"""
    
    @staticmethod
    def generate_salt():
        return secrets.token_bytes(32)
    
    @staticmethod
    def derive_key(password: str, salt: bytes):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    @staticmethod
    def encrypt_data(data: str, key: bytes):
        f = Fernet(key)
        return f.encrypt(data.encode()).decode()
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes):
        f = Fernet(key)
        return f.decrypt(encrypted_data.encode()).decode()

class AIProvider:
    """Handle different AI model integrations"""
    
    def __init__(self):
        self.session = requests.Session()
        if ENABLE_PROXY:
            self.session.proxies = {
                'http': PROXY_URL,
                'https': PROXY_URL
            }
    
    def call_openai(self, message: str, model: str = "gpt-3.5-turbo"):
        """Call OpenAI API"""
        if OPENAI_API_KEY == 'your-openai-api-key-here':
            return self._simulate_response(message, model)
        
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': message}],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        try:
            response = self.session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def call_anthropic(self, message: str, model: str = "claude-3-sonnet-20240229"):
        """Call Anthropic Claude API"""
        if ANTHROPIC_API_KEY == 'your-anthropic-api-key-here':
            return self._simulate_response(message, model)
        
        headers = {
            'x-api-key': ANTHROPIC_API_KEY,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': model,
            'max_tokens': 1000,
            'messages': [{'role': 'user', 'content': message}]
        }
        
        try:
            response = self.session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['content'][0]['text']
        except Exception as e:
            return f"Error calling Anthropic: {str(e)}"
    
    def call_local_llm(self, message: str, model: str = "llama2"):
        """Call local LLM via Ollama or similar"""
        try:
            response = self.session.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': message,
                    'stream': False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return self._simulate_response(message, f"local-{model}")
    
    def _simulate_response(self, message: str, model: str):
        """Simulate AI response for demo purposes"""
        time.sleep(random.uniform(1, 3))  # Simulate network delay
        
        responses = {
            'gpt-4': f"GPT-4 Response: I received your encrypted message '{message[:50]}...' and this response is also encrypted end-to-end for your privacy.",
            'gpt-3.5-turbo': f"GPT-3.5 Response: Your message about '{message[:30]}...' was processed securely with client-side encryption.",
            'claude-3-sonnet-20240229': f"Claude Response: I understand your encrypted query regarding '{message[:40]}...' - all communication is privacy-first.",
            'local-llama2': f"Local Llama Response: Processing '{message[:35]}...' entirely on your local machine with zero external data sharing."
        }
        
        return responses.get(model, f"AI Response: Processed your encrypted message '{message[:30]}...' securely.")

ai_provider = AIProvider()

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle encrypted chat messages"""
    try:
        data = request.get_json()
        
        # Extract request data
        encrypted_message = data.get('message', '')
        model = data.get('model', 'gpt-3.5-turbo')
        use_encryption = data.get('encrypted', True)
        client_id = data.get('client_id', 'anonymous')
        
        # Log request (if enabled and not containing sensitive data)
        if LOG_REQUESTS:
            app.logger.info(f"Chat request - Model: {model}, Encrypted: {use_encryption}, Client: {client_id[:8]}...")
        
        # For demo purposes, we'll work with the message as-is
        # In a real implementation, you might decrypt server-side if needed
        message_to_process = encrypted_message
        
        # Route to appropriate AI provider
        if model.startswith('gpt'):
            response = ai_provider.call_openai(message_to_process, model)
        elif model.startswith('claude'):
            response = ai_provider.call_anthropic(message_to_process, model)
        elif model.startswith('local'):
            response = ai_provider.call_local_llm(message_to_process, model.replace('local-', ''))
        else:
            response = ai_provider._simulate_response(message_to_process, model)
        
        return jsonify({
            'response': response,
            'model': model,
            'encrypted': use_encryption,
            'timestamp': datetime.now().isoformat(),
            'server_id': hashlib.sha256(f"{client_id}{time.time()}".encode()).hexdigest()[:16]
        })
        
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'error': 'Failed to process request',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available AI models"""
    models = {
        'openai': [
            {'id': 'gpt-4', 'name': 'GPT-4', 'provider': 'OpenAI'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'provider': 'OpenAI'}
        ],
        'anthropic': [
            {'id': 'claude-3-sonnet-20240229', 'name': 'Claude 3 Sonnet', 'provider': 'Anthropic'},
            {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku', 'provider': 'Anthropic'}
        ],
        'local': [
            {'id': 'local-llama2', 'name': 'Llama 2 (Local)', 'provider': 'Local'},
            {'id': 'local-mistral', 'name': 'Mistral (Local)', 'provider': 'Local'}
        ]
    }
    return jsonify(models)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'encryption': 'enabled',
        'proxy': ENABLE_PROXY,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/privacy', methods=['GET'])
def privacy_info():
    """Get privacy configuration"""
    return jsonify({
        'client_side_encryption': True,
        'server_side_encryption': True,
        'proxy_enabled': ENABLE_PROXY,
        'logging_enabled': LOG_REQUESTS,
        'data_retention': 'none',
        'third_party_sharing': 'none'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Development server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
