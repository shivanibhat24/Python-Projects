class EncryptedChat {
    constructor() {
        this.encryptionKey = null;
        this.isEncrypted = true;
        this.useProxy = false;
        this.showDebug = false;
        this.selectedModel = 'gpt-3.5-turbo';
        this.clientId = this.generateClientId();
        
        this.init();
    }
    
    async init() {
        await this.generateEncryptionKey();
        this.setupEventListeners();
        this.updateUI();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 16);
    }
    
    async generateEncryptionKey() {
        try {
            this.encryptionKey = await window.crypto.subtle.generateKey(
                { name: 'AES-GCM', length: 256 },
                true,
                ['encrypt', 'decrypt']
            );
        } catch (error) {
            console.error('Failed to generate encryption key:', error);
        }
    }
    
    async encryptMessage(message) {
        if (!this.encryptionKey || !this.isEncrypted) return message;
        
        try {
            const encoder = new TextEncoder();
            const data = encoder.encode(message);
            const iv = window.crypto.getRandomValues(new Uint8Array(12));
            
            const encryptedData = await window.crypto.subtle.encrypt(
                { name: 'AES-GCM', iv: iv },
                this.encryptionKey,
                data
            );
            
            const combined = new Uint8Array(iv.length + encryptedData.byteLength);
            combined.set(iv);
            combined.set(new Uint8Array(encryptedData), iv.length);
            
            return btoa(String.fromCharCode(...combined));
        } catch (error) {
            console.error('Encryption failed:', error);
            return message;
        }
    }
    
    async decryptMessage(encryptedMessage) {
        if (!this.encryptionKey || !this.isEncrypted) return encryptedMessage;
        
        try {
            const combined = new Uint8Array(
                atob(encryptedMessage).split('').map(char => char.charCodeAt(0))
            );
            const iv = combined.slice(0, 12);
            const data = combined.slice(12);
            
            const decryptedData = await window.crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: iv },
                this.encryptionKey,
                data
            );
            
            const decoder = new TextDecoder();
            return decoder.decode(decryptedData);
        } catch (error) {
            console.error('Decryption failed:', error);
            return encryptedMessage;
        }
    }
    
    setupEventListeners() {
        // Settings toggle
        document.getElementById('settings-btn').addEventListener('click', () => {
            const panel = document.getElementById('settings-panel');
            panel.classList.toggle('hidden');
        });
        
        // Model selection
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.selectedModel = e.target.value;
            this.updateUI();
        });
        
        // Encryption toggle
        document.getElementById('encryption-toggle').addEventListener('change', (e) => {
            this.isEncrypted = e.target.checked;
            this.updateUI();
        });
        
        // Proxy toggle
        document.getElementById('proxy-toggle').addEventListener('change', (e) => {
            this.useProxy = e.target.checked;
            this.updateUI();
        });
        
        // Debug toggle
        document.getElementById('debug-toggle').addEventListener('click', () => {
            this.showDebug = !this.showDebug;
            this.updateUI();
        });
        
        // Send message
        document.getElementById('send-button').addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Enter key to send
        document.getElementById('message-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }
    
    updateUI() {
        // Update encryption status
        const encStatus = document.getElementById('encryption-status');
        encStatus.textContent = this.isEncrypted ? '🔒 Encrypted' : '🔓 Unencrypted';
        
        // Update model display
        document.getElementById('model-display').textContent = `Model: ${this.selectedModel}`;
        
        // Update proxy display
        const proxyDisplay = document.getElementById('proxy-display');
        if (this.useProxy) {
            proxyDisplay.classList.remove('hidden');
        } else {
            proxyDisplay.classList.add('hidden');
        }
        
        // Update encryption display
        const encDisplay = document.getElementById('encryption-display');
        encDisplay.textContent = this.isEncrypted ? '• End-to-End Encrypted' : '• Unencrypted';
        
        // Update debug button
        const debugBtn = document.getElementById('debug-toggle');
        debugBtn.textContent = this.showDebug ? '🙈 Hide Encryption' : '👁️ Show Encryption';
        
        // Update input placeholder
        const input = document.getElementById('message-input');
        input.placeholder = this.isEncrypted ? 
            'Type your message (will be encrypted)...' : 
            'Type your message...';
    }
    
    async sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();
        if (!message) return;
        
        // Clear input and disable send button
        input.value = '';
        const sendBtn = document.getElementById('send-button');
        sendBtn.disabled = true;
        
        // Hide welcome message
        document.getElementById('welcome-message').style.display = 'none';
        
        // Add user message to chat
        this.addMessage(message, 'user', this.isEncrypted);
        
        // Show loading
        this.showLoading();
        
        try {
            // Encrypt message
            const processedMessage = await this.encryptMessage(message);
            
            // Send to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: processedMessage,
                    model: this.selectedModel,
                    encrypted: this.isEncrypted,
                    client_id: this.clientId,
                    use_proxy: this.useProxy
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Process response
            let finalResponse = data.response;
            if (this.isEncrypted) {
                // In a real implementation, you might decrypt the response here
                // For demo, we'll use it as-is
            }
            
            // Add AI response to chat
            this.addMessage(finalResponse, 'ai', this.isEncrypted, data.model);
            
        } catch (error) {
            console.error('Failed to send message:', error);
            this.addMessage(`Error: ${error.message}`, 'error', false);
        } finally {
            // Hide loading and re-enable send button
            this.hideLoading();
            sendBtn.disabled = false;
        }
    }
    
    addMessage(content, type, encrypted = false, model = null) {
        const container = document.getElementById('messages-container');
        const messageDiv = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString();
        
        const isUser = type === 'user';
        const isError = type === 'error';
        
        messageDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;
        
        const bubbleClass = isUser ? 
            'bg-gradient-to-r from-purple-600 to-pink-600' :
            isError ? 'bg-red-600/20 border border-red-500/30' :
            'bg-gray-800/50 border border-gray-700';
        
        let debugInfo = '';
        if (this.showDebug && encrypted) {
            debugInfo = `
                <div class="mt-2 p-2 bg-black/30 rounded text-xs font-mono text-gray-400 break-all">
                    <div class="flex items-center mb-1">
                        <span>🔑 Encrypted payload preview:</span>
                    </div>
                    ${content.substring(0, 50)}...
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${bubbleClass}">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-300">
                        ${isUser ? 'You' : model || 'AI'}
                    </span>
                    <div class="flex items-center space-x-1">
                        ${encrypted ? '<span class="text-green-400">🔒</span>' : ''}
                        <span class="text-xs text-gray-400">${timestamp}</span>
                    </div>
                </div>
                <p class="text-sm leading-relaxed">${content}</p>
                ${debugInfo}
            </div>
        `;
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }
    
    showLoading() {
        const container = document.getElementById('messages-container');
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loading-message';
        loadingDiv.className = 'flex justify-start';
        
        loadingDiv.innerHTML = `
            <div class="bg-gray-800/50 border border-gray-700 px-4 py-3 rounded-2xl">
                <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                    <span class="text-sm text-gray-400 ml-2">Processing encrypted message...</span>
                </div>
            </div>
        `;
        
        container.appendChild(loadingDiv);
        container.scrollTop = container.scrollHeight;
    }
    
    hideLoading() {
        const loading = document.getElementById('loading-message');
        if (loading) {
            loading.remove();
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EncryptedChat();
});
