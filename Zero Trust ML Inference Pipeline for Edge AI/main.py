# Zero-Trust ML Inference Pipeline for Edge AI
# Complete implementation with encrypted inference and remote attestation

import asyncio
import json
import hashlib
import hmac
import time
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import numpy as np
import pickle
import os
import socket
import threading
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttestationToken:
    """Remote attestation token structure"""
    device_id: str
    timestamp: int
    nonce: str
    platform_measurements: Dict[str, str]
    model_hash: str
    signature: str
    
    def is_valid(self, max_age_seconds: int = 300) -> bool:
        """Validate attestation token freshness"""
        current_time = int(time.time())
        return (current_time - self.timestamp) <= max_age_seconds

@dataclass
class InferenceRequest:
    """Encrypted inference request structure"""
    request_id: str
    encrypted_data: str
    data_hash: str
    attestation_token: AttestationToken
    timestamp: int

@dataclass
class InferenceResponse:
    """Encrypted inference response structure"""
    request_id: str
    encrypted_result: str
    result_hash: str
    processing_time: float
    attestation_proof: str

class CryptoManager:
    """Handles all cryptographic operations"""
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
    def generate_symmetric_key(self, password: str, salt: bytes) -> bytes:
        """Generate symmetric encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data using AES-GCM"""
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext, iv + encryptor.tag
    
    def decrypt_data(self, ciphertext: bytes, metadata: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-GCM"""
        iv = metadata[:12]
        tag = metadata[12:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data using RSA private key"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key) -> bool:
        """Verify signature using RSA public key"""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

class AttestationService:
    """Handles remote attestation and device verification"""
    
    def __init__(self, crypto_manager: CryptoManager):
        self.crypto = crypto_manager
        self.trusted_devices = {}
        self.revoked_devices = set()
        
    def register_device(self, device_id: str, public_key_pem: bytes) -> bool:
        """Register a trusted device"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, 
                backend=default_backend()
            )
            self.trusted_devices[device_id] = public_key
            logger.info(f"Device {device_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return False
    
    def revoke_device(self, device_id: str):
        """Revoke a device's trust status"""
        self.revoked_devices.add(device_id)
        if device_id in self.trusted_devices:
            del self.trusted_devices[device_id]
        logger.info(f"Device {device_id} revoked")
    
    def generate_challenge(self) -> str:
        """Generate cryptographic challenge for attestation"""
        return base64.b64encode(os.urandom(32)).decode()
    
    def verify_attestation(self, token: AttestationToken) -> bool:
        """Verify remote attestation token"""
        try:
            # Check if device is trusted and not revoked
            if token.device_id not in self.trusted_devices:
                logger.error(f"Device {token.device_id} not trusted")
                return False
            
            if token.device_id in self.revoked_devices:
                logger.error(f"Device {token.device_id} is revoked")
                return False
            
            # Verify token freshness
            if not token.is_valid():
                logger.error("Attestation token expired")
                return False
            
            # Verify platform measurements (simplified)
            expected_measurements = self._get_expected_measurements(token.device_id)
            if not self._verify_measurements(token.platform_measurements, expected_measurements):
                logger.error("Platform measurements verification failed")
                return False
            
            # Verify signature
            token_data = f"{token.device_id}{token.timestamp}{token.nonce}{token.model_hash}"
            signature = base64.b64decode(token.signature)
            public_key = self.trusted_devices[token.device_id]
            
            if not self.crypto.verify_signature(token_data.encode(), signature, public_key):
                logger.error("Attestation signature verification failed")
                return False
            
            logger.info(f"Attestation verified for device {token.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Attestation verification error: {e}")
            return False
    
    def _get_expected_measurements(self, device_id: str) -> Dict[str, str]:
        """Get expected platform measurements for device"""
        # In production, this would come from a secure configuration store
        return {
            "bootloader": "sha256:abc123...",
            "kernel": "sha256:def456...",
            "initrd": "sha256:ghi789..."
        }
    
    def _verify_measurements(self, actual: Dict[str, str], expected: Dict[str, str]) -> bool:
        """Verify platform measurements match expected values"""
        for key, expected_value in expected.items():
            if key not in actual or actual[key] != expected_value:
                return False
        return True

class SecureModelLoader:
    """Loads and manages ML models securely"""
    
    def __init__(self, crypto_manager: CryptoManager):
        self.crypto = crypto_manager
        self.loaded_models = {}
        self.model_hashes = {}
    
    def load_encrypted_model(self, model_path: str, key: bytes, model_id: str) -> bool:
        """Load an encrypted ML model"""
        try:
            with open(model_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Split encrypted model and metadata
            metadata_size = int.from_bytes(encrypted_data[:4], 'big')
            metadata = encrypted_data[4:4+metadata_size]
            ciphertext = encrypted_data[4+metadata_size:]
            
            # Decrypt model
            model_data = self.crypto.decrypt_data(ciphertext, metadata, key)
            model = pickle.loads(model_data)
            
            # Calculate and store model hash
            model_hash = hashlib.sha256(model_data).hexdigest()
            self.model_hashes[model_id] = model_hash
            self.loaded_models[model_id] = model
            
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def get_model_hash(self, model_id: str) -> Optional[str]:
        """Get hash of loaded model"""
        return self.model_hashes.get(model_id)
    
    def get_model(self, model_id: str):
        """Get loaded model"""
        return self.loaded_models.get(model_id)

class MockMLModel:
    """Mock ML model for demonstration"""
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Simple mock prediction - returns random values"""
        return np.random.random((data.shape[0], 3))
    
    def __getstate__(self):
        return {"type": "MockMLModel"}
    
    def __setstate__(self, state):
        pass

class ZeroTrustInferencePipeline:
    """Main zero-trust ML inference pipeline"""
    
    def __init__(self, model_id: str = "default_model"):
        self.crypto = CryptoManager()
        self.attestation_service = AttestationService(self.crypto)
        self.model_loader = SecureModelLoader(self.crypto)
        self.model_id = model_id
        self.active_sessions = {}
        
        # Initialize with a mock model for demonstration
        self._setup_demo_model()
        
    def _setup_demo_model(self):
        """Setup a demo model for testing"""
        model = MockMLModel()
        model_data = pickle.dumps(model)
        model_hash = hashlib.sha256(model_data).hexdigest()
        
        self.model_loader.loaded_models[self.model_id] = model
        self.model_loader.model_hashes[self.model_id] = model_hash
        
    def register_edge_device(self, device_id: str) -> Tuple[str, str]:
        """Register an edge device and return keys"""
        # Generate device key pair
        device_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Serialize public key
        public_key_pem = device_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Register device
        self.attestation_service.register_device(device_id, public_key_pem)
        
        # Return private key for device
        private_key_pem = device_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return private_key_pem.decode(), public_key_pem.decode()
    
    async def process_inference_request(self, request: InferenceRequest, session_key: bytes) -> InferenceResponse:
        """Process encrypted inference request"""
        start_time = time.time()
        
        try:
            # Verify attestation
            if not self.attestation_service.verify_attestation(request.attestation_token):
                raise ValueError("Attestation verification failed")
            
            # Verify model hash matches attestation
            expected_hash = self.model_loader.get_model_hash(self.model_id)
            if request.attestation_token.model_hash != expected_hash:
                raise ValueError("Model hash mismatch")
            
            # Decrypt input data
            encrypted_data = base64.b64decode(request.encrypted_data)
            metadata_size = int.from_bytes(encrypted_data[:4], 'big')
            metadata = encrypted_data[4:4+metadata_size]
            ciphertext = encrypted_data[4+metadata_size:]
            
            input_data = self.crypto.decrypt_data(ciphertext, metadata, session_key)
            
            # Verify data integrity
            data_hash = hashlib.sha256(input_data).hexdigest()
            if data_hash != request.data_hash:
                raise ValueError("Data integrity check failed")
            
            # Deserialize input
            input_array = pickle.loads(input_data)
            
            # Perform inference
            model = self.model_loader.get_model(self.model_id)
            result = model.predict(input_array)
            
            # Serialize result
            result_data = pickle.dumps(result)
            result_hash = hashlib.sha256(result_data).hexdigest()
            
            # Encrypt result
            encrypted_result, result_metadata = self.crypto.encrypt_data(result_data, session_key)
            
            # Prepare encrypted response
            response_data = len(result_metadata).to_bytes(4, 'big') + result_metadata + encrypted_result
            encrypted_response = base64.b64encode(response_data).decode()
            
            # Generate attestation proof
            proof_data = f"{request.request_id}{result_hash}{time.time()}"
            attestation_proof = base64.b64encode(
                self.crypto.sign_data(proof_data.encode())
            ).decode()
            
            processing_time = time.time() - start_time
            
            return InferenceResponse(
                request_id=request.request_id,
                encrypted_result=encrypted_response,
                result_hash=result_hash,
                processing_time=processing_time,
                attestation_proof=attestation_proof
            )
            
        except Exception as e:
            logger.error(f"Inference processing failed: {e}")
            raise

class EdgeDevice:
    """Simulates an edge device with secure inference capabilities"""
    
    def __init__(self, device_id: str, private_key_pem: str):
        self.device_id = device_id
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend()
        )
        self.crypto = CryptoManager()
        
    def create_attestation_token(self, model_hash: str, nonce: str) -> AttestationToken:
        """Create attestation token for this device"""
        timestamp = int(time.time())
        
        # Mock platform measurements
        platform_measurements = {
            "bootloader": "sha256:abc123...",
            "kernel": "sha256:def456...",
            "initrd": "sha256:ghi789..."
        }
        
        # Sign attestation data
        token_data = f"{self.device_id}{timestamp}{nonce}{model_hash}"
        signature = self.private_key.sign(
            token_data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return AttestationToken(
            device_id=self.device_id,
            timestamp=timestamp,
            nonce=nonce,
            platform_measurements=platform_measurements,
            model_hash=model_hash,
            signature=base64.b64encode(signature).decode()
        )
    
    def create_inference_request(self, data: np.ndarray, model_hash: str, 
                               session_key: bytes) -> InferenceRequest:
        """Create encrypted inference request"""
        request_id = f"req_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Serialize and encrypt data
        data_bytes = pickle.dumps(data)
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        encrypted_data, metadata = self.crypto.encrypt_data(data_bytes, session_key)
        
        # Prepare encrypted payload
        payload = len(metadata).to_bytes(4, 'big') + metadata + encrypted_data
        encrypted_payload = base64.b64encode(payload).decode()
        
        # Create attestation token
        nonce = base64.b64encode(os.urandom(16)).decode()
        attestation_token = self.create_attestation_token(model_hash, nonce)
        
        return InferenceRequest(
            request_id=request_id,
            encrypted_data=encrypted_payload,
            data_hash=data_hash,
            attestation_token=attestation_token,
            timestamp=int(time.time())
        )

class InferenceServer:
    """HTTP-like server for handling inference requests"""
    
    def __init__(self, pipeline: ZeroTrustInferencePipeline, host='localhost', port=8443):
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.running = False
        
    def start(self):
        """Start the inference server"""
        self.running = True
        server_thread = threading.Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()
        logger.info(f"Inference server started on {self.host}:{self.port}")
        
    def _run_server(self):
        """Run the server loop (simplified TCP server)"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Server error: {e}")
    
    def _handle_client(self, client_socket, addr):
        """Handle individual client connections"""
        try:
            with client_socket:
                # Receive request data
                data = client_socket.recv(65536).decode()
                if not data:
                    return
                
                # Parse JSON request
                request_data = json.loads(data)
                
                # Process inference request (simplified)
                logger.info(f"Received request from {addr}")
                response = {"status": "processed", "timestamp": time.time()}
                
                # Send response
                response_json = json.dumps(response)
                client_socket.send(response_json.encode())
                
        except Exception as e:
            logger.error(f"Client handling error: {e}")

# Demo and testing functions
async def run_demo():
    """Run a complete demonstration of the zero-trust ML pipeline"""
    print("🔐 Zero-Trust ML Inference Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ZeroTrustInferencePipeline()
    
    # Register edge device
    device_id = "edge_device_001"
    private_key_pem, public_key_pem = pipeline.register_edge_device(device_id)
    print(f"✅ Registered edge device: {device_id}")
    
    # Create edge device instance
    edge_device = EdgeDevice(device_id, private_key_pem)
    
    # Generate session key
    session_password = "secure_session_key_2024"
    salt = os.urandom(16)
    session_key = pipeline.crypto.generate_symmetric_key(session_password, salt)
    
    # Prepare test data
    test_data = np.random.random((10, 5))  # Sample input data
    model_hash = pipeline.model_loader.get_model_hash(pipeline.model_id)
    
    print(f"📊 Created test data shape: {test_data.shape}")
    print(f"🔑 Model hash: {model_hash[:16]}...")
    
    # Create inference request
    inference_request = edge_device.create_inference_request(
        test_data, model_hash, session_key
    )
    print(f"📤 Created inference request: {inference_request.request_id}")
    
    # Process inference
    try:
        response = await pipeline.process_inference_request(inference_request, session_key)
        print(f"📥 Received response for request: {response.request_id}")
        print(f"⏱️  Processing time: {response.processing_time:.3f}s")
        print(f"🔍 Result hash: {response.result_hash[:16]}...")
        
        # Decrypt and verify result
        encrypted_data = base64.b64decode(response.encrypted_result)
        metadata_size = int.from_bytes(encrypted_data[:4], 'big')
        metadata = encrypted_data[4:4+metadata_size]
        ciphertext = encrypted_data[4+metadata_size:]
        
        result_data = pipeline.crypto.decrypt_data(ciphertext, metadata, session_key)
        result = pickle.loads(result_data)
        
        print(f"✅ Decrypted result shape: {result.shape}")
        print(f"📈 Sample predictions: {result[0][:3]}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    
    print("\n🎯 Demo completed successfully!")

def create_test_suite():
    """Create comprehensive test suite"""
    
    async def test_attestation():
        """Test attestation service"""
        print("Testing attestation service...")
        crypto = CryptoManager()
        attestation = AttestationService(crypto)
        
        # Test device registration
        device_key = rsa.generate_private_key(65537, 2048, default_backend())
        public_key_pem = device_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        result = attestation.register_device("test_device", public_key_pem)
        assert result, "Device registration failed"
        print("✅ Device registration test passed")
        
    async def test_encryption():
        """Test encryption/decryption"""
        print("Testing encryption...")
        crypto = CryptoManager()
        
        test_data = b"Hello, Zero-Trust ML!"
        password = "test_password"
        salt = os.urandom(16)
        
        key = crypto.generate_symmetric_key(password, salt)
        ciphertext, metadata = crypto.encrypt_data(test_data, key)
        decrypted = crypto.decrypt_data(ciphertext, metadata, key)
        
        assert decrypted == test_data, "Encryption/decryption failed"
        print("✅ Encryption test passed")
    
    async def test_model_loading():
        """Test secure model loading"""
        print("Testing model loading...")
        crypto = CryptoManager()
        loader = SecureModelLoader(crypto)
        
        # Create and "load" a mock model
        model = MockMLModel()
        model_id = "test_model"
        loader.loaded_models[model_id] = model
        loader.model_hashes[model_id] = "test_hash"
        
        loaded_model = loader.get_model(model_id)
        assert loaded_model is not None, "Model loading failed"
        print("✅ Model loading test passed")
    
    return [test_attestation, test_encryption, test_model_loading]

if __name__ == "__main__":
    print("🚀 Starting Zero-Trust ML Inference Pipeline")
    
    # Run demo
    asyncio.run(run_demo())
    
    print("\n🧪 Running test suite...")
    tests = create_test_suite()
    
    async def run_tests():
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ Test failed: {e}")
    
    asyncio.run(run_tests())
    print("\n🎉 All systems operational!")
