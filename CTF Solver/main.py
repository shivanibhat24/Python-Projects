#!/usr/bin/env python3
"""
CTF Challenge Solver - Machine Learning Approach
A tool to assist in solving various types of CTF challenges using machine learning techniques.
"""

import argparse
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
import binascii
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import joblib
import logging
from PIL import Image
import pytesseract
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTFSolver:
    def __init__(self, model_dir="models"):
        """Initialize the CTF solver with various ML models"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.cipher_classifier = None
        self.stego_detector = None
        self.binary_classifier = None
        self.web_vuln_detector = None
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists(f"{self.model_dir}/cipher_classifier.pkl"):
                self.cipher_classifier = joblib.load(f"{self.model_dir}/cipher_classifier.pkl")
                logger.info("Loaded cipher classification model")
            
            if os.path.exists(f"{self.model_dir}/stego_detector.h5"):
                self.stego_detector = load_model(f"{self.model_dir}/stego_detector.h5")
                logger.info("Loaded steganography detection model")
            
            if os.path.exists(f"{self.model_dir}/binary_classifier.pkl"):
                self.binary_classifier = joblib.load(f"{self.model_dir}/binary_classifier.pkl")
                logger.info("Loaded binary vulnerability classification model")
            
            if os.path.exists(f"{self.model_dir}/web_vuln_detector.pkl"):
                self.web_vuln_detector = joblib.load(f"{self.model_dir}/web_vuln_detector.pkl")
                logger.info("Loaded web vulnerability detection model")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def train_cipher_classifier(self, training_data, labels):
        """Train a classifier to identify cipher types from encrypted text"""
        logger.info("Training cipher classification model...")
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
        X = vectorizer.fit_transform(training_data)
        
        self.cipher_classifier = RandomForestClassifier(n_estimators=100)
        self.cipher_classifier.fit(X, labels)
        
        # Save the vectorizer with the model
        joblib.dump((self.cipher_classifier, vectorizer), f"{self.model_dir}/cipher_classifier.pkl")
        logger.info("Cipher classification model trained and saved")
    
    def identify_cipher(self, encrypted_text):
        """Identify the likely cipher used for the encrypted text"""
        if self.cipher_classifier is None:
            logger.error("Cipher classifier model not loaded or trained")
            return None
        
        # Load the model and vectorizer
        classifier, vectorizer = joblib.load(f"{self.model_dir}/cipher_classifier.pkl")
        
        # Transform the input text
        X = vectorizer.transform([encrypted_text])
        
        # Predict cipher type and return probabilities
        cipher_type = classifier.predict(X)[0]
        probs = classifier.predict_proba(X)[0]
        
        # Get top 3 most likely cipher types
        top_indices = np.argsort(probs)[::-1][:3]
        results = [(classifier.classes_[i], probs[i]) for i in top_indices]
        
        return {
            "most_likely": cipher_type,
            "probabilities": results
        }
    
    def solve_crypto_challenge(self, data, cipher_type=None):
        """Attempt to solve a cryptographic challenge"""
        if cipher_type is None and self.cipher_classifier is not None:
            cipher_info = self.identify_cipher(data)
            cipher_type = cipher_info["most_likely"]
            logger.info(f"Detected cipher type: {cipher_type}")
        
        results = {}
        
        # Try simple decodings
        try:
            results["base64"] = base64.b64decode(data.encode()).decode('utf-8', errors='ignore')
        except:
            results["base64"] = "Failed to decode"
        
        try:
            results["hex"] = bytes.fromhex(data).decode('utf-8', errors='ignore')
        except:
            results["hex"] = "Failed to decode"
        
        try:
            results["binary"] = ''.join(chr(int(data[i:i+8], 2)) for i in range(0, len(data), 8))
        except:
            results["binary"] = "Failed to decode"
        
        # Caesar cipher brute force
        caesar_results = []
        if cipher_type == "caesar" or cipher_type is None:
            for shift in range(26):
                result = ''.join(chr((ord(c) - ord('a') + shift) % 26 + ord('a')) if c.islower() else 
                               (chr((ord(c) - ord('A') + shift) % 26 + ord('A')) if c.isupper() else c) 
                               for c in data)
                caesar_results.append((shift, result))
            results["caesar"] = caesar_results
        
        # XOR brute force (basic)
        if cipher_type == "xor" or cipher_type is None:
            xor_results = []
            try:
                data_bytes = data.encode('utf-8')
                for key in range(256):
                    result = bytes([b ^ key for b in data_bytes])
                    xor_results.append((key, result.decode('utf-8', errors='ignore')))
                results["xor"] = xor_results[:5]  # Show only top 5 possibilities
            except:
                results["xor"] = "Failed to process XOR"
        
        return results
    
    def train_stego_detector(self, image_paths, labels):
        """Train a model to detect steganography in images"""
        logger.info("Training steganography detection model...")
        
        # Create a CNN model for steganography detection
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # Binary classification: stego vs clean
        ])
        
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        # Load and preprocess images
        X = []
        y = []
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                img = img.resize((100, 100))
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(labels[i])
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
        
        X = np.array(X)
        y = to_categorical(np.array(y), num_classes=2)
        
        # Train the model
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        
        # Save the model
        model.save(f"{self.model_dir}/stego_detector.h5")
        self.stego_detector = model
        logger.info("Steganography detection model trained and saved")
    
    def analyze_image(self, image_path):
        """Analyze an image for hidden data or steganography"""
        results = {}
        
        # Check if the image exists
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Extract EXIF data
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = image._getexif()
                for tag, value in exif.items():
                    exif_data[tag] = str(value)
            results["exif"] = exif_data
            
            # Run OCR to extract any visible text
            try:
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    results["ocr_text"] = ocr_text
            except:
                results["ocr_text"] = "OCR processing failed"
            
            # Check for steganography if model is loaded
            if self.stego_detector is not None:
                # Preprocess the image
                img = image.resize((100, 100))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                
                # Predict
                prediction = self.stego_detector.predict(img_array)[0]
                stego_probability = prediction[1]  # Assuming index 1 is the stego class
                results["stego_probability"] = float(stego_probability)
                results["has_steganography"] = stego_probability > 0.7
            
            # Basic LSB check (simplified)
            try:
                pixels = list(image.getdata())
                lsb_analysis = self._analyze_lsb(pixels[:1000])  # Analyze first 1000 pixels
                results["lsb_analysis"] = lsb_analysis
            except:
                results["lsb_analysis"] = "LSB analysis failed"
            
        except Exception as e:
            results["error"] = f"Error analyzing image: {str(e)}"
        
        return results
    
    def _analyze_lsb(self, pixels):
        """Analyze least significant bits of image pixels for potential steganography"""
        # Simple analysis - just count 0s and 1s in LSBs
        lsb_0s = 0
        lsb_1s = 0
        
        for pixel in pixels:
            if isinstance(pixel, tuple):
                for value in pixel:
                    if value % 2 == 0:
                        lsb_0s += 1
                    else:
                        lsb_1s += 1
        
        ratio = lsb_1s / (lsb_0s + lsb_1s) if (lsb_0s + lsb_1s) > 0 else 0
        
        # If ratio is close to 0.5, it might be random data which could indicate steganography
        # If it's significantly different, it might be a normal image
        return {
            "lsb_0s": lsb_0s,
            "lsb_1s": lsb_1s,
            "ratio": ratio,
            "potential_hidden_data": 0.45 < ratio < 0.55
        }
    
    def train_binary_classifier(self, binary_samples, labels):
        """Train a model to detect vulnerability patterns in binary files"""
        logger.info("Training binary vulnerability classification model...")
        
        # Extract features from binary files
        X = []
        for sample in binary_samples:
            features = self._extract_binary_features(sample)
            X.append(features)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Train a random forest classifier
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X, y)
        
        # Save the model
        joblib.dump(classifier, f"{self.model_dir}/binary_classifier.pkl")
        self.binary_classifier = classifier
        logger.info("Binary vulnerability classification model trained and saved")
    
    def _extract_binary_features(self, binary_data):
        """Extract features from binary data for vulnerability detection"""
        # This is a simplified version - real implementation would be more complex
        features = []
        
        # Feature 1: Entropy
        entropy = self._calculate_entropy(binary_data)
        features.append(entropy)
        
        # Feature 2: Executable sections count
        exec_sections = binary_data.count(b"\x55\x8B\xEC")  # x86 function prologue
        features.append(exec_sections)
        
        # Feature 3: Potential buffer operations
        buffer_ops = binary_data.count(b"strcpy") + binary_data.count(b"strcat")
        features.append(buffer_ops)
        
        # Feature 4: Potential format string vulnerabilities
        format_strings = binary_data.count(b"printf") + binary_data.count(b"scanf")
        features.append(format_strings)
        
        return features
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        entropy = 0
        for x in range(256):
            p_x = data.count(bytes([x])) / len(data)
            if p_x > 0:
                entropy += -p_x * np.log2(p_x)
        
        return entropy
    
    def analyze_binary(self, binary_path):
        """Analyze a binary file for potential vulnerabilities"""
        if not os.path.exists(binary_path):
            return {"error": "Binary file not found"}
        
        results = {}
        
        try:
            # Read the binary file
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Basic analysis
            results["file_size"] = len(binary_data)
            results["entropy"] = self._calculate_entropy(binary_data)
            
            # Check for common function signatures
            signatures = {
                "strcpy": binary_data.count(b"strcpy"),
                "strcat": binary_data.count(b"strcat"),
                "gets": binary_data.count(b"gets"),
                "system": binary_data.count(b"system"),
                "exec": binary_data.count(b"exec"),
            }
            results["dangerous_functions"] = signatures
            
            # Check for hardcoded strings that might be interesting
            strings = re.findall(b'[\x20-\x7E]{5,}', binary_data)
            strings = [s.decode('utf-8', errors='ignore') for s in strings]
            results["strings"] = strings[:20]  # Limit to first 20 strings
            
            # Check for potential flag formats
            flag_patterns = [
                r'flag\{[^}]+\}',
                r'CTF\{[^}]+\}',
                r'key\{[^}]+\}'
            ]
            potential_flags = []
            for pattern in flag_patterns:
                for string in strings:
                    matches = re.findall(pattern, string, re.IGNORECASE)
                    potential_flags.extend(matches)
            
            if potential_flags:
                results["potential_flags"] = potential_flags
            
            # Use the trained model if available
            if self.binary_classifier is not None:
                features = self._extract_binary_features(binary_data)
                vulnerability_pred = self.binary_classifier.predict_proba([features])[0]
                results["vulnerability_probability"] = float(vulnerability_pred[1])  # Assuming index 1 is vulnerable class
            
        except Exception as e:
            results["error"] = f"Error analyzing binary: {str(e)}"
        
        return results
    
    def train_web_vuln_detector(self, request_samples, labels):
        """Train a model to detect web vulnerabilities in HTTP requests"""
        logger.info("Training web vulnerability detection model...")
        
        # Convert requests to feature vectors
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        X = vectorizer.fit_transform(request_samples)
        
        # Train a random forest classifier
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X, labels)
        
        # Save the model and vectorizer
        joblib.dump((classifier, vectorizer), f"{self.model_dir}/web_vuln_detector.pkl")
        self.web_vuln_detector = classifier
        logger.info("Web vulnerability detection model trained and saved")
    
    def analyze_web_request(self, request_text):
        """Analyze an HTTP request for potential vulnerabilities"""
        results = {}
        
        # Basic analysis for common web vulnerabilities
        vuln_patterns = {
            "sql_injection": [
                r"'\s*OR\s*'1'='1", 
                r"--\s", 
                r";\s*--",
                r"UNION\s+SELECT"
            ],
            "xss": [
                r"<script>", 
                r"onerror=", 
                r"javascript:",
                r"on\w+\s*="
            ],
            "directory_traversal": [
                r"\.\.\/",
                r"%2e%2e%2f",
                r"\.\.\\",
            ],
            "command_injection": [
                r";\s*\w+\s",
                r"\|\s*\w+",
                r"`\w+`",
            ]
        }
        
        detected_vulns = {}
        for vuln_type, patterns in vuln_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, request_text, re.IGNORECASE)
                if found:
                    matches.extend(found)
            
            if matches:
                detected_vulns[vuln_type] = matches
        
        results["detected_vulnerabilities"] = detected_vulns
        
        # Use the ML model if available
        if self.web_vuln_detector is not None:
            classifier, vectorizer = joblib.load(f"{self.model_dir}/web_vuln_detector.pkl")
            
            # Transform the request text
            X = vectorizer.transform([request_text])
            
            # Predict vulnerability types
            vuln_probs = classifier.predict_proba(X)[0]
            
            # Get top 3 most likely vulnerability types
            top_indices = np.argsort(vuln_probs)[::-1][:3]
            vuln_results = [(classifier.classes_[i], float(vuln_probs[i])) for i in top_indices]
            
            results["ml_vulnerability_analysis"] = vuln_results
        
        return results
    
    def generate_payloads(self, vuln_type, target_info=None):
        """Generate potential payloads based on vulnerability type"""
        payloads = {
            "sql_injection": [
                "' OR '1'='1",
                "' OR '1'='1' --",
                "1'; DROP TABLE users; --",
                "' UNION SELECT 1,2,3,4,5,6,7,8,9,10 --",
                "' UNION SELECT NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL --",
                "' UNION SELECT table_name,NULL FROM information_schema.tables --",
                "' OR 1=1 LIMIT 1 OFFSET 1 --"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "\"><script>alert('XSS')</script>",
                "';alert('XSS');//"
            ],
            "directory_traversal": [
                "../../../etc/passwd",
                "../../../../etc/shadow",
                "../../Windows/system.ini",
                "..%2f..%2f..%2fetc%2fpasswd",
                "..\\..\\..\\Windows\\system.ini"
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "`id`",
                "$(whoami)",
                "& dir",
                "&& whoami",
                "|| whoami"
            ]
        }
        
        if vuln_type in payloads:
            return payloads[vuln_type]
        else:
            return []

def main():
    parser = argparse.ArgumentParser(description="CTF Challenge Solver using Machine Learning")
    parser.add_argument("--mode", choices=["crypto", "stego", "binary", "web"], required=True, 
                        help="Challenge type to analyze")
    parser.add_argument("--file", help="Path to the file to analyze")
    parser.add_argument("--text", help="Text data to analyze")
    parser.add_argument("--train", action="store_true", help="Train a model instead of analyzing")
    parser.add_argument("--training-data", help="Path to training data (CSV format)")
    parser.add_argument("--model-dir", default="models", help="Directory to store/load models")
    
    args = parser.parse_args()
    
    solver = CTFSolver(model_dir=args.model_dir)
    
    if args.train:
        if not args.training_data:
            logger.error("Training data must be provided with --training-data when using --train")
            return
        
        # Training mode
        if args.mode == "crypto":
            # Assume training data is CSV with text,label format
            import csv
            texts = []
            labels = []
            with open(args.training_data, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        texts.append(row[0])
                        labels.append(row[1])
            
            solver.train_cipher_classifier(texts, labels)
        
        # Add other training modes here
    
    else:
        # Analysis mode
        if args.mode == "crypto":
            if args.text:
                results = solver.solve_crypto_challenge(args.text)
                print("Cryptographic Analysis Results:")
                for cipher, result in results.items():
                    print(f"\n{cipher.upper()}:")
                    if isinstance(result, list):
                        for i, (param, text) in enumerate(result[:5]):  # Show only first 5 results
                            print(f"  Option {i+1} (shift={param}): {text[:50]}..." if len(text) > 50 else f"  Option {i+1} (shift={param}): {text}")
                    else:
                        print(f"  {result[:100]}..." if len(result) > 100 else f"  {result}")
            else:
                logger.error("Text data must be provided with --text for crypto mode")
        
        elif args.mode == "stego":
            if args.file:
                results = solver.analyze_image(args.file)
                print("Steganography Analysis Results:")
                for key, value in results.items():
                    if key == "exif":
                        print("\nEXIF Data:")
                        for exif_key, exif_val in value.items():
                            print(f"  {exif_key}: {exif_val}")
                    elif key == "ocr_text" and value != "OCR processing failed":
                        print(f"\nOCR Text:\n{value[:200]}..." if len(value) > 200 else f"\nOCR Text:\n{value}")
                    elif key == "lsb_analysis":
                        print("\nLSB Analysis:")
                        for lsb_key, lsb_val in value.items():
                            print(f"  {lsb_key}: {lsb_val}")
                    else:
                        print(f"\n{key}: {value}")
            else:
                logger.error("File must be provided with --file for stego mode")
        
        elif args.mode == "binary":
            if args.file:
                results = solver.analyze_binary(args.file)
                print("Binary Analysis Results:")
                for key, value in results.items():
                    if key == "strings":
                        print("\nStrings found:")
                        for i, string in enumerate(value):
                            print(f"  {i+1}. {string}")
                    elif key == "dangerous_functions":
                        print("\nPotentially dangerous functions:")
                        for func, count in value.items():
                            if count > 0:
                                print(f"  {func}: {count} occurrences")
                    elif key == "potential_flags":
                        print("\nPotential flags found:")
                        for flag in value:
                            print(f"  {flag}")
                    else:
                        print(f"\n{key}: {value}")
            else:
                logger.error("File must be provided with --file for binary mode")
        
        elif args.mode == "web":
            if args.text:
                results = solver.analyze_web_request(args.text)
                print("Web Vulnerability Analysis Results:")
                for key, value in results.items():
                    if key == "detected_vulnerabilities":
                        print("\nVulnerabilities detected:")
                        for vuln_type, matches in value.items():
                            print(f"  {vuln_type}:")
                            for match in matches:
                                print(f"    - {match}")
                            
                            # Generate potential payloads
                            payloads = solver.generate_payloads(vuln_type)
                            if payloads:
                                print(f"\n  Suggested {vuln_type} payloads:")
                                for i, payload in enumerate(payloads[:5]):  # Show only first 5 payloads
                                    print(f"    {i+1}. {payload}")
                    else:
                        print(f"\n{key}:")
                        print(value)
            else:
                logger.error("Text data must be provided with --text for web mode")

if __name__ == "__main__":
    main()
