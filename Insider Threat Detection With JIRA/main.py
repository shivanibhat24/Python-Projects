import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import datetime
import logging
import threading
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for app modes
APP_MODE = os.environ.get("APP_MODE", "streamlit")  # "streamlit" or "flask"

# Common logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("insider_threat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("insider_threat_detector")

# ============================
# Core Functionality Classes
# ============================

class UserBehaviorAnalyzer:
    """Analyze user behavior patterns and detect anomalies"""
    
    def __init__(self, model_path="models"):
        """Initialize the analyzer with model paths"""
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.isolation_forest = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)
        
    def preprocess_data(self, data):
        """Preprocess user activity data"""
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Convert timestamp to hour of day and day of week
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour_of_day'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Encode categorical features
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = pd.factorize(data[col])[0]
        
        # Drop non-numeric columns
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        return scaled_data, numeric_data.columns
    
    def train_models(self, data):
        """Train anomaly detection models"""
        scaled_data, _ = self.preprocess_data(data)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest model...")
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        self.isolation_forest.fit(scaled_data)
        
        # Reduce dimensions for clustering
        reduced_data = self.pca.fit_transform(scaled_data)
        
        # Train KMeans for user behavior clustering
        logger.info("Training KMeans model for behavior clustering...")
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.kmeans.fit(reduced_data)
        
        # Save the models
        self._save_models()
        logger.info("Models trained successfully")
    
    def _save_models(self):
        """Save the trained models to disk"""
        if self.isolation_forest and self.kmeans:
            with open(f"{self.model_path}/isolation_forest.pkl", 'wb') as f:
                pickle.dump(self.isolation_forest, f)
            with open(f"{self.model_path}/kmeans.pkl", 'wb') as f:
                pickle.dump(self.kmeans, f)
            with open(f"{self.model_path}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(f"{self.model_path}/pca.pkl", 'wb') as f:
                pickle.dump(self.pca, f)
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            with open(f"{self.model_path}/isolation_forest.pkl", 'rb') as f:
                self.isolation_forest = pickle.load(f)
            with open(f"{self.model_path}/kmeans.pkl", 'rb') as f:
                self.kmeans = pickle.load(f)
            with open(f"{self.model_path}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{self.model_path}/pca.pkl", 'rb') as f:
                self.pca = pickle.load(f)
            logger.info("Models loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("Models not found. Please train models first.")
            return False
    
    def detect_anomalies(self, user_data):
        """Detect anomalies in user behavior"""
        if not self.isolation_forest:
            loaded = self.load_models()
            if not loaded:
                raise ValueError("Models not trained. Please train models first.")
        
        scaled_data, feature_names = self.preprocess_data(user_data)
        
        # Predict anomalies using Isolation Forest (-1 for anomalies, 1 for normal)
        anomaly_scores = self.isolation_forest.decision_function(scaled_data)
        predictions = self.isolation_forest.predict(scaled_data)
        
        # Add anomaly information to original data
        user_data = user_data.copy()
        user_data['anomaly_score'] = anomaly_scores
        user_data['is_anomaly'] = np.where(predictions == -1, 1, 0)
        
        # Get cluster assignments
        reduced_data = self.pca.transform(scaled_data)
        user_data['cluster'] = self.kmeans.predict(reduced_data)
        
        # Identify the specific type of anomaly
        user_data['anomaly_type'] = self._classify_anomaly_type(user_data)
        
        return user_data[user_data['is_anomaly'] == 1]
    
    def _classify_anomaly_type(self, data):
        """Classify the type of anomaly based on features"""
        anomaly_types = []
        
        for _, row in data.iterrows():
            if row['is_anomaly'] == 0:
                anomaly_types.append("normal")
                continue
            
            # Check for data exfiltration indicators
            if 'data_volume' in row and row['data_volume'] > 100:
                anomaly_types.append("data_exfiltration")
            # Check for privilege escalation
            elif 'permission_level_change' in row and row['permission_level_change'] > 0:
                anomaly_types.append("privilege_escalation")
            # Check for unauthorized access
            elif 'failed_login_attempts' in row and row['failed_login_attempts'] > 3:
                anomaly_types.append("unauthorized_access")
            # Check for unusual login times
            elif 'hour_of_day' in row and (row['hour_of_day'] < 6 or row['hour_of_day'] > 20):
                anomaly_types.append("unusual_timing")
            else:
                anomaly_types.append("unknown_anomaly")
        
        return anomaly_types


class LogIngestor:
    """Ingest and parse various log formats for analysis"""
    
    def __init__(self):
        """Initialize log ingestor"""
        self.supported_formats = ['csv', 'json', 'syslog']
    
    def ingest_logs(self, file_path, format_type=None):
        """Ingest logs from a file"""
        if not format_type:
            # Determine format from file extension
            _, ext = os.path.splitext(file_path)
            format_type = ext[1:].lower() if ext else None
        
        if format_type not in self.supported_formats:
            logger.error(f"Unsupported log format: {format_type}")
            return None
        
        try:
            if format_type == 'csv':
                return pd.read_csv(file_path)
            elif format_type == 'json':
                return pd.read_json(file_path)
            elif format_type == 'syslog':
                return self._parse_syslog(file_path)
        except Exception as e:
            logger.error(f"Error ingesting logs: {str(e)}")
            return None
    
    def _parse_syslog(self, file_path):
        """Parse syslog format logs"""
        log_data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Simple syslog parser
                    parts = line.split(' ', 3)
                    if len(parts) >= 4:
                        timestamp = f"{parts[0]} {parts[1]} {parts[2]}"
                        message = parts[3].strip()
                        
                        # Extract username, action, and resource if possible
                        # This would need to be customized based on your syslog format
                        log_entry = {
                            'timestamp': timestamp,
                            'raw_message': message
                        }
                        
                        # Look for user information
                        if 'user=' in message:
                            user_part = message.split('user=')[1].split(' ')[0]
                            log_entry['username'] = user_part
                        
                        # Look for action information
                        if 'action=' in message:
                            action_part = message.split('action=')[1].split(' ')[0]
                            log_entry['action'] = action_part
                        
                        log_data.append(log_entry)
                except Exception as e:
                    logger.warning(f"Could not parse log line: {line.strip()} - {str(e)}")
        
        return pd.DataFrame(log_data)


class InsiderThreatDetector:
    """Main class to orchestrate the insider threat detection process"""
    
    def __init__(self, config_file="config.json"):
        """Initialize the detector with configuration"""
        self.config_file = config_file
        self.config = self._load_config()
        
        self.log_ingestor = LogIngestor()
        self.behavior_analyzer = UserBehaviorAnalyzer(model_path=self.config.get('model_path', 'models'))
        
        self.user_info_db = self._load_user_info()
        self.monitoring_thread = None
        self.stop_monitoring = False
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_file} not found. Using default configuration.")
            return {
                'model_path': 'models',
                'jira_config': 'jira_config.json',
                'user_info_db': 'user_info.json',
                'alert_threshold': -0.5,
                'scan_interval': 3600  # seconds
            }
    
    def _load_user_info(self):
        """Load user information database"""
        try:
            with open(self.config.get('user_info_db', 'user_info.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("User info database not found. Using empty database.")
            return {}
    
    def train_detection_models(self, training_data_path):
        """Train anomaly detection models with historical data"""
        try:
            # Determine file format
            _, ext = os.path.splitext(training_data_path)
            format_type = ext[1:].lower()
            
            # Ingest training data
            training_data = self.log_ingestor.ingest_logs(training_data_path, format_type)
            
            if training_data is not None and not training_data.empty:
                # Train the models
                self.behavior_analyzer.train_models(training_data)
                logger.info(f"Successfully trained models with {len(training_data)} records")
                return True
            else:
                logger.error("Failed to load training data")
                return False
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def scan_logs(self, log_path, format_type=None):
        """Scan logs for suspicious activities"""
        # Ingest logs
        log_data = self.log_ingestor.ingest_logs(log_path, format_type)
        
        if log_data is None or log_data.empty:
            logger.error("No valid log data to analyze")
            return []
        
        # Enrich log data with user information
        log_data = self._enrich_log_data(log_data)
        
        # Detect anomalies
        anomalies = self.behavior_analyzer.detect_anomalies(log_data)
        
        if len(anomalies) > 0:
            logger.info(f"Detected {len(anomalies)} potential insider threats")
            
        return anomalies
    
    def _enrich_log_data(self, log_data):
        """Enrich log data with additional information"""
        # Make a copy to avoid modifying the original
        log_data = log_data.copy()
        
        # Add derived features for better anomaly detection
        
        # Add time-based features if timestamp is available
        if 'timestamp' in log_data.columns:
            log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
            log_data['hour_of_day'] = log_data['timestamp'].dt.hour
            log_data['day_of_week'] = log_data['timestamp'].dt.dayofweek
            log_data['is_weekend'] = log_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            log_data['is_after_hours'] = log_data['hour_of_day'].apply(lambda x: 1 if x < 8 or x > 18 else 0)
        
        # Add user role level if username is available
        if 'username' in log_data.columns:
            log_data['user_role_level'] = log_data['username'].apply(
                lambda x: self.user_info_db.get(x, {}).get('role_level', 0)
            )
        
        # Calculate session features if session IDs are available
        if 'session_id' in log_data.columns:
            session_counts = log_data.groupby('session_id').size().reset_index(name='session_action_count')
            log_data = log_data.merge(session_counts, on='session_id')
        
        return log_data
    
    def run_continuous_monitoring(self, log_directory, scan_interval=None):
        """Run continuous monitoring of log files in a separate thread"""
        if scan_interval is None:
            scan_interval = self.config.get('scan_interval', 3600)  # Default to hourly
            
        def monitoring_thread(log_dir, interval):
            self.stop_monitoring = False
            logger.info(f"Starting continuous monitoring of {log_dir} every {interval} seconds")
            
            while not self.stop_monitoring:
                try:
                    # Scan all log files in the directory
                    files_scanned = 0
                    anomalies_found = 0
                    
                    for file in os.listdir(log_dir):
                        if self.stop_monitoring:
                            break
                            
                        if file.endswith(('.log', '.csv', '.json')):
                            file_path = os.path.join(log_dir, file)
                            logger.info(f"Scanning {file_path}")
                            
                            # Determine format from file extension
                            _, ext = os.path.splitext(file)
                            format_type = ext[1:].lower()
                            
                            # Scan the log file
                            try:
                                anomalies = self.scan_logs(file_path, format_type)
                                if anomalies is not None:
                                    anomalies_found += len(anomalies)
                                files_scanned += 1
                            except Exception as e:
                                logger.error(f"Error scanning {file_path}: {str(e)}")
                    
                    logger.info(f"Monitoring cycle completed. Scanned {files_scanned} files, found {anomalies_found} anomalies.")
                    
                    # Wait for the next scan interval
                    for _ in range(interval):
                        if self.stop_monitoring:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {str(e)}")
                    time.sleep(interval)  # Wait and retry
        
        # Stop any existing monitoring thread
        self.stop_monitoring_thread()
        
        # Start a new monitoring thread
        self.monitoring_thread = threading.Thread(
            target=monitoring_thread, 
            args=(log_directory, scan_interval)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        return True
    
    def stop_monitoring_thread(self):
        """Stop the continuous monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Stopping monitoring thread...")
            self.stop_monitoring = True
            self.monitoring_thread.join(timeout=5)
            logger.info("Monitoring thread stopped")
        self.monitoring_thread = None


def create_example_data():
    """Create example log data for testing"""
    import random
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create sample user activity data
    users = ['jdoe', 'asmith', 'bjohnson', 'mwilliams', 'rjones']
    resources = ['financial_report.xlsx', 'customer_data.csv', 'product_roadmap.docx', 
                'hr_policies.pdf', 'source_code.zip', 'database_backup.sql']
    actions = ['view', 'download', 'edit', 'share', 'print', 'delete']
    
    # Generate normal activity data
    normal_data = []
    
    # Current time
    end_time = datetime.datetime.now()
    # Start time (30 days ago)
    start_time = end_time - datetime.timedelta(days=30)
    
    # Generate timestamps between start and end time
    timestamps = []
    current = start_time
    while current < end_time:
        # Add more activity during business hours
        if 8 <= current.hour <= 18 and current.weekday() < 5:
            # Generate up to 5 events per hour during business hours
            for _ in range(random.randint(1, 5)):
                timestamps.append(current + datetime.timedelta(minutes=random.randint(0, 59)))
        else:
            # Occasionally generate after-hours events
            if random.random() < 0.1:
                timestamps.append(current + datetime.timedelta(minutes=random.randint(0, 59)))
        
        current += datetime.timedelta(hours=1)
    
    # Generate normal user activities
    for ts in timestamps:
        user = random.choice(users)
        resource = random.choice(resources)
        action = random.choice(actions)
        
        # Calculate data volume based on resource and action
        if 'database' in resource or '.zip' in resource:
            data_volume = random.randint(50, 200)
        elif action == 'download':
            data_volume = random.randint(10, 50)
        else:
            data_volume = random.randint(1, 10)
        
        # Failed login attempts (usually 0, occasionally 1)
        failed_logins = random.choices([0, 1], weights=[0.95, 0.05])[0]
        
        # Permission level change (usually 0)
        permission_change = random.choices([0, 1], weights=[0.99, 0.01])[0]
        
        normal_data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'username': user,
            'resource': resource,
            'action': action,
            'data_volume': data_volume,
            'failed_login_attempts': failed_logins,
            'permission_level_change': permission_change,
            'ip_address': f'10.0.{random.randint(1, 255)}.{random.randint(1, 255)}'
        })
    
    # Generate a few anomalous activities
    anomalous_data = []
    
    # Data exfiltration (large data volume)
    anomalous_data.append({
        'timestamp': (end_time - datetime.timedelta(days=random.randint(1, 7), 
                                                  hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M:%S'),
        'username': random.choice(users),
        'resource': 'customer_database.sql',
        'action': 'download',
        'data_volume': random.randint(500, 1000),  # Suspiciously large
        'failed_login_attempts': 0,
        'permission_level_change': 0,
        'ip_address': f'10.0.{random.randint(1, 255)}.{random.randint(1, 255)}'
    })
    
    # Privilege escalation
    anomalous_data.append({
        'timestamp': (end_time - datetime.timedelta(days=random.randint(1, 7), 
                                                  hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M:%S'),
        'username': random.choice(users),
        'resource': 'admin_console',
        'action': 'modify',
        'data_volume': 5,
        'failed_login_attempts': 0,
        'permission_level_change': 2,  # Suspicious permission change
        'ip_address': f'10.0.{random.randint(1, 255)}.{random.randint(1, 255)}'
    })
    
    # Unauthorized access (multiple failed logins)
    anomalous_data.append({
        'timestamp': (end_time - datetime.timedelta(days=random.randint(1, 7), 
                                                  hours=random.randint(0, 23))).strftime('%Y-%m-%d %H:%M:%S'),
        'username': random.choice(users),
        'resource': 'financial_report.xlsx',
        'action': 'view',
        'data_volume': 3,
        'failed_login_attempts': 5,  # Suspicious failed login attempts
        'permission_level_change': 0,
        'ip_address': f'192.168.{random.randint(1, 255)}.{random.randint(1, 255)}'  # Different subnet
    })
    
    # After-hours activity
    late_night = end_time.replace(hour=3, minute=random.randint(0, 59))
    anomalous_data.append({
        'timestamp': late_night.strftime('%Y-%m-%d %H:%M:%S'),
        'username': random.choice(users),
        'resource': 'hr_database.sql',
        'action': 'download',
        'data_volume': 75,
        'failed_login_attempts': 1,
        'permission_level_change': 0,
        'ip_address': f'10.0.{random.randint(1, 255)}.{random.randint(1, 255)}'
    })
    
    # Combine normal and anomalous data
    all_data = normal_data + anomalous_data
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    df.to_csv('data/sample_user_activity.csv', index=False)
    
    logger.info("Example data created: data/sample_user_activity.csv")
    logger.info(f"Total records: {len(all_data)} ({len(normal_data)} normal, {len(anomalous_data)} anomalous)")
    
    # Create user info database
    user_info = {
        'jdoe': {
            'name': 'John Doe',
            'email': 'jdoe@company.com',
            'department': 'Engineering',
            'manager': 'Jane Smith',
            'manager_email': 'jsmith@company.com',
            'role_level': 3
        },
        'asmith': {
            'name': 'Alice Smith',
            'email': 'asmith@company.com',
            'department': 'Finance',
            'manager': 'Bob Johnson',
            'manager_email': 'bjohnson@company.com',
            'role_level': 4
        },
        'bjohnson': {
            'name': 'Bob Johnson',
            'email': 'bjohnson@company.com',
            'department': 'Finance',
            'manager': 'Sarah Lee',
            'manager_email': 'slee@company.com',
            'role_level': 5
        },
        'mwilliams': {
            'name': 'Mike Williams',
            'email': 'mwilliams@company.com',
            'department': 'IT',
            'manager': 'John Doe',
            'manager_email': 'jdoe@company.com',
            'role_level': 3
        },
        'rjones': {
            'name': 'Rachel Jones',
            'email': 'rjones@company.com',
            'department': 'HR',
            'manager': 'Sarah Lee',
            'manager_email': 'slee@company.com',
            'role_level': 4
        }
    }
    
    with open('data/user_info.json', 'w') as f:
        json.dump(user_info, f, indent=4)
    
    logger.info("Example user database created: data/user_info.json")
    
    # Create config file
    config = {
        'model_path': 'models',
        'user_info_db': 'data/user_info.json',
        'alert_threshold': -0.5,
        'scan_interval': 3600  # seconds
    }
    
    with open('data/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("Example config created: data/config.json")
    
    return 'data/sample_user_activity.csv'


# ============================
# Flask API Backend
# ============================

def run_flask_app():
    from flask import Flask, request, jsonify, send_from_directory
    from werkzeug.utils import secure_filename
    
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize detector
    detector = InsiderThreatDetector(config_file="data/config.json")
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok", "message": "Insider Threat Detection API is running"})
    
    @app.route('/api/train', methods=['POST'])
    def train_model():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            success = detector.train_detection_models(file_path)
            
            if success:
                return jsonify({"message": "Model trained successfully"})
            else:
                return jsonify({"error": "Failed to train model"}), 500
    
    @app.route('/api/detect', methods=['POST'])
    def detect_threats():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                anomalies = detector.scan_logs(file_path)
                
                # Convert DataFrame to dict for JSON response
                if anomalies is not None and not anomalies.empty:
                    # Convert timestamps to string for JSON serialization
                    if 'timestamp' in anomalies.columns:
                        anomalies['timestamp'] = anomalies['timestamp'].astype(str)
                    
                    result = anomalies.to_dict(orient='records')
                    return jsonify({
                        "anomalies_found": len(result),
                        "anomalies": result
                    })
                else:
                    return jsonify({
                        "anomalies_found": 0,
                        "anomalies": []
                    })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    @app.route('/api/monitor/start', methods=['POST'])
    def start_monitoring():
        data = request.json
        directory = data.get('directory', 'logs')
        interval = data.get('interval', 3600)
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        if detector.run_continuous_monitoring(directory, interval):
            return jsonify({"message": f"Monitoring started for directory: {directory}"})
        else:
            return jsonify({"error": "Failed to start monitoring"}), 500
    
    @app.route('/api/monitor/stop', methods=['POST'])
    def stop_monitoring():
        detector.stop_monitoring_thread()
        return jsonify({"message": "Monitoring stopped"})
    
    @app.route('/api/sample-data', methods=['GET'])
    def generate_sample_data():
        file_path = create_example_data()
        return jsonify({"message": f"Sample data created at {file_path}"})
    
    @app.route('/data/<path:filename>')
    def download_file(filename):
        return send_from_directory('data', filename)
    
    print("Starting Flask API server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)


# ============================
# Streamlit UI
# ============================

def run_streamlit_app():
    import streamlit as st
    
    # Set page configuration
    st.set_page_config(
        page_title="Insider Threat Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize detector
    @st.cache_resource
    def get_detector():
        return InsiderThreatDetector(config_file="data/config.json")
    
    detector = get_detector()
    
    # Sidebar navigation
    st.sidebar.title("🛡️ Insider Threat Detection")
    
    pages = [
        "Dashboard", 
        "Upload & Analyze Logs", 
        "Train Models", 
        "Continuous Monitoring",
        "Generate Sample Data"
    ]
    
    selected_page = st.sidebar.radio("Navigation", pages)
    
    # Add info about the app
    with st.sidebar.expander("About this app"):
        st.write("""
        This application helps security teams detect potential insider threats 
        by analyzing user behavior patterns and identifying anomalies.
        
        It uses machine learning algorithms including Isolation Forest and K-means 
        clustering to identify suspicious activities.
        """)
    
    # Dashboard page
    if selected_page == "Dashboard":
        st.title("Insider Threat Detection Dashboard")
        
        # Display app status
        st.markdown("### System Status")
        
        col1, col2, col3 = st.columns(3)
        
        # Check if models are trained
        models_trained = os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl")
        
        with col1:
            if models_trained:
                st.success("Models: Trained ✓")
            else:
                st.error("Models: Not Trained ✗")
        
        # Check if monitoring is active
        with col2:
            if detector.monitoring_thread and detector.monitoring_thread.is_alive():
                st.success("Monitoring: Active ✓")
            else:
                st.warning("Monitoring: Inactive ⚠")
        
        # Check if sample data exists
        with col3:
            if os.path.exists("data/sample_user_activity.csv"):
                st.success("Sample Data: Available ✓")
            else:
                st.info("Sample Data: Not Generated ℹ")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        quick_action_cols = st.columns(3)
        
        with quick_action_cols[0]:
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    file_path = create_example_data()
                    st.success(f"Sample data created at {file_path}")
                    st.experimental_rerun()
        
        with quick_action_cols[1]:
            if not models_trained:
                if st.button("Train Models with Sample Data"):
                    if os.path.exists("data/sample_user_activity.csv"):
                        with st.spinner("Training models..."):
                            success = detector.train_detection_models("data/sample_user_activity.csv")
                            if success:
                                st.success("Models trained successfully!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to train models.")
                    else:
                        st.error("Sample data not found. Please generate it first.")
            else:
                if st.button("Retrain Models"):
                    if os.path.exists("data/sample_user_activity.csv"):
                        with st.spinner("Retraining models..."):
                            success = detector.train_detection_models("data/sample_user_activity.csv")
                            if success:
                                st.success("Models retrained successfully!")
                            else:
                                st.error("Failed to retrain models.")
                    else:
                        st.error("Sample data not found. Please generate it first.")
        
        with quick_action_cols[2]:
            if detector.monitoring_thread and detector.monitoring_thread.is_alive():
                if st.button("Stop Monitoring"):
                    detector.stop_monitoring_thread()
                    st.success("Monitoring stopped.")
                    st.experimental_rerun()
            else:
                if st.button("Start Monitoring"):
                    if not os.path.exists("logs"):
                        os.makedirs("logs")
                    detector.run_continuous_monitoring("logs")
                    st.success("Monitoring started.")
                    st.experimental_rerun()
        
        # Display a dummy visualization
        if models_trained and os.path.exists("data/sample_user_activity.csv"):
            st.markdown("### Recent Anomalies")
            
            try:
                # Load and analyze sample data for visualization
                sample_data = pd.read_csv("data/sample_user_activity.csv")
                anomalies = detector.scan_logs("data/sample_user_activity.csv")
                
                if anomalies is not None and not anomalies.empty:
                    st.warning(f"Found {len(anomalies)} potential insider threats in sample data.")
                    
                    # Display anomalies in a table
                    st.dataframe(anomalies[['timestamp', 'username', 'resource', 'action', 'anomaly_type', 'anomaly_score']].sort_values('anomaly_score'))
                    
                    # Plot anomaly distribution
                    st.markdown("### Anomaly Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Anomalies by Type")
                        type_counts = anomalies['anomaly_type'].value_counts().reset_index()
                        type_counts.columns = ['Anomaly Type', 'Count']
                        st.bar_chart(type_counts.set_index('Anomaly Type'))
                    
                    with col2:
                        st.subheader("Anomalies by User")
                        user_counts = anomalies['username'].value_counts().reset_index()
                        user_counts.columns = ['Username', 'Count']
                        st.bar_chart(user_counts.set_index('Username'))
                else:
                    st.success("No anomalies detected in the sample data.")
            
            except Exception as e:
                st.error(f"Error analyzing sample data: {str(e)}")
        
        elif not models_trained:
            st.info("Please train models to view anomaly detection results.")
    
    # Upload & Analyze Logs page
    elif selected_page == "Upload & Analyze Logs":
        st.title("Upload & Analyze Logs")
        
        st.markdown("""
        Upload user activity logs to scan for potential insider threats.
        Supported formats: CSV, JSON, Syslog
        """)
        
        # Check if models are trained
        models_trained = os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl")
        
        if not models_trained:
            st.warning("Models are not trained yet. Please go to the 'Train Models' page to train models first.")
        else:
            uploaded_file = st.file_uploader("Upload log file", type=['csv', 'json', 'log', 'txt'])
            
            if uploaded_file is not None:
                # Save the uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                os.makedirs("uploads", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                
                # Determine file format
                _, ext = os.path.splitext(uploaded_file.name)
                format_type = ext[1:].lower()
                
                # Add a button to scan the uploaded file
                if st.button("Scan for Threats"):
                    with st.spinner("Analyzing logs..."):
                        try:
                            anomalies = detector.scan_logs(file_path, format_type)
                            
                            if anomalies is not None and not anomalies.empty:
                                st.warning(f"Found {len(anomalies)} potential insider threats!")
                                
                                # Display anomalies in a table
                                st.dataframe(anomalies)
                                
                                # Show more detailed analysis
                                if 'anomaly_type' in anomalies.columns:
                                    st.subheader("Threat Categories")
                                    anomaly_types = anomalies['anomaly_type'].value_counts()
                                    st.bar_chart(anomaly_types)
                                
                                # Allow downloading the results
                                csv = anomalies.to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name="threat_detection_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.success("No anomalies detected in the uploaded logs.")
                        except Exception as e:
                            st.error(f"Error analyzing logs: {str(e)}")
    
    # Train Models page
    elif selected_page == "Train Models":
        st.title("Train Detection Models")
        
        st.markdown("""
        Train the machine learning models used for anomaly detection.
        You can use either sample data or upload your own historical log data.
        """)
        
        train_option = st.radio(
            "Select training data source:",
            ["Use Sample Data", "Upload Training Data"]
        )
        
        if train_option == "Use Sample Data":
            if os.path.exists("data/sample_user_activity.csv"):
                st.info("Sample data is available and ready for training.")
                
                if st.button("Train with Sample Data"):
                    with st.spinner("Training models with sample data..."):
                        success = detector.train_detection_models("data/sample_user_activity.csv")
                        
                        if success:
                            st.success("Models trained successfully!")
                        else:
                            st.error("Failed to train models with sample data.")
            else:
                st.warning("Sample data not found. Please generate it first.")
                if st.button("Generate Sample Data"):
                    with st.spinner("Generating sample data..."):
                        file_path = create_example_data()
                        st.success(f"Sample data created at {file_path}")
                        st.experimental_rerun()
        
        else:  # Upload Training Data
            st.markdown("Upload historical log data for training:")
            uploaded_file = st.file_uploader("Upload training data", type=['csv', 'json'])
            
            if uploaded_file is not None:
                # Save the uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                os.makedirs("uploads", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                
                if st.button("Train Models"):
                    with st.spinner("Training models..."):
                        success = detector.train_detection_models(file_path)
                        
                        if success:
                            st.success("Models trained successfully!")
                        else:
                            st.error("Failed to train models with uploaded data.")
        
        # Model information
        st.markdown("### Model Information")
        
        models_exist = os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl")
        
        if models_exist:
            # Get model files and their creation time
            model_dir = detector.config.get('model_path', 'models')
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            if model_files:
                st.success("Models are trained and ready to use.")
                
                # Display model information
                model_info = []
                for file in model_files:
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path) / 1024  # Size in KB
                    file_time = os.path.getmtime(file_path)
                    file_time_str = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                    
                    model_info.append({
                        "Model": file,
                        "Size (KB)": f"{file_size:.2f}",
                        "Last Updated": file_time_str
                    })
                
                st.table(pd.DataFrame(model_info))
                
                # Option to reset models
                if st.button("Reset Models"):
                    with st.spinner("Deleting existing models..."):
                        for file in model_files:
                            os.remove(os.path.join(model_dir, file))
                        st.success("Models reset successfully.")
                        st.experimental_rerun()
            else:
                st.warning("Model directory exists but no model files found.")
        else:
            st.warning("Models are not trained yet.")
    
    # Continuous Monitoring page
    elif selected_page == "Continuous Monitoring":
        st.title("Continuous Log Monitoring")
        
        st.markdown("""
        Set up continuous monitoring of log files to automatically detect potential insider threats.
        """)
        
        # Check if models are trained
        models_trained = os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl")
        
        if not models_trained:
            st.warning("Models are not trained yet. Please go to the 'Train Models' page to train models first.")
        else:
            # Current monitoring status
            if detector.monitoring_thread and detector.monitoring_thread.is_alive():
                st.success("Monitoring is currently active.")
                
                if st.button("Stop Monitoring"):
                    detector.stop_monitoring_thread()
                    st.success("Monitoring stopped.")
                    st.experimental_rerun()
            else:
                st.info("Monitoring is currently inactive.")
                
                # Form for monitoring settings
                with st.form("monitoring_form"):
                    st.markdown("### Monitoring Settings")
                    
                    log_dir = st.text_input("Log Directory", value="logs")
                    interval = st.slider("Scan Interval (seconds)", 
                                        min_value=60, max_value=3600, 
                                        value=600, step=60)
                    
                    submit = st.form_submit_button("Start Monitoring")
                    
                    if submit:
                        # Make sure log directory exists
                        os.makedirs(log_dir, exist_ok=True)
                        
                        detector.run_continuous_monitoring(log_dir, interval)
                        st.success(f"Monitoring started for directory: {log_dir}")
                        st.experimental_rerun()
    
    # Generate Sample Data page
    elif selected_page == "Generate Sample Data":
        st.title("Generate Sample Data")
        
        st.markdown("""
        Generate synthetic user activity data for testing and training the models.
        The generated data will include normal user activities as well as a few anomalous patterns.
        """)
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating synthetic user activity data..."):
                file_path = create_example_data()
                st.success(f"Sample data created successfully at {file_path}")
                
                # Display sample of the generated data
                if os.path.exists(file_path):
                    sample_data = pd.read_csv(file_path)
                    st.markdown("### Sample of Generated Data")
                    st.dataframe(sample_data.head(10))
                    
                    # Display data statistics
                    st.markdown("### Data Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(sample_data))
                    
                    with col2:
                        st.metric("Users", len(sample_data['username'].unique()))
                    
                    with col3:
                        st.metric("Date Range", f"{sample_data['timestamp'].min().split()[0]} to {sample_data['timestamp'].max().split()[0]}")
                    
                    # Allow downloading the data
                    csv = sample_data.to_csv(index=False)
                    st.download_button(
                        label="Download Sample Data",
                        data=csv,
                        file_name="sample_user_activity.csv",
                        mime="text/csv"
                    )


# ============================
# Main Execution
# ============================

def main():
    """Main execution function"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Insider Threat Detection System')
    parser.add_argument('--mode', choices=['streamlit', 'flask', 'cli'], 
                      default=APP_MODE, help='Application mode')
    parser.add_argument('--generate-data', action='store_true',
                      help='Generate example data and exit')
    parser.add_argument('--train', help='Train models with the provided data file')
    parser.add_argument('--scan', help='Scan the provided log file for threats')
    parser.add_argument('--monitor', help='Monitor the provided directory for log files')
    parser.add_argument('--interval', type=int, default=3600,
                      help='Scan interval in seconds for monitoring (default: 3600)')
    args = parser.parse_args()
    
    # Generate example data if requested
    if args.generate_data:
        file_path = create_example_data()
        print(f"Example data created at: {file_path}")
        return
    
    # CLI mode operations
    if args.train or args.scan or args.monitor:
        detector = InsiderThreatDetector(config_file="data/config.json")
        
        if args.train:
            print(f"Training models with {args.train}...")
            success = detector.train_detection_models(args.train)
            print("Training completed successfully!" if success else "Training failed.")
        
        if args.scan:
            if not os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl"):
                print("Models are not trained. Please train models first.")
            else:
                print(f"Scanning {args.scan} for threats...")
                anomalies = detector.scan_logs(args.scan)
                
                if anomalies is not None and not anomalies.empty:
                    print(f"Found {len(anomalies)} potential insider threats:")
                    print(anomalies[['username', 'action', 'resource', 'anomaly_type']])
                else:
                    print("No anomalies detected.")
        
        if args.monitor:
            if not os.path.exists(f"{detector.config.get('model_path', 'models')}/isolation_forest.pkl"):
                print("Models are not trained. Please train models first.")
            else:
                print(f"Starting continuous monitoring of {args.monitor}...")
                detector.run_continuous_monitoring(args.monitor, args.interval)
                
                try:
                    # Keep the main thread running
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Stopping monitoring...")
                    detector.stop_monitoring_thread()
        
        return
    
    # Run the appropriate app based on mode
    if args.mode == "streamlit":
        run_streamlit_app()
    elif args.mode == "flask":
        run_flask_app()
    else:
        print("Invalid mode. Please choose 'streamlit', 'flask', or 'cli'.")


if __name__ == "__main__":
    main()
