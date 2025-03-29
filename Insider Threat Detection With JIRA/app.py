# INSIDER THREAT DETECTION SYSTEM
# -------------------------------
# A comprehensive solution to detect suspicious insider activities 
# with machine learning capabilities and Jira integration

import pandas as pd
import numpy as np
import os
import json
import logging
import datetime
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from jira import JIRA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("insider_threat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("insider_threat_detector")

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
        # Convert timestamp to hour of day and day of week
        if 'timestamp' in data.columns:
            data['hour_of_day'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        
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


class JiraIntegrator:
    """Integrate with Jira for incident management"""
    
    def __init__(self, config_file="jira_config.json"):
        """Initialize Jira connection from config file"""
        self.config_file = config_file
        self.jira = None
        self.project_key = None
        self._connect_to_jira()
    
    def _connect_to_jira(self):
        """Establish connection to Jira"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.jira = JIRA(
                server=config['server_url'],
                basic_auth=(config['username'], config['api_token'])
            )
            self.project_key = config['project_key']
            logger.info("Successfully connected to Jira")
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
    
    def create_incident(self, anomaly_data, user_info):
        """Create a Jira incident for a detected anomaly"""
        if not self.jira:
            logger.error("Jira connection not established")
            return None
        
        # Create issue description
        description = f"""
        Insider Threat Alert
        
        User: {user_info['name']} ({user_info['email']})
        Department: {user_info['department']}
        Manager: {user_info['manager']}
        
        Anomaly Type: {anomaly_data['anomaly_type']}
        Anomaly Score: {anomaly_data['anomaly_score']:.2f}
        
        Activity Details:
        - Timestamp: {anomaly_data.get('timestamp', 'N/A')}
        - Resource Accessed: {anomaly_data.get('resource', 'N/A')}
        - Action: {anomaly_data.get('action', 'N/A')}
        
        This incident was automatically generated by the Insider Threat Detection System.
        """
        
        # Determine issue priority based on anomaly score
        priority = "High" if anomaly_data['anomaly_score'] < -0.5 else "Medium"
        
        # Create the issue
        try:
            issue_dict = {
                'project': {'key': self.project_key},
                'summary': f"Insider Threat Alert: {anomaly_data['anomaly_type']} detected for {user_info['name']}",
                'description': description,
                'issuetype': {'name': 'Security Incident'},
                'priority': {'name': priority},
                'labels': ['insider-threat', anomaly_data['anomaly_type']],
            }
            
            issue = self.jira.create_issue(fields=issue_dict)
            logger.info(f"Created Jira incident: {issue.key}")
            
            # Add the user's manager as a watcher
            if user_info.get('manager_email'):
                self.jira.add_watcher(issue.key, user_info['manager_email'])
            
            return issue.key
        except Exception as e:
            logger.error(f"Failed to create Jira incident: {str(e)}")
            return None
    
    def update_incident_status(self, issue_key, status, comment=None):
        """Update the status of an incident"""
        if not self.jira:
            logger.error("Jira connection not established")
            return False
        
        try:
            # Get available transitions
            transitions = self.jira.transitions(issue_key)
            transition_id = None
            
            # Find the transition ID for the desired status
            for t in transitions:
                if t['name'].lower() == status.lower():
                    transition_id = t['id']
                    break
            
            if transition_id:
                self.jira.transition_issue(issue_key, transition_id)
                
                # Add a comment if provided
                if comment:
                    self.jira.add_comment(issue_key, comment)
                
                logger.info(f"Updated incident {issue_key} to status: {status}")
                return True
            else:
                logger.warning(f"Transition to status '{status}' not available for {issue_key}")
                return False
        except Exception as e:
            logger.error(f"Failed to update incident status: {str(e)}")
            return False
    
    def get_open_incidents(self):
        """Get all open security incidents"""
        if not self.jira:
            logger.error("Jira connection not established")
            return []
        
        try:
            jql = f'project = {self.project_key} AND issuetype = "Security Incident" AND status != Closed AND labels = insider-threat'
            issues = self.jira.search_issues(jql)
            return issues
        except Exception as e:
            logger.error(f"Failed to retrieve open incidents: {str(e)}")
            return []


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
        self.jira_integrator = JiraIntegrator(config_file=self.config.get('jira_config', 'jira_config.json'))
        
        self.user_info_db = self._load_user_info()
    
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
            
            # Create Jira incidents for each anomaly
            for _, anomaly in anomalies.iterrows():
                username = anomaly.get('username', 'unknown')
                user_info = self.user_info_db.get(username, {
                    'name': username,
                    'email': f"{username}@company.com",
                    'department': 'Unknown',
                    'manager': 'Unknown'
                })
                
                # Create Jira incident
                incident_key = self.jira_integrator.create_incident(anomaly, user_info)
                
                # Alert security team for high-risk anomalies
                if incident_key and anomaly['anomaly_score'] < self.config.get('alert_threshold', -0.5):
                    self._alert_security_team(anomaly, user_info, incident_key)
        
        return anomalies
    
    def _enrich_log_data(self, log_data):
        """Enrich log data with additional information"""
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
    
    def _alert_security_team(self, anomaly, user_info, incident_key):
        """Alert security team about high-risk anomalies"""
        # In a real system, this might send an email, SMS, or use a messaging API
        alert_message = f"""
        HIGH RISK INSIDER THREAT DETECTED
        
        User: {user_info['name']} ({user_info.get('email', 'N/A')})
        Department: {user_info.get('department', 'N/A')}
        Anomaly Type: {anomaly['anomaly_type']}
        Anomaly Score: {anomaly['anomaly_score']:.2f}
        
        Jira Incident: {incident_key}
        
        Please investigate immediately.
        """
        
        logger.warning(f"SECURITY ALERT: {alert_message}")
        # In a production system, implement actual alerting mechanism here
    
    def run_continuous_monitoring(self, log_directory, scan_interval=None):
        """Run continuous monitoring of log files"""
        if scan_interval is None:
            scan_interval = self.config.get('scan_interval', 3600)  # Default to hourly
        
        logger.info(f"Starting continuous monitoring of {log_directory}")
        
        try:
            while True:
                # Scan all log files in the directory
                for file in os.listdir(log_directory):
                    if file.endswith(('.log', '.csv', '.json')):
                        file_path = os.path.join(log_directory, file)
                        logger.info(f"Scanning {file_path}")
                        
                        # Determine format from file extension
                        _, ext = os.path.splitext(file)
                        format_type = ext[1:].lower()
                        
                        # Scan the log file
                        self.scan_logs(file_path, format_type)
                
                # Wait for the next scan interval
                logger.info(f"Waiting {scan_interval} seconds until next scan")
                import time
                time.sleep(scan_interval)
        except KeyboardInterrupt:
            logger.info("Continuous monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {str(e)}")


# Example configuration files

def create_example_config():
    """Create example configuration files"""
    # Main config
    config = {
        'model_path': 'models',
        'jira_config': 'jira_config.json',
        'user_info_db': 'user_info.json',
        'alert_threshold': -0.5,
        'scan_interval': 3600  # seconds
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Jira config
    jira_config = {
        'server_url': 'https://your-company.atlassian.net',
        'username': 'your-jira-email@company.com',
        'api_token': 'your-jira-api-token',
        'project_key': 'SEC'
    }
    
    with open('jira_config.json', 'w') as f:
        json.dump(jira_config, f, indent=4)
    
    # User info database
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
        }
    }
    
    with open('user_info.json', 'w') as f:
        json.dump(user_info, f, indent=4)
    
    print("Example configuration files created:")
    print("1. config.json - Main application configuration")
    print("2. jira_config.json - Jira integration settings")
    print("3. user_info.json - User information database")


def create_example_data():
    """Create example log data for testing"""
    import random
    
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
    df.to_csv('sample_user_activity.csv', index=False)
    
    print("Example data created: sample_user_activity.csv")
    print(f"Total records: {len(all_data)} ({len(normal_data)} normal, {len(anomalous_data)} anomalous)")


# Example usage of the insider threat detection system
# Example usage of the insider threat detection system
def main():
    """Main function to demonstrate the system"""
    print("Insider Threat Detection System")
    print("------------------------------")
    
    # Create example configuration and data
    create_example_config()
    create_example_data()
    
    # Initialize the detector
    detector = InsiderThreatDetector(config_file="config.json")
    
    # Train the models
    print("\nTraining detection models...")
    detector.train_detection_models('sample_user_activity.csv')
    
    # Scan for threats
    print("\nScanning for insider threats...")
    anomalies = detector.scan_logs('sample_user_activity.csv')
    
    # Display results
    if len(anomalies) > 0:
        print(f"\nDetected {len(anomalies)} potential insider threats:")
        for i, (_, anomaly) in enumerate(anomalies.iterrows(), 1):
            print(f"\nThreat #{i}:")
            print(f"User: {anomaly.get('username', 'unknown')}")
            print(f"Anomaly Type: {anomaly.get('anomaly_type', 'unknown')}")
            print(f"Anomaly Score: {anomaly.get('anomaly_score', 0):.2f}")
            print(f"Timestamp: {anomaly.get('timestamp', 'N/A')}")
            print(f"Resource: {anomaly.get('resource', 'N/A')}")
            print(f"Action: {anomaly.get('action', 'N/A')}")
    else:
        print("\nNo insider threats detected.")
    
    print("\nTo run continuous monitoring:")
    print("detector.run_continuous_monitoring('/path/to/logs')")


if __name__ == "__main__":
    main()
