# RansomGuard: Real-time Ransomware Detection System
# main.py

import os
import time
import hashlib
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import logging
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ransomguard.log'
)
logger = logging.getLogger('RansomGuard')

class FileActivityMonitor(FileSystemEventHandler):
    """Monitors file system events and extracts features for ML detection"""
    
    def __init__(self, paths_to_monitor, event_queue, sample_interval=0.1):
        self.paths_to_monitor = paths_to_monitor
        self.event_queue = event_queue
        self.sample_interval = sample_interval
        self.file_operations = {}
        self.entropy_samples = {}
        self.file_extension_counters = {}
        
        # Initialize counters for various file operations
        self.operations_count = {
            'created': 0,
            'modified': 0,
            'deleted': 0,
            'moved': 0,
            'file_count': 0,
            'dir_count': 0,
        }
        
        # Maintain a sliding window of historical data
        self.history_window = deque(maxlen=1000)
        
    def start(self):
        self.observer = Observer()
        for path in self.paths_to_monitor:
            self.observer.schedule(self, path, recursive=True)
        self.observer.start()
        logger.info(f"Started monitoring: {', '.join(self.paths_to_monitor)}")
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._periodic_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def stop(self):
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped file monitoring")
    
    def _calculate_entropy(self, file_path):
        """Calculate Shannon entropy for a file to detect encryption"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(8192)  # Read sample of file to save resources
                if not data:
                    return 0
                
                entropy = 0
                byte_counts = {}
                file_size = len(data)
                
                # Count occurrences of each byte
                for byte in data:
                    byte_counts[byte] = byte_counts.get(byte, 0) + 1
                
                # Calculate entropy
                for count in byte_counts.values():
                    probability = count / file_size
                    entropy -= probability * np.log2(probability)
                
                return entropy
        except (IOError, PermissionError):
            return 0
    
    def _extract_features(self):
        """Extract features for ML model"""
        current_time = time.time()
        features = {
            'created_rate': self.operations_count['created'] / 10,
            'modified_rate': self.operations_count['modified'] / 10,
            'deleted_rate': self.operations_count['deleted'] / 10,
            'entropy_avg': np.mean(list(self.entropy_samples.values()) or [0]),
            'entropy_std': np.std(list(self.entropy_samples.values()) or [0]),
            'operation_intensity': sum(self.operations_count.values()) / 10,
            'unique_extensions': len(self.file_extension_counters),
            'exe_dll_ratio': self.file_extension_counters.get('.exe', 0) + 
                             self.file_extension_counters.get('.dll', 0),
            'timestamp': current_time
        }
        
        # Reset counters for next window
        for key in self.operations_count:
            self.operations_count[key] = 0
        
        return features
    
    def _periodic_analysis(self):
        """Periodically analyze collected data for anomalies"""
        while True:
            time.sleep(10)  # Analyze every 10 seconds
            features = self._extract_features()
            self.history_window.append(features)
            self.event_queue.append((
                'features', 
                features
            ))
    
    def on_created(self, event):
        if not event.is_directory:
            self.operations_count['created'] += 1
            _, ext = os.path.splitext(event.src_path)
            if ext:
                self.file_extension_counters[ext] = self.file_extension_counters.get(ext, 0) + 1
            
            # Sample entropy for new files
            entropy = self._calculate_entropy(event.src_path)
            self.entropy_samples[event.src_path] = entropy
            
            self.event_queue.append((
                'file_event', 
                {'type': 'created', 'path': event.src_path, 'entropy': entropy}
            ))
        else:
            self.operations_count['dir_count'] += 1
    
    def on_modified(self, event):
        if not event.is_directory:
            self.operations_count['modified'] += 1
            
            # Recalculate entropy for modified files
            entropy = self._calculate_entropy(event.src_path)
            self.entropy_samples[event.src_path] = entropy
            
            self.event_queue.append((
                'file_event', 
                {'type': 'modified', 'path': event.src_path, 'entropy': entropy}
            ))
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.operations_count['deleted'] += 1
            if event.src_path in self.entropy_samples:
                del self.entropy_samples[event.src_path]
            
            self.event_queue.append((
                'file_event', 
                {'type': 'deleted', 'path': event.src_path}
            ))
        else:
            self.operations_count['dir_count'] -= 1
    
    def on_moved(self, event):
        self.operations_count['moved'] += 1
        if not event.is_directory:
            # Update entropy sample key
            if event.src_path in self.entropy_samples:
                self.entropy_samples[event.dest_path] = self.entropy_samples[event.src_path]
                del self.entropy_samples[event.src_path]
        
        self.event_queue.append((
            'file_event', 
            {'type': 'moved', 'src_path': event.src_path, 'dest_path': event.dest_path}
        ))


class RansomwareDetector:
    """ML-based ransomware detector that analyzes file system features"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.scaler = StandardScaler()
        
        # Initialize models
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with a basic model if no saved model exists
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize models for detection"""
        # Random Forest for classification (if labeled data available)
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        
        # Isolation Forest for anomaly detection (unsupervised)
        self.if_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        
        logger.info("Initialized new ML models")
    
    def load_model(self, model_path):
        """Load saved models from disk"""
        try:
            models = joblib.load(model_path)
            self.rf_model = models.get('rf_model')
            self.if_model = models.get('if_model')
            self.scaler = models.get('scaler', StandardScaler())
            logger.info(f"Loaded models from {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.initialize_model()
    
    def save_model(self, model_path):
        """Save models to disk"""
        models = {
            'rf_model': self.rf_model,
            'if_model': self.if_model,
            'scaler': self.scaler
        }
        try:
            joblib.dump(models, model_path)
            logger.info(f"Saved models to {model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def train(self, X, y=None):
        """Train models with collected data"""
        if len(X) < 10:
            logger.warning("Not enough data for training")
            return False
        
        # Prepare feature data
        feature_columns = [
            'created_rate', 'modified_rate', 'deleted_rate',
            'entropy_avg', 'entropy_std', 'operation_intensity',
            'unique_extensions', 'exe_dll_ratio'
        ]
        
        X_features = X[feature_columns]
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train models
        if y is not None and len(np.unique(y)) > 1:
            # Supervised learning if labels are available
            self.rf_model.fit(X_scaled, y)
            logger.info("Trained Random Forest classifier")
        
        # Always train anomaly detector on normal data
        self.if_model.fit(X_scaled)
        logger.info("Trained Isolation Forest anomaly detector")
        
        return True
    
    def detect(self, features):
        """Detect ransomware activity from features"""
        # Extract and scale features
        feature_columns = [
            'created_rate', 'modified_rate', 'deleted_rate',
            'entropy_avg', 'entropy_std', 'operation_intensity',
            'unique_extensions', 'exe_dll_ratio'
        ]
        
        # Handle missing features
        for col in feature_columns:
            if col not in features:
                features[col] = 0
        
        X_features = np.array([[features[col] for col in feature_columns]])
        X_scaled = self.scaler.transform(X_features)
        
        # Get anomaly score from Isolation Forest
        anomaly_score = -self.if_model.score_samples(X_scaled)[0]
        
        # Get probability if Random Forest is available
        if hasattr(self, 'rf_model') and hasattr(self.rf_model, 'predict_proba'):
            try:
                rf_probas = self.rf_model.predict_proba(X_scaled)[0]
                malicious_probability = rf_probas[1] if len(rf_probas) > 1 else 0
            except:
                malicious_probability = 0
        else:
            malicious_probability = 0
        
        # Calculate threat score based on anomaly and classification
        threat_score = 0.7 * anomaly_score + 0.3 * malicious_probability
        
        # Determine alert level
        if threat_score > 0.8:
            alert_level = "CRITICAL"
        elif threat_score > 0.6:
            alert_level = "HIGH"
        elif threat_score > 0.4:
            alert_level = "MEDIUM"
        elif threat_score > 0.2:
            alert_level = "LOW"
        else:
            alert_level = "NORMAL"
        
        result = {
            'threat_score': threat_score,
            'anomaly_score': anomaly_score,
            'malicious_probability': malicious_probability,
            'alert_level': alert_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result


class RansomGuardUI:
    """Modern UI for the RansomGuard system"""
    
    def __init__(self, root, event_queue):
        self.root = root
        self.event_queue = event_queue
        self.alert_history = []
        self.file_events = []
        self.setup_ui()
        
        # Start UI update timer
        self.root.after(100, self.update_ui)
    
    def setup_ui(self):
        """Set up the main UI components"""
        self.root.title("RansomGuard - Advanced Ransomware Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e2e')
        style.configure('TLabel', background='#1e1e2e', foreground='#cdd6f4')
        style.configure('TButton', background='#1e1e2e', foreground='#cdd6f4')
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), foreground='#cdd6f4')
        style.configure('Status.TLabel', font=('Arial', 12), foreground='#cdd6f4')
        
        # Create main containers
        self.header_frame = ttk.Frame(self.root, style='TFrame')
        self.header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.main_frame = ttk.Frame(self.root, style='TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.status_frame = ttk.Frame(self.root, style='TFrame')
        self.status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Header with title and status
        self.title_label = ttk.Label(
            self.header_frame, 
            text="RansomGuard", 
            style='Header.TLabel'
        )
        self.title_label.pack(side=tk.LEFT, padx=10)
        
        self.status_indicator = tk.Canvas(
            self.header_frame, 
            width=20, 
            height=20, 
            bg="#1e1e2e", 
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.RIGHT, padx=10)
        self.status_indicator.create_oval(2, 2, 18, 18, fill="#a6e3a1", outline="")
        
        self.status_label = ttk.Label(
            self.header_frame, 
            text="System: NORMAL", 
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Main content with tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        self.dashboard_tab = ttk.Frame(self.tab_control, style='TFrame')
        self.alerts_tab = ttk.Frame(self.tab_control, style='TFrame')
        self.activity_tab = ttk.Frame(self.tab_control, style='TFrame')
        self.settings_tab = ttk.Frame(self.tab_control, style='TFrame')
        
        self.tab_control.add(self.dashboard_tab, text="Dashboard")
        self.tab_control.add(self.alerts_tab, text="Alerts")
        self.tab_control.add(self.activity_tab, text="File Activity")
        self.tab_control.add(self.settings_tab, text="Settings")
        
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Dashboard content
        self.setup_dashboard()
        
        # Alerts content
        self.setup_alerts_tab()
        
        # Activity content
        self.setup_activity_tab()
        
        # Settings content
        self.setup_settings_tab()
        
        # Status Bar
        self.cpu_label = ttk.Label(
            self.status_frame, 
            text="CPU: 2%", 
            style='Status.TLabel'
        )
        self.cpu_label.pack(side=tk.LEFT, padx=10)
        
        self.memory_label = ttk.Label(
            self.status_frame, 
            text="Memory: 124MB", 
            style='Status.TLabel'
        )
        self.memory_label.pack(side=tk.LEFT, padx=10)
        
        self.files_monitored_label = ttk.Label(
            self.status_frame, 
            text="Files Monitored: 0", 
            style='Status.TLabel'
        )
        self.files_monitored_label.pack(side=tk.LEFT, padx=10)
    
    def setup_dashboard(self):
        """Set up the dashboard tab with graphs and stats"""
        # Create dashboard layout
        left_frame = ttk.Frame(self.dashboard_tab, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(self.dashboard_tab, style='TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Threat level gauge
        threat_frame = ttk.Frame(left_frame, style='TFrame')
        threat_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        threat_label = ttk.Label(
            threat_frame, 
            text="Current Threat Level", 
            style='Header.TLabel'
        )
        threat_label.pack(pady=5)
        
        # Create matplotlib figure for gauge
        self.gauge_figure = Figure(figsize=(4, 4), facecolor='#1e1e2e')
        self.gauge_ax = self.gauge_figure.add_subplot(111)
        self.gauge_canvas = FigureCanvasTkAgg(self.gauge_figure, threat_frame)
        self.gauge_canvas.get_tk_widget().configure(bg='#1e1e2e', highlightthickness=0)
        self.gauge_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up initial gauge
        self.setup_gauge(0.1)
        
        # Activity monitoring charts
        activity_frame = ttk.Frame(right_frame, style='TFrame')
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        activity_label = ttk.Label(
            activity_frame, 
            text="System Activity", 
            style='Header.TLabel'
        )
        activity_label.pack(pady=5)
        
        # Create matplotlib figure for activity
        self.activity_figure = Figure(figsize=(6, 3), facecolor='#1e1e2e')
        self.activity_ax = self.activity_figure.add_subplot(111)
        self.activity_canvas = FigureCanvasTkAgg(self.activity_figure, activity_frame)
        self.activity_canvas.get_tk_widget().configure(bg='#1e1e2e', highlightthickness=0)
        self.activity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up initial activity chart
        self.setup_activity_chart()
        
        # Stats section
        stats_frame = ttk.Frame(left_frame, style='TFrame')
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        stats_label = ttk.Label(
            stats_frame, 
            text="Detection Statistics", 
            style='Header.TLabel'
        )
        stats_label.pack(pady=5)
        
        self.stats_text = tk.Text(
            stats_frame, 
            height=10, 
            bg='#313244', 
            fg='#cdd6f4',
            relief=tk.FLAT,
            font=('Consolas', 11)
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.insert(tk.END, "System started at: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        self.stats_text.insert(tk.END, "Files scanned: 0\n")
        self.stats_text.insert(tk.END, "Anomalies detected: 0\n")
        self.stats_text.insert(tk.END, "Average entropy: 0.0\n")
        self.stats_text.insert(tk.END, "File operations: 0\n")
        self.stats_text.config(state=tk.DISABLED)
    
    def setup_alerts_tab(self):
        """Set up the alerts tab"""
        # Create alerts treeview
        columns = ('timestamp', 'level', 'score', 'description')
        
        self.alerts_tree = ttk.Treeview(
            self.alerts_tab, 
            columns=columns, 
            show='headings',
            style='Treeview'
        )
        
        # Define column headings
        self.alerts_tree.heading('timestamp', text='Time')
        self.alerts_tree.heading('level', text='Alert Level')
        self.alerts_tree.heading('score', text='Threat Score')
        self.alerts_tree.heading('description', text='Description')
        
        # Define column widths
        self.alerts_tree.column('timestamp', width=150)
        self.alerts_tree.column('level', width=100)
        self.alerts_tree.column('score', width=100)
        self.alerts_tree.column('description', width=500)
        
        # Add scrollbar
        alerts_scrollbar = ttk.Scrollbar(self.alerts_tab, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=alerts_scrollbar.set)
        
        # Pack widgets
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_activity_tab(self):
        """Set up the file activity tab"""
        # Create file activity treeview
        columns = ('timestamp', 'operation', 'path', 'entropy')
        
        self.activity_tree = ttk.Treeview(
            self.activity_tab, 
            columns=columns, 
            show='headings',
            style='Treeview'
        )
        
        # Define column headings
        self.activity_tree.heading('timestamp', text='Time')
        self.activity_tree.heading('operation', text='Operation')
        self.activity_tree.heading('path', text='File Path')
        self.activity_tree.heading('entropy', text='Entropy')
        
        # Define column widths
        self.activity_tree.column('timestamp', width=150)
        self.activity_tree.column('operation', width=100)
        self.activity_tree.column('path', width=500)
        self.activity_tree.column('entropy', width=100)
        
        # Add scrollbar
        activity_scrollbar = ttk.Scrollbar(self.activity_tab, orient=tk.VERTICAL, command=self.activity_tree.yview)
        self.activity_tree.configure(yscrollcommand=activity_scrollbar.set)
        
        # Pack widgets
        self.activity_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        activity_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_settings_tab(self):
        """Set up the settings tab"""
        settings_frame = ttk.Frame(self.settings_tab, style='TFrame')
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Paths to monitor
        paths_label = ttk.Label(
            settings_frame, 
            text="Directories to Monitor", 
            style='Header.TLabel'
        )
        paths_label.grid(row=0, column=0, sticky=tk.W, pady=10)
        
        self.paths_text = tk.Text(
            settings_frame, 
            height=5, 
            bg='#313244', 
            fg='#cdd6f4',
            relief=tk.FLAT,
            font=('Consolas', 11)
        )
        self.paths_text.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.paths_text.insert(tk.END, os.path.expanduser("~/Documents") + "\n")
        
        # Sensitivity slider
        sensitivity_label = ttk.Label(
            settings_frame, 
            text="Detection Sensitivity", 
            style='Header.TLabel'
        )
        sensitivity_label.grid(row=2, column=0, sticky=tk.W, pady=10)
        
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_slider = ttk.Scale(
            settings_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.sensitivity_var,
            length=300
        )
        sensitivity_slider.grid(row=3, column=0, sticky=tk.EW, padx=5, pady=5)
        
        sensitivity_value_label = ttk.Label(
            settings_frame, 
            textvariable=self.sensitivity_var, 
            style='Status.TLabel'
        )
        sensitivity_value_label.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Training options
        training_label = ttk.Label(
            settings_frame, 
            text="Training Options", 
            style='Header.TLabel'
        )
        training_label.grid(row=4, column=0, sticky=tk.W, pady=10)
        
        self.auto_train_var = tk.BooleanVar(value=True)
        auto_train_check = ttk.Checkbutton(
            settings_frame,
            text="Auto-train model with new data",
            variable=self.auto_train_var
        )
        auto_train_check.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(settings_frame, style='TFrame')
        button_frame.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=20)
        
        save_button = ttk.Button(
            button_frame,
            text="Save Settings",
            command=self.save_settings
        )
        save_button.pack(side=tk.LEFT, padx=5)
        
        train_button = ttk.Button(
            button_frame,
            text="Train Model Now",
            command=self.train_model
        )
        train_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=0)
        
    def setup_gauge(self, value=0.0):
        """Set up the threat level gauge"""
        self.gauge_ax.clear()
        
        # Create a semi-circle gauge
        self.gauge_ax.set_theta_offset(3*np.pi/2)
        self.gauge_ax.set_theta_direction(-1)
        
        # Set up colors based on value
        if value < 0.2:
            color = '#a6e3a1'  # Green
        elif value < 0.4:
            color = '#f9e2af'  # Yellow
        elif value < 0.6:
            color = '#fab387'  # Orange
        else:
            color = '#f38ba8'  # Red
        
        # Plot the gauge
        self.gauge_ax.pie(
            [value, 1-value], 
            radius=0.8,
            startangle=180, 
            counterclock=False,
            colors=[color, '#313244'],
            wedgeprops=dict(width=0.3, edgecolor='none')
        )
        
        # Add text in the center
        threat_text = "LOW"
        if value > 0.2:
            threat_text = "MEDIUM"
        if value > 0.4:
            threat_text = "HIGH"
        if value > 0.6:
            threat_text = "CRITICAL"
        
        self.gauge_ax.text(
            0, 0, 
            f"{threat_text}\n{value:.2f}", 
            ha='center', 
            va='center', 
            fontsize=16,
            color='#cdd6f4'
        )
        
        # Customize appearance
        self.gauge_ax.set_facecolor('#1e1e2e')
        self.gauge_figure.patch.set_facecolor('#1e1e2e')
        
        # Remove axes and labels
        self.gauge_ax.axis('off')
        
        self.gauge_canvas.draw()
    
    def setup_activity_chart(self):
        """Set up the activity monitoring chart"""
        self.activity_ax.clear()
        
        # Create empty data
        self.time_points = list(range(60))
        self.file_ops_data = [0] * 60
        self.entropy_data = [0] * 60
        
        # Plot initial data
        self.file_ops_line, = self.activity_ax.plot(
            self.time_points, 
            self.file_ops_data, 
            '-', 
            label='File Operations', 
            color='#f5c2e7'
        )
        
        self.entropy_line, = self.activity_ax.plot(
            self.time_points, 
            self.entropy_data, 
            '-', 
            label='Entropy', 
            color='#89dceb'
        )
        
        # Customize appearance
        self.activity_ax.set_facecolor('#313244')
        self.activity_figure.patch.set_facecolor('#1e1e2e')
        self.activity_ax.tick_params(axis='x', colors='#cdd6f4')
        self.activity_ax.tick_params(axis='y', colors='#cdd6f4')
        self.activity_ax.grid(True, color='#6c7086', linestyle='--', alpha=0.3)
        
        # Add legend
        self.activity_ax.legend(loc='upper right', facecolor='#313244', edgecolor='none', labelcolor='#cdd6f4')
        
        # Set labels
        self.activity_ax.set_title('System Activity Monitor', color='#cdd6f4')
        self.activity_ax.set_xlabel('Time (s)', color='#cdd6f4')
        self.activity_ax.set_ylabel('Activity Level', color='#cdd6f4')
        
        self.activity_canvas.draw()
    
    def update_gauge(self, value):
        """Update the threat level gauge with a new value"""
        self.setup_gauge(value)
    
    def update_activity_chart(self, file_ops, entropy):
        """Update the activity chart with new data"""
        # Shift data to the left
        self.file_ops_data = self.file_ops_data[1:] + [file_ops]
        self.entropy_data = self.entropy_data[1:] + [entropy]
        
        # Update plot data
        self.file_ops_line.set_ydata(self.file_ops_data)
        self.entropy_line.set_ydata(self.entropy_data)
        
        # Adjust y-axis if needed
        max_y = max(max(self.file_ops_data), max(self.entropy_data)) * 1.1
        if max_y > 0:
            self.activity_ax.set_ylim(0, max_y)
        
        # Redraw
        self.activity_canvas.draw()
    
    def update_stats(self, stats):
        """Update the stats text box with new information"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        for key, value in stats.items():
            self.stats_text.insert(tk.END, f"{key}: {value}\n")
        
        self.stats_text.config(state=tk.DISABLED)
    
    def add_alert(self, alert):
        """Add an alert to the alerts treeview"""
        # Add to list
        self.alert_history.append(alert)
        
        # Add to treeview
        level = alert.get('alert_level', 'UNKNOWN')
        score = alert.get('threat_score', 0)
        timestamp = alert.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Generate description based on features
        description = f"Anomalous file activity detected"
        if 'trigger_files' in alert and alert['trigger_files']:
            description += f" involving {len(alert['trigger_files'])} files"
        
        item_id = self.alerts_tree.insert(
            '', 'end', 
            values=(timestamp, level, f"{score:.2f}", description)
        )
        
        # Set tag color based on level
        if level == "CRITICAL":
            self.alerts_tree.item(item_id, tags=('critical',))
        elif level == "HIGH":
            self.alerts_tree.item(item_id, tags=('high',))
        elif level == "MEDIUM":
            self.alerts_tree.item(item_id, tags=('medium',))
        
        # Ensure the newest alert is visible
        self.alerts_tree.see(item_id)
    
    def add_file_event(self, event):
        """Add a file event to the activity treeview"""
        # Add to list
        self.file_events.append(event)
        
        # Add to treeview
        event_type = event.get('type', 'UNKNOWN')
        path = event.get('path', 'Unknown path')
        entropy = event.get('entropy', 0)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        item_id = self.activity_tree.insert(
            '', 'end', 
            values=(timestamp, event_type.upper(), path, f"{entropy:.2f}")
        )
        
        # Set tag color based on entropy
        if entropy > 7.0:
            self.activity_tree.item(item_id, tags=('high_entropy',))
        
        # Ensure the newest event is visible
        self.activity_tree.see(item_id)
        
        # Limit displayed items to 1000
        if len(self.file_events) > 1000:
            self.file_events.pop(0)
            try:
                first_item = self.activity_tree.get_children()[0]
                self.activity_tree.delete(first_item)
            except IndexError:
                pass
    
    def update_status(self, status):
        """Update the system status indicator"""
        level = status.get('alert_level', 'NORMAL')
        
        # Update status text
        self.status_label.config(text=f"System: {level}")
        
        # Update status indicator color
        color = "#a6e3a1"  # Green/Normal
        if level == "LOW":
            color = "#f9e2af"  # Yellow
        elif level == "MEDIUM":
            color = "#fab387"  # Orange
        elif level == "HIGH":
            color = "#f38ba8"  # Red
        elif level == "CRITICAL":
            color = "#f38ba8"  # Red
            # Make indicator blink for critical alerts
            current_color = self.status_indicator.itemcget(1, "fill")
            color = "#1e1e2e" if current_color == "#f38ba8" else "#f38ba8"
        
        self.status_indicator.itemconfig(1, fill=color)
    
    def update_ui(self):
        """Process events and update UI elements"""
        # Process up to 10 events per update
        for _ in range(10):
            if self.event_queue:
                try:
                    event_type, event_data = self.event_queue.popleft()
                    
                    if event_type == 'alert':
                        self.add_alert(event_data)
                        self.update_status(event_data)
                        self.update_gauge(event_data.get('threat_score', 0))
                    
                    elif event_type == 'file_event':
                        self.add_file_event(event_data)
                    
                    elif event_type == 'features':
                        # Update activity chart
                        file_ops = event_data.get('operation_intensity', 0)
                        entropy = event_data.get('entropy_avg', 0)
                        self.update_activity_chart(file_ops, entropy)
                    
                    elif event_type == 'stats':
                        self.update_stats(event_data)
                        
                        # Update hardware stats
                        if 'cpu' in event_data:
                            self.cpu_label.config(text=f"CPU: {event_data['cpu']}%")
                        if 'memory' in event_data:
                            self.memory_label.config(text=f"Memory: {event_data['memory']}MB")
                        if 'files_monitored' in event_data:
                            self.files_monitored_label.config(text=f"Files Monitored: {event_data['files_monitored']}")
                
                except IndexError:
                    break
                except Exception as e:
                    logger.error(f"Error updating UI: {e}")
        
        # Schedule next update
        self.root.after(100, self.update_ui)
    
    def save_settings(self):
        """Save user settings"""
        # Not fully implemented in this demo
        messagebox = tk.messagebox.showinfo(
            "Settings", 
            "Settings saved successfully!"
        )
    
    def train_model(self):
        """Manually trigger model training"""
        # Not fully implemented in this demo
        messagebox = tk.messagebox.showinfo(
            "Training", 
            "Model training initiated. This may take a few minutes."
        )


class RansomGuard:
    """Main application class for RansomGuard"""
    
    def __init__(self, paths_to_monitor=None):
        if paths_to_monitor is None:
            # Default to Documents folder if none specified
            paths_to_monitor = [os.path.expanduser("~/Documents")]
        
        self.paths_to_monitor = paths_to_monitor
        self.event_queue = deque(maxlen=10000)
        self.stats = {
            'system_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'files_scanned': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'average_entropy': 0.0,
            'file_operations': 0,
            'cpu': 0,
            'memory': 0,
            'files_monitored': 0
        }
        
        # Initialize components
        self.monitor = FileActivityMonitor(paths_to_monitor, self.event_queue)
        self.detector = RansomwareDetector(model_path='ransomguard_model.joblib')
        
        # Start monitoring thread
        self.monitor.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start stats thread
        self.stats_thread = threading.Thread(target=self._stats_loop)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        logger.info("RansomGuard initialized and started monitoring")
    
    def _detection_loop(self):
        """Main detection loop that processes events and triggers alerts"""
        features_buffer = []
        alert_cooldown = 0
        
        while True:
            try:
                # Sleep to reduce CPU usage
                time.sleep(1)
                
                # Process feature events for detection
                for _ in range(min(50, len(self.event_queue))):
                    if not self.event_queue:
                        break
                    
                    event_type, event_data = self.event_queue[0]
                    
                    if event_type == 'features':
                        # Pop the event since we're processing it
                        self.event_queue.popleft()
                        
                        # Store feature for analysis
                        features_buffer.append(event_data)
                        
                        # Detect anomalies
                        if alert_cooldown <= 0:
                            detection_result = self.detector.detect(event_data)
                            
                            # Update stats
                            self.stats['average_entropy'] = event_data.get('entropy_avg', 0)
                            self.stats['file_operations'] += event_data.get('operation_intensity', 0)
                            
                            # Generate alert if threat score is significant
                            if detection_result['threat_score'] > 0.2:
                                # Add some context to the alert
                                detection_result['trigger_features'] = event_data
                                self.event_queue.append(('alert', detection_result))
                                
                                self.stats['alerts_generated'] += 1
                                if detection_result['threat_score'] > 0.4:
                                    self.stats['anomalies_detected'] += 1
                                
                                # Set cooldown to avoid alert spam
                                if detection_result['threat_score'] > 0.6:
                                    alert_cooldown = 60  # 1 minute for high severity
                                else:
                                    alert_cooldown = 30  # 30 seconds for lower severity
                    else:
                        # Leave other event types for the UI to process
                        pass
                
                # Train model if we have enough data (every ~30 minutes)
                if len(features_buffer) >= 180:
                    # Convert to DataFrame for training
                    train_df = pd.DataFrame(features_buffer)
                    
                    # Train the model with the new data
                    if self.detector.train(train_df):
                        # Save the updated model
                        self.detector.save_model('ransomguard_model.joblib')
                        logger.info(f"Model trained with {len(features_buffer)} samples")
                    
                    # Clear buffer after training
                    features_buffer = []
                
                # Decrement alert cooldown
                if alert_cooldown > 0:
                    alert_cooldown -= 1
            
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
    
    def _stats_loop(self):
        """Update system statistics periodically"""
        while True:
            try:
                time.sleep(10)  # Update every 10 seconds
                
                # Update CPU usage (simple simulation for demo)
                self.stats['cpu'] = min(100, max(1, self.stats['cpu'] + random.randint(-2, 3)))
                
                # Update memory usage (simple simulation for demo)
                self.stats['memory'] = min(1024, max(100, self.stats['memory'] + random.randint(-10, 15)))
                
                # Update files monitored count
                self.stats['files_monitored'] = len(self.monitor.entropy_samples)
                
                # Queue stats update
                self.event_queue.append(('stats', self.stats.copy()))
            
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")
    
    def stop(self):
        """Stop all monitoring and detection activities"""
        self.monitor.stop()
        logger.info("RansomGuard stopped")


def main():
    """Main entry point for RansomGuard application"""
    # Set up paths to monitor
    paths = [
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Downloads")
    ]
    
    # Initialize the ransomware guard
    guard = RansomGuard(paths_to_monitor=paths)
    
    # Set up the UI
    root = tk.Tk()
    app = RansomGuardUI(root, guard.event_queue)
    
    # Set custom icon and title
    root.title("RansomGuard - Real-time Ransomware Detection")
    try:
        # This would be replaced with a real icon in a production app
        root.iconbitmap('ransomguard.ico')
    except:
        pass
    
    try:
        # Start the UI main loop
        root.mainloop()
    finally:
        # Make sure to stop monitoring when the app is closed
        guard.stop()


if __name__ == "__main__":
    main()
