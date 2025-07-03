"""
Web Dashboard for AI Exam Monitor
Real-time monitoring dashboard with live video feed and alerts
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import base64
import numpy as np
from exam_monitor import ExamMonitor, DetectionResult, SuspiciousActivity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'exam_monitor_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
monitor = None
is_monitoring = False
current_frame = None
frame_lock = threading.Lock()

class DatabaseManager:
    def __init__(self, db_path="exam_monitor.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_violations INTEGER,
                status TEXT
            )
        ''')
        
        # Create violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP,
                activity TEXT,
                confidence REAL,
                description TEXT,
                FOREIGN KEY (session_id) REFERENCES monitoring_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self):
        """Create a new monitoring session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO monitoring_sessions (start_time, status)
            VALUES (?, ?)
        ''', (datetime.now(), 'active'))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id, violation_count):
        """End a monitoring session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE monitoring_sessions
            SET end_time = ?, total_violations = ?, status = ?
            WHERE id = ?
        ''', (datetime.now(), violation_count, 'completed', session_id))
        
        conn.commit()
        conn.close()
    
    def add_violation(self, session_id, violation):
        """Add a violation to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO violations (session_id, timestamp, activity, confidence, description)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.fromtimestamp(violation.timestamp),
            violation.activity.value,
            violation.confidence,
            violation.description
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_sessions(self, limit=10):
        """Get recent monitoring sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM monitoring_sessions
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        return sessions

db_manager = DatabaseManager()
current_session_id = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start the exam monitoring"""
    global monitor, is_monitoring, current_session_id
    
    if not is_monitoring:
        monitor = ExamMonitor()
        current_session_id = db_manager.create_session()
        is_monitoring = True
        
        # Start monitoring in a separate thread
        monitoring_thread = threading.Thread(target=run_monitoring)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        return jsonify({"status": "success", "message": "Monitoring started"})
    
    return jsonify({"status": "error", "message": "Already monitoring"})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop the exam monitoring"""
    global is_monitoring, current_session_id
    
    if is_monitoring:
        is_monitoring = False
        
        if monitor and current_session_id:
            violation_count = len(monitor.violations)
            db_manager.end_session(current_session_id, violation_count)
        
        return jsonify({"status": "success", "message": "Monitoring stopped"})
    
    return jsonify({"status": "error", "message": "Not currently monitoring"})

@app.route('/api/status')
def get_status():
    """Get current monitoring status"""
    violation_count = len(monitor.violations) if monitor else 0
    
    return jsonify({
        "is_monitoring": is_monitoring,
        "violation_count": violation_count,
        "session_id": current_session_id
    })

@app.route('/api/violations')
def get_violations():
    """Get recent violations"""
    if not monitor:
        return jsonify([])
    
    violations = []
    for violation in monitor.violations[-20:]:  # Last 20 violations
        violations.append({
            "timestamp": violation.timestamp,
            "activity": violation.activity.value,
            "confidence": violation.confidence,
            "description": violation.description
        })
    
    return jsonify(violations)

@app.route('/api/sessions')
def get_sessions():
    """Get monitoring sessions"""
    sessions = db_manager.get_recent_sessions()
    
    session_list = []
    for session in sessions:
        session_list.append({
            "id": session[0],
            "start_time": session[1],
            "end_time": session[2],
            "total_violations": session[3],
            "status": session[4]
        })
    
    return jsonify(session_list)

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame, frame_lock
    
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # 10 FPS for web streaming

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_monitoring():
    """Run the monitoring system"""
    global current_frame, frame_lock, is_monitoring, monitor, current_session_id
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while is_monitoring:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detections = monitor._process_frame(frame)
        
        # Handle detections
        for detection in detections:
            monitor._handle_detection(detection)
            
            # Add to database
            if current_session_id:
                db_manager.add_violation(current_session_id, detection)
            
            # Emit real-time alert
            socketio.emit('violation_alert', {
                'timestamp': detection.timestamp,
                'activity': detection.activity.value,
                'confidence': detection.confidence,
                'description': detection.description
            })
        
        # Annotate frame
        annotated_frame = monitor._annotate_frame(frame, detections)
        
        # Update current frame for streaming
        with frame_lock:
            current_frame = annotated_frame
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to exam monitor'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
