"""
AI Exam Monitor - Main Application
Real-time exam monitoring system using computer vision and deep learning
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import queue
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exam_monitor.log'),
        logging.StreamHandler()
    ]
)

class SuspiciousActivity(Enum):
    LOOKING_AWAY = "looking_away"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE_DETECTED = "no_face_detected"
    SUSPICIOUS_OBJECTS = "suspicious_objects"
    UNUSUAL_MOVEMENT = "unusual_movement"
    PHONE_DETECTED = "phone_detected"
    PAPER_DETECTED = "paper_detected"

@dataclass
class DetectionResult:
    timestamp: float
    activity: SuspiciousActivity
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    description: str = ""

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in the image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                face_info = {
                    'bbox': (
                        int(bbox.xmin * w),
                        int(bbox.ymin * h),
                        int(bbox.width * w),
                        int(bbox.height * h)
                    ),
                    'confidence': detection.score[0]
                }
                faces.append(face_info)
        
        return faces

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """Detect human pose landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return {'landmarks': landmarks}
        
        return None

class ObjectDetector:
    def __init__(self, model_path: str = None):
        """Initialize object detector with YOLO or custom model"""
        self.model = None
        self.class_names = ['phone', 'paper', 'book', 'pen', 'calculator']
        
        # If no model provided, use a simple template matching approach
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                logging.info(f"Loaded object detection model from {model_path}")
            except Exception as e:
                logging.warning(f"Could not load model: {e}")
                self.model = None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect suspicious objects in the image"""
        detected_objects = []
        
        # Simple template matching for demonstration
        # In production, you'd use a trained YOLO or similar model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phone detection using simple features
        phone_detected = self._detect_phone_simple(gray)
        if phone_detected:
            detected_objects.append({
                'class': 'phone',
                'confidence': 0.8,
                'bbox': phone_detected
            })
        
        return detected_objects
    
    def _detect_phone_simple(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Simple phone detection using edge detection"""
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:  # Phone-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.4 < aspect_ratio < 0.8:  # Phone-like aspect ratio
                    return (x, y, w, h)
        
        return None

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key facial landmarks for gaze detection
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
    def track_gaze(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """Track gaze direction"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        gaze_info = {
            'looking_forward': True,
            'gaze_direction': 'center',
            'confidence': 0.0
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate gaze direction based on eye landmarks
            left_eye_center = self._get_eye_center(face_landmarks, self.LEFT_EYE)
            right_eye_center = self._get_eye_center(face_landmarks, self.RIGHT_EYE)
            
            # Simple gaze direction calculation
            h, w = image.shape[:2]
            
            # Check if eyes are looking towards center of screen
            eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
            eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
            
            # Normalize coordinates
            normalized_x = eye_center_x / w
            normalized_y = eye_center_y / h
            
            # Determine gaze direction
            if normalized_x < 0.3:
                gaze_info['gaze_direction'] = 'left'
                gaze_info['looking_forward'] = False
            elif normalized_x > 0.7:
                gaze_info['gaze_direction'] = 'right'
                gaze_info['looking_forward'] = False
            
            gaze_info['confidence'] = 0.8
        
        return gaze_info
    
    def _get_eye_center(self, landmarks, eye_indices):
        """Calculate the center of an eye"""
        x_coords = [landmarks.landmark[i].x for i in eye_indices]
        y_coords = [landmarks.landmark[i].y for i in eye_indices]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (center_x, center_y)

class ExamMonitor:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the exam monitoring system"""
        self.config = self._load_config(config_path)
        
        # Initialize detectors
        self.face_detector = FaceDetector()
        self.pose_detector = PoseDetector()
        self.object_detector = ObjectDetector()
        self.gaze_tracker = GazeTracker()
        
        # Monitoring state
        self.is_monitoring = False
        self.detection_queue = queue.Queue()
        self.violations = []
        
        # Thresholds
        self.gaze_away_threshold = 3.0  # seconds
        self.no_face_threshold = 2.0    # seconds
        
        # Tracking variables
        self.last_face_time = time.time()
        self.gaze_away_start = None
        
        logging.info("Exam Monitor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "detection_confidence": 0.5,
            "max_faces": 1,
            "alert_threshold": 3,
            "recording_enabled": True,
            "output_dir": "monitoring_output"
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def start_monitoring(self, camera_id: int = 0):
        """Start real-time monitoring"""
        self.is_monitoring = True
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logging.error(f"Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logging.info("Starting exam monitoring...")
        
        while self.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                break
            
            # Process frame
            detections = self._process_frame(frame)
            
            # Handle detections
            for detection in detections:
                self._handle_detection(detection)
            
            # Display frame with annotations
            annotated_frame = self._annotate_frame(frame, detections)
            cv2.imshow('Exam Monitor', annotated_frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Monitoring stopped")
    
    def _process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process a single frame and detect suspicious activities"""
        detections = []
        current_time = time.time()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Check for multiple faces
        if len(faces) > 1:
            detections.append(DetectionResult(
                timestamp=current_time,
                activity=SuspiciousActivity.MULTIPLE_FACES,
                confidence=0.9,
                description=f"Detected {len(faces)} faces"
            ))
        
        # Check for no face
        elif len(faces) == 0:
            if current_time - self.last_face_time > self.no_face_threshold:
                detections.append(DetectionResult(
                    timestamp=current_time,
                    activity=SuspiciousActivity.NO_FACE_DETECTED,
                    confidence=0.8,
                    description="No face detected for extended period"
                ))
        else:
            self.last_face_time = current_time
            
            # Track gaze for the detected face
            face = faces[0]
            gaze_info = self.gaze_tracker.track_gaze(frame, face['bbox'])
            
            if not gaze_info['looking_forward']:
                if self.gaze_away_start is None:
                    self.gaze_away_start = current_time
                elif current_time - self.gaze_away_start > self.gaze_away_threshold:
                    detections.append(DetectionResult(
                        timestamp=current_time,
                        activity=SuspiciousActivity.LOOKING_AWAY,
                        confidence=gaze_info['confidence'],
                        description=f"Looking {gaze_info['gaze_direction']}"
                    ))
            else:
                self.gaze_away_start = None
        
        # Detect objects
        objects = self.object_detector.detect_objects(frame)
        for obj in objects:
            if obj['class'] == 'phone':
                detections.append(DetectionResult(
                    timestamp=current_time,
                    activity=SuspiciousActivity.PHONE_DETECTED,
                    confidence=obj['confidence'],
                    bbox=obj['bbox'],
                    description="Phone detected"
                ))
        
        return detections
    
    def _handle_detection(self, detection: DetectionResult):
        """Handle a suspicious activity detection"""
        self.violations.append(detection)
        
        # Log the detection
        logging.warning(
            f"VIOLATION: {detection.activity.value} - "
            f"Confidence: {detection.confidence:.2f} - "
            f"Description: {detection.description}"
        )
        
        # Add to queue for real-time processing
        self.detection_queue.put(detection)
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated_frame = frame.copy()
        
        # Draw face bounding boxes
        faces = self.face_detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Face: {face['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detection alerts
        y_offset = 30
        for detection in detections:
            color = (0, 0, 255) if detection.confidence > 0.7 else (0, 255, 255)
            text = f"{detection.activity.value}: {detection.confidence:.2f}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Draw object bounding boxes
        for detection in detections:
            if detection.bbox:
                x, y, w, h = detection.bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, detection.description, 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return annotated_frame
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
    
    def generate_report(self) -> Dict:
        """Generate a monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_violations': len(self.violations),
            'violation_summary': {},
            'violations': []
        }
        
        # Count violations by type
        for violation in self.violations:
            activity = violation.activity.value
            if activity not in report['violation_summary']:
                report['violation_summary'][activity] = 0
            report['violation_summary'][activity] += 1
            
            report['violations'].append({
                'timestamp': violation.timestamp,
                'activity': activity,
                'confidence': violation.confidence,
                'description': violation.description
            })
        
        return report

def main():
    """Main function to run the exam monitor"""
    monitor = ExamMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logging.info("Monitoring interrupted by user")
    finally:
        # Generate and save report
        report = monitor.generate_report()
        
        with open(f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Report generated: {report['total_violations']} violations detected")

if __name__ == "__main__":
    main()
