import os
import cv2
import mediapipe as mp
import numpy as np
import time
import torch
from gtts import gTTS
from ultralytics import YOLO
import pygame

class SignLanguageTranslator:
    def __init__(self):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize YOLO model for sign language detection
        self.model = YOLO("yolov8n.pt")  # Start with pre-trained model
        # Note: In real implementation, you would train or fine-tune on sign language dataset
        
        # Dictionary mapping gestures to meanings
        self.gesture_dict = {
            "palm_open": "hello",
            "fist_closed": "yes",
            "index_pointing": "I/me",
            "thumbs_up": "good",
            "palm_down": "no",
            "victory": "peace",
            "pinky_thumb": "call me",
            "index_thumb": "drink"
        }
        
        # Variables for gesture tracking
        self.current_gesture = None
        self.gesture_start_time = 0
        self.gesture_threshold = 1.0  # seconds
        self.sentence = []
        self.last_spoken_time = 0
        self.speak_cooldown = 3.0  # seconds
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
    
    def detect_hand_gesture(self, hand_landmarks):
        """Analyze hand landmarks to determine the gesture"""
        # Extract key points 
        if not hand_landmarks:
            return "unknown"
            
        # Get landmark positions
        landmarks = []
        for point in hand_landmarks.landmark:
            landmarks.append([point.x, point.y, point.z])
        
        # Simple gesture recognition based on finger positions
        # This is a simplified version - a real system would use more complex analysis
        
        # Thumb tip, index tip, middle tip, ring tip, pinky tip
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        # Thumb IP, index PIP, middle PIP, ring PIP, pinky PIP (middle joints)
        middle_joints = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
        # Wrist position
        wrist = landmarks[0]
        
        # Check if fingers are extended
        fingers_extended = []
        
        # Special check for thumb
        thumb_tip = fingertips[0]
        thumb_mcp = landmarks[2]  # Thumb MCP joint
        wrist_to_mcp_x = thumb_mcp[0] - wrist[0]
        mcp_to_tip_x = thumb_tip[0] - thumb_mcp[0]
        # If the thumb is extended outward
        fingers_extended.append(wrist_to_mcp_x * mcp_to_tip_x > 0 and abs(thumb_tip[0] - wrist[0]) > 0.1)
        
        # For other fingers, check if fingertip is above the middle joint
        for i in range(1, 5):
            fingers_extended.append(fingertips[i][1] < middle_joints[i][1])
        
        # Identify gestures based on extended fingers
        if all(fingers_extended):
            return "palm_open"
        elif not any(fingers_extended):
            return "fist_closed"
        elif fingers_extended[1] and not any(fingers_extended[2:]) and not fingers_extended[0]:
            return "index_pointing"
        elif fingers_extended[0] and not any(fingers_extended[1:]):
            return "thumbs_up"
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not fingers_extended[3] and not fingers_extended[4]:
            return "victory"
        elif fingers_extended[0] and fingers_extended[4] and not any(fingers_extended[1:4]):
            return "pinky_thumb"
        elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
            return "index_thumb"
        else:
            return "unknown"
    
    def text_to_speech(self, text):
        """Convert text to speech and play it"""
        if not text:
            return
            
        # Create temporary file
        tts = gTTS(text=text, lang='en')
        temp_file = "temp_speech.mp3"
        tts.save(temp_file)
        
        # Play the sound
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up the temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    def process_frame(self, frame):
        """Process each video frame"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture
                gesture = self.detect_hand_gesture(hand_landmarks)
                
                # Track gestures over time
                current_time = time.time()
                
                if gesture != "unknown":
                    if self.current_gesture == gesture:
                        # If the gesture has been held long enough
                        if current_time - self.gesture_start_time >= self.gesture_threshold:
                            if gesture in self.gesture_dict and self.gesture_dict[gesture] not in self.sentence:
                                self.sentence.append(self.gesture_dict[gesture])
                                print(f"Recognized: {self.gesture_dict[gesture]}")
                                
                                # Reset timer after adding to sentence
                                self.gesture_start_time = current_time
                    else:
                        # New gesture detected
                        self.current_gesture = gesture
                        self.gesture_start_time = current_time
                        
                # Display detected gesture on frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.current_gesture = None
        
        # Display current sentence
        sentence_text = " ".join(self.sentence)
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check if it's time to speak the sentence
        current_time = time.time()
        if self.sentence and current_time - self.last_spoken_time >= self.speak_cooldown:
            if len(self.sentence) >= 2:  # Speak when we have at least 2 words
                self.text_to_speech(sentence_text)
                self.last_spoken_time = current_time
                self.sentence = []  # Clear after speaking
        
        # Also run YOLO detector for additional sign detection
        # (to show integration - in practice would use specialized sign language model)
        results = self.model(frame)
        
        # Draw YOLO detections
        annotated_frame = results[0].plot()
        
        return annotated_frame
    
    def run(self):
        """Main loop to capture and process video"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display controls info
            cv2.putText(processed_frame, "Press 'c' to clear sentence", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(processed_frame, "Press 's' to speak sentence", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(processed_frame, "Press 'q' to quit", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Sign Language Translator', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence = []
            elif key == ord('s'):
                sentence_text = " ".join(self.sentence)
                if sentence_text:
                    self.text_to_speech(sentence_text)
                    self.sentence = []
                    self.last_spoken_time = time.time()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

class SignLanguageDataCollector:
    """Tool to help collect training data for sign language recognition"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create directory for data collection
        self.data_dir = "sign_language_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_data(self, sign_name, frames_to_capture=50):
        """Collect data for a specific sign"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        frame_count = 0
        countdown = 3
        countdown_start = time.time()
        collecting = False
        
        while frame_count < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display countdown before starting to collect
            current_time = time.time()
            if not collecting:
                time_left = int(countdown - (current_time - countdown_start))
                if time_left <= 0:
                    collecting = True
                    print(f"Starting to collect data for '{sign_name}'")
                else:
                    cv2.putText(frame, f"Starting in {time_left}...", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Process frame with MediaPipe hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if collecting:
                # Save frame and landmark data
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    frame_count += 1
                    
                    # Save image
                    image_path = os.path.join(sign_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(image_path, frame)
                    
                    # Save landmark data
                    landmark_path = os.path.join(sign_dir, f"landmarks_{frame_count}.txt")
                    with open(landmark_path, 'w') as f:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for point in hand_landmarks.landmark:
                                f.write(f"{point.x},{point.y},{point.z}\n")
                    
                    cv2.putText(frame, f"Capturing: {frame_count}/{frames_to_capture}", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection complete for '{sign_name}'. Captured {frame_count} frames.")

class SignLanguageTrainer:
    """Train a model for sign language recognition"""
    def __init__(self, data_dir="sign_language_data"):
        self.data_dir = data_dir
        
    def prepare_yolo_dataset(self):
        """Prepare dataset for YOLO training"""
        # In a real implementation, this would:
        # 1. Convert collected images to YOLO format
        # 2. Create label files
        # 3. Set up training config
        print("Preparing YOLO dataset...")
        
    def train_model(self):
        """Train the YOLO model"""
        # In a real implementation, this would:
        # 1. Set up YOLO training config
        # 2. Run training on the dataset
        print("Training model...")
        # model = YOLO("yolov8n.yaml")  # build a new model from scratch
        # model.train(data="dataset.yaml", epochs=100)

def main():
    print("Sign Language Translator")
    print("=======================")
    print("1. Run Sign Language Translator")
    print("2. Collect Training Data")
    print("3. Train Model")
    print("0. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == "1":
        translator = SignLanguageTranslator()
        translator.run()
    elif choice == "2":
        collector = SignLanguageDataCollector()
        sign_name = input("Enter sign name to collect data for: ")
        frames = int(input("Enter number of frames to capture (default 50): ") or "50")
        collector.collect_data(sign_name, frames)
    elif choice == "3":
        trainer = SignLanguageTrainer()
        trainer.prepare_yolo_dataset()
        trainer.train_model()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
