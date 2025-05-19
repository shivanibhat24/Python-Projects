import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import argparse
import os

class WhiteboardCapture:
    def __init__(self, yolo_model_path=None):
        """
        Initialize the whiteboard capture system.
        
        Args:
            yolo_model_path: Path to a custom YOLO model. If None, uses a pre-trained model.
        """
        # Load YOLO model - either use a custom whiteboard-trained model or the general YOLOv8 model
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.model = YOLO(yolo_model_path)
            print(f"Loaded custom YOLO model from {yolo_model_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Use YOLOv8 nano by default
            print("Loaded default YOLOv8n model")
        
        # Configure pytesseract path if needed (uncomment and modify for Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def detect_whiteboard(self, image):
        """
        Detect whiteboards in the image using YOLO.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of bounding boxes for detected whiteboards
        """
        # In the default YOLO model, class 73 is 'laptop', 62 is 'tv', and 72 is 'refrigerator'
        # We'll use these as a proxy for detecting whiteboard-like rectangles
        # Or use a custom-trained YOLO model specifically for whiteboards
        
        results = self.model(image)
        detections = []
        
        # Parse results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # If using default model, look for rectangular objects
                class_id = int(box.cls.item())
                conf = box.conf.item()
                
                # If using the default model, filter for objects that might be whiteboard-like
                # If using a custom model trained on whiteboards, you'd check for whiteboard class
                if ((class_id in [62, 72, 73] and conf > 0.4) or  # Default model: TV, fridge, laptop 
                    (class_id == 0 and conf > 0.5)):  # Custom model: whiteboard would be class 0
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': class_id
                    })
        
        # If no objects detected with default approach, fall back to detecting rectangular whitish regions
        if not detections:
            detections = self.detect_rectangular_whitish_regions(image)
            
        return detections
    
    def detect_rectangular_whitish_regions(self, image):
        """
        Fallback method to detect whiteboard-like regions based on color and shape.
        
        Args:
            image: Input image
            
        Returns:
            List of potential whiteboard bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find bright regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (avoid small regions)
            if w > image.shape[1] * 0.15 and h > image.shape[0] * 0.15:
                # Filter by aspect ratio (whiteboards are typically rectangular)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:
                    # Check if region is whitish
                    roi = image[y:y+h, x:x+w]
                    mean_color = np.mean(roi, axis=(0, 1))
                    
                    # High value in all channels suggests whitish color
                    if np.all(mean_color > 180):
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.5,
                            'class_id': -1  # Custom class for whiteboard detection
                        })
        
        return detections

    def enhance_whiteboard(self, image, bbox):
        """
        Enhance the whiteboard region for better text recognition.
        
        Args:
            image: Original image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Enhanced whiteboard image
        """
        x1, y1, x2, y2 = bbox
        whiteboard = image[y1:y2, x1:x2].copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(whiteboard, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle lighting variations
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Morphological operations to enhance text
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        # Invert back for OCR (black text on white background)
        enhanced = cv2.bitwise_not(dilated)
        
        return enhanced

    def extract_text(self, image):
        """
        Extract text from the enhanced whiteboard image.
        
        Args:
            image: Enhanced whiteboard image
            
        Returns:
            Extracted text
        """
        # OCR configuration for better whiteboard text recognition
        custom_config = r'--oem 3 --psm 6 -l eng'
        
        # Extract text
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text

    def process_image(self, image_path):
        """
        Process an image to detect whiteboards and extract text.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with results including detected boards and text
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect whiteboards
        detections = self.detect_whiteboard(image)
        
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            # Enhance whiteboard region
            enhanced = self.enhance_whiteboard(image, bbox)
            
            # Extract text
            text = self.extract_text(enhanced)
            
            # Save enhanced whiteboard image
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            enhanced_path = f"{output_dir}/{basename}_whiteboard_{i}.jpg"
            cv2.imwrite(enhanced_path, enhanced)
            
            # Draw bounding box on original image
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            results.append({
                'bbox': bbox,
                'text': text,
                'enhanced_image_path': enhanced_path
            })
        
        # Save annotated image
        annotated_path = f"{output_dir}/{basename}_annotated.jpg"
        cv2.imwrite(annotated_path, image)
        
        return {
            'original_image': image_path,
            'annotated_image': annotated_path,
            'whiteboards': results
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Smart Whiteboard Note Capture')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', help='Path to custom YOLO model (optional)')
    args = parser.parse_args()
    
    # Initialize whiteboard capture system
    whiteboard_capture = WhiteboardCapture(args.model)
    
    try:
        # Process image
        results = whiteboard_capture.process_image(args.image)
        
        # Print results
        print(f"\nProcessed image: {results['original_image']}")
        print(f"Annotated image saved to: {results['annotated_image']}")
        print(f"Detected {len(results['whiteboards'])} whiteboard(s)")
        
        for i, board in enumerate(results['whiteboards']):
            print(f"\nWhiteboard {i+1}:")
            print(f"Enhanced image saved to: {board['enhanced_image_path']}")
            print(f"Extracted text:\n{board['text']}")
    
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
