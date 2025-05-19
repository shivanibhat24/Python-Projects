import os
import cv2
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from torchvision import transforms
from rasterio.windows import Window
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle

class SatelliteDebrisDetector:
    """
    Satellite Debris Detection System
    
    This class provides functionality to detect space debris from satellite camera footage
    using a combination of deep learning (YOLOv8) and traditional computer vision techniques.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25, device=None):
        """
        Initialize the debris detector
        
        Args:
            model_path (str): Path to a pre-trained model. If None, will use YOLOv8n.
            confidence_threshold (float): Detection confidence threshold
            device (str): Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.conf_threshold = confidence_threshold
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load or initialize the model
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading default YOLOv8n model")
            self.model = YOLO('yolov8n.pt')
            
        # Create an output directory for saving results
        self.results_dir = os.path.join(os.getcwd(), "debris_detection_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def preprocess_image(self, image):
        """
        Preprocess satellite image to enhance potential debris features
        
        Args:
            image (numpy.ndarray): Input satellite image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply noise reduction
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel image for model input
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return processed

    def detect_debris(self, image_path, save_results=True, enhance_preprocessing=False):
        """
        Detect space debris in a satellite image
        
        Args:
            image_path (str): Path to the satellite image
            save_results (bool): Whether to save detection results
            enhance_preprocessing (bool): Apply advanced preprocessing
            
        Returns:
            dict: Detection results including bounding boxes and confidence scores
        """
        # Check if the file is a geospatial raster
        is_geospatial = Path(image_path).suffix.lower() in ['.tif', '.tiff']
        
        if is_geospatial:
            # Open geospatial image with rasterio
            with rasterio.open(image_path) as src:
                # Read the image data
                image = src.read()
                # Convert from (bands, height, width) to (height, width, bands)
                image = np.transpose(image, (1, 2, 0))
                # If more than 3 bands, take only RGB (if available)
                if image.shape[2] > 3:
                    image = image[:, :, 0:3]
                # Normalize to 0-255 range for OpenCV
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Save geospatial metadata for later use
                geo_transform = src.transform
                crs = src.crs
        else:
            # Read regular image file
            image = cv2.imread(image_path)
            geo_transform = None
            crs = None
            
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")
        
        # Apply preprocessing if requested
        if enhance_preprocessing:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image.copy()
            
        # Detect objects using YOLOv8
        detections = self.model.predict(
            source=processed_image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Extract results
        result = {
            'image_path': image_path,
            'original_image': image,
            'processed_image': processed_image,
            'detections': [],
            'geo_transform': geo_transform,
            'crs': crs
        }
        
        # Process detection results
        if len(detections.boxes) > 0:
            for i, box in enumerate(detections.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = detections.names[class_id]
                
                # Filter for likely debris (small, bright objects)
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Extract the region of the detected object
                roi = image[y1:y2, x1:x2]
                
                # Additional verification (e.g., checking for brightness)
                is_debris = self.verify_debris(roi, class_name)
                
                if is_debris:
                    result['detections'].append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'width': box_width,
                        'height': box_height
                    })
        
        # Apply custom debris detection method (brightness-based)
        custom_detections = self.detect_bright_spots(image)
        for detection in custom_detections:
            # Check if this detection overlaps with existing ones
            if not self._is_overlapping(detection, result['detections']):
                result['detections'].append(detection)
        
        # Save results if requested
        if save_results:
            self._save_detection_results(image_path, result)
        
        return result
    
    def verify_debris(self, roi, class_name):
        """
        Additional verification to confirm if detection is likely space debris
        
        Args:
            roi (numpy.ndarray): Region of interest (cropped detection)
            class_name (str): Class name from YOLO model
            
        Returns:
            bool: True if likely debris, False otherwise
        """
        # For generic objects, apply additional checks
        if class_name not in ['debris', 'satellite', 'spacecraft']:
            # Check brightness - debris often appears bright against dark space
            if len(roi.shape) == 3:  # Color image
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
                
            # Calculate brightness metrics
            mean_brightness = np.mean(gray_roi)
            max_brightness = np.max(gray_roi)
            
            # Check shape properties (debris often small and point-like)
            height, width = roi.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            # Decision criteria for likely debris
            is_bright = mean_brightness > 150 or max_brightness > 200
            is_compact = 0.5 < aspect_ratio < 2.0 and width < 100 and height < 100
            
            return is_bright and is_compact
        
        # If the model already classified it as debris-related, trust it
        return True
    
    def detect_bright_spots(self, image, min_brightness=200, min_area=5, max_area=100):
        """
        Detect bright spots in the image that might be space debris
        
        Args:
            image (numpy.ndarray): Input image
            min_brightness (int): Minimum brightness threshold
            min_area (int): Minimum area of debris
            max_area (int): Maximum area of debris
            
        Returns:
            list: List of detected debris objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply thresholding to identify bright spots
        _, thresh = cv2.threshold(gray, min_brightness, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        debris_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid and average brightness
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                
                debris_detections.append({
                    'box': [x, y, x+w, y+h],
                    'confidence': mean_brightness / 255.0,  # Normalize to 0-1
                    'class_id': 0,  # Custom detection class
                    'class_name': 'bright_debris',
                    'width': w,
                    'height': h
                })
                
        return debris_detections
    
    def _is_overlapping(self, detection, existing_detections, iou_threshold=0.3):
        """
        Check if a detection overlaps with any existing detections
        
        Args:
            detection (dict): New detection
            existing_detections (list): List of existing detections
            iou_threshold (float): IoU threshold for considering overlap
            
        Returns:
            bool: True if overlapping, False otherwise
        """
        new_box = detection['box']
        
        for existing in existing_detections:
            existing_box = existing['box']
            
            # Calculate intersection
            x_left = max(new_box[0], existing_box[0])
            y_top = max(new_box[1], existing_box[1])
            x_right = min(new_box[2], existing_box[2])
            y_bottom = min(new_box[3], existing_box[3])
            
            if x_right < x_left or y_bottom < y_top:
                continue  # No overlap
                
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas
            new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            existing_area = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
            
            # Calculate IoU
            union = new_area + existing_area - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                return True
                
        return False
        
    def _save_detection_results(self, image_path, result):
        """
        Save detection results, including annotated images
        
        Args:
            image_path (str): Path to the original image
            result (dict): Detection results
        """
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_basename = f"{base_filename}_{timestamp}"
        
        # Create annotated image
        annotated = result['original_image'].copy()
        
        # Draw bounding boxes
        for detection in result['detections']:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(
                annotated, 
                (box[0], box[1]), 
                (box[2], box[3]), 
                (0, 255, 0), 
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Save annotated image
        annotated_path = os.path.join(self.results_dir, f"{output_basename}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        
        # If there were detections, save detection data
        if result['detections']:
            # Save detection information to a text file
            info_path = os.path.join(self.results_dir, f"{output_basename}_detections.txt")
            with open(info_path, 'w') as f:
                f.write(f"Debris Detection Results for {image_path}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of detections: {len(result['detections'])}\n\n")
                
                for i, detection in enumerate(result['detections']):
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Class: {detection['class_name']}\n")
                    f.write(f"  Confidence: {detection['confidence']:.4f}\n")
                    f.write(f"  Bounding Box: {detection['box']}\n")
                    f.write(f"  Dimensions: {detection['width']}x{detection['height']} pixels\n\n")
        
        print(f"Results saved to {self.results_dir}")

    def train_custom_model(self, dataset_path, epochs=50, img_size=640):
        """
        Train a custom YOLOv8 model for debris detection
        
        Args:
            dataset_path (str): Path to the YOLOv8 format dataset
            epochs (int): Number of training epochs
            img_size (int): Image size for training
            
        Returns:
            str: Path to the trained model
        """
        # Load a small YOLOv8 model as base
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=img_size,
            device=self.device,
            name='debris_detector',
            verbose=True
        )
        
        # Path to the best model
        best_model_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
        
        # Update the detector with the new model
        self.model = YOLO(best_model_path)
        
        return best_model_path
    
    def process_video(self, video_path, output_path=None, frame_interval=1):
        """
        Process a video for debris detection
        
        Args:
            video_path (str): Path to the video file
            output_path (str): Path for the output video
            frame_interval (int): Process every nth frame
            
        Returns:
            str: Path to the output video
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"{video_name}_{timestamp}_debris.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track debris across frames
        tracked_debris = []
        frame_count = 0
        
        print(f"Processing video with {total_frames} frames...")
        
        # Process the video
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Detect debris in the current frame
                frame_copy = frame.copy()
                result = self.model.predict(
                    source=frame_copy,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False
                )[0]
                
                # Draw bounding boxes
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    # Verify if it's likely debris
                    roi = frame[y1:y2, x1:x2]
                    if self.verify_debris(roi, class_name):
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
                        # Track this debris
                        tracked_debris.append({
                            'frame': frame_count,
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': class_name
                        })
            
            # Write the frame to output video
            writer.write(frame)
            
            # Update progress indicator
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
        
        # Release resources
        cap.release()
        writer.release()
        
        # Save tracking data
        tracking_path = os.path.join(self.results_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_tracking.txt")
        with open(tracking_path, 'w') as f:
            f.write(f"Debris Tracking Results for {video_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of detections: {len(tracked_debris)}\n\n")
            
            for i, debris in enumerate(tracked_debris):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Frame: {debris['frame']}\n")
                f.write(f"  Class: {debris['class_name']}\n")
                f.write(f"  Confidence: {debris['confidence']:.4f}\n")
                f.write(f"  Bounding Box: {debris['box']}\n\n")
        
        print(f"Video processing complete. Output saved to {output_path}")
        return output_path
    
    def process_folder(self, folder_path, save_results=True, recursive=False):
        """
        Process all images in a folder
        
        Args:
            folder_path (str): Path to the folder containing images
            save_results (bool): Whether to save results
            recursive (bool): Process images in subfolders
            
        Returns:
            dict: Dictionary mapping image paths to detection results
        """
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        if recursive:
            image_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if os.path.splitext(file.lower())[1] in image_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
        
        results = {}
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_path}")
            try:
                result = self.detect_debris(image_path, save_results=save_results)
                results[image_path] = result
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        return results
    
    def generate_report(self, results, output_path=None):
        """
        Generate a comprehensive report of debris detections
        
        Args:
            results (dict): Detection results from process_folder
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"debris_report_{timestamp}.html")
        
        # Count total detections
        total_detections = sum(len(result['detections']) for result in results.values())
        
        # Generate HTML report
        with open(output_path, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('    <title>Space Debris Detection Report</title>\n')
            f.write('    <style>\n')
            f.write('        body { font-family: Arial, sans-serif; margin: 20px; }\n')
            f.write('        h1, h2 { color: #2c3e50; }\n')
            f.write('        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }\n')
            f.write('        .detection { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }\n')
            f.write('        table { border-collapse: collapse; width: 100%; }\n')
            f.write('        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
            f.write('        th { background-color: #f2f2f2; }\n')
            f.write('        .image-container { margin-top: 15px; }\n')
            f.write('    </style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            
            # Header
            f.write(f'    <h1>Space Debris Detection Report</h1>\n')
            f.write(f'    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            
            # Summary
            f.write('    <div class="summary">\n')
            f.write(f'        <h2>Summary</h2>\n')
            f.write(f'        <p>Total images processed: {len(results)}</p>\n')
            f.write(f'        <p>Total debris detections: {total_detections}</p>\n')
            f.write('    </div>\n')
            
            # Detailed results
            f.write('    <h2>Detection Details</h2>\n')
            
            for image_path, result in results.items():
                f.write('    <div class="detection">\n')
                f.write(f'        <h3>{os.path.basename(image_path)}</h3>\n')
                f.write(f'        <p>Number of detections: {len(result["detections"])}</p>\n')
                
                if result['detections']:
                    f.write('        <table>\n')
                    f.write('            <tr><th>ID</th><th>Class</th><th>Confidence</th><th>Size (px)</th></tr>\n')
                    
                    for i, detection in enumerate(result['detections']):
                        f.write(f'            <tr>\n')
                        f.write(f'                <td>{i+1}</td>\n')
                        f.write(f'                <td>{detection["class_name"]}</td>\n')
                        f.write(f'                <td>{detection["confidence"]:.4f}</td>\n')
                        f.write(f'                <td>{detection["width"]}x{detection["height"]}</td>\n')
                        f.write(f'            </tr>\n')
                    
                    f.write('        </table>\n')
                    
                    # Add thumbnail link to full-size annotated image
                    annotated_filename = os.path.join(
                        self.results_dir, 
                        f"{os.path.splitext(os.path.basename(image_path))[0]}_*_annotated.jpg"
                    )
                    
                    # Simplify by just showing a placeholder for the image
                    f.write('        <div class="image-container">\n')
                    f.write('            <p>Annotated image available in results directory</p>\n')
                    f.write('        </div>\n')
                
                f.write('    </div>\n')
            
            f.write('</body>\n')
            f.write('</html>\n')
        
        print(f"Report generated at {output_path}")
        return output_path

# Example usage script
def main():
    # Initialize the detector
    detector = SatelliteDebrisDetector(confidence_threshold=0.3)
    
    # Example: Process a single image
    result = detector.detect_debris("sample_satellite_image.jpg")
    print(f"Detected {len(result['detections'])} potential debris objects")
    
    # Example: Process a folder of images
    # results = detector.process_folder("satellite_images")
    # detector.generate_report(results)
    
    # Example: Process a video
    # detector.process_video("satellite_footage.mp4")
    
    # Example: Train a custom model (if you have labeled data)
    # detector.train_custom_model("debris_dataset")

if __name__ == "__main__":
    main()
