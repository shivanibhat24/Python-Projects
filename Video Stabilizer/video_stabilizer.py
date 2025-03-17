import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QSlider, QSpinBox, QProgressBar, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread


class VideoStabilizer:
    def __init__(self, smoothing_radius=30, border_crop=0):
        """
        Initialize the video stabilizer with configurable parameters.
        
        Args:
            smoothing_radius (int): Radius for smoothing the motion trajectory.
            border_crop (int): Number of pixels to crop from the borders to remove black edges.
        """
        self.smoothing_radius = smoothing_radius
        self.border_crop = border_crop
        self.prev_gray = None
        self.prev_to_cur_transform = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.transforms = []
        
    def _detect_features(self, frame_gray):
        """Detect features in the frame using Shi-Tomasi corner detection."""
        features = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        return features
    
    def _track_features(self, prev_gray, cur_gray, prev_features):
        """Track features using Lucas-Kanade optical flow."""
        cur_features, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_features, None
        )
        
        # Keep only valid points
        status = status.reshape(status.shape[0])
        prev_points = prev_features[status == 1]
        cur_points = cur_features[status == 1]
        
        return prev_points, cur_points
    
    def _estimate_transform(self, prev_points, cur_points):
        """Estimate the rigid transformation between two sets of points."""
        if len(prev_points) < 4 or len(cur_points) < 4:
            return None
        
        # Find transformation matrix
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            prev_points, cur_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        
        return transform_matrix
    
    def _update_trajectory(self, transform):
        """Update the trajectory based on the current transformation."""
        if transform is None:
            if not self.trajectory:
                self.trajectory.append([0, 0, 0])  # dx, dy, da
            else:
                self.trajectory.append(self.trajectory[-1])
            return
        
        # Extract translation and rotation
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        
        # Store the transformation
        self.transforms.append(transform)
        
        # Update trajectory
        if not self.trajectory:
            self.trajectory.append([dx, dy, da])
        else:
            prev_dx, prev_dy, prev_da = self.trajectory[-1]
            self.trajectory.append([prev_dx + dx, prev_dy + dy, prev_da + da])
    
    def _smooth_trajectory(self):
        """Smooth the trajectory using Savitzky-Golay filter."""
        trajectory = np.array(self.trajectory)
        
        # Smooth the trajectory
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            if len(trajectory) > self.smoothing_radius:
                smoothed_trajectory[:, i] = savgol_filter(
                    trajectory[:, i], 
                    window_length=self.smoothing_radius, 
                    polyorder=3
                )
            else:
                # If not enough points for Savitzky-Golay, use moving average
                kernel = np.ones(min(len(trajectory), 5)) / min(len(trajectory), 5)
                smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')
        
        self.smoothed_trajectory = smoothed_trajectory
    
    def _calculate_smooth_transform(self, index):
        """Calculate the smoothed transformation for the given frame index."""
        if index >= len(self.transforms):
            return None
        
        # Get the current and smoothed trajectories
        cur_trajectory = self.trajectory[index]
        smooth_trajectory = self.smoothed_trajectory[index]
        
        # Calculate the difference
        diff = smooth_trajectory - cur_trajectory
        
        # Calculate the smooth transform
        dx, dy, da = diff
        
        # Get the original transform
        transform = self.transforms[index]
        
        # Update the transform
        smooth_transform = np.copy(transform)
        smooth_transform[0, 2] += dx
        smooth_transform[1, 2] += dy
        
        # Adjust rotation
        ca = np.cos(da)
        sa = np.sin(da)
        rotation_matrix = np.array([[ca, -sa], [sa, ca]])
        
        smooth_transform[:2, :2] = np.matmul(rotation_matrix, transform[:2, :2])
        
        return smooth_transform
    
    def _apply_transform(self, frame, transform):
        """Apply the transformation to the frame."""
        h, w = frame.shape[:2]
        
        # Apply the transformation
        stabilized_frame = cv2.warpAffine(
            frame, transform, (w, h), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        
        # Crop borders if needed
        if self.border_crop > 0:
            stabilized_frame = stabilized_frame[
                self.border_crop:-self.border_crop,
                self.border_crop:-self.border_crop
            ]
        
        return stabilized_frame
    
    def stabilize_frame(self, frame):
        """Stabilize a single frame."""
        # Convert to grayscale
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame if this is the first frame
        if self.prev_gray is None:
            self.prev_gray = cur_gray
            return frame
        
        # Detect features in the previous frame
        prev_features = self._detect_features(self.prev_gray)
        
        if prev_features is None or len(prev_features) < 4:
            self.prev_gray = cur_gray
            return frame
        
        # Track features
        prev_points, cur_points = self._track_features(self.prev_gray, cur_gray, prev_features)
        
        if len(prev_points) < 4:
            self.prev_gray = cur_gray
            return frame
        
        # Estimate transformation
        transform = self._estimate_transform(prev_points, cur_points)
        
        # Update trajectory
        self._update_trajectory(transform)
        
        # Smooth trajectory
        self._smooth_trajectory()
        
        # Calculate smooth transform
        smooth_transform = self._calculate_smooth_transform(len(self.trajectory) - 1)
        
        if smooth_transform is None:
            self.prev_gray = cur_gray
            return frame
        
        # Apply transformation
        stabilized_frame = self._apply_transform(frame, smooth_transform)
        
        # Update previous frame
        self.prev_gray = cur_gray
        
        return stabilized_frame
    
    def reset(self):
        """Reset the stabilizer state."""
        self.prev_gray = None
        self.prev_to_cur_transform = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.transforms = []
        
    def plot_trajectory(self):
        """Plot the original and smoothed trajectories for analysis."""
        if not self.trajectory:
            print("No trajectory data available. Process a video first.")
            return None
        
        trajectory = np.array(self.trajectory)
        smoothed_trajectory = np.array(self.smoothed_trajectory)
        
        plt.figure(figsize=(10, 6))
        
        # Plot x, y, and rotation
        labels = ['X', 'Y', 'Rotation']
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(trajectory[:, i], 'b-', label='Original')
            plt.plot(smoothed_trajectory[:, i], 'r-', label='Smoothed')
            plt.title(f'{labels[i]} Trajectory')
            plt.legend()
            plt.grid()
        
        plt.tight_layout()
        
        # Save to a temporary file
        plot_path = 'trajectory_plot.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class StabilizeThread(QThread):
    progress_update = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    frame_processed = pyqtSignal(np.ndarray, np.ndarray)
    
    def __init__(self, stabilizer, input_path, output_path, parent=None):
        super().__init__(parent)
        self.stabilizer = stabilizer
        self.input_path = input_path
        self.output_path = output_path
        self.running = True
        
    def stop(self):
        self.running = False
        
    def run(self):
        # Reset stabilizer
        self.stabilizer.reset()
        
        # Open the video file
        cap = cv2.VideoCapture(self.input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            self.output_path, fourcc, fps, 
            (width - 2 * self.stabilizer.border_crop, height - 2 * self.stabilizer.border_crop) 
            if self.stabilizer.border_crop > 0 else (width, height)
        )
        
        # Process the video
        processed_frames = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stabilize the frame
            stabilized_frame = self.stabilizer.stabilize_frame(frame)
            
            # Emit signal for preview
            if processed_frames % 5 == 0:  # Send every 5th frame to reduce overhead
                self.frame_processed.emit(frame, stabilized_frame)
            
            # Write the stabilized frame
            out.write(stabilized_frame)
            
            processed_frames += 1
            progress = int((processed_frames / frame_count) * 100)
            self.progress_update.emit(progress)
            
        # Release everything
        cap.release()
        out.release()
        
        self.finished_signal.emit(self.output_path)


class VideoStabilizerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stabilizer = VideoStabilizer()
        self.input_path = ""
        self.output_path = ""
        self.stabilize_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Video Stabilizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Top section - File selection
        file_section = QWidget()
        file_layout = QHBoxLayout()
        file_section.setLayout(file_layout)
        
        self.input_label = QLabel("No video selected")
        self.select_btn = QPushButton("Select Video")
        self.select_btn.clicked.connect(self.select_video)
        
        file_layout.addWidget(QLabel("Input Video:"))
        file_layout.addWidget(self.input_label, 1)
        file_layout.addWidget(self.select_btn)
        
        main_layout.addWidget(file_section)
        
        # Middle section - Parameters
        param_section = QWidget()
        param_layout = QHBoxLayout()
        param_section.setLayout(param_layout)
        
        # Smoothing radius
        smooth_layout = QVBoxLayout()
        smooth_label = QLabel("Smoothing Radius:")
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(5)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setValue(30)
        self.smooth_slider.setTickPosition(QSlider.TicksBelow)
        self.smooth_slider.setTickInterval(10)
        self.smooth_spinbox = QSpinBox()
        self.smooth_spinbox.setMinimum(5)
        self.smooth_spinbox.setMaximum(100)
        self.smooth_spinbox.setValue(30)
        
        self.smooth_slider.valueChanged.connect(self.update_smooth_spinbox)
        self.smooth_spinbox.valueChanged.connect(self.update_smooth_slider)
        
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(self.smooth_slider)
        smooth_layout.addWidget(self.smooth_spinbox)
        
        # Border crop
        crop_layout = QVBoxLayout()
        crop_label = QLabel("Border Crop:")
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setMinimum(0)
        self.crop_slider.setMaximum(100)
        self.crop_slider.setValue(50)
        self.crop_slider.setTickPosition(QSlider.TicksBelow)
        self.crop_slider.setTickInterval(10)
        self.crop_spinbox = QSpinBox()
        self.crop_spinbox.setMinimum(0)
        self.crop_spinbox.setMaximum(100)
        self.crop_spinbox.setValue(50)
        
        self.crop_slider.valueChanged.connect(self.update_crop_spinbox)
        self.crop_spinbox.valueChanged.connect(self.update_crop_slider)
        
        crop_layout.addWidget(crop_label)
        crop_layout.addWidget(self.crop_slider)
        crop_layout.addWidget(self.crop_spinbox)
        
        param_layout.addLayout(smooth_layout)
        param_layout.addLayout(crop_layout)
        
        main_layout.addWidget(param_section)
        
        # Preview section
        preview_section = QSplitter(Qt.Horizontal)
        
        # Original video preview
        self.original_preview = QLabel("Original Video")
        self.original_preview.setAlignment(Qt.AlignCenter)
        self.original_preview.setMinimumSize(400, 300)
        self.original_preview.setStyleSheet("border: 1px solid #cccccc;")
        
        # Stabilized video preview
        self.stabilized_preview = QLabel("Stabilized Video")
        self.stabilized_preview.setAlignment(Qt.AlignCenter)
        self.stabilized_preview.setMinimumSize(400, 300)
        self.stabilized_preview.setStyleSheet("border: 1px solid #cccccc;")
        
        preview_section.addWidget(self.original_preview)
        preview_section.addWidget(self.stabilized_preview)
        
        main_layout.addWidget(preview_section, 1)
        
        # Bottom section - Controls and progress
        control_section = QWidget()
        control_layout = QHBoxLayout()
        control_section.setLayout(control_layout)
        
        self.stabilize_btn = QPushButton("Stabilize Video")
        self.stabilize_btn.clicked.connect(self.stabilize_video)
        self.stabilize_btn.setEnabled(False)
        
        self.plot_btn = QPushButton("Plot Trajectory")
        self.plot_btn.clicked.connect(self.plot_trajectory)
        self.plot_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_stabilization)
        self.stop_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        control_layout.addWidget(self.stabilize_btn)
        control_layout.addWidget(self.plot_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(control_section)
        
    def update_smooth_spinbox(self, value):
        self.smooth_spinbox.setValue(value)
        self.stabilizer.smoothing_radius = value
        
    def update_smooth_slider(self, value):
        self.smooth_slider.setValue(value)
        self.stabilizer.smoothing_radius = value
        
    def update_crop_spinbox(self, value):
        self.crop_spinbox.setValue(value)
        self.stabilizer.border_crop = value
        
    def update_crop_slider(self, value):
        self.crop_slider.setValue(value)
        self.stabilizer.border_crop = value
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        
        if file_path:
            self.input_path = file_path
            self.input_label.setText(os.path.basename(file_path))
            self.stabilize_btn.setEnabled(True)
            
            # Generate default output path
            base_name, ext = os.path.splitext(file_path)
            self.output_path = f"{base_name}_stabilized{ext}"
            
    def stabilize_video(self):
        # Check if input path is valid
        if not self.input_path or not os.path.exists(self.input_path):
            return
        
        # Ask for output path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Stabilized Video", self.output_path, "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        
        if not file_path:
            return
            
        self.output_path = file_path
        
        # Update UI elements
        self.stabilize_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.plot_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create and start the stabilization thread
        self.stabilizer.smoothing_radius = self.smooth_spinbox.value()
        self.stabilizer.border_crop = self.crop_spinbox.value()
        
        self.stabilize_thread = StabilizeThread(
            self.stabilizer, self.input_path, self.output_path, self
        )
        self.stabilize_thread.progress_update.connect(self.update_progress)
        self.stabilize_thread.finished_signal.connect(self.stabilization_finished)
        self.stabilize_thread.frame_processed.connect(self.update_preview)
        self.stabilize_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_preview(self, original_frame, stabilized_frame):
        # Convert frames to QImage for display
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        stabilized_rgb = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2RGB)
        
        # Scale down for preview if needed
        h, w = original_rgb.shape[:2]
        preview_h = self.original_preview.height()
        preview_w = self.original_preview.width()
        scale = min(preview_w / w, preview_h / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            original_rgb = cv2.resize(original_rgb, (new_w, new_h))
            stabilized_rgb = cv2.resize(stabilized_rgb, (new_w, new_h))
        
        # Convert to QImage
        h, w, c = original_rgb.shape
        original_qimg = QImage(original_rgb.data, w, h, w * c, QImage.Format_RGB888)
        original_pixmap = QPixmap.fromImage(original_qimg)
        self.original_preview.setPixmap(original_pixmap)
        
        h, w, c = stabilized_rgb.shape
        stabilized_qimg = QImage(stabilized_rgb.data, w, h, w * c, QImage.Format_RGB888)
        stabilized_pixmap = QPixmap.fromImage(stabilized_qimg)
        self.stabilized_preview.setPixmap(stabilized_pixmap)
        
    def stabilization_finished(self, output_path):
        # Update UI elements
        self.stabilize_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.plot_btn.setEnabled(True)
        
    def stop_stabilization(self):
        if self.stabilize_thread and self.stabilize_thread.isRunning():
            self.stabilize_thread.stop()
            self.stabilize_thread.wait()
            
        self.stabilize_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def plot_trajectory(self):
        plot_path = self.stabilizer.plot_trajectory()
        if plot_path and os.path.exists(plot_path):
            # Show plot in a new window
            pixmap = QPixmap(plot_path)
            
            # Create a new window to display the plot
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle("Trajectory Plot")
            plot_window.setGeometry(200, 200, 800, 600)
            
            # Create a label to display the plot
            plot_label = QLabel()
            plot_label.setPixmap(pixmap)
            plot_label.setAlignment(Qt.AlignCenter)
            plot_label.setScaledContents(True)
            
            plot_window.setCentralWidget(plot_label)
            plot_window.show()


def main():
    app = QApplication(sys.argv)
    window = VideoStabilizerUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
