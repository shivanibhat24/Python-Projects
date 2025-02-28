import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QWidget, 
                            QLabel, QComboBox, QSlider, QCheckBox, QGroupBox,
                            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                            QSplitter, QFrame, QProgressBar, QSpinBox, QGridLayout,
                            QLineEdit, QTextEdit, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont
import time
import matplotlib
matplotlib.use('Qt5Agg')

# Import your model
try:
    from model import RenewableEnergyPredictiveMaintenance
except ImportError:
    QMessageBox.critical(None, "Import Error", 
                         "Could not import RenewableEnergyPredictiveMaintenance from model.py. "
                         "Please ensure the file is in the same directory as this application.")
    sys.exit(1)

# Set up dark matplotlib style
plt.style.use('dark_background')

class ModelTrainingThread(QThread):
    """Worker thread for training models without freezing UI"""
    update_progress = pyqtSignal(int)
    training_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, pm_system, model_type, epochs):
        super().__init__()
        self.pm_system = pm_system
        self.model_type = model_type
        self.epochs = epochs
        
    def run(self):
        try:
            history = None
            
            # Update progress in chunks to give visual feedback
            progress_step = 100 // self.epochs
            
            if self.model_type == "autoencoder":
                class ProgressCallback(tensorflow.keras.callbacks.Callback):
                    def __init__(self, thread):
                        super().__init__()
                        self.thread = thread
                        
                    def on_epoch_end(self, epoch, logs=None):
                        self.thread.update_progress.emit((epoch + 1) * progress_step)
                
                callback = ProgressCallback(self)
                history = self.pm_system.train_autoencoder(epochs=self.epochs, 
                                                          callbacks=[callback])
                
            elif self.model_type == "rul":
                class ProgressCallback(tensorflow.keras.callbacks.Callback):
                    def __init__(self, thread):
                        super().__init__()
                        self.thread = thread
                        
                    def on_epoch_end(self, epoch, logs=None):
                        self.thread.update_progress.emit((epoch + 1) * progress_step)
                
                callback = ProgressCallback(self)
                history = self.pm_system.train_rul_model(epochs=self.epochs,
                                                        callbacks=[callback])
            
            self.update_progress.emit(100)
            self.training_complete.emit({"history": history, "model_type": self.model_type})
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class RenewableEnergyPredictiveMaintenanceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pm_system = None
        self.current_data = None
        self.anomaly_results = None
        self.rul_results = None
        self.maintenance_plan = None
        self.dark_mode = True  # Start with dark mode by default
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('Renewable Energy Predictive Maintenance')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header with app title and theme toggle
        header_layout = QHBoxLayout()
        
        # App title with icon
        title_label = QLabel("Renewable Energy Predictive Maintenance")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label, 1)
        
        # Theme toggle switch
        self.theme_toggle = QPushButton()
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(self.dark_mode)
        self.theme_toggle.setFixedSize(100, 30)
        self.update_theme_button()
        self.theme_toggle.clicked.connect(self.toggle_theme)
        
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Light"))
        theme_layout.addWidget(self.theme_toggle)
        theme_layout.addWidget(QLabel("Dark"))
        theme_container = QWidget()
        theme_container.setLayout(theme_layout)
        header_layout.addWidget(theme_container, 0)
        
        main_layout.addLayout(header_layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Data tab
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data Import & Preprocessing")
        self.setup_data_tab()
        
        # Model Training tab
        self.model_tab = QWidget()
        self.tabs.addTab(self.model_tab, "Model Training")
        self.setup_model_tab()
        
        # Analysis tab
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis & Results")
        self.setup_analysis_tab()
        
        # Set the initial theme
        self.apply_theme()
        
        # Disable tabs that require data to be loaded first
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
    
    def update_theme_button(self):
        """Update the theme toggle button text based on current theme"""
        if self.dark_mode:
            self.theme_toggle.setText("Dark Mode")
        else:
            self.theme_toggle.setText("Light Mode")
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.dark_mode = self.theme_toggle.isChecked()
        self.update_theme_button()
        self.apply_theme()
        
        # Also update all matplotlib figures
        for canvas in self.findChildren(FigureCanvas):
            fig = canvas.figure
            if self.dark_mode:
                fig.set_facecolor("#2D2D30")
                for ax in fig.get_axes():
                    ax.set_facecolor("#2D2D30")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    if hasattr(ax, 'legend_') and ax.legend_ is not None:
                        ax.legend_.set_frame_on(True)
                        ax.legend_.get_frame().set_facecolor("#2D2D30")
                        ax.legend_.get_frame().set_edgecolor("gray")
                        for text in ax.legend_.get_texts():
                            text.set_color("white")
            else:
                fig.set_facecolor("white")
                for ax in fig.get_axes():
                    ax.set_facecolor("white")
                    ax.tick_params(colors="black")
                    ax.xaxis.label.set_color("black")
                    ax.yaxis.label.set_color("black")
                    ax.title.set_color("black")
                    if hasattr(ax, 'legend_') and ax.legend_ is not None:
                        ax.legend_.set_frame_on(True)
                        ax.legend_.get_frame().set_facecolor("white")
                        ax.legend_.get_frame().set_edgecolor("black")
                        for text in ax.legend_.get_texts():
                            text.set_color("black")
            canvas.draw()
    
    def apply_theme(self):
        """Apply the current theme to the application"""
        app = QApplication.instance()
        palette = QPalette()
        
        if self.dark_mode:
            # Dark theme
            palette.setColor(QPalette.Window, QColor(45, 45, 48))
            palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
            palette.setColor(QPalette.Base, QColor(30, 30, 30))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, QColor(27, 27, 27))
            palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
            palette.setColor(QPalette.Text, QColor(212, 212, 212))
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
            palette.setColor(QPalette.Link, QColor(3, 155, 229))
            palette.setColor(QPalette.Highlight, QColor(3, 155, 229))
            palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
            palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
            palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))
            
            # Set stylesheet for additional components
            self.setStyleSheet("""
                QTabWidget::pane { 
                    border: 1px solid #3E3E40; 
                    background-color: #2D2D30; 
                }
                QTabBar::tab {
                    background-color: #3E3E40;
                    color: #D4D4D4;
                    border: 1px solid #3E3E40;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #2D2D30;
                    border-bottom-color: #2D2D30;
                }
                QGroupBox {
                    border: 1px solid #3E3E40;
                    border-radius: 5px;
                    margin-top: 1em;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #1C86EE;
                }
                QPushButton:pressed {
                    background-color: #0063B1;
                }
                QPushButton:disabled {
                    background-color: #4A4A4A;
                    color: #9D9D9D;
                }
                QComboBox, QSpinBox, QLineEdit {
                    background-color: #333333;
                    color: #D4D4D4;
                    border: 1px solid #3E3E40;
                    border-radius: 4px;
                    padding: 5px;
                }
                QTableWidget {
                    gridline-color: #3E3E40;
                    background-color: #252526;
                    border: 1px solid #3E3E40;
                }
                QTableWidget::item {
                    padding: 5px;
                }
                QTableWidget::item:selected {
                    background-color: #0078D7;
                }
                QHeaderView::section {
                    background-color: #3E3E40;
                    color: #D4D4D4;
                    padding: 5px;
                    border: 1px solid #252526;
                }
                QProgressBar {
                    border: 1px solid #3E3E40;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #252526;
                }
                QProgressBar::chunk {
                    background-color: #0078D7;
                    width: 1px;
                }
                QSplitter::handle {
                    background-color: #3E3E40;
                }
                QTextEdit {
                    background-color: #252526;
                    color: #D4D4D4;
                    border: 1px solid #3E3E40;
                }
            """)
            
            # Set matplotlib dark style
            plt.style.use('dark_background')
            
        else:
            # Light theme
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
            palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
            palette.setColor(QPalette.Link, QColor(0, 122, 204))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            palette.setColor(QPalette.Active, QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
            palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
            palette.setColor(QPalette.Disabled, QPalette.Light, QColor(180, 180, 180))
            
            # Set stylesheet for additional components
            self.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #C0C0C0;
                    background-color: #F0F0F0;
                }
                QTabBar::tab {
                    background-color: #E1E1E1;
                    color: #333333;
                    border: 1px solid #C0C0C0;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #F0F0F0;
                    border-bottom-color: #F0F0F0;
                }
                QGroupBox {
                    border: 1px solid #C0C0C0;
                    border-radius: 5px;
                    margin-top: 1em;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #1C86EE;
                }
                QPushButton:pressed {
                    background-color: #0063B1;
                }
                QPushButton:disabled {
                    background-color: #E1E1E1;
                    color: #A0A0A0;
                }
                QComboBox, QSpinBox, QLineEdit {
                    background-color: white;
                    color: black;
                    border: 1px solid #C0C0C0;
                    border-radius: 4px;
                    padding: 5px;
                }
                QTableWidget {
                    gridline-color: #D0D0D0;
                    background-color: white;
                    border: 1px solid #C0C0C0;
                }
                QTableWidget::item {
                    padding: 5px;
                }
                QTableWidget::item:selected {
                    background-color: #0078D7;
                    color: white;
                }
                QHeaderView::section {
                    background-color: #E1E1E1;
                    color: #333333;
                    padding: 5px;
                    border: 1px solid #D0D0D0;
                }
                QProgressBar {
                    border: 1px solid #C0C0C0;
                    border-radius: 4px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0078D7;
                    width: 1px;
                }
                QSplitter::handle {
                    background-color: #C0C0C0;
                }
                QTextEdit {
                    background-color: white;
                    color: black;
                    border: 1px solid #C0C0C0;
                }
            """)
            
            # Set matplotlib light style
            plt.style.use('default')
            
        app.setPalette(palette)
        
    def setup_data_tab(self):
        """Set up the data import and preprocessing tab"""
        layout = QVBoxLayout(self.data_tab)
        
        # Data import section
        import_group = QGroupBox("Data Import")
        import_layout = QVBoxLayout()
        
        # File selection row
        file_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select SCADA data file (CSV)...")
        self.file_path_input.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_input, 3)
        file_layout.addWidget(browse_button, 1)
        import_layout.addLayout(file_layout)
        
        # Sample data option
        sample_layout = QHBoxLayout()
        sample_label = QLabel("Or use sample data:")
        self.use_sample_data = QPushButton("Load Sample Data")
        self.use_sample_data.clicked.connect(self.load_sample_data)
        sample_layout.addWidget(sample_label)
        sample_layout.addWidget(self.use_sample_data)
        sample_layout.addStretch()
        import_layout.addLayout(sample_layout)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        # Create table for data preview
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)
        preview_layout.addWidget(self.data_table)
        
        # Data stats
        stats_layout = QHBoxLayout()
        self.data_stats_label = QLabel("No data loaded")
        stats_layout.addWidget(self.data_stats_label)
        stats_layout.addStretch()
        preview_layout.addLayout(stats_layout)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Preprocessing options section
        preprocess_group = QGroupBox("Preprocessing Options")
        preprocess_layout = QGridLayout()
        
        # Handling missing values
        missing_values_label = QLabel("Handle Missing Values:")
        self.missing_values_combo = QComboBox()
        self.missing_values_combo.addItems(["Interpolate", "Forward Fill", "Drop Rows", "Mean"])
        preprocess_layout.addWidget(missing_values_label, 0, 0)
        preprocess_layout.addWidget(self.missing_values_combo, 0, 1)
        
        # Handling outliers
        outliers_label = QLabel("Handle Outliers:")
        self.outliers_combo = QComboBox()
        self.outliers_combo.addItems(["IQR Method", "Z-Score", "None"])
        preprocess_layout.addWidget(outliers_label, 1, 0)
        preprocess_layout.addWidget(self.outliers_combo, 1, 1)
        
        # Feature engineering options
        feature_eng_label = QLabel("Feature Engineering:")
        self.create_lag_features = QCheckBox("Create Lag Features")
        self.create_lag_features.setChecked(True)
        self.create_rolling_features = QCheckBox("Create Rolling Statistics")
        self.create_rolling_features.setChecked(True)
        self.create_time_features = QCheckBox("Create Time Features")
        self.create_time_features.setChecked(True)
        
        preprocess_layout.addWidget(feature_eng_label, 2, 0)
        feature_eng_container = QWidget()
        feature_eng_layout = QHBoxLayout(feature_eng_container)
        feature_eng_layout.addWidget(self.create_lag_features)
        feature_eng_layout.addWidget(self.create_rolling_features)
        feature_eng_layout.addWidget(self.create_time_features)
        feature_eng_layout.setContentsMargins(0, 0, 0, 0)
        preprocess_layout.addWidget(feature_eng_container, 2, 1)
        
        # Scaling method
        scaling_label = QLabel("Scaling Method:")
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["Min-Max Scaling", "Standardization (Z-score)", "None"])
        preprocess_layout.addWidget(scaling_label, 3, 0)
        preprocess_layout.addWidget(self.scaling_combo, 3, 1)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        self.process_data_button = QPushButton("Preprocess Data")
        self.process_data_button.clicked.connect(self.preprocess_data)
        self.process_data_button.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.process_data_button)
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def setup_model_tab(self):
        """Set up the model training tab"""
        layout = QVBoxLayout(self.model_tab)
        
        # Create splitter for left and right panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Model configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # Autoencoder configuration
        autoencoder_group = QGroupBox("Autoencoder Configuration")
        autoencoder_layout = QGridLayout()
        
        # Encoding dimension
        encoding_dim_label = QLabel("Encoding Dimension:")
        self.encoding_dim_spin = QSpinBox()
        self.encoding_dim_spin.setRange(2, 50)
        self.encoding_dim_spin.setValue(10)
        autoencoder_layout.addWidget(encoding_dim_label, 0, 0)
        autoencoder_layout.addWidget(self.encoding_dim_spin, 0, 1)
        
        # Epochs
        ae_epochs_label = QLabel("Epochs:")
        self.ae_epochs_spin = QSpinBox()
        self.ae_epochs_spin.setRange(10, 200)
        self.ae_epochs_spin.setValue(50)
        autoencoder_layout.addWidget(ae_epochs_label, 1, 0)
        autoencoder_layout.addWidget(self.ae_epochs_spin, 1, 1)
        
        # Batch size
        ae_batch_label = QLabel("Batch Size:")
        self.ae_batch_spin = QSpinBox()
        self.ae_batch_spin.setRange(8, 256)
        self.ae_batch_spin.setSingleStep(8)
        self.ae_batch_spin.setValue(32)
        autoencoder_layout.addWidget(ae_batch_label, 2, 0)
        autoencoder_layout.addWidget(self.ae_batch_spin, 2, 1)
        
        # Validation split
        ae_val_split_label = QLabel("Validation Split:")
        self.ae_val_split_spin = QSpinBox()
        self.ae_val_split_spin.setRange(10, 40)
        self.ae_val_split_spin.setValue(20)
        self.ae_val_split_spin.setSuffix("%")
        autoencoder_layout.addWidget(ae_val_split_label, 3, 0)
        autoencoder_layout.addWidget(self.ae_val_split_spin, 3, 1)
        
        # Progress bar and train button
        ae_progress_label = QLabel("Training Progress:")
        self.ae_progress_bar = QProgressBar()
        self.ae_progress_bar.setValue(0)
        self.train_ae_button = QPushButton("Train Autoencoder")
        self.train_ae_button.clicked.connect(self.train_autoencoder)
        
        autoencoder_layout.addWidget(ae_progress_label, 4, 0)
        autoencoder_layout.addWidget(self.ae_progress_bar, 4, 1)
        autoencoder_layout.addWidget(self.train_ae_button, 5, 0, 1, 2)
        
        autoencoder_group.setLayout(autoencoder_layout)
        left_layout.addWidget(autoencoder_group)
        
        # RUL model configuration
        rul_group = QGroupBox("RUL Model Configuration")
        rul_layout = QGridLayout()
        
        # Sequence length
        seq_length_label = QLabel("Sequence Length:")
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(6, 48)
        self.seq_length_spin.setValue(24)
        rul_layout.addWidget(seq_length_label, 0, 0)
        rul_layout.addWidget(self.seq_length_spin, 0, 1)
        
        # Epochs
        rul_epochs_label = QLabel("Epochs:")
        self.rul_epochs_spin = QSpinBox()
        self.rul_epochs_spin.setRange(10, 200)
        self.rul_epochs_spin.setValue(50)
        rul_layout.addWidget(rul_epochs_label, 1, 0)
        rul_layout.addWidget(self.rul_epochs_spin, 1, 1)
        
        # Batch size
        rul_batch_label = QLabel("Batch Size:")
        self.rul_batch_spin = QSpinBox()
        self.rul_batch_spin.setRange(8, 256)
        self.rul_batch_spin.setSingleStep(8)
        self.rul_batch_spin.setValue(32)
        rul_layout.addWidget(rul_batch_label, 2, 0)
        rul_layout.addWidget(self.rul_batch_spin, 2, 1)
        
        # Validation split
        rul_val_split_label = QLabel("Validation Split:")
        self.rul_val_split_spin = QSpinBox()
        self.rul_val_split_spin.setRange(10, 40)
        self.rul_val_split_spin.setValue(20)
        self.rul_val_split_spin.setSuffix("%")
        rul_layout.addWidget(rul_val_split_label, 3, 0)
        rul_layout.addWidget(self.rul_val_split_spin, 3, 1)
        
        # Progress bar and train button
        rul_progress_label = QLabel("Training Progress:")
        self.rul_progress_bar = QProgressBar()
        self.rul_progress_bar.setValue(0)
        self.train_rul_button = QPushButton("Train RUL Model")
        self.train_rul_button.clicked.connect(self.train_rul_model)
        self.train_rul_button.setEnabled(False)  # Disabled until autoencoder is trained
        
        rul_layout.addWidget(rul_progress_label, 4, 0)
        rul_layout.addWidget(self.rul_progress_bar, 4, 1)
        rul_layout.addWidget(self.train_rul_button, 5, 0, 1, 2)
        
        rul_group.setLayout(rul_layout)
        left_layout.addWidget(rul_group)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Right panel - Model performance visualizations
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
      # Tabs for different plots
        plot_tabs = QTabWidget()
        
        # Autoencoder performance tab
        self.ae_plot_tab = QWidget()
        ae_plot_layout = QVBoxLayout(self.ae_plot_tab)
        
        # Create figure for autoencoder loss plot
        self.ae_figure = Figure(figsize=(5, 4), dpi=100)
        self.ae_canvas = FigureCanvas(self.ae_figure)
        ae_plot_layout.addWidget(self.ae_canvas)
        
        plot_tabs.addTab(self.ae_plot_tab, "Autoencoder Training")
        
        # RUL model performance tab
        self.rul_plot_tab = QWidget()
        rul_plot_layout = QVBoxLayout(self.rul_plot_tab)
        
        # Create figure for RUL model loss plot
        self.rul_figure = Figure(figsize=(5, 4), dpi=100)
        self.rul_canvas = FigureCanvas(self.rul_figure)
        rul_plot_layout.addWidget(self.rul_canvas)
        
        plot_tabs.addTab(self.rul_plot_tab, "RUL Model Training")
        
        # Add the plot tabs to the right panel
        right_layout.addWidget(plot_tabs)
        
        # Add a model summary text area
        model_summary_group = QGroupBox("Model Summary")
        model_summary_layout = QVBoxLayout()
        self.model_summary_text = QTextEdit()
        self.model_summary_text.setReadOnly(True)
        model_summary_layout.addWidget(self.model_summary_text)
        model_summary_group.setLayout(model_summary_layout)
        right_layout.addWidget(model_summary_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
        # Proceed to analysis button (bottom of page)
        proceed_layout = QHBoxLayout()
        self.proceed_to_analysis_button = QPushButton("Proceed to Analysis")
        self.proceed_to_analysis_button.clicked.connect(self.go_to_analysis_tab)
        self.proceed_to_analysis_button.setEnabled(False)
        proceed_layout.addStretch()
        proceed_layout.addWidget(self.proceed_to_analysis_button)
        layout.addLayout(proceed_layout)
    
    def setup_analysis_tab(self):
        """Set up the analysis and results tab"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # Create tabs for different analysis views
        analysis_tabs = QTabWidget()
        
        # Anomaly Detection tab
        self.anomaly_tab = QWidget()
        anomaly_layout = QVBoxLayout(self.anomaly_tab)
        
        # Anomaly detection controls
        anomaly_controls = QHBoxLayout()
        
        threshold_label = QLabel("Anomaly Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(95)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_value_label = QLabel("95%")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v}%"))
        self.threshold_slider.valueChanged.connect(self.update_anomaly_plot)
        
        anomaly_controls.addWidget(threshold_label)
        anomaly_controls.addWidget(self.threshold_slider)
        anomaly_controls.addWidget(self.threshold_value_label)
        
        detect_button = QPushButton("Detect Anomalies")
        detect_button.clicked.connect(self.detect_anomalies)
        anomaly_controls.addWidget(detect_button)
        
        anomaly_layout.addLayout(anomaly_controls)
        
        # Anomaly time series plot
        self.anomaly_figure = Figure(figsize=(5, 4), dpi=100)
        self.anomaly_canvas = FigureCanvas(self.anomaly_figure)
        anomaly_layout.addWidget(self.anomaly_canvas)
        
        # Anomaly results table
        self.anomaly_table = QTableWidget()
        self.anomaly_table.setColumnCount(4)
        self.anomaly_table.setHorizontalHeaderLabels(
            ["Timestamp", "Error Score", "Is Anomaly", "Affected Components"])
        header = self.anomaly_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        anomaly_layout.addWidget(self.anomaly_table)
        
        analysis_tabs.addTab(self.anomaly_tab, "Anomaly Detection")
        
        # RUL Prediction tab
        self.rul_tab = QWidget()
        rul_layout = QVBoxLayout(self.rul_tab)
        
        # RUL prediction controls
        rul_controls = QHBoxLayout()
        
        asset_label = QLabel("Select Asset:")
        self.asset_combo = QComboBox()
        self.asset_combo.currentIndexChanged.connect(self.update_rul_plot)
        
        predict_horizon_label = QLabel("Prediction Horizon (days):")
        self.predict_horizon_spin = QSpinBox()
        self.predict_horizon_spin.setRange(1, 90)
        self.predict_horizon_spin.setValue(30)
        
        predict_button = QPushButton("Predict RUL")
        predict_button.clicked.connect(self.predict_rul)
        
        rul_controls.addWidget(asset_label)
        rul_controls.addWidget(self.asset_combo)
        rul_controls.addWidget(predict_horizon_label)
        rul_controls.addWidget(self.predict_horizon_spin)
        rul_controls.addWidget(predict_button)
        
        rul_layout.addLayout(rul_controls)
        
        # RUL time series plot
        self.rul_pred_figure = Figure(figsize=(5, 4), dpi=100)
        self.rul_pred_canvas = FigureCanvas(self.rul_pred_figure)
        rul_layout.addWidget(self.rul_pred_canvas)
        
        # RUL results table
        self.rul_table = QTableWidget()
        self.rul_table.setColumnCount(5)
        self.rul_table.setHorizontalHeaderLabels(
            ["Asset ID", "Current Health", "Estimated RUL (days)", 
             "Confidence", "Maintenance Priority"])
        header = self.rul_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        rul_layout.addWidget(self.rul_table)
        
        analysis_tabs.addTab(self.rul_tab, "RUL Prediction")
        
        # Maintenance Planning tab
        self.maintenance_tab = QWidget()
        maintenance_layout = QVBoxLayout(self.maintenance_tab)
        
        # Maintenance settings
        maintenance_settings = QHBoxLayout()
        
        maintenance_horizon_label = QLabel("Planning Horizon (days):")
        self.maintenance_horizon_spin = QSpinBox()
        self.maintenance_horizon_spin.setRange(7, 180)
        self.maintenance_horizon_spin.setValue(60)
        
        maintenance_threshold_label = QLabel("Maintenance Threshold (RUL days):")
        self.maintenance_threshold_spin = QSpinBox()
        self.maintenance_threshold_spin.setRange(1, 30)
        self.maintenance_threshold_spin.setValue(14)
        
        generate_plan_button = QPushButton("Generate Maintenance Plan")
        generate_plan_button.clicked.connect(self.generate_maintenance_plan)
        
        maintenance_settings.addWidget(maintenance_horizon_label)
        maintenance_settings.addWidget(self.maintenance_horizon_spin)
        maintenance_settings.addWidget(maintenance_threshold_label)
        maintenance_settings.addWidget(self.maintenance_threshold_spin)
        maintenance_settings.addWidget(generate_plan_button)
        
        maintenance_layout.addLayout(maintenance_settings)
        
        # Maintenance Gantt chart
        self.maintenance_figure = Figure(figsize=(5, 4), dpi=100)
        self.maintenance_canvas = FigureCanvas(self.maintenance_figure)
        maintenance_layout.addWidget(self.maintenance_canvas)
        
        # Maintenance plan table
        self.maintenance_table = QTableWidget()
        self.maintenance_table.setColumnCount(6)
        self.maintenance_table.setHorizontalHeaderLabels(
            ["Asset ID", "Scheduled Date", "Estimated Duration (hours)", 
             "Required Resources", "Priority", "Expected Impact"])
        header = self.maintenance_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        maintenance_layout.addWidget(self.maintenance_table)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        export_results_button = QPushButton("Export Results")
        export_results_button.clicked.connect(self.export_results)
        export_plan_button = QPushButton("Export Maintenance Plan")
        export_plan_button.clicked.connect(self.export_maintenance_plan)
        
        export_layout.addWidget(export_results_button)
        export_layout.addWidget(export_plan_button)
        
        maintenance_layout.addLayout(export_layout)
        
        analysis_tabs.addTab(self.maintenance_tab, "Maintenance Planning")
        
        # Add the tabs to the main layout
        layout.addWidget(analysis_tabs)
    
    def browse_file(self):
        """Open file dialog to select data file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SCADA Data File", "", 
            "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            self.file_path_input.setText(file_path)
            self.load_data(file_path)
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            # Try to locate sample data in resources
            sample_data_path = os.path.join(
                os.path.dirname(__file__), "sample_data", "wind_turbine_data.csv")
            
            if not os.path.exists(sample_data_path):
                # Create synthetic data if sample file doesn't exist
                self.statusBar.showMessage("Generating synthetic data...", 5000)
                data = self.generate_synthetic_data()
                self.file_path_input.setText("Sample Data (Synthetic)")
                self.current_data = data
                self.update_data_preview()
                self.process_data_button.setEnabled(True)
            else:
                self.file_path_input.setText(sample_data_path)
                self.load_data(sample_data_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load sample data: {str(e)}")
    
    def generate_synthetic_data(self):
        """Generate synthetic data for demonstration"""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        
        # Number of assets
        n_assets = 5
        
        # Create a dataframe
        data = pd.DataFrame()
        
        for asset_id in range(1, n_assets + 1):
            # Base values with some randomness
            wind_speed = np.random.normal(8, 2, len(dates)) + np.sin(np.linspace(0, 10*np.pi, len(dates)))
            wind_speed = np.clip(wind_speed, 0, 20)
            
            # Power output dependent on wind speed
            power_output = 0.5 * wind_speed ** 3 * (wind_speed > 3) * (wind_speed < 15)
            
            # Add degradation trend
            degradation = np.linspace(0, 1, len(dates)) ** 2
            
            # Vibration increases with degradation
            vibration = 0.1 + degradation * 0.9 + np.random.normal(0, 0.1, len(dates))
            
            # Gearbox temperature increases with power and degradation
            gearbox_temp = 50 + 0.1 * power_output + 20 * degradation + np.random.normal(0, 5, len(dates))
            
            # Generator temperature
            generator_temp = 45 + 0.15 * power_output + 15 * degradation + np.random.normal(0, 3, len(dates))
            
            # Randomly inject a few anomalies
            anomaly_points = np.random.choice(len(dates), size=5, replace=False)
            for point in anomaly_points:
                # Anomalous high vibration
                vibration[point:point+24] += np.random.uniform(0.5, 1.5)
                
                # Anomalous temperature
                gearbox_temp[point:point+24] += np.random.uniform(10, 30)
                
            # Create asset dataframe
            asset_data = pd.DataFrame({
                'timestamp': dates,
                'asset_id': asset_id,
                'wind_speed': wind_speed,
                'power_output': power_output,
                'vibration': vibration,
                'gearbox_temperature': gearbox_temp,
                'generator_temperature': generator_temp,
                'ambient_temperature': 15 + 10 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + np.random.normal(0, 3, len(dates)),
                'humidity': 60 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 5, len(dates)),
                'nacelle_orientation': np.random.uniform(0, 360, len(dates))
            })
            
            # Append to main dataframe
            data = pd.concat([data, asset_data], ignore_index=True)
        
        # Sort by timestamp and asset_id
        data = data.sort_values(['timestamp', 'asset_id']).reset_index(drop=True)
        
        # Introduce some missing values (1%)
        for col in data.columns:
            if col not in ['timestamp', 'asset_id']:
                mask = np.random.random(len(data)) < 0.01
                data.loc[mask, col] = np.nan
        
        return data
    
    def load_data(self, file_path):
        """Load data from the selected file"""
        try:
            # Load the data
            data = pd.read_csv(file_path)
            
            # Check if the data contains the required columns
            required_columns = ['timestamp']
            
            # Check if timestamp column exists
            if not all(col in data.columns for col in required_columns):
                QMessageBox.warning(self, "Invalid Data Format", 
                                    "The data must contain a 'timestamp' column")
                return
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Store the data
            self.current_data = data
            
            # Update the data preview
            self.update_data_preview()
            
            # Enable the preprocess button
            self.process_data_button.setEnabled(True)
            
            self.statusBar.showMessage(f"Data loaded successfully: {len(data)} rows", 5000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load data: {str(e)}")
    
    def update_data_preview(self):
        """Update the data preview table with loaded data"""
        if self.current_data is None:
            return
        
        # Get a sample of the data
        sample_data = self.current_data.head(10)
        
        # Update the table
        self.data_table.setColumnCount(len(sample_data.columns))
        self.data_table.setRowCount(len(sample_data))
        
        # Set headers
        self.data_table.setHorizontalHeaderLabels(sample_data.columns)
        
        # Fill table with data
        for i, row in enumerate(sample_data.itertuples()):
            for j, value in enumerate(row[1:]):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()
        
        # Update stats label
        self.update_data_stats()
    
    def update_data_stats(self):
        """Update data statistics label"""
        if self.current_data is None:
            self.data_stats_label.setText("No data loaded")
            return
        
        # Calculate stats
        num_rows = len(self.current_data)
        num_cols = len(self.current_data.columns)
        missing_cells = self.current_data.isna().sum().sum()
        missing_pct = missing_cells / (num_rows * num_cols) * 100
        
        # Get time range
        time_range = f"{self.current_data['timestamp'].min()} to {self.current_data['timestamp'].max()}"
        
        # Format stats text
        stats_text = (f"Rows: {num_rows} | Columns: {num_cols} | "
                      f"Missing Values: {missing_cells} ({missing_pct:.2f}%) | "
                      f"Time Range: {time_range}")
        
        self.data_stats_label.setText(stats_text)
