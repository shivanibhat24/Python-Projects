import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QLabel, QSlider, QPushButton, 
                            QCheckBox, QGroupBox, QGridLayout, QSpinBox, QSplitter,
                            QTabWidget, QLineEdit, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib
matplotlib.use('Qt5Agg')
from scipy import signal

class ModulationVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set dark theme for the application
        self.setDarkTheme()
        
        # Main window settings
        self.setWindowTitle("Digital Modulation Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize modulation parameters
        self.num_bits = 5000
        self.snr_values = [10, 15, 20]
        self.selected_modulations = ["BPSK", "QPSK", "16-PSK", "16-QAM"]
        
        # Modulation colors
        self.mod_colors = {
            'BPSK': '#39FF14',    # Neon Green
            'QPSK': '#7DF9FF',    # Electric Blue
            '16-PSK': '#FFFF00',  # Yellow
            '16-QAM': '#BF00FF'   # Neon Purple
        }
        
        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create UI components
        self.createTopBar()
        self.createMainContent()
        self.createStatusBar()
        
        # Initialize timer for animation effects
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateAnimation)
        
        # Generate initial data and plot
        self.generateData()
        self.updatePlots()

    def setDarkTheme(self):
        # Set dark theme for the application
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(dark_palette)
        
        # Set stylesheet for additional components
        style_sheet = """
        QGroupBox {
            border: 1px solid gray;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #2a82da;
            color: white;
            border-radius: 4px;
            padding: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3a92ea;
        }
        QComboBox {
            border: 1px solid gray;
            border-radius: 3px;
            padding: 1px 18px 1px 3px;
            min-width: 6em;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 4px;
            top: -1px;
        }
        QTabBar::tab {
            background: #333;
            border: 1px solid #444;
            padding: 5px;
            min-width: 80px;
        }
        QTabBar::tab:selected {
            background: #444;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999;
            height: 8px;
            background: #333;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #2a82da;
            border: 1px solid #2a82da;
            width: 16px;
            margin: -2px 0;
            border-radius: 8px;
        }
        QFrame#separator {
            background-color: #444;
        }
        """
        self.setStyleSheet(style_sheet)

    def createTopBar(self):
        # Create top bar for controls
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        # Bits control
        bits_group = QGroupBox("Data Settings")
        bits_layout = QVBoxLayout(bits_group)
        
        bits_input_layout = QHBoxLayout()
        bits_input_layout.addWidget(QLabel("Number of Bits:"))
        self.bits_spinbox = QSpinBox()
        self.bits_spinbox.setRange(1000, 100000)
        self.bits_spinbox.setValue(self.num_bits)
        self.bits_spinbox.setSingleStep(1000)
        bits_input_layout.addWidget(self.bits_spinbox)
        
        self.random_data_checkbox = QCheckBox("Random Data")
        self.random_data_checkbox.setChecked(True)
        
        generate_button = QPushButton("Generate Data")
        generate_button.clicked.connect(self.generateData)
        
        bits_layout.addLayout(bits_input_layout)
        bits_layout.addWidget(self.random_data_checkbox)
        bits_layout.addWidget(generate_button)
        
        # SNR control
        snr_group = QGroupBox("SNR Settings")
        snr_layout = QVBoxLayout(snr_group)
        
        snr_input_layout = QHBoxLayout()
        snr_input_layout.addWidget(QLabel("SNR Values (comma separated):"))
        self.snr_input = QLineEdit(",".join(map(str, self.snr_values)))
        snr_input_layout.addWidget(self.snr_input)
        
        snr_slider_layout = QHBoxLayout()
        snr_slider_layout.addWidget(QLabel("Add SNR:"))
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setRange(0, 30)
        self.snr_slider.setValue(15)
        self.snr_slider_value = QLabel("15 dB")
        self.snr_slider.valueChanged.connect(self.updateSnrSliderValue)
        snr_slider_layout.addWidget(self.snr_slider)
        snr_slider_layout.addWidget(self.snr_slider_value)
        
        add_snr_button = QPushButton("Add SNR")
        add_snr_button.clicked.connect(self.addSnrValue)
        
        snr_layout.addLayout(snr_input_layout)
        snr_layout.addLayout(snr_slider_layout)
        snr_layout.addWidget(add_snr_button)
        
        # Modulation control
        mod_group = QGroupBox("Modulation Schemes")
        mod_layout = QGridLayout(mod_group)
        
        self.mod_checkboxes = {}
        mod_types = ["BPSK", "QPSK", "16-PSK", "16-QAM"]
        
        for i, mod_type in enumerate(mod_types):
            self.mod_checkboxes[mod_type] = QCheckBox(mod_type)
            self.mod_checkboxes[mod_type].setChecked(mod_type in self.selected_modulations)
            row, col = divmod(i, 2)
            mod_layout.addWidget(self.mod_checkboxes[mod_type], row, col)
        
        update_button = QPushButton("Update Plots")
        update_button.clicked.connect(self.updatePlotsFromUi)
        mod_layout.addWidget(update_button, 2, 0, 1, 2)
        
        # Add all groups to top bar
        top_layout.addWidget(bits_group)
        top_layout.addWidget(snr_group)
        top_layout.addWidget(mod_group)
        
        # Animation control
        animation_group = QGroupBox("Visualization")
        animation_layout = QVBoxLayout(animation_group)
        
        self.animation_checkbox = QCheckBox("Enable Animation")
        self.animation_checkbox.setChecked(False)
        self.animation_checkbox.stateChanged.connect(self.toggleAnimation)
        
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Cyberpunk"])
        self.theme_combo.setCurrentIndex(0)
        self.theme_combo.currentIndexChanged.connect(self.changeTheme)
        theme_layout.addWidget(self.theme_combo)
        
        animation_layout.addWidget(self.animation_checkbox)
        animation_layout.addLayout(theme_layout)
        
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.exportResults)
        animation_layout.addWidget(export_button)
        
        top_layout.addWidget(animation_group)
        
        self.main_layout.addWidget(top_bar)
        
        # Add separator line
        separator = QFrame()
        separator.setObjectName("separator")
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(separator)

    def createMainContent(self):
        # Create tabs for different visualizations
        tab_widget = QTabWidget()
        
        # Constellation tab
        constellation_tab = QWidget()
        constellation_layout = QVBoxLayout(constellation_tab)
        
        # Create matplotlib canvas for constellation diagrams
        self.constellation_figure = Figure(figsize=(5, 5), dpi=100)
        self.constellation_figure.patch.set_facecolor('#333333')
        self.constellation_canvas = FigureCanvas(self.constellation_figure)
        constellation_layout.addWidget(self.constellation_canvas)
        
        # Eye diagram tab
        eye_diagram_tab = QWidget()
        eye_layout = QVBoxLayout(eye_diagram_tab)
        
        # Create matplotlib canvas for eye diagrams
        self.eye_figure = Figure(figsize=(5, 5), dpi=100)
        self.eye_figure.patch.set_facecolor('#333333')
        self.eye_canvas = FigureCanvas(self.eye_figure)
        eye_layout.addWidget(self.eye_canvas)
        
        # BER performance tab
        ber_tab = QWidget()
        ber_layout = QVBoxLayout(ber_tab)
        
        # Create matplotlib canvas for BER curves
        self.ber_figure = Figure(figsize=(5, 5), dpi=100)
        self.ber_figure.patch.set_facecolor('#333333')
        self.ber_canvas = FigureCanvas(self.ber_figure)
        ber_layout.addWidget(self.ber_canvas)
        
        # Add tabs to tab widget
        tab_widget.addTab(constellation_tab, "Constellation Diagrams")
        tab_widget.addTab(eye_diagram_tab, "Eye Diagrams")
        tab_widget.addTab(ber_tab, "BER Performance")
        
        self.main_layout.addWidget(tab_widget)

    def createStatusBar(self):
        # Create a status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(5, 0, 5, 5)
        
        status_label = QLabel("Ready")
        self.status_value = QLabel("")
        
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_value)
        status_layout.addStretch()
        
        # Version info
        version_label = QLabel("v1.0.0")
        status_layout.addWidget(version_label)
        
        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        
        self.main_layout.addWidget(status_widget)

    def generateData(self):
        # Get number of bits from UI
        self.num_bits = self.bits_spinbox.value()
        
        # Make sure the number of bits is a multiple of 4 for 16-QAM and 16-PSK
        if self.num_bits % 4 != 0:
            self.num_bits = self.num_bits + (4 - (self.num_bits % 4))
            self.bits_spinbox.setValue(self.num_bits)
        
        # Generate binary data
        if self.random_data_checkbox.isChecked():
            self.data = np.random.randint(0, 2, self.num_bits)
        else:
            # Generate deterministic sequence for testing
            self.data = np.zeros(self.num_bits, dtype=int)
            for i in range(0, self.num_bits, 8):
                self.data[i:i+4] = 1
        
        # Update status
        self.status_value.setText(f"Generated {self.num_bits} bits of data")
        
        # Update plots
        self.updatePlots()

    def updatePlotsFromUi(self):
        # Get SNR values from UI
        snr_text = self.snr_input.text()
        try:
            self.snr_values = [int(snr.strip()) for snr in snr_text.split(',')]
        except ValueError:
            # Reset to default if invalid input
            self.snr_values = [10, 15, 20]
            self.snr_input.setText(",".join(map(str, self.snr_values)))
        
        # Get selected modulation schemes
        self.selected_modulations = [mod for mod, checkbox in self.mod_checkboxes.items() if checkbox.isChecked()]
        
        # Update plots
        self.updatePlots()

    def updatePlots(self):
        # Clear existing plots
        self.constellation_figure.clear()
        self.eye_figure.clear()
        self.ber_figure.clear()
        
        # Skip if no modulations are selected
        if not self.selected_modulations:
            self.constellation_canvas.draw()
            self.eye_canvas.draw()
            self.ber_canvas.draw()
            return
        
        # Define modulation functions
        mod_schemes = {
            'BPSK': self.bpsk_modulation,
            'QPSK': self.qpsk_modulation,
            '16-PSK': self.psk16_modulation,
            '16-QAM': self.qam16_modulation
        }
        
        # Set up constellation subplot grid
        rows = len(self.selected_modulations)
        cols = len(self.snr_values) + 1
        self.constellation_figure.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Create plots for each modulation scheme
        mod_data = {}
        for mod_idx, mod_name in enumerate(self.selected_modulations):
            color = self.mod_colors[mod_name]
            
            # Perform modulation
            if mod_name in ['BPSK', 'QPSK']:
                mod_data[mod_name] = mod_schemes[mod_name](self.data[:self.num_bits if mod_name == 'BPSK' else self.num_bits//2])
            elif mod_name in ['16-PSK', '16-QAM']:
                mod_data[mod_name] = mod_schemes[mod_name](self.data[:self.num_bits])
            
            # Plot constellation before noise
            ax = self.constellation_figure.add_subplot(rows, cols, mod_idx * cols + 1)
            self.plot_constellation(mod_data[mod_name], f'{mod_name} (No Noise)', ax, color)
            
            # Add AWGN noise for different SNRs and plot
            for snr_idx, snr in enumerate(self.snr_values):
                noisy_data = self.add_awgn_noise(mod_data[mod_name], snr)
                ax = self.constellation_figure.add_subplot(rows, cols, mod_idx * cols + snr_idx + 2)
                self.plot_constellation(noisy_data, f'{mod_name} @ {snr} dB', ax, color)
        
        # Plot eye diagrams
        self.plot_eye_diagrams(mod_data)
        
        # Plot BER curves
        self.plot_ber_curves(mod_data)
        
        # Draw all canvases
        self.constellation_canvas.draw()
        self.eye_canvas.draw()
        self.ber_canvas.draw()

    def plot_constellation(self, mod_data, title, ax, color):
        ax.scatter(mod_data.real, mod_data.imag, color=color, s=10, alpha=0.7)
        ax.set_title(title, color='white', fontsize=9)
        ax.grid(True, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.5)
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.tick_params(axis='both', colors='white', labelsize=8)
        ax.set_facecolor('#333333')

    def plot_eye_diagrams(self, mod_data):
        rows = len(self.selected_modulations)
        self.eye_figure.subplots_adjust(hspace=0.5)
        
        for idx, mod_name in enumerate(self.selected_modulations):
            color = self.mod_colors[mod_name]
            
            # Get I and Q components
            I_data = mod_data[mod_name].real
            Q_data = mod_data[mod_name].imag
            
            # Simple eye diagram (just for demonstration)
            ax1 = self.eye_figure.add_subplot(rows, 2, idx*2 + 1)
            self.plot_eye(I_data, f'{mod_name} I-Channel Eye', ax1, color)
            
            ax2 = self.eye_figure.add_subplot(rows, 2, idx*2 + 2)
            self.plot_eye(Q_data, f'{mod_name} Q-Channel Eye', ax2, color)

    def plot_eye(self, data, title, ax, color):
        # Simple eye diagram simulation (this is simplified for demonstration)
        samples_per_symbol = 8
        span = 2  # Number of symbols to display
        
        # Upsample and filter (simplified)
        upsampled = np.zeros(len(data) * samples_per_symbol)
        upsampled[::samples_per_symbol] = data
        
        # Apply pulse shaping filter (simplified)
        h = np.sinc(np.linspace(-3, 3, 21))  # Simple sinc filter
        filtered = np.convolve(upsampled, h, 'same')
        
        # Create eye diagram
        symbols_to_plot = min(100, len(filtered) // samples_per_symbol - span)
        
        for i in range(symbols_to_plot):
            start = i * samples_per_symbol
            ax.plot(filtered[start:start + samples_per_symbol * span], 
                   color=color, alpha=0.3)
        
        ax.set_title(title, color='white', fontsize=9)
        ax.grid(True, color='gray', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', colors='white', labelsize=8)
        ax.set_facecolor('#333333')

    def plot_ber_curves(self, mod_data):
        ax = self.ber_figure.add_subplot(111)
        
        # Theoretical BER curves (simplified)
        snr_db_range = np.linspace(0, 20, 100)
        snr_linear = 10**(snr_db_range / 10)
        
        for mod_name in self.selected_modulations:
            color = self.mod_colors[mod_name]
            
            # Simplified theoretical BER curves (not exact)
            if mod_name == 'BPSK':
                ber = 0.5 * np.exp(-snr_linear)
            elif mod_name == 'QPSK':
                ber = 0.5 * np.exp(-0.5 * snr_linear)
            elif mod_name == '16-PSK':
                ber = 0.5 * np.exp(-0.2 * snr_linear)
            elif mod_name == '16-QAM':
                ber = 0.5 * np.exp(-0.1 * snr_linear)
            
            ax.semilogy(snr_db_range, ber, color=color, label=mod_name)
            
            # Add markers for the specific SNR values we're using
            measured_ber = []
            for snr in self.snr_values:
                # Simplified BER calculation (not accurate)
                if mod_name == 'BPSK':
                    ber_point = 0.5 * np.exp(-10**(snr / 10))
                elif mod_name == 'QPSK':
                    ber_point = 0.5 * np.exp(-0.5 * 10**(snr / 10))
                elif mod_name == '16-PSK':
                    ber_point = 0.5 * np.exp(-0.2 * 10**(snr / 10))
                elif mod_name == '16-QAM':
                    ber_point = 0.5 * np.exp(-0.1 * 10**(snr / 10))
                
                measured_ber.append(ber_point)
            
            ax.scatter(self.snr_values, measured_ber, color=color, s=50, edgecolors='white')
        
        ax.set_title('Bit Error Rate Performance', color='white')
        ax.set_xlabel('SNR (dB)', color='white')
        ax.set_ylabel('Bit Error Rate (BER)', color='white')
        ax.grid(True, color='gray', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('#333333')
        ax.legend(loc='lower left')

    def updateSnrSliderValue(self):
        value = self.snr_slider.value()
        self.snr_slider_value.setText(f"{value} dB")

    def addSnrValue(self):
        value = self.snr_slider.value()
        if value not in self.snr_values:
            self.snr_values.append(value)
            self.snr_values.sort()
            self.snr_input.setText(",".join(map(str, self.snr_values)))
            self.updatePlots()

    def toggleAnimation(self, state):
        if state == Qt.Checked:
            self.timer.start(50)  # Update every 50ms
        else:
            self.timer.stop()

    def updateAnimation(self):
        # Simple animation effect for constellation points
        self.updatePlots()  # For a real implementation, you'd want to modify existing plot data instead

    def changeTheme(self, index):
        theme = self.theme_combo.currentText()
        if theme == "Light":
            # Update plot colors for light theme
            for fig in [self.constellation_figure, self.eye_figure, self.ber_figure]:
                fig.patch.set_facecolor('#f0f0f0')
            self.updatePlots()
        elif theme == "Dark":
            # Update plot colors for dark theme
            for fig in [self.constellation_figure, self.eye_figure, self.ber_figure]:
                fig.patch.set_facecolor('#333333')
            self.updatePlots()
        elif theme == "Cyberpunk":
            # Update plot colors for cyberpunk theme
            for fig in [self.constellation_figure, self.eye_figure, self.ber_figure]:
                fig.patch.set_facecolor('#0a0a2a')
            self.updatePlots()

    def exportResults(self):
        # Save plot figures
        self.constellation_figure.savefig('constellation_diagrams.png')
        self.eye_figure.savefig('eye_diagrams.png')
        self.ber_figure.savefig('ber_curves.png')
        self.status_value.setText("Exported plots to PNG files")

    # Modulation functions
    def bpsk_modulation(self, data):
        return 2*data - 1

    def qpsk_modulation(self, data):
        data_reshaped = data.reshape((-1, 2))
        symbols = (2*data_reshaped[:, 0] - 1) + 1j * (2*data_reshaped[:, 1] - 1)
        return symbols / np.sqrt(2)

    def psk16_modulation(self, data):
        M = 16
        data_reshaped = data.reshape((-1, 4))
        data_symbols = np.zeros(data_reshaped.shape[0], dtype=int)
        
        for i in range(4):
            data_symbols += data_reshaped[:, i] * (2 ** (3 - i))
            
        phase = (2 * np.pi * data_symbols) / M
        return np.exp(1j * phase)

    def qam16_modulation(self, data):
        data_reshaped = data.reshape((-1, 4))
        
        # Map 4 bits to 16-QAM constellation
        I_bits = (2 * data_reshaped[:, 0] - 1) * 2 + (2 * data_reshaped[:, 1] - 1)
        Q_bits = (2 * data_reshaped[:, 2] - 1) * 2 + (2 * data_reshaped[:, 3] - 1)
        
        # Normalize average power to 1
        symbols = (I_bits + 1j * Q_bits) / np.sqrt(10)
        return symbols

    def add_awgn_noise(self, signal, snr_db):
        avg_signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = avg_signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise

# Create the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModulationVisualizer()
    window.show()
    sys.exit(app.exec_())
