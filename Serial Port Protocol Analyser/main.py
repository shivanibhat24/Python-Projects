import sys
import time
import threading
import datetime
import json
import binascii
import re
from queue import Queue

import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

class SerialSniffer:
    def __init__(self, root):
        self.root = root
        self.root.title("Serial Port Protocol Analyzer")
        self.root.geometry("900x700")
        
        # Serial connection variables
        self.serial_port = None
        self.is_connected = False
        self.is_monitoring = False
        self.capture_buffer = []
        self.command_history = []
        self.history_position = 0
        self.packet_queue = Queue()
        
        # Display settings
        self.display_mode = tk.StringVar(value="ASCII")
        self.log_enabled = tk.BooleanVar(value=False)
        self.log_file = None
        
        self._init_ui()
        self._update_port_list()
        
        # Set up periodic UI updates for received data
        self.root.after(100, self._process_packet_queue)
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Create main frame with tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Tab frames
        self.monitor_frame = ttk.Frame(self.notebook)
        self.send_frame = ttk.Frame(self.notebook)
        self.fuzzer_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.monitor_frame, text="Monitor")
        self.notebook.add(self.send_frame, text="Send Commands")
        self.notebook.add(self.fuzzer_frame, text="Fuzzer")
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Set up the connection panel (shared at top of all tabs)
        self._setup_connection_panel()
        
        # Set up the individual tab contents
        self._setup_monitor_tab()
        self._setup_send_tab()
        self._setup_fuzzer_tab()
    
    def _setup_connection_panel(self):
        """Create the serial port connection panel"""
        # Connection frame at the top of the window
        connection_frame = ttk.LabelFrame(self.root, text="Serial Connection")
        connection_frame.pack(fill="x", padx=5, pady=5)
        
        # Port selection
        port_frame = ttk.Frame(connection_frame)
        port_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(port_frame, text="Port:").pack(side="left", padx=5)
        self.port_combobox = ttk.Combobox(port_frame, width=20)
        self.port_combobox.pack(side="left", padx=5)
        
        refresh_button = ttk.Button(port_frame, text="Refresh", command=self._update_port_list)
        refresh_button.pack(side="left", padx=5)
        
        # Connection settings
        settings_frame = ttk.Frame(connection_frame)
        settings_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Baud Rate:").pack(side="left", padx=5)
        self.baud_combobox = ttk.Combobox(settings_frame, width=10)
        self.baud_combobox["values"] = ("1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200")
        self.baud_combobox.current(3)  # Default to 9600 baud
        self.baud_combobox.pack(side="left", padx=5)
        
        ttk.Label(settings_frame, text="Data Bits:").pack(side="left", padx=5)
        self.data_bits_combobox = ttk.Combobox(settings_frame, width=5)
        self.data_bits_combobox["values"] = ("5", "6", "7", "8")
        self.data_bits_combobox.current(3)  # Default to 8 data bits
        self.data_bits_combobox.pack(side="left", padx=5)
        
        ttk.Label(settings_frame, text="Parity:").pack(side="left", padx=5)
        self.parity_combobox = ttk.Combobox(settings_frame, width=5)
        self.parity_combobox["values"] = ("None", "Even", "Odd", "Mark", "Space")
        self.parity_combobox.current(0)  # Default to no parity
        self.parity_combobox.pack(side="left", padx=5)
        
        ttk.Label(settings_frame, text="Stop Bits:").pack(side="left", padx=5)
        self.stop_bits_combobox = ttk.Combobox(settings_frame, width=5)
        self.stop_bits_combobox["values"] = ("1", "1.5", "2")
        self.stop_bits_combobox.current(0)  # Default to 1 stop bit
        self.stop_bits_combobox.pack(side="left", padx=5)
        
        # Connect button
        self.connect_button = ttk.Button(connection_frame, text="Connect", command=self._toggle_connection)
        self.connect_button.pack(side="left", padx=5, pady=5)
        
        # Status indicator
        self.status_label = ttk.Label(connection_frame, text="Disconnected")
        self.status_label.pack(side="right", padx=5, pady=5)
    
    def _setup_monitor_tab(self):
        """Set up the monitor tab UI"""
        # Top controls
        control_frame = ttk.Frame(self.monitor_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.monitor_button = ttk.Button(control_frame, text="Start Monitoring", command=self._toggle_monitoring)
        self.monitor_button.pack(side="left", padx=5)
        
        self.clear_button = ttk.Button(control_frame, text="Clear", command=self._clear_monitor)
        self.clear_button.pack(side="left", padx=5)
        
        # Display options
        display_frame = ttk.LabelFrame(control_frame, text="Display Format")
        display_frame.pack(side="left", padx=10)
        
        ttk.Radiobutton(display_frame, text="ASCII", variable=self.display_mode, value="ASCII").pack(side="left", padx=5)
        ttk.Radiobutton(display_frame, text="HEX", variable=self.display_mode, value="HEX").pack(side="left", padx=5)
        ttk.Radiobutton(display_frame, text="Both", variable=self.display_mode, value="BOTH").pack(side="left", padx=5)
        
        # Logging controls
        log_frame = ttk.Frame(control_frame)
        log_frame.pack(side="right", padx=5)
        
        ttk.Checkbutton(log_frame, text="Log to file", variable=self.log_enabled).pack(side="left")
        self.log_button = ttk.Button(log_frame, text="Select Log File", command=self._select_log_file)
        self.log_button.pack(side="left", padx=5)
        
        # Data capture display
        capture_frame = ttk.Frame(self.monitor_frame)
        capture_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.monitor_text = scrolledtext.ScrolledText(capture_frame, wrap=tk.WORD, width=80, height=20)
        self.monitor_text.pack(expand=True, fill="both")
        self.monitor_text.config(state=tk.DISABLED)
        
        # Status bar for monitor
        self.monitor_status = ttk.Label(self.monitor_frame, text="Ready")
        self.monitor_status.pack(side="left", padx=5, pady=5)
        
        # Export button
        self.export_button = ttk.Button(self.monitor_frame, text="Export Data", command=self._export_capture)
        self.export_button.pack(side="right", padx=5, pady=5)
    
    def _setup_send_tab(self):
        """Set up the command sending tab UI"""
        # Command input area
        input_frame = ttk.LabelFrame(self.send_frame, text="Command Input")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        format_frame = ttk.Frame(input_frame)
        format_frame.pack(fill="x", padx=5, pady=5)
        
        self.input_format = tk.StringVar(value="ASCII")
        ttk.Radiobutton(format_frame, text="ASCII", variable=self.input_format, value="ASCII").pack(side="left", padx=5)
        ttk.Radiobutton(format_frame, text="HEX", variable=self.input_format, value="HEX").pack(side="left", padx=5)
        
        self.input_text = ttk.Entry(input_frame, width=50)
        self.input_text.pack(fill="x", padx=5, pady=5)
        self.input_text.bind("<Return>", lambda e: self._send_command())
        self.input_text.bind("<Up>", self._history_up)
        self.input_text.bind("<Down>", self._history_down)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.send_button = ttk.Button(button_frame, text="Send", command=self._send_command)
        self.send_button.pack(side="left", padx=5)
        
        self.add_newline = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Add Newline", variable=self.add_newline).pack(side="left", padx=5)
        
        # Command history display
        history_frame = ttk.LabelFrame(self.send_frame, text="Command History")
        history_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, width=80, height=15)
        self.history_text.pack(expand=True, fill="both")
        self.history_text.config(state=tk.DISABLED)
        
        # Saved commands section
        saved_frame = ttk.LabelFrame(self.send_frame, text="Saved Commands")
        saved_frame.pack(fill="x", padx=5, pady=5)
        
        self.saved_commands = []
        
        saved_buttons_frame = ttk.Frame(saved_frame)
        saved_buttons_frame.pack(fill="x", padx=5, pady=5)
        
        self.save_command_button = ttk.Button(saved_buttons_frame, text="Save Current Command", 
                                              command=self._save_current_command)
        self.save_command_button.pack(side="left", padx=5)
        
        self.saved_combobox = ttk.Combobox(saved_buttons_frame, width=40)
        self.saved_combobox.pack(side="left", padx=5)
        
        self.load_saved_button = ttk.Button(saved_buttons_frame, text="Load", 
                                            command=self._load_saved_command)
        self.load_saved_button.pack(side="left", padx=5)
        
        self.delete_saved_button = ttk.Button(saved_buttons_frame, text="Delete", 
                                              command=self._delete_saved_command)
        self.delete_saved_button.pack(side="left", padx=5)
    
    def _setup_fuzzer_tab(self):
        """Set up the fuzzer tab UI"""
        # Fuzzer configuration
        config_frame = ttk.LabelFrame(self.fuzzer_frame, text="Fuzzer Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Base command
        base_frame = ttk.Frame(config_frame)
        base_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(base_frame, text="Base Command:").pack(side="left", padx=5)
        self.base_command = ttk.Entry(base_frame, width=50)
        self.base_command.pack(side="left", padx=5, fill="x", expand=True)
        
        format_frame = ttk.Frame(base_frame)
        format_frame.pack(side="right", padx=5)
        
        self.fuzzer_format = tk.StringVar(value="ASCII")
        ttk.Radiobutton(format_frame, text="ASCII", variable=self.fuzzer_format, value="ASCII").pack(side="left", padx=5)
        ttk.Radiobutton(format_frame, text="HEX", variable=self.fuzzer_format, value="HEX").pack(side="left", padx=5)
        
        # Fuzzing configuration
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill="x", padx=5, pady=5)
        
        # Target byte position
        pos_frame = ttk.Frame(params_frame)
        pos_frame.pack(side="left", padx=10)
        
        ttk.Label(pos_frame, text="Target Position:").pack(side="left", padx=5)
        self.target_position = ttk.Spinbox(pos_frame, from_=0, to=100, width=5)
        self.target_position.set(0)
        self.target_position.pack(side="left", padx=5)
        
        # Byte range
        range_frame = ttk.Frame(params_frame)
        range_frame.pack(side="left", padx=10)
        
        ttk.Label(range_frame, text="Range:").pack(side="left", padx=5)
        ttk.Label(range_frame, text="From:").pack(side="left", padx=5)
        self.range_start = ttk.Spinbox(range_frame, from_=0, to=255, width=5)
        self.range_start.set(0)
        self.range_start.pack(side="left", padx=5)
        
        ttk.Label(range_frame, text="To:").pack(side="left", padx=5)
        self.range_end = ttk.Spinbox(range_frame, from_=0, to=255, width=5)
        self.range_end.set(255)
        self.range_end.pack(side="left", padx=5)
        
        # Timing
        timing_frame = ttk.Frame(params_frame)
        timing_frame.pack(side="left", padx=10)
        
        ttk.Label(timing_frame, text="Delay (ms):").pack(side="left", padx=5)
        self.delay_ms = ttk.Spinbox(timing_frame, from_=0, to=5000, width=7)
        self.delay_ms.set(100)
        self.delay_ms.pack(side="left", padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.start_fuzzer_button = ttk.Button(control_frame, text="Start Fuzzing", command=self._start_fuzzing)
        self.start_fuzzer_button.pack(side="left", padx=5)
        
        self.stop_fuzzer_button = ttk.Button(control_frame, text="Stop", command=self._stop_fuzzing)
        self.stop_fuzzer_button.pack(side="left", padx=5)
        self.stop_fuzzer_button.config(state=tk.DISABLED)
        
        # Fuzzing results
        results_frame = ttk.LabelFrame(self.fuzzer_frame, text="Fuzzing Results")
        results_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.fuzzer_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=80, height=15)
        self.fuzzer_text.pack(expand=True, fill="both")
        self.fuzzer_text.config(state=tk.DISABLED)
        
        # Status bar for fuzzer
        status_frame = ttk.Frame(self.fuzzer_frame)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self.fuzzer_status = ttk.Label(status_frame, text="Ready")
        self.fuzzer_status.pack(side="left", padx=5)
        
        self.fuzzer_progress = ttk.Progressbar(status_frame, orient="horizontal", length=200, mode="determinate")
        self.fuzzer_progress.pack(side="right", padx=5)
        
        # Fuzzing state variables
        self.is_fuzzing = False
        self.fuzzer_thread = None
    
    def _update_port_list(self):
        """Update the list of available serial ports"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combobox["values"] = ports
        if ports:
            self.port_combobox.current(0)
    
    def _toggle_connection(self):
        """Connect to or disconnect from the serial port"""
        if not self.is_connected:
            # Connect to the serial port
            try:
                port = self.port_combobox.get()
                baudrate = int(self.baud_combobox.get())
                
                # Get parity setting
                parity_val = self.parity_combobox.get()
                parity_dict = {"None": serial.PARITY_NONE, "Even": serial.PARITY_EVEN, 
                              "Odd": serial.PARITY_ODD, "Mark": serial.PARITY_MARK, 
                              "Space": serial.PARITY_SPACE}
                parity = parity_dict.get(parity_val, serial.PARITY_NONE)
                
                # Get data bits
                databits = int(self.data_bits_combobox.get())
                
                # Get stop bits
                stopbits_val = self.stop_bits_combobox.get()
                stopbits_dict = {"1": serial.STOPBITS_ONE, "1.5": serial.STOPBITS_ONE_POINT_FIVE, 
                                "2": serial.STOPBITS_TWO}
                stopbits = stopbits_dict.get(stopbits_val, serial.STOPBITS_ONE)
                
                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    bytesize=databits,
                    parity=parity,
                    stopbits=stopbits,
                    timeout=0.1
                )
                
                self.is_connected = True
                self.connect_button.config(text="Disconnect")
                self.status_label.config(text=f"Connected to {port}")
                
                # Enable monitoring and send controls
                self.monitor_button.config(state=tk.NORMAL)
                self.send_button.config(state=tk.NORMAL)
                self.start_fuzzer_button.config(state=tk.NORMAL)
                
            except (serial.SerialException, ValueError) as e:
                messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
                return
        else:
            # Disconnect
            self._stop_monitoring()
            if self.serial_port:
                self.serial_port.close()
            self.serial_port = None
            self.is_connected = False
            self.connect_button.config(text="Connect")
            self.status_label.config(text="Disconnected")
            
            # Disable controls
            self.monitor_button.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            self.start_fuzzer_button.config(state=tk.DISABLED)
    
    def _toggle_monitoring(self):
        """Start or stop monitoring the serial port"""
        if not self.is_monitoring:
            # Start monitoring
            self.is_monitoring = True
            self.monitor_button.config(text="Stop Monitoring")
            self.monitor_status.config(text="Monitoring active")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_task, daemon=True)
            self.monitor_thread.start()
        else:
            # Stop monitoring
            self._stop_monitoring()
    
    def _stop_monitoring(self):
        """Stop the monitoring thread"""
        if self.is_monitoring:
            self.is_monitoring = False
            self.monitor_button.config(text="Start Monitoring")
            self.monitor_status.config(text="Monitoring stopped")
    
    def _monitoring_task(self):
        """Background task to read from the serial port"""
        while self.is_monitoring and self.is_connected:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # Queue the data for UI display
                        self.packet_queue.put(("RX", timestamp, data))
                        
                        # Add to capture buffer
                        self.capture_buffer.append({
                            "type": "RX",
                            "timestamp": timestamp,
                            "data": binascii.hexlify(data).decode('ascii'),
                            "ascii": ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
                        })
                        
                        # Log to file if enabled
                        if self.log_enabled.get() and self.log_file:
                            self._write_to_log("RX", timestamp, data)
            except serial.SerialException:
                self._stop_monitoring()
                messagebox.showerror("Error", "Serial port disconnected or error occurred.")
                break
                
            # Small delay to prevent CPU hogging
            time.sleep(0.01)
    
    def _process_packet_queue(self):
        """Process and display queued packets on the UI thread"""
        while not self.packet_queue.empty():
            try:
                direction, timestamp, data = self.packet_queue.get_nowait()
                self._display_data(direction, timestamp, data)
            except Exception as e:
                print(f"Error processing packet: {e}")
        
        # Schedule the next queue check
        self.root.after(100, self._process_packet_queue)
    
    def _display_data(self, direction, timestamp, data):
        """Display received data in the monitor window"""
        if not data:
            return
            
        self.monitor_text.config(state=tk.NORMAL)
        
        # Display direction indicator
        direction_tag = "rx" if direction == "RX" else "tx"
        self.monitor_text.insert(tk.END, f"[{timestamp}] {direction}: ", direction_tag)
        
        # Display data according to selected format
        display_format = self.display_mode.get()
        if display_format == "ASCII" or display_format == "BOTH":
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
            self.monitor_text.insert(tk.END, ascii_str, f"{direction_tag}_ascii")
            
        if display_format == "HEX" or display_format == "BOTH":
            if display_format == "BOTH":
                self.monitor_text.insert(tk.END, " (", f"{direction_tag}_plain")
                
            hex_str = binascii.hexlify(data).decode('ascii')
            # Insert spaces between bytes for readability
            hex_str = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
            self.monitor_text.insert(tk.END, hex_str, f"{direction_tag}_hex")
            
            if display_format == "BOTH":
                self.monitor_text.insert(tk.END, ")", f"{direction_tag}_plain")
        
        self.monitor_text.insert(tk.END, "\n")
        self.monitor_text.see(tk.END)
        self.monitor_text.config(state=tk.DISABLED)
    
    def _clear_monitor(self):
        """Clear the monitor display and capture buffer"""
        self.monitor_text.config(state=tk.NORMAL)
        self.monitor_text.delete(1.0, tk.END)
        self.monitor_text.config(state=tk.DISABLED)
        self.capture_buffer = []
    
    def _select_log_file(self):
        """Select a log file for saving captured data"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.log_file = open(file_path, "a")
                self.log_enabled.set(True)
                self.log_button.config(text=f"Logging to: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open log file: {str(e)}")
                self.log_enabled.set(False)
    
    def _write_to_log(self, direction, timestamp, data):
        """Write captured data to log file"""
        if not self.log_file:
            return
            
        try:
            hex_data = binascii.hexlify(data).decode('ascii')
            ascii_data = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
            
            log_line = f"[{timestamp}] {direction}: HEX={hex_data} ASCII={ascii_data}\n"
            self.log_file.write(log_line)
            self.log_file.flush()
        except Exception as e:
            print(f"Error writing to log: {e}")
            self.log_enabled.set(False)
    
    def _export_capture(self):
        """Export the captured data to a file"""
        if not self.capture_buffer:
            messagebox.showinfo("Export", "No data to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.capture_buffer, f, indent=2)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _send_command(self):
        """Send a command through the serial port"""
        if not self.is_connected or not self.serial_port:
            messagebox.showerror("Error", "Not connected to a serial port")
            return
            
        command = self.input_text.get().strip()
        if not command:
            return
            
        try:
            # Process the command based on input format
            if self.input_format.get() == "HEX":
                # Remove spaces and validate hex string
                command = command.replace(" ", "")
                if not re.match(r'^[0-9a-fA-F]+$', command):
                    messagebox.showerror("Error", "Invalid hex format")
                    return
                    
                # Convert hex string to bytes
                data = binascii.unhexlify(command)
            else:
                # ASCII format
                data = command.encode('utf-8')
                
            # Add newline if requested
            if self.add_newline.get():
                data += b'\r\n'
                
            # Send the data
            self.serial_port.write(data)
            
            # Add to command history
            self.command_history.append(command)
            self.history_position = len(self.command_history)
            
            # Display in history window
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            self.history_text.config(state=tk.NORMAL)
            self.history_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            
            if self.input_format.get() == "HEX":
                # Format hex for display
                formatted_hex = ' '.join(command[i:i+2] for i in range(0, len(command), 2))
                self.history_text.insert(tk.END, f"HEX: {formatted_hex}\n", "command")
            else:
                self.history_text.insert(tk.END, f"ASCII: {command}\n", "command")
                
            self.history_text.see(tk.END)
            self.history_text.config(state=tk.DISABLED)
            
            # Queue the data for monitor display if monitoring
            if self.is_monitoring:
                self.packet_queue.put(("TX", timestamp, data))
                
                # Add to capture buffer
                self.capture_buffer.append({
                    "type": "TX",
                    "timestamp": timestamp,
                    "data": binascii.hexlify(data).decode('ascii'),
                    "ascii": ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
                })
                
                # Log to file if enabled
                if self.log_enabled.get() and self.log_file:
                    self._write_to_log("TX", timestamp, data)
                    
            # Clear the input field
            self.input_text.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Send Error", f"Failed to send command: {str(e)}")
    
    def _history_up(self, event):
        """Navigate up through command history"""
        if not self.command_history or self.history_position <= 0:
            return
            
        self.history_position -= 1
        self.input_text.delete(0, tk.END)
        self.input_text.insert(0, self.command_history[self.history_position])
    
    def _history_down(self, event):
        """Navigate down through command history"""
        if not self.command_history or self.history_position >= len(self.command_history):
            return
            
        self.history_position += 1
        self.input_text.delete(0, tk.END)
        if self.history_position < len(self.command_history):
            self.input_text.insert(0, self.command_history[self.history_position])
    
    def _save_current_command(self):
        """Save the current command to the saved commands list"""
        command = self.input_text.get().strip()
        if not command:
            return
            
        name = simpledialog.askstring("Save Command", "Enter a name for this command:")
        if name:
            self.saved_commands.append({"name": name, "command": command, "format": self.input_format.get()})
            self._update_saved_commands_list()
    
    def _update_saved_commands_list(self):
        """Update the saved commands dropdown"""
        self.saved_combobox["values"] = [cmd["name"] for cmd in self.saved_commands]
        if self.saved_commands:
            self.saved_combobox.current(0)
    
    def _load_saved_command(self):
        """Load a saved command into the input field"""
        selected = self.saved_combobox.get()
        for cmd in self.saved_commands:
            if cmd["name"] == selected:
                self.input_text.delete(0, tk.END)
                self.input_text.insert(0, cmd["command"])
                self.input_format.set(cmd["format"])
                break
    
    def _delete_saved_command(self):
        """Delete a saved command"""
        selected = self.saved_combobox.get()
        self.saved_commands = [cmd for cmd in self.saved_commands if cmd["name"] != selected]
        self._update_saved_commands_list()
    
    def _start_fuzzing(self):
        """Start the fuzzing process"""
        if not self.is_connected or not self.serial_port:
            messagebox.showerror("Error", "Not connected to a serial port")
            return
            
        base_command = self.base_command.get().strip()
        if not base_command:
            messagebox.showerror("Error", "Base command is required")
            return
            
        try:
            # Process the base command
            if self.fuzzer_format.get() == "HEX":
                base_command = base_command.replace(" ", "")
                if not re.match(r'^[0-9a-fA-F]+, base_command):
                    messagebox.showerror("Error", "Invalid hex format for base command")
                    return
                    
                # Convert to bytes
                cmd_bytes = bytearray(binascii.unhexlify(base_command))
            else:
                # ASCII format
                cmd_bytes = bytearray(base_command.encode('utf-8'))
                
            # Check target position
            target_pos = int(self.target_position.get())
            if target_pos >= len(cmd_bytes):
                messagebox.showerror("Error", f"Target position {target_pos} exceeds command length {len(cmd_bytes)}")
                return
                
            # Get range
            start_val = int(self.range_start.get())
            end_val = int(self.range_end.get())
            if start_val > end_val:
                start_val, end_val = end_val, start_val
                
            # Validate values
            if start_val < 0 or end_val > 255:
                messagebox.showerror("Error", "Valid byte range is 0-255")
                return
                
            # Get delay
            delay_ms = int(self.delay_ms.get())
            
            # Disable controls
            self.start_fuzzer_button.config(state=tk.DISABLED)
            self.stop_fuzzer_button.config(state=tk.NORMAL)
            self.base_command.config(state=tk.DISABLED)
            self.target_position.config(state=tk.DISABLED)
            self.range_start.config(state=tk.DISABLED)
            self.range_end.config(state=tk.DISABLED)
            self.delay_ms.config(state=tk.DISABLED)
            
            # Clear results
            self.fuzzer_text.config(state=tk.NORMAL)
            self.fuzzer_text.delete(1.0, tk.END)
            self.fuzzer_text.config(state=tk.DISABLED)
            
            # Set up progress bar
            total_iterations = (end_val - start_val) + 1
            self.fuzzer_progress["maximum"] = total_iterations
            self.fuzzer_progress["value"] = 0
            
            # Start fuzzing thread
            self.is_fuzzing = True
            self.fuzzer_status.config(text="Fuzzing in progress...")
            self.fuzzer_thread = threading.Thread(
                target=self._fuzzing_task, 
                args=(cmd_bytes, target_pos, start_val, end_val, delay_ms),
                daemon=True
            )
            self.fuzzer_thread.start()
            
        except Exception as e:
            messagebox.showerror("Fuzzer Error", f"Failed to start fuzzing: {str(e)}")
            self._stop_fuzzing()
    
    def _fuzzing_task(self, base_cmd, target_pos, start_val, end_val, delay_ms):
        """Background task for fuzzing"""
        total_values = (end_val - start_val) + 1
        progress_count = 0
        
        try:
            for val in range(start_val, end_val + 1):
                if not self.is_fuzzing:
                    break
                    
                # Create modified command
                cmd_copy = bytearray(base_cmd)
                cmd_copy[target_pos] = val
                
                # Send command
                self.serial_port.write(cmd_copy)
                
                # Log command
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                progress_count += 1
                
                # Update UI
                self.root.after(10, self._update_fuzzer_progress, progress_count, total_values)
                self.root.after(10, self._log_fuzzer_command, timestamp, val, cmd_copy)
                
                # Wait for response
                time.sleep(delay_ms / 1000.0)
                
            # Completed
            if self.is_fuzzing:
                self.root.after(0, self._fuzzing_completed)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Fuzzer Error", f"Error during fuzzing: {str(e)}"))
            self.root.after(0, self._stop_fuzzing)
    
    def _update_fuzzer_progress(self, current, total):
        """Update the fuzzer progress bar"""
        self.fuzzer_progress["value"] = current
        percent = int((current / total) * 100)
        self.fuzzer_status.config(text=f"Fuzzing in progress... {percent}% ({current}/{total})")
    
    def _log_fuzzer_command(self, timestamp, val, cmd_bytes):
        """Log a fuzzer command to the results window"""
        self.fuzzer_text.config(state=tk.NORMAL)
        
        # Format command for display
        if self.fuzzer_format.get() == "HEX":
            cmd_str = binascii.hexlify(cmd_bytes).decode('ascii')
            cmd_str = ' '.join(cmd_str[i:i+2] for i in range(0, len(cmd_str), 2))
            self.fuzzer_text.insert(tk.END, f"[{timestamp}] Value: {val:03d} (0x{val:02X}) Command: {cmd_str}\n")
        else:
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in cmd_bytes)
            self.fuzzer_text.insert(tk.END, f"[{timestamp}] Value: {val:03d} (0x{val:02X}) Command: {ascii_str}\n")
            
        self.fuzzer_text.see(tk.END)
        self.fuzzer_text.config(state=tk.DISABLED)
    
    def _fuzzing_completed(self):
        """Called when fuzzing is completed"""
        self.fuzzer_status.config(text="Fuzzing completed")
        self._stop_fuzzing()
    
    def _stop_fuzzing(self):
        """Stop the fuzzing process"""
        self.is_fuzzing = False
        
        # Re-enable controls
        self.start_fuzzer_button.config(state=tk.NORMAL)
        self.stop_fuzzer_button.config(state=tk.DISABLED)
        self.base_command.config(state=tk.NORMAL)
        self.target_position.config(state=tk.NORMAL)
        self.range_start.config(state=tk.NORMAL)
        self.range_end.config(state=tk.NORMAL)
        self.delay_ms.config(state=tk.NORMAL)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = SerialSniffer(root)
    
    # Configure some basic styling
    style = ttk.Style()
    style.configure("TButton", padding=5)
    style.configure("TLabel", padding=2)
    
    # Set up text styles
    app.monitor_text.tag_configure("rx", foreground="blue")
    app.monitor_text.tag_configure("tx", foreground="green")
    app.monitor_text.tag_configure("rx_ascii", foreground="blue")
    app.monitor_text.tag_configure("tx_ascii", foreground="green")
    app.monitor_text.tag_configure("rx_hex", foreground="darkblue")
    app.monitor_text.tag_configure("tx_hex", foreground="darkgreen")
    app.monitor_text.tag_configure("rx_plain", foreground="blue")
    app.monitor_text.tag_configure("tx_plain", foreground="green")
    
    app.history_text.tag_configure("timestamp", foreground="gray")
    app.history_text.tag_configure("command", foreground="black")
    
    root.mainloop()
    
    # Clean up resources
    if app.serial_port and app.serial_port.is_open:
        app.serial_port.close()
    if app.log_file:
        app.log_file.close()


if __name__ == "__main__":
    try:
        from tkinter import simpledialog
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure you have the required packages installed:")
        print("pip install pyserial")
        sys.exit(1)
