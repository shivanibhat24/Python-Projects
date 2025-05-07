#!/usr/bin/env python3
"""
UART Security Monitor - A tool for monitoring and analyzing UART communications
For authorized security testing purposes only.
"""

import serial
import argparse
import time
import re
import logging
import os
from datetime import datetime

class UARTMonitor:
    def __init__(self, port, baudrate, timeout=1, log_file=None):
        """Initialize the UART monitor with specified parameters."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.patterns = {
            'login': re.compile(r'(?i)(login|username|user)[\s]*:'),
            'password': re.compile(r'(?i)(password|pwd|pass)[\s]*:'),
            'shell_prompt': re.compile(r'[$#>]\s*$')
        }
        self.credentials = {}
        self.current_context = None
        self.buffer = ""
        
        # Setup logging
        self.setup_logging(log_file)
        
    def setup_logging(self, log_file):
        """Configure logging for the application."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logging.basicConfig(
                filename=log_file, 
                level=logging.INFO,
                format=log_format
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists('logs'):
                os.makedirs('logs')
            log_file = f"logs/uart_monitor_{timestamp}.log"
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format=log_format
            )
        
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        logging.info(f"Starting UART monitoring on {self.port} at {self.baudrate} baud")
    
    def start_monitoring(self):
        """Begin monitoring the UART interface."""
        try:
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                logging.info(f"Successfully connected to {self.port}")
                print(f"\nMonitoring UART traffic on {self.port} at {self.baudrate} baud...")
                print("Press Ctrl+C to stop.\n")
                
                while True:
                    if ser.in_waiting:
                        data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
                        self.process_data(data)
                    time.sleep(0.1)
                    
        except serial.SerialException as e:
            logging.error(f"Serial port error: {e}")
            print(f"Error: Could not open port {self.port}. {e}")
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
            print("\nMonitoring stopped.")
            self.show_results()
    
    def process_data(self, data):
        """Process incoming data and look for credential patterns."""
        self.buffer += data
        print(data, end='', flush=True)
        
        # Check for login prompt
        if self.patterns['login'].search(self.buffer):
            self.current_context = 'username'
            logging.info("Login prompt detected")
            self.buffer = ""
            return
            
        # Check for password prompt
        if self.patterns['password'].search(self.buffer):
            self.current_context = 'password'
            logging.info("Password prompt detected")
            self.buffer = ""
            return
            
        # Check for shell prompt (indicating successful login)
        if self.patterns['shell_prompt'].search(self.buffer):
            if 'username' in self.credentials and 'password' in self.credentials:
                logging.info(f"Possible successful login detected for user: {self.credentials['username']}")
                print(f"\n[!] Possible successful login: {self.credentials['username']}:{self.credentials['password']}")
            self.credentials = {}  # Reset for next login attempt
            self.current_context = None
            self.buffer = ""
            return
            
        # Capture input after prompts (when Enter key is pressed)
        if self.current_context and '\r' in self.buffer or '\n' in self.buffer:
            # Clean up the input
            input_text = self.buffer.strip().replace('\r', '').replace('\n', '')
            
            if input_text and self.current_context == 'username':
                self.credentials['username'] = input_text
                logging.info(f"Username captured: {input_text}")
                print(f"\n[+] Username captured: {input_text}")
                
            elif input_text and self.current_context == 'password':
                self.credentials['password'] = input_text
                logging.info(f"Password captured: {input_text}")
                print(f"\n[+] Password captured: {input_text}")
                
            self.buffer = ""
            
        # Prevent buffer from growing too large
        if len(self.buffer) > 4096:
            self.buffer = self.buffer[-2048:]
    
    def show_results(self):
        """Display a summary of the monitoring session."""
        if 'username' in self.credentials or 'password' in self.credentials:
            print("\n--- Captured Credentials ---")
            username = self.credentials.get('username', 'Not captured')
            password = self.credentials.get('password', 'Not captured')
            print(f"Username: {username}")
            print(f"Password: {password}")
            logging.info(f"Session ended with credentials - Username: {username}, Password: {password}")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='UART Security Monitoring Tool')
    parser.add_argument('-p', '--port', required=True, help='Serial port to monitor')
    parser.add_argument('-b', '--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('-t', '--timeout', type=float, default=1, help='Read timeout in seconds (default: 1)')
    parser.add_argument('-l', '--log', help='Log file path')
    
    args = parser.parse_args()
    
    # Display ethical usage disclaimer
    print("\n" + "="*60)
    print("UART Security Monitoring Tool - For authorized testing only")
    print("="*60)
    print("WARNING: This tool should only be used for legitimate security")
    print("testing with proper authorization. Unauthorized use may violate")
    print("applicable laws and regulations.")
    print("="*60 + "\n")
    
    # Prompt for confirmation
    confirm = input("I confirm I am using this tool for authorized security testing (y/n): ")
    if confirm.lower() != 'y':
        print("Exiting without monitoring.")
        return
    
    # Start monitoring
    monitor = UARTMonitor(args.port, args.baudrate, args.timeout, args.log)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
