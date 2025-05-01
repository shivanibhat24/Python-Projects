#!/usr/bin/env python3
"""
Serial Device Simulator Farm

This script creates virtual serial port pairs and simulates different types of serial devices
(modems, GPS units, etc.) on one end while allowing test applications to connect to the other.

Requires:
- Python 3.6+
- pyserial
- pyserial-virtport (Windows) or socat (Linux/macOS)
"""

import os
import sys
import time
import threading
import random
import argparse
import json
import logging
from datetime import datetime
from enum import Enum
import serial
import signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SerialFarm')

# Check platform and import appropriate modules
if sys.platform.startswith('win'):
    import virtualportpair
    PLATFORM = "Windows"
else:
    import subprocess
    import shlex
    PLATFORM = "Unix"


class DeviceType(Enum):
    MODEM = "modem"
    GPS = "gps"
    TELEMETRY = "telemetry"
    CUSTOM = "custom"


class SerialPortPair:
    """Creates and manages a pair of virtual serial ports"""
    
    def __init__(self, name):
        self.name = name
        self.port_a = None
        self.port_b = None
        self.socat_process = None
        
    def create(self):
        """Create a pair of virtual serial ports"""
        if PLATFORM == "Windows":
            self.port_a, self.port_b = virtualportpair.create_port_pair()
            logger.info(f"Created virtual port pair: {self.port_a} <-> {self.port_b}")
            return self.port_a, self.port_b
        else:
            # Create a temporary socat command for Unix-based systems
            # Format: /dev/pts/X and /dev/pts/Y
            cmd = "socat -d -d pty,raw,echo=0 pty,raw,echo=0"
            self.socat_process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Get port names from socat output
            for _ in range(2):  # We expect 2 lines of output for the 2 PTYs
                line = self.socat_process.stderr.readline().strip()
                if "PTY is /dev/" in line:
                    port_path = line.split("PTY is ")[1].strip()
                    if self.port_a is None:
                        self.port_a = port_path
                    else:
                        self.port_b = port_path
            
            logger.info(f"Created virtual port pair: {self.port_a} <-> {self.port_b}")
            return self.port_a, self.port_b
    
    def destroy(self):
        """Destroy the virtual serial port pair"""
        if PLATFORM == "Windows":
            # Close and remove the serial port pair
            if self.port_a and self.port_b:
                virtualportpair.remove_port_pair(self.port_a, self.port_b)
                logger.info(f"Removed virtual port pair: {self.port_a} <-> {self.port_b}")
        else:
            # Kill the socat process
            if self.socat_process:
                self.socat_process.terminate()
                self.socat_process.wait()
                logger.info(f"Terminated socat process for port pair: {self.port_a} <-> {self.port_b}")


class SerialDeviceSimulator:
    """Base class for serial device simulators"""
    
    def __init__(self, port, device_type, config=None):
        self.port = port
        self.device_type = device_type
        self.config = config or {}
        self.running = False
        self.serial = None
        self.thread = None
        
    def start(self):
        """Start the device simulator"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.config.get('baudrate', 9600),
                bytesize=self.config.get('bytesize', serial.EIGHTBITS),
                parity=self.config.get('parity', serial.PARITY_NONE),
                stopbits=self.config.get('stopbits', serial.STOPBITS_ONE),
                timeout=self.config.get('timeout', 1)
            )
            
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info(f"Started {self.device_type.value} simulator on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start {self.device_type.value} simulator on {self.port}: {e}")
            return False
    
    def stop(self):
        """Stop the device simulator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial:
            self.serial.close()
        logger.info(f"Stopped {self.device_type.value} simulator on {self.port}")
    
    def run(self):
        """Main loop for the device simulator - to be implemented by subclasses"""
        pass
    
    def handle_command(self, command):
        """Handle a command from the serial port - to be implemented by subclasses"""
        pass


class ModemSimulator(SerialDeviceSimulator):
    """Simulates a Hayes AT command compatible modem"""
    
    def __init__(self, port, config=None):
        super().__init__(port, DeviceType.MODEM, config)
        self.config.setdefault('echo', True)
        self.config.setdefault('verbose', True)
        self.config.setdefault('response_delay', 0.1)
        self.connection_status = "disconnected"  # disconnected, connecting, connected
        self.buffer = ""
        
    def run(self):
        """Main loop for the modem simulator"""
        while self.running:
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting).decode('ascii', errors='replace')
                
                # Echo characters if enabled
                if self.config['echo']:
                    self.serial.write(data.encode('ascii'))
                
                self.buffer += data
                
                # Process complete commands (ending with CR or LF)
                if '\r' in self.buffer or '\n' in self.buffer:
                    commands = self.buffer.replace('\r', '\n').split('\n')
                    self.buffer = commands.pop()  # Keep the incomplete command (if any)
                    
                    for cmd in commands:
                        cmd = cmd.strip()
                        if cmd:
                            self.handle_command(cmd)
            
            # Simulate random events like incoming calls if connected
            if self.connection_status == "connected" and random.random() < 0.001:
                self.serial.write(b"\r\nRING\r\n")
            
            time.sleep(0.05)
    
    def handle_command(self, command):
        """Handle AT commands"""
        # Wait for configured response delay
        time.sleep(self.config['response_delay'])
        
        # Strip AT prefix for processing
        cmd = command.upper()
        if not cmd.startswith("AT"):
            self.send_response("ERROR")
            return
        
        cmd = cmd[2:].strip()  # Remove AT prefix
        
        # Process special cases or pass to generic handler
        if cmd == "":  # Basic AT command
            self.send_response("OK")
        elif cmd == "Z":  # Reset
            self.send_response("OK")
        elif cmd == "I":  # Information
            self.send_response("SerialFarm Virtual Modem v1.0\r\nOK")
        elif cmd == "H0":  # Hang up
            self.connection_status = "disconnected"
            self.send_response("OK")
        elif cmd.startswith("DT") or cmd.startswith("D"):  # Dial
            number = cmd[2:] if cmd.startswith("DT") else cmd[1:]
            self.connection_status = "connecting"
            self.send_response("CONNECT 9600")
            self.connection_status = "connected"
        elif cmd == "H1":  # Off hook
            self.send_response("OK")
        elif cmd == "O":  # Return to online state
            if self.connection_status == "connected":
                self.send_response("CONNECT 9600")
            else:
                self.send_response("ERROR")
        elif cmd == "H":  # Hook control
            self.send_response("OK")
        elif cmd.startswith("&F"):  # Factory defaults
            self.send_response("OK")
        elif cmd.startswith("E"):  # Command echo
            if len(cmd) > 1:
                self.config['echo'] = cmd[1] == "1"
            self.send_response("OK")
        elif cmd.startswith("V"):  # Verbose responses
            if len(cmd) > 1:
                self.config['verbose'] = cmd[1] == "1"
            self.send_response("OK")
        elif cmd.startswith("S"):  # S-registers
            self.send_response("OK")
        elif cmd.startswith("+"):  # Extended AT commands
            if cmd.startswith("+GMM"):  # Model
                self.send_response("SerialFarm-Modem\r\nOK")
            elif cmd.startswith("+GMR"):  # Revision
                self.send_response("1.0.0\r\nOK")
            elif cmd.startswith("+CSQ"):  # Signal quality
                self.send_response("+CSQ: 25,0\r\nOK")
            else:
                self.send_response("OK")
        else:
            self.send_response("OK")
    
    def send_response(self, response):
        """Send a response to the client"""
        if self.config['verbose']:
            self.serial.write(f"\r\n{response}\r\n".encode('ascii'))
        else:
            # Numeric responses
            if response == "OK":
                self.serial.write(b"\r\n0\r\n")
            elif response == "ERROR":
                self.serial.write(b"\r\n4\r\n")
            else:
                self.serial.write(f"\r\n{response}\r\n".encode('ascii'))


class GPSSimulator(SerialDeviceSimulator):
    """Simulates a GPS device sending NMEA sentences"""
    
    def __init__(self, port, config=None):
        super().__init__(port, DeviceType.GPS, config)
        self.config.setdefault('update_rate', 1.0)  # seconds
        self.config.setdefault('start_lat', 37.7749)  # San Francisco
        self.config.setdefault('start_lon', -122.4194)
        self.config.setdefault('movement', True)
        self.config.setdefault('movement_range', 0.001)  # Degrees
        
        self.latitude = self.config['start_lat']
        self.longitude = self.config['start_lon']
        self.speed = 0.0
        self.course = 0.0
        self.last_update = time.time()
    
    def run(self):
        """Main loop for the GPS simulator"""
        while self.running:
            current_time = time.time()
            
            # Check if it's time to send a new update
            if current_time - self.last_update >= self.config['update_rate']:
                self.last_update = current_time
                
                # Update position if movement is enabled
                if self.config['movement']:
                    self.update_position()
                
                # Generate and send NMEA sentences
                self.send_nmea_sentences()
            
            # Check for incoming commands
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting).decode('ascii', errors='replace')
                # Most GPS units ignore incoming data, but you could implement commands here
            
            time.sleep(0.05)
    
    def update_position(self):
        """Update the simulated GPS position"""
        # Randomly change course occasionally
        if random.random() < 0.1:
            self.course = random.uniform(0, 360)
            self.speed = random.uniform(0, 60)  # knots
        
        # Calculate movement based on speed and course
        if self.speed > 0:
            # Convert speed from knots to degrees per update
            # This is a very simplified calculation
            movement = self.speed * 0.0001
            
            # Calculate new position (very simplified)
            self.latitude += movement * math.cos(math.radians(self.course))
            self.longitude += movement * math.sin(math.radians(self.course))
        
        # Add some random noise
        self.latitude += random.uniform(-0.0001, 0.0001)
        self.longitude += random.uniform(-0.0001, 0.0001)
    
    def send_nmea_sentences(self):
        """Generate and send NMEA sentences"""
        now = datetime.utcnow()
        time_str = now.strftime("%H%M%S.%f")[:-3]
        date_str = now.strftime("%d%m%y")
        
        # Generate GPRMC sentence (Recommended Minimum Navigation Information)
        lat_deg = int(abs(self.latitude))
        lat_min = (abs(self.latitude) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:06.3f}"
        lat_dir = "N" if self.latitude >= 0 else "S"
        
        lon_deg = int(abs(self.longitude))
        lon_min = (abs(self.longitude) - lon_deg) * 60
        lon_str = f"{lon_deg:03d}{lon_min:06.3f}"
        lon_dir = "E" if self.longitude >= 0 else "W"
        
        speed_knots = f"{self.speed:.1f}"
        course = f"{self.course:.1f}"
        
        # Build the GPRMC sentence
        gprmc = f"GPRMC,{time_str},{self.get_status()},{lat_str},{lat_dir},{lon_str},{lon_dir},{speed_knots},{course},{date_str},,,"
        checksum = self.calculate_checksum(gprmc)
        gprmc_sentence = f"${gprmc}*{checksum}\r\n"
        
        # Build the GPGGA sentence (Global Positioning System Fix Data)
        alt = f"{random.uniform(0, 100):.1f}"
        satellites = str(random.randint(4, 12))
        hdop = f"{random.uniform(0.8, 2.0):.1f}"
        
        gpgga = f"GPGGA,{time_str},{lat_str},{lat_dir},{lon_str},{lon_dir},{1},{satellites},{hdop},{alt},M,0.0,M,,"
        checksum = self.calculate_checksum(gpgga)
        gpgga_sentence = f"${gpgga}*{checksum}\r\n"
        
        # Send the sentences
        self.serial.write(gprmc_sentence.encode('ascii'))
        self.serial.write(gpgga_sentence.encode('ascii'))
    
    def get_status(self):
        """Return GPS status: A=active, V=void"""
        # Simulate occasional GPS signal loss
        return "A" if random.random() > 0.05 else "V"
    
    def calculate_checksum(self, sentence):
        """Calculate NMEA checksum"""
        checksum = 0
        for char in sentence:
            checksum ^= ord(char)
        return f"{checksum:02X}"


class TelemetrySimulator(SerialDeviceSimulator):
    """Simulates a telemetry device sending sensor data"""
    
    def __init__(self, port, config=None):
        super().__init__(port, DeviceType.TELEMETRY, config)
        self.config.setdefault('update_rate', 1.0)  # seconds
        self.config.setdefault('format', 'json')  # 'json', 'csv', or 'binary'
        self.config.setdefault('sensors', {
            'temperature': {'min': 20.0, 'max': 30.0, 'unit': 'C'},
            'humidity': {'min': 40.0, 'max': 60.0, 'unit': '%'},
            'pressure': {'min': 990.0, 'max': 1010.0, 'unit': 'hPa'},
            'battery': {'min': 3.0, 'max': 4.2, 'unit': 'V'}
        })
        
        self.sensor_values = {}
        for sensor, config in self.config['sensors'].items():
            self.sensor_values[sensor] = random.uniform(config['min'], config['max'])
        
        self.last_update = time.time()
        self.buffer = ""
    
    def run(self):
        """Main loop for the telemetry simulator"""
        while self.running:
            current_time = time.time()
            
            # Check if it's time to send a new update
            if current_time - self.last_update >= self.config['update_rate']:
                self.last_update = current_time
                
                # Update sensor values
                self.update_sensor_values()
                
                # Send telemetry data
                self.send_telemetry_data()
            
            # Check for incoming commands
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting).decode('ascii', errors='replace')
                self.buffer += data
                
                # Process complete commands (ending with CR or LF)
                if '\r' in self.buffer or '\n' in self.buffer:
                    commands = self.buffer.replace('\r', '\n').split('\n')
                    self.buffer = commands.pop()  # Keep the incomplete command (if any)
                    
                    for cmd in commands:
                        cmd = cmd.strip()
                        if cmd:
                            self.handle_command(cmd)
            
            time.sleep(0.05)
    
    def update_sensor_values(self):
        """Update simulated sensor values"""
        for sensor, config in self.config['sensors'].items():
            # Calculate a random delta within a small range
            delta = random.uniform(-0.1, 0.1) * (config['max'] - config['min'])
            
            # Update the sensor value
            self.sensor_values[sensor] += delta
            
            # Keep values within configured range
            self.sensor_values[sensor] = max(config['min'], min(config['max'], self.sensor_values[sensor]))
    
    def send_telemetry_data(self):
        """Send telemetry data in the configured format"""
        if self.config['format'] == 'json':
            # Create a dictionary with sensor values and timestamp
            data = {
                'timestamp': datetime.now().isoformat(),
                'sensors': {}
            }
            
            for sensor, value in self.sensor_values.items():
                data['sensors'][sensor] = {
                    'value': round(value, 2),
                    'unit': self.config['sensors'][sensor]['unit']
                }
            
            # Send as JSON
            json_data = json.dumps(data) + "\r\n"
            self.serial.write(json_data.encode('ascii'))
            
        elif self.config['format'] == 'csv':
            # Create CSV header if it's the first run
            if not hasattr(self, 'csv_header_sent'):
                header = "timestamp," + ",".join(self.sensor_values.keys()) + "\r\n"
                self.serial.write(header.encode('ascii'))
                self.csv_header_sent = True
            
            # Create CSV line
            timestamp = datetime.now().isoformat()
            values = [str(round(v, 2)) for v in self.sensor_values.values()]
            line = timestamp + "," + ",".join(values) + "\r\n"
            self.serial.write(line.encode('ascii'))
            
        elif self.config['format'] == 'binary':
            # Create a simple binary packet
            # Format: [0xAA][0x55][timestamp(4)][sensor1(4)][sensor2(4)]...
            packet = bytearray([0xAA, 0x55])
            
            # Add timestamp (4 bytes representing seconds since epoch)
            timestamp = int(time.time())
            packet.extend(timestamp.to_bytes(4, byteorder='big'))
            
            # Add sensor values (4 bytes each, as float converted to int with scaling)
            for value in self.sensor_values.values():
                # Scale float to int for simplicity
                scaled_value = int(value * 100)
                packet.extend(scaled_value.to_bytes(4, byteorder='big'))
            
            # Calculate simple checksum
            checksum = sum(packet) & 0xFF
            packet.append(checksum)
            
            # Send binary packet
            self.serial.write(packet)
    
    def handle_command(self, command):
        """Handle commands received from the client"""
        # Some example commands
        if command.upper() == "GET_DATA":
            self.send_telemetry_data()
        elif command.upper() == "GET_CONFIG":
            config_json = json.dumps(self.config) + "\r\n"
            self.serial.write(config_json.encode('ascii'))
        elif command.upper().startswith("SET_RATE:"):
            try:
                new_rate = float(command.split(":")[1])
                if 0.1 <= new_rate <= 60:
                    self.config['update_rate'] = new_rate
                    self.serial.write(b"OK\r\n")
                else:
                    self.serial.write(b"ERROR: Rate must be between 0.1 and 60 seconds\r\n")
            except:
                self.serial.write(b"ERROR: Invalid rate format\r\n")
        elif command.upper().startswith("SET_FORMAT:"):
            new_format = command.split(":")[1].lower()
            if new_format in ["json", "csv", "binary"]:
                self.config['format'] = new_format
                self.serial.write(b"OK\r\n")
            else:
                self.serial.write(b"ERROR: Format must be json, csv, or binary\r\n")
        else:
            self.serial.write(b"ERROR: Unknown command\r\n")


class CustomDeviceSimulator(SerialDeviceSimulator):
    """Simulates a custom device with user-defined behavior"""
    
    def __init__(self, port, config=None):
        super().__init__(port, DeviceType.CUSTOM, config)
        self.config.setdefault('commands', {})
        self.config.setdefault('auto_messages', [])
        self.config.setdefault('auto_message_interval', 5.0)
        self.buffer = ""
        self.last_auto_message = time.time()
    
    def run(self):
        """Main loop for the custom device simulator"""
        while self.running:
            # Check for incoming commands
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting).decode('ascii', errors='replace')
                self.buffer += data
                
                # Process complete commands (ending with CR or LF)
                if '\r' in self.buffer or '\n' in self.buffer:
                    commands = self.buffer.replace('\r', '\n').split('\n')
                    self.buffer = commands.pop()  # Keep the incomplete command (if any)
                    
                    for cmd in commands:
                        cmd = cmd.strip()
                        if cmd:
                            self.handle_command(cmd)
            
            # Send auto messages if configured
            current_time = time.time()
            if (current_time - self.last_auto_message >= self.config['auto_message_interval'] and 
                self.config['auto_messages']):
                self.last_auto_message = current_time
                message = random.choice(self.config['auto_messages'])
                self.serial.write(f"{message}\r\n".encode('ascii'))
            
            time.sleep(0.05)
    
    def handle_command(self, command):
        """Handle custom commands based on configuration"""
        # Look for command in the configured commands
        for cmd_pattern, response in self.config['commands'].items():
            if cmd_pattern in command:
                self.serial.write(f"{response}\r\n".encode('ascii'))
                return
        
        # Default response if no command matched
        self.serial.write(b"ERROR: Unknown command\r\n")


class SerialFarm:
    """Main class for managing the Serial Device Simulator Farm"""
    
    def __init__(self):
        self.port_pairs = {}
        self.simulators = {}
        self.running = False
    
    def create_device(self, device_name, device_type, config=None):
        """Create a new virtual device"""
        if device_name in self.port_pairs:
            logger.error(f"Device {device_name} already exists")
            return None, None
        
        # Create a virtual port pair
        port_pair = SerialPortPair(device_name)
        port_a, port_b = port_pair.create()
        
        # Store the port pair
        self.port_pairs[device_name] = port_pair
        
        # Create the appropriate simulator based on device type
        if device_type == DeviceType.MODEM:
            simulator = ModemSimulator(port_a, config)
        elif device_type == DeviceType.GPS:
            simulator = GPSSimulator(port_a, config)
        elif device_type == DeviceType.TELEMETRY:
            simulator = TelemetrySimulator(port_a, config)
        elif device_type == DeviceType.CUSTOM:
            simulator = CustomDeviceSimulator(port_a, config)
        else:
            logger.error(f"Unknown device type: {device_type}")
            port_pair.destroy()
            del self.port_pairs[device_name]
            return None, None
        
        # Store the simulator
        self.simulators[device_name] = simulator
        
        # Return the client port for the application to connect to
        return simulator, port_b
    
    def start_device(self, device_name):
        """Start a specific device simulator"""
        if device_name in self.simulators:
            return self.simulators[device_name].start()
        else:
            logger.error(f"Device {device_name} does not exist")
            return False
    
    def stop_device(self, device_name):
        """Stop a specific device simulator"""
        if device_name in self.simulators:
            self.simulators[device_name].stop()
            return True
        else:
            logger.error(f"Device {device_name} does not exist")
            return False
    
    def destroy_device(self, device_name):
        """Destroy a specific device simulator and its port pair"""
        if device_name in self.simulators:
            # Stop the simulator if it's running
            self.simulators[device_name].stop()
            
            # Destroy the port pair
            self.port_pairs[device_name].destroy()
            
            # Remove from dictionaries
            del self.simulators[device_name]
            del self.port_pairs[device_name]
            
            logger.info(f"Destroyed device {device_name}")
            return True
        else:
            logger.error(f"Device {device_name} does not exist")
            return False
    
    def list_devices(self):
        """List all created devices"""
        devices = []
        for name, simulator in self.simulators.items():
            device_info = {
                'name': name,
                'type': simulator.device_type.value,
                'port': simulator.port,
                'client_port': self.port_pairs[name].port_b,
                'running': simulator.running
            }
            devices.append(device_info)
        return devices
    
    def get_device_info(self, device_name):
        """Get detailed information about a specific device"""
        if device_name in self.simulators:
            simulator = self.simulators[device_name]
            device_info = {
                'name': device_name,
                'type': simulator.device_type.value,
                'port': simulator.port,
                'client_port': self.port_pairs[device_name].port_b,
                'running': simulator.running,
                'config': simulator.config
            }
            return device_info
        else:
            logger.error(f"Device {device_name} does not exist")
            return None
    
    def start_all(self):
        """Start all device simulators"""
        for name in self.simulators:
            self.start_device(name)
        self.running = True
    
    def stop_all(self):
        """Stop all device simulators"""
        for name in self.simulators:
            self.stop_device(name)
        self.running = False
    
    def cleanup(self):
        """Clean up all resources"""
        self.stop_all()
        for name in list(self.simulators.keys()):
            self.destroy_device(name)


# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Serial Device Simulator Farm")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create device command
    create_parser = subparsers.add_parser("create", help="Create a new virtual device")
    create_parser.add_argument("name", help="Name of the device")
    create_parser.add_argument("type", choices=["modem", "gps", "telemetry", "custom"], 
                              help="Type of device to create")
    create_parser.add_argument("--config", type=str, help="Configuration file (JSON)")
    
    # Start device command
    start_parser = subparsers.add_parser("start", help="Start a device simulator")
    start_parser.add_argument("name", help="Name of the device to start")
    
    # Stop device command
    stop_parser = subparsers.add_parser("stop", help="Stop a device simulator")
    stop_parser.add_argument("name", help="Name of the device to stop")
    
    # Destroy device command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy a device simulator")
    destroy_parser.add_argument("name", help="Name of the device to destroy")
    
    # List devices command
    subparsers.add_parser("list", help="List all created devices")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get detailed information about a device")
    info_parser.add_argument("name", help="Name of the device")
    
    # Start all command
    subparsers.add_parser("start-all", help="Start all device simulators")
    
    # Stop all command
    subparsers.add_parser("stop-all", help="Stop all device simulators")
    
    # Interactive mode command
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create the serial farm
    farm = SerialFarm()
    
    try:
        # Handle commands
        if args.command == "create":
            config = None
            if args.config:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            
            simulator, client_port = farm.create_device(args.name, DeviceType(args.type), config)
            if simulator:
                simulator.start()
                print(f"Created and started {args.type} device '{args.name}'")
                print(f"Your application can connect to: {client_port}")
            else:
                print(f"Failed to create device '{args.name}'")
        
        elif args.command == "start":
            if farm.start_device(args.name):
                print(f"Started device '{args.name}'")
            else:
                print(f"Failed to start device '{args.name}'")
        
        elif args.command == "stop":
            if farm.stop_device(args.name):
                print(f"Stopped device '{args.name}'")
            else:
                print(f"Failed to stop device '{args.name}'")
        
        elif args.command == "destroy":
            if farm.destroy_device(args.name):
                print(f"Destroyed device '{args.name}'")
            else:
                print(f"Failed to destroy device '{args.name}'")
        
        elif args.command == "list":
            devices = farm.list_devices()
            if devices:
                print("Created devices:")
                print("-" * 80)
                print(f"{'Name':<15} {'Type':<10} {'Device Port':<20} {'Client Port':<20} {'Status':<10}")
                print("-" * 80)
                for dev in devices:
                    status = "Running" if dev['running'] else "Stopped"
                    print(f"{dev['name']:<15} {dev['type']:<10} {dev['port']:<20} {dev['client_port']:<20} {status:<10}")
            else:
                print("No devices created")
        
        elif args.command == "info":
            info = farm.get_device_info(args.name)
            if info:
                print(f"Device: {info['name']} ({info['type']})")
                print(f"Device Port: {info['port']}")
                print(f"Client Port: {info['client_port']}")
                print(f"Status: {'Running' if info['running'] else 'Stopped'}")
                print("Configuration:")
                for key, value in info['config'].items():
                    print(f"  {key}: {value}")
            else:
                print(f"Device '{args.name}' not found")
        
        elif args.command == "start-all":
            farm.start_all()
            print("Started all devices")
        
        elif args.command == "stop-all":
            farm.stop_all()
            print("Stopped all devices")
        
        elif args.command == "interactive":
            # Simple interactive console
            farm.running = True
            print("Serial Device Simulator Farm - Interactive Mode")
            print("Type 'help' for available commands")
            
            while farm.running:
                try:
                    cmd = input("> ").strip().lower()
                    
                    if cmd == "help":
                        print("Available commands:")
                        print("  create <name> <type> [config_file] - Create a new device")
                        print("  start <name> - Start a device")
                        print("  stop <name> - Stop a device")
                        print("  destroy <name> - Destroy a device")
                        print("  list - List all devices")
                        print("  info <name> - Get detailed information about a device")
                        print("  start-all - Start all devices")
                        print("  stop-all - Stop all devices")
                        print("  exit - Exit interactive mode")
                    
                    elif cmd.startswith("create "):
                        parts = cmd.split()
                        if len(parts) < 3:
                            print("Usage: create <name> <type> [config_file]")
                            continue
                        
                        name = parts[1]
                        dev_type = parts[2]
                        
                        if dev_type not in [t.value for t in DeviceType]:
                            print(f"Invalid device type. Available types: {', '.join(t.value for t in DeviceType)}")
                            continue
                        
                        config = None
                        if len(parts) > 3:
                            config_file = parts[3]
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                            except Exception as e:
                                print(f"Error loading config file: {e}")
                                continue
                        
                        simulator, client_port = farm.create_device(name, DeviceType(dev_type), config)
                        if simulator:
                            simulator.start()
                            print(f"Created and started {dev_type} device '{name}'")
                            print(f"Your application can connect to: {client_port}")
                        else:
                            print(f"Failed to create device '{name}'")
                    
                    elif cmd.startswith("start "):
                        name = cmd.split()[1]
                        if farm.start_device(name):
                            print(f"Started device '{name}'")
                        else:
                            print(f"Failed to start device '{name}'")
                    
                    elif cmd.startswith("stop "):
                        name = cmd.split()[1]
                        if farm.stop_device(name):
                            print(f"Stopped device '{name}'")
                        else:
                            print(f"Failed to stop device '{name}'")
                    
                    elif cmd.startswith("destroy "):
                        name = cmd.split()[1]
                        if farm.destroy_device(name):
                            print(f"Destroyed device '{name}'")
                        else:
                            print(f"Failed to destroy device '{name}'")
                    
                    elif cmd == "list":
                        devices = farm.list_devices()
                        if devices:
                            print("Created devices:")
                            print("-" * 80)
                            print(f"{'Name':<15} {'Type':<10} {'Device Port':<20} {'Client Port':<20} {'Status':<10}")
                            print("-" * 80)
                            for dev in devices:
                                status = "Running" if dev['running'] else "Stopped"
                                print(f"{dev['name']:<15} {dev['type']:<10} {dev['port']:<20} {dev['client_port']:<20} {status:<10}")
                        else:
                            print("No devices created")
                    
                    elif cmd.startswith("info "):
                        name = cmd.split()[1]
                        info = farm.get_device_info(name)
                        if info:
                            print(f"Device: {info['name']} ({info['type']})")
                            print(f"Device Port: {info['port']}")
                            print(f"Client Port: {info['client_port']}")
                            print(f"Status: {'Running' if info['running'] else 'Stopped'}")
                            print("Configuration:")
                            for key, value in info['config'].items():
                                print(f"  {key}: {value}")
                        else:
                            print(f"Device '{name}' not found")
                    
                    elif cmd == "start-all":
                        farm.start_all()
                        print("Started all devices")
                    
                    elif cmd == "stop-all":
                        farm.stop_all()
                        print("Stopped all devices")
                    
                    elif cmd in ["exit", "quit"]:
                        farm.running = False
                        print("Exiting interactive mode")
                    
                    else:
                        print("Unknown command. Type 'help' for available commands")
                
                except KeyboardInterrupt:
                    print("\nExiting interactive mode")
                    farm.running = False
                
                except Exception as e:
                    print(f"Error: {e}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up resources
        farm.cleanup()


# Handle signals
def signal_handler(sig, frame):
    print("\nReceived signal to terminate. Cleaning up...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Import math module needed for GPS simulator
    import math
    
    # Start the program
    main()
