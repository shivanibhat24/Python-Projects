#!/usr/bin/env python3
"""
Serial Port Command Line Interface
A command-line tool for interacting with serial devices, logging traffic,
and analyzing protocols.
"""

import sys
import time
import argparse
import threading
import binascii
import datetime
import re
import json
import os
import signal
import cmd
import readline
from queue import Queue

import serial
import serial.tools.list_ports


class SerialCLI(cmd.Cmd):
    """Interactive CLI for serial port communication"""
    
    intro = "Serial Port Protocol Analyzer CLI. Type help or ? to list commands.\n"
    prompt = "serial> "
    
    def __init__(self, port=None, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.1):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.is_monitoring = False
        self.monitor_thread = None
        self.log_file = None
        self.capture_buffer = []
        self.display_format = "BOTH"  # BOTH, ASCII, or HEX
        self.command_history = []
        
        # Connect if port is provided
        if port:
            self.do_connect(f"{port} {baudrate} {bytesize} {parity} {stopbits}")
    
    def do_list(self, arg):
        """List available serial ports"""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found")
            return
        
        print("Available serial ports:")
        for i, port in enumerate(ports):
            print(f"  {i}: {port.device} - {port.description}")
    
    def do_connect(self, arg):
        """Connect to a serial port: connect PORT [BAUD] [BYTESIZE] [PARITY] [STOPBITS]
        Example: connect /dev/ttyUSB0 9600 8 N 1"""
        if self.is_connected:
            print("Already connected. Use 'disconnect' first.")
            return
        
        args = arg.split()
        if not args:
            print("Port argument required. Use 'list' to see available ports.")
            return
        
        port = args[0]
        baudrate = int(args[1]) if len(args) > 1 else 9600
        bytesize_map = {'5': serial.FIVEBITS, '6': serial.SIXBITS, 
                      '7': serial.SEVENBITS, '8': serial.EIGHTBITS}
        bytesize = bytesize_map.get(args[2] if len(args) > 2 else '8', serial.EIGHTBITS)
        
        parity_map = {'N': serial.PARITY_NONE, 'E': serial.PARITY_EVEN, 
                     'O': serial.PARITY_ODD, 'M': serial.PARITY_MARK, 
                     'S': serial.PARITY_SPACE}
        parity = parity_map.get(args[3].upper() if len(args) > 3 else 'N', serial.PARITY_NONE)
        
        stopbits_map = {'1': serial.STOPBITS_ONE, '1.5': serial.STOPBITS_ONE_POINT_FIVE, 
                       '2': serial.STOPBITS_TWO}
        stopbits = stopbits_map.get(args[4] if len(args) > 4 else '1', serial.STOPBITS_ONE)
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=0.1
            )
            self.is_connected = True
            print(f"Connected to {port} at {baudrate} baud")
            
            # Update prompt to show connected port
            self.prompt = f"serial({port})> "
        except (serial.SerialException, ValueError) as e:
            print(f"Failed to connect: {str(e)}")
    
    def do_disconnect(self, arg):
        """Disconnect from the serial port"""
        if not self.is_connected:
            print("Not connected")
            return
        
        self.do_stop(arg)  # Stop monitoring if active
        
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        
        self.is_connected = False
        print("Disconnected")
        self.prompt = "serial> "
    
    def do_monitor(self, arg):
        """Start monitoring the serial port: monitor [log_file]
        If log_file is provided, will log all traffic to that file"""
        if not self.is_connected:
            print("Not connected to a port. Use 'connect' first.")
            return
        
        if self.is_monitoring:
            print("Already monitoring. Use 'stop' to stop monitoring.")
            return
        
        # Check for log file argument
        if arg:
            try:
                self.log_file = open(arg, 'a')
                print(f"Logging to {arg}")
            except Exception as e:
                print(f"Error opening log file: {str(e)}")
                return
        
        self.is_monitoring = True
        print("Monitoring started. Press Ctrl+C to stop.")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_task, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_task(self):
        """Background task to monitor the serial port"""
        try:
            while self.is_monitoring and self.is_connected:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self._display_data("RX", timestamp, data)
                        
                        # Add to capture buffer
                        self.capture_buffer.append({
                            "type": "RX",
                            "timestamp": timestamp,
                            "data": binascii.hexlify(data).decode('ascii'),
                            "ascii": ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
                        })
                        
                        # Log to file if enabled
                        if self.log_file:
                            self._write_to_log("RX", timestamp, data)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
        except Exception as e:
            print(f"\nMonitoring error: {str(e)}")
            self.is_monitoring = False
    
    def do_stop(self, arg):
        """Stop monitoring the serial port"""
        if not self.is_monitoring:
            print("Not monitoring")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(1.0)
        
        print("Monitoring stopped")
        
        # Close log file if open
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def do_send(self, arg):
        """Send data to the serial port: send [-h] DATA
        By default, DATA is sent as ASCII. Use -h flag to send hex data.
        Examples:
            send Hello World    # Sends ASCII "Hello World"
            send -h 48656C6C6F  # Sends hex bytes "Hello"
        """
        if not self.is_connected:
            print("Not connected to a port. Use 'connect' first.")
            return
        
        if not arg:
            print("No data provided")
            return
        
        try:
            # Check if we're sending hex data
            if arg.startswith("-h "):
                hex_data = arg[3:].strip().replace(" ", "")
                if not re.match(r'^[0-9a-fA-F]+$', hex_data):
                    print("Invalid hex format")
                    return
                
                data = binascii.unhexlify(hex_data)
                data_type = "HEX"
            else:
                data = arg.encode('utf-8')
                data_type = "ASCII"
            
            # Send the data
            self.serial_port.write(data)
            
            # Add to command history
            self.command_history.append(arg)
            
            # Display and log the sent data
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._display_data("TX", timestamp, data)
            
            # Add to capture buffer
            self.capture_buffer.append({
                "type": "TX",
                "timestamp": timestamp,
                "data": binascii.hexlify(data).decode('ascii'),
                "ascii": ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
            })
            
            # Log to file if enabled
            if self.log_file:
                self._write_to_log("TX", timestamp, data)
                
        except Exception as e:
            print(f"Send error: {str(e)}")
    
    def do_sendline(self, arg):
        """Send data followed by newline: sendline [-h] DATA
        Same as 'send' but adds a newline (CR+LF) at the end."""
        if not arg:
            self.do_send("\r\n")
        else:
            self.do_send(f"{arg}\r\n")
    
    def do_display(self, arg):
        """Set display format: display [ASCII|HEX|BOTH]"""
        formats = ["ASCII", "HEX", "BOTH"]
        arg = arg.upper()
        
        if not arg or arg not in formats:
            print(f"Current display format: {self.display_format}")
            print(f"Available formats: {', '.join(formats)}")
            return
        
        self.display_format = arg
        print(f"Display format set to {self.display_format}")
    
    def do_clear(self, arg):
        """Clear the capture buffer"""
        self.capture_buffer = []
        print("Capture buffer cleared")
    
    def do_export(self, arg):
        """Export captured data to a file: export FILENAME"""
        if not self.capture_buffer:
            print("No data to export")
            return
        
        if not arg:
            print("Filename required")
            return
        
        try:
            with open(arg, "w") as f:
                json.dump(self.capture_buffer, f, indent=2)
            print(f"Data exported to {arg}")
        except Exception as e:
            print(f"Export error: {str(e)}")
    
    def do_fuzz(self, arg):
        """Fuzz a command by varying a byte: fuzz [-h] BASE_CMD POS MIN MAX DELAY
        
        Arguments:
          BASE_CMD - Base command to fuzz (ASCII or hex with -h flag)
          POS      - Position of the byte to modify (0-based)
          MIN      - Minimum value for the byte (0-255)
          MAX      - Maximum value for the byte (0-255)
          DELAY    - Delay between commands in milliseconds
          
        Example:
          fuzz "AT+CMD?" 3 0 255 100
          fuzz -h "415443" 2 0 255 100
        """
        if not self.is_connected:
            print("Not connected to a port. Use 'connect' first.")
            return
        
        args = arg.split()
        if len(args) < 5:
            print("Insufficient arguments. See 'help fuzz' for usage.")
            return
        
        try:
            # Parse arguments
            if args[0] == "-h":
                hex_cmd = args[1].replace(" ", "")
                if not re.match(r'^[0-9a-fA-F]+$', hex_cmd):
                    print("Invalid hex format")
                    return
                
                cmd_bytes = bytearray(binascii.unhexlify(hex_cmd))
                pos = int(args[2])
                min_val = int(args[3])
                max_val = int(args[4])
                delay_ms = int(args[5]) if len(args) > 5 else 100
            else:
                cmd_bytes = bytearray(args[0].encode('utf-8'))
                pos = int(args[1])
                min_val = int(args[2])
                max_val = int(args[3])
                delay_ms = int(args[4]) if len(args) > 4 else 100
            
            # Validate parameters
            if pos >= len(cmd_bytes):
                print(f"Position {pos} exceeds command length {len(cmd_bytes)}")
                return
            
            if min_val < 0 or max_val > 255 or min_val > max_val:
                print("Invalid range. MIN and MAX must be between 0-255")
                return
            
            # Start fuzzing
            print(f"Fuzzing byte at position {pos} from {min_val} to {max_val}")
            print("Press Ctrl+C to stop")
            
            try:
                for val in range(min_val, max_val + 1):
                    # Create modified command
                    cmd_copy = bytearray(cmd_bytes)
                    cmd_copy[pos] = val
                    
                    # Send command
                    self.serial_port.write(cmd_copy)
                    
                    # Log command
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    hex_str = binascii.hexlify(cmd_copy).decode('ascii')
                    ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in cmd_copy)
                    
                    print(f"[{timestamp}] Value: {val:03d} (0x{val:02X}) Sent: HEX={hex_str} ASCII={ascii_str}")
                    
                    # Add to capture buffer
                    self.capture_buffer.append({
                        "type": "TX",
                        "timestamp": timestamp,
                        "data": hex_str,
                        "ascii": ascii_str,
                        "fuzz": {"position": pos, "value": val}
                    })
                    
                    # Log to file if enabled
                    if self.log_file:
                        self._write_to_log("TX", timestamp, cmd_copy, fuzz_info=f"Fuzz pos={pos} val={val}")
                    
                    # Wait for response
                    time.sleep(delay_ms / 1000.0)
                
                print("Fuzzing completed")
                
            except KeyboardInterrupt:
                print("\nFuzzing stopped by user")
            
        except Exception as e:
            print(f"Fuzzing error: {str(e)}")
    
    def do_exit(self, arg):
        """Exit the program"""
        if self.is_connected:
            self.do_disconnect("")
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the program"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on EOF (Ctrl+D)"""
        print()  # Print a newline before exiting
        return self.do_exit(arg)
    
    def _display_data(self, direction, timestamp, data):
        """Display received or sent data"""
        if not data:
            return
        
        hex_str = binascii.hexlify(data).decode('ascii')
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
        
        if self.display_format == "ASCII":
            print(f"[{timestamp}] {direction}: {ascii_str}")
        elif self.display_format == "HEX":
            # Format hex with spaces for readability
            formatted_hex = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
            print(f"[{timestamp}] {direction}: {formatted_hex}")
        else:  # BOTH
            # Format hex with spaces for readability
            formatted_hex = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
            print(f"[{timestamp}] {direction}: {ascii_str} ({formatted_hex})")
    
    def _write_to_log(self, direction, timestamp, data, fuzz_info=None):
        """Write data to log file"""
        if not self.log_file:
            return
        
        try:
            hex_data = binascii.hexlify(data).decode('ascii')
            ascii_data = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
            
            if fuzz_info:
                log_line = f"[{timestamp}] {direction}: HEX={hex_data} ASCII={ascii_data} {fuzz_info}\n"
            else:
                log_line = f"[{timestamp}] {direction}: HEX={hex_data} ASCII={ascii_data}\n"
                
            self.log_file.write(log_line)
            self.log_file.flush()
        except Exception as e:
            print(f"Error writing to log: {str(e)}")
    
    def do_history(self, arg):
        """Show command history"""
        if not self.command_history:
            print("No command history")
            return
        
        print("Command history:")
        for i, cmd in enumerate(self.command_history):
            print(f"  {i}: {cmd}")
    
    def do_repeat(self, arg):
        """Repeat a command from history: repeat INDEX"""
        if not self.command_history:
            print("No command history")
            return
        
        try:
            index = int(arg)
            if 0 <= index < len(self.command_history):
                cmd = self.command_history[index]
                print(f"Repeating: {cmd}")
                self.do_send(cmd)
            else:
                print(f"Invalid index. Use 'history' to see valid indices")
        except ValueError:
            print("Invalid index. Please provide a number")
    
    def do_status(self, arg):
        """Show current connection status"""
        if self.is_connected:
            port_name = self.serial_port.name
            baudrate = self.serial_port.baudrate
            bytesize = self.serial_port.bytesize
            parity = self.serial_port.parity
            stopbits = self.serial_port.stopbits
            
            parity_names = {
                serial.PARITY_NONE: "None",
                serial.PARITY_EVEN: "Even",
                serial.PARITY_ODD: "Odd",
                serial.PARITY_MARK: "Mark",
                serial.PARITY_SPACE: "Space"
            }
            
            stopbits_names = {
                serial.STOPBITS_ONE: "1",
                serial.STOPBITS_ONE_POINT_FIVE: "1.5",
                serial.STOPBITS_TWO: "2"
            }
            
            print(f"Connected to: {port_name}")
            print(f"Baud rate: {baudrate}")
            print(f"Data bits: {bytesize}")
            print(f"Parity: {parity_names.get(parity, 'Unknown')}")
            print(f"Stop bits: {stopbits_names.get(stopbits, 'Unknown')}")
            print(f"Monitoring: {'Active' if self.is_monitoring else 'Inactive'}")
            print(f"Display format: {self.display_format}")
            print(f"Logging: {'Enabled' if self.log_file else 'Disabled'}")
            print(f"Capture buffer: {len(self.capture_buffer)} entries")
        else:
            print("Not connected")
    
    def do_analyze(self, arg):
        """Analyze captured data for patterns"""
        if not self.capture_buffer:
            print("No data to analyze")
            return
        
        # Count message types
        tx_count = sum(1 for item in self.capture_buffer if item["type"] == "TX")
        rx_count = sum(1 for item in self.capture_buffer if item["type"] == "RX")
        
        print(f"Analysis of {len(self.capture_buffer)} captured messages:")
        print(f"  TX messages: {tx_count}")
        print(f"  RX messages: {rx_count}")
        
        # Find common patterns in RX data
        if rx_count > 0:
            rx_data = [item["data"] for item in self.capture_buffer if item["type"] == "RX"]
            common_prefixes = self._find_common_prefixes(rx_data)
            
            if common_prefixes:
                print("\nCommon prefixes in received data:")
                for prefix, count in common_prefixes:
                    prefix_hex = ' '.join(prefix[i:i+2] for i in range(0, len(prefix), 2))
                    try:
                        prefix_ascii = binascii.unhexlify(prefix).decode('ascii', errors='replace')
                    except:
                        prefix_ascii = "N/A"
                    
                    print(f"  {prefix_hex} ({prefix_ascii}) - {count} occurrences")
    
    def _find_common_prefixes(self, hex_strings, min_length=4, min_occurrences=2):
        """Find common prefixes in a list of hex strings"""
        if not hex_strings:
            return []
        
        # Count occurrences of each prefix
        prefix_counts = {}
        for hex_str in hex_strings:
            # Consider prefixes of different lengths
            for length in range(min_length, min(len(hex_str), 20) + 1, 2):  # Increment by 2 for hex byte pairs
                prefix = hex_str[:length]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        # Filter by minimum occurrences
        common_prefixes = [(prefix, count) for prefix, count in prefix_counts.items() 
                          if count >= min_occurrences]
        
        # Sort by occurrence count (descending)
        common_prefixes.sort(key=lambda x: x[1], reverse=True)
        
        return common_prefixes[:5]  # Return top 5 common prefixes


def main():
    """Main function to parse arguments and start the CLI"""
    parser = argparse.ArgumentParser(description="Serial Port Protocol Analyzer CLI")
    parser.add_argument('-p', '--port', help='Serial port to connect to')
    parser.add_argument('-b', '--baud', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('-d', '--databits', type=int, choices=[5, 6, 7, 8], default=8, 
                        help='Data bits (default: 8)')
    parser.add_argument('-a', '--parity', choices=['N', 'E', 'O', 'M', 'S'], default='N', 
                        help='Parity (N=None, E=Even, O=Odd, M=Mark, S=Space) (default: N)')
    parser.add_argument('-s', '--stopbits', choices=['1', '1.5', '2'], default='1', 
                        help='Stop bits (default: 1)')
    parser.add_argument('-m', '--monitor', action='store_true', 
                        help='Start monitoring immediately after connecting')
    parser.add_argument('-l', '--log', help='Log file to record all traffic')
    parser.add_argument('-f', '--format', choices=['ASCII', 'HEX', 'BOTH'], default='BOTH', 
                        help='Display format (default: BOTH)')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = SerialCLI(port=args.port, 
                   baudrate=args.baud, 
                   bytesize=args.databits,
                   parity=args.parity,
                   stopbits=args.stopbits)
    
    # Set display format
    cli.display_format = args.format
    
    # Start monitoring if requested
    if args.monitor and cli.is_connected:
        cli.do_monitor(args.log if args.log else "")
    
    # Set up signal handlers for clean exit
    def signal_handler(sig, frame):
        print("\nInterrupt received, exiting...")
        if cli.is_connected:
            cli.do_disconnect("")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the CLI
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        if cli.is_connected:
            cli.do_disconnect("")
    except Exception as e:
        print(f"Error: {str(e)}")
        if cli.is_connected:
            cli.do_disconnect("")


if __name__ == "__main__":
    main()
