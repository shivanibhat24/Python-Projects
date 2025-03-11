"""
Comprehensive Deception-Based Network Security System
Featuring honeypots, tarpits, and monitoring capabilities
"""

import logging
import socket
import threading
import time
import json
import os
import random
import datetime
import ipaddress
from scapy.all import sniff, IP, TCP, UDP
import psutil
import sqlite3
from flask import Flask, render_template, jsonify
import paramiko
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deception_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeceptionSystem")

# Database setup
def setup_database():
    conn = sqlite3.connect('deception_events.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source_ip TEXT,
        destination_ip TEXT,
        service TEXT,
        activity TEXT,
        payload BLOB,
        classification TEXT
    )
    ''')
    conn.commit()
    return conn

# Configuration management
class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "network": {
                    "honeypot_subnet": "192.168.100.0/24",
                    "production_subnets": ["192.168.1.0/24", "10.0.0.0/8"]
                },
                "honeypots": {
                    "ssh": {
                        "enabled": True,
                        "ports": [22, 2222],
                        "interaction_level": "medium"
                    },
                    "web": {
                        "enabled": True,
                        "ports": [80, 443, 8080],
                        "interaction_level": "high"
                    },
                    "ftp": {
                        "enabled": True,
                        "ports": [21],
                        "interaction_level": "low"
                    },
                    "smb": {
                        "enabled": True,
                        "ports": [445],
                        "interaction_level": "medium"
                    }
                },
                "tarpits": {
                    "tcp": {
                        "enabled": True,
                        "ports": [25, 110, 143]
                    },
                    "http": {
                        "enabled": True,
                        "ports": [8081]
                    }
                },
                "monitoring": {
                    "packet_capture": True,
                    "alert_threshold": 5,
                    "admin_email": "admin@example.com"
                }
            }
            self.save_config()
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_config(self):
        return self.config

# Event management
class EventManager:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.cursor = self.conn.cursor()
    
    def log_event(self, source_ip, destination_ip, service, activity, payload=None, classification="unknown"):
        timestamp = datetime.datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO events (timestamp, source_ip, destination_ip, service, activity, payload, classification) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, source_ip, destination_ip, service, activity, payload, classification)
        )
        self.conn.commit()
        logger.info(f"Event logged: {source_ip} -> {destination_ip} {service} {activity} [{classification}]")
    
    def get_recent_events(self, limit=100):
        self.cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        return self.cursor.fetchall()
    
    def get_events_by_ip(self, ip_address):
        self.cursor.execute("SELECT * FROM events WHERE source_ip = ? ORDER BY timestamp DESC", (ip_address,))
        return self.cursor.fetchall()

# Base Honeypot class
class BaseHoneypot:
    def __init__(self, service_name, ports, interaction_level="low"):
        self.service_name = service_name
        self.ports = ports
        self.interaction_level = interaction_level
        self.running = False
        self.threads = []
        logger.info(f"Initialized {interaction_level}-interaction {service_name} honeypot on ports {ports}")
    
    def start(self):
        self.running = True
        for port in self.ports:
            thread = threading.Thread(target=self.run_server, args=(port,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            logger.info(f"Started {self.service_name} honeypot on port {port}")
    
    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1)
        logger.info(f"Stopped {self.service_name} honeypot")
    
    def run_server(self, port):
        # To be implemented by specific honeypot types
        pass

    def get_banner(self):
        banners = {
            "ssh": "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.4",
            "ftp": "220 ProFTPD 1.3.5e Server",
            "http": "Apache/2.4.41 (Ubuntu) Server",
            "smtp": "220 mail.example.com ESMTP Postfix",
            "telnet": "\r\nUbuntu 20.04.4 LTS\r\nlogin: ",
            "smb": "Windows Server 2019 10.0",
            "mysql": "5.7.33-0ubuntu0.16.04.1"
        }
        return banners.get(self.service_name.lower(), f"{self.service_name} Service")

# SSH Honeypot
class SSHHoneypot(BaseHoneypot):
    def __init__(self, ports, interaction_level="medium"):
        super().__init__("SSH", ports, interaction_level)
        self.banner = self.get_banner()
        self.fake_shell_commands = {
            "ls": "file1.txt  file2.txt  config.conf  backup  logs",
            "pwd": "/home/user",
            "whoami": "user",
            "id": "uid=1000(user) gid=1000(user) groups=1000(user),4(adm),24(cdrom),27(sudo)",
            "cat": "Permission denied",
            "ps": " PID TTY          TIME CMD\n 1234 pts/0    00:00:00 bash\n 2345 pts/0    00:00:00 ps",
            "netstat": "Active Internet connections (w/o servers)\nProto Recv-Q Send-Q Local Address           Foreign Address         State      \ntcp        0      0 192.168.1.50:22        10.0.0.1:55425         ESTABLISHED",
            "uname": "Linux honeypot 5.4.0-97-generic #110-Ubuntu SMP Wed Feb 2 12:15:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux"
        }
        
    def run_server(self, port):
        if self.interaction_level == "low":
            self._run_low_interaction(port)
        else:
            self._run_medium_interaction(port)
    
    def _run_low_interaction(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)
        
        while self.running:
            try:
                client, addr = sock.accept()
                client.send(f"{self.banner}\r\n".encode())
                client_thread = threading.Thread(target=self._handle_client_low, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                logger.error(f"Error in SSH honeypot: {e}")
        
        sock.close()
    
    def _handle_client_low(self, client, addr):
        source_ip = addr[0]
        event_manager.log_event(source_ip, "0.0.0.0", "SSH", "connection", None, "reconnaissance")
        try:
            # Ask for username
            client.send(b"login as: ")
            username = client.recv(1024).strip().decode('utf-8', errors='ignore')
            
            # Ask for password
            client.send(b"Password: ")
            password = client.recv(1024).strip().decode('utf-8', errors='ignore')
            
            event_manager.log_event(source_ip, "0.0.0.0", "SSH", f"login_attempt", f"user:{username} pass:{password}", "credential_harvesting")
            
            # Always reject
            client.send(b"Access denied\r\n")
            time.sleep(1)
            client.close()
        except Exception as e:
            logger.error(f"Error handling SSH client: {e}")
            client.close()
    
    def _run_medium_interaction(self, port):
        # For medium interaction, we'll use paramiko to create a more realistic SSH server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        
        # Generate an RSA key for the SSH server
        host_key = paramiko.RSAKey.generate(2048)
        
        class SSHServerHandler(paramiko.ServerInterface):
            def __init__(self, addr, honeypot):
                self.addr = addr
                self.honeypot = honeypot
                self.event = threading.Event()
            
            def check_channel_request(self, kind, chanid):
                if kind == 'session':
                    return paramiko.OPEN_SUCCEEDED
                return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
            
            def check_auth_password(self, username, password):
                event_manager.log_event(self.addr[0], "0.0.0.0", "SSH", f"login_attempt", 
                                       f"user:{username} pass:{password}", "credential_harvesting")
                # Allow authentication after several attempts
                if random.random() < 0.1:  # 10% chance of success
                    return paramiko.AUTH_SUCCESSFUL
                return paramiko.AUTH_FAILED
            
            def get_allowed_auths(self, username):
                return 'password'
            
            def check_channel_shell_request(self, channel):
                self.event.set()
                return True
            
            def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
                return True

        while self.running:
            try:
                client, addr = server_socket.accept()
                logger.info(f"SSH connection from {addr[0]}:{addr[1]}")
                
                transport = paramiko.Transport(client)
                transport.local_version = self.banner
                transport.add_server_key(host_key)
                
                handler = SSHServerHandler(addr, self)
                transport.start_server(server=handler)
                
                # Wait for auth
                channel = transport.accept(20)
                if channel is None:
                    transport.close()
                    continue
                
                handler.event.wait(10)
                
                # Start the shell simulation
                client_thread = threading.Thread(target=self._handle_shell, args=(channel, addr))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                logger.error(f"Error in medium-interaction SSH: {e}")
        
        server_socket.close()
    
    def _handle_shell(self, channel, addr):
        try:
            channel.send(f"Welcome to Ubuntu 20.04.4 LTS (GNU/Linux 5.4.0-97-generic x86_64)\r\n")
            channel.send(f"\r\n$ ")
            
            buffer = ""
            while True:
                char = channel.recv(1).decode('utf-8', errors='ignore')
                if not char:
                    break
                
                if char == '\r':
                    channel.send('\r\n')
                    command = buffer.strip()
                    
                    if command:
                        event_manager.log_event(addr[0], "0.0.0.0", "SSH", f"command", command, "exploitation")
                        
                        if command in self.fake_shell_commands:
                            response = self.fake_shell_commands[command]
                        elif command.startswith("cat "):
                            filename = command[4:].strip()
                            response = f"cat: {filename}: No such file or directory"
                        elif command.startswith("cd "):
                            response = ""
                        elif command == "exit":
                            channel.send("logout\r\n")
                            break
                        else:
                            response = f"{command}: command not found"
                        
                        channel.send(f"{response}\r\n$ ")
                    else:
                        channel.send("$ ")
                    
                    buffer = ""
                elif char == '\x03':  # Ctrl+C
                    channel.send('^C\r\n$ ')
                    buffer = ""
                elif char in ('\x7f', '\x08'):  # Backspace
                    if buffer:
                        buffer = buffer[:-1]
                        channel.send('\b \b')  # Erase character
                else:
                    buffer += char
                    channel.send(char)
                    
        except Exception as e:
            logger.error(f"Error in SSH shell: {e}")
        finally:
            channel.close()

# HTTP Honeypot
class HTTPHoneypot(BaseHoneypot):
    def __init__(self, ports, interaction_level="medium"):
        super().__init__("HTTP", ports, interaction_level)
        self.fake_urls = {
            "/": self._generate_index_page,
            "/login": self._generate_login_page,
            "/admin": self._generate_admin_page,
            "/api": self._generate_api_page,
            "/robots.txt": self._generate_robots_txt
        }
    
    def run_server(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # For HTTPS ports, use SSL
        use_ssl = port in [443, 8443]
        
        if use_ssl:
            # Generate self-signed certificate if it doesn't exist
            if not os.path.exists("server.key") or not os.path.exists("server.crt"):
                self._generate_self_signed_cert()
            
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain("server.crt", "server.key")
            sock = context.wrap_socket(sock, server_side=True)
        
        try:
            sock.bind(('0.0.0.0', port))
            sock.listen(5)
            
            while self.running:
                client, addr = sock.accept()
                client_thread = threading.Thread(target=self._handle_http_client, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
                
        except Exception as e:
            logger.error(f"Error in HTTP honeypot on port {port}: {e}")
        finally:
            sock.close()
    
    def _generate_self_signed_cert(self):
        # This is a simplified version, in a real system use cryptography library
        logger.info("Generating self-signed certificate for HTTPS")
        os.system("openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 "
                  "-subj '/CN=www.example.com' -keyout server.key -out server.crt")
    
    def _handle_http_client(self, client, addr):
        try:
            # Set a timeout to prevent hanging connections
            client.settimeout(15)
            data = client.recv(4096)
            if not data:
                client.close()
                return
            
            # Parse the HTTP request
            request = data.decode('utf-8', errors='ignore')
            request_lines = request.split('\r\n')
            if not request_lines:
                client.close()
                return
            
            # Extract method, path and headers
            first_line = request_lines[0].split(' ')
            if len(first_line) < 3:
                client.close()
                return
            
            method, path, _ = first_line
            
            # Extract headers
            headers = {}
            for line in request_lines[1:]:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    headers[key.lower()] = value
            
            # Extract body for POST requests
            body = ""
            if method == "POST" and "content-length" in headers:
                content_length = int(headers["content-length"])
                body_start = request.find("\r\n\r\n") + 4
                body = request[body_start:body_start + content_length]
            
            # Log the request
            event_manager.log_event(
                addr[0], "0.0.0.0", "HTTP", 
                f"{method} {path}", 
                json.dumps({"headers": headers, "body": body}),
                "reconnaissance"
            )
            
            # Generate response
            if path in self.fake_urls:
                content = self.fake_urls[path]()
            elif path.endswith('.php'):
                content = self._handle_php_request(path, method, body)
            elif '/wp-admin' in path or '/wp-login' in path:
                content = self._handle_wordpress_request(path)
            else:
                content = "<html><body><h1>404 Not Found</h1><p>The requested URL was not found on this server.</p></body></html>"
                self._send_response(client, "404 Not Found", content)
                client.close()
                return
            
            # Send response with deliberate delay for tarpit effect
            if random.random() < 0.3:  # 30% chance of delay
                time.sleep(random.uniform(1, 5))
            
            self._send_response(client, "200 OK", content)
        
        except Exception as e:
            logger.error(f"Error handling HTTP client: {e}")
        finally:
            client.close()
    
    def _send_response(self, client, status, content):
        response = f"HTTP/1.1 {status}\r\n"
        response += f"Server: {self.get_banner()}\r\n"
        response += "Content-Type: text/html\r\n"
        response += f"Content-Length: {len(content)}\r\n"
        response += "Connection: close\r\n\r\n"
        response += content
        
        client.send(response.encode())
    
    def _generate_index_page(self):
        return """
        <html>
        <head><title>Welcome to Example Corp</title></head>
        <body>
            <h1>Example Corporation</h1>
            <p>Welcome to our corporate website.</p>
            <ul>
                <li><a href="/login">Login</a></li>
                <li><a href="/admin">Admin Panel</a></li>
                <li><a href="/api">API Documentation</a></li>
            </ul>
        </body>
        </html>
        """
    
    def _generate_login_page(self):
        return """
        <html>
        <head><title>Login - Example Corp</title></head>
        <body>
            <h1>Login</h1>
            <form method="post" action="/login">
                <div>
                    <label>Username:</label>
                    <input type="text" name="username">
                </div>
                <div>
                    <label>Password:</label>
                    <input type="password" name="password">
                </div>
                <div>
                    <button type="submit">Login</button>
                </div>
            </form>
        </body>
        </html>
        """
    
    def _generate_admin_page(self):
        return """
        <html>
        <head><title>Admin Panel - Example Corp</title></head>
        <body>
            <h1>Admin Panel</h1>
            <p>Access Restricted. Please login first.</p>
            <a href="/login">Login</a>
        </body>
        </html>
        """
    
    def _generate_api_page(self):
        return """
        <html>
        <head><title>API Documentation - Example Corp</title></head>
        <body>
            <h1>API Documentation</h1>
            <h2>Available Endpoints</h2>
            <ul>
                <li>/api/v1/users</li>
                <li>/api/v1/products</li>
                <li>/api/v1/orders</li>
            </ul>
            <p>For authentication, use Basic Auth or API key in the header:</p>
            <pre>Authorization: Bearer &lt;api_key&gt;</pre>
        </body>
        </html>
        """
    
    def _generate_robots_txt(self):
        return """
        User-agent: *
        Disallow: /admin/
        Disallow: /backup/
        Disallow: /config/
        Disallow: /db/
        """
    
    def _handle_php_request(self, path, method, body):
        # Simulate common PHP vulnerabilities for research purposes
        if 'shell' in path or 'cmd' in path or 'exec' in path:
            event_manager.log_event(
                "unknown", "0.0.0.0", "HTTP", 
                f"Potential RCE attempt: {path}", 
                body,
                "exploitation"
            )
            return "<html><body><h1>Internal Server Error</h1></body></html>"
        
        elif 'login' in path or 'admin' in path:
            return """
            <html>
            <head><title>PHP Login</title></head>
            <body>
                <h1>Login</h1>
                <form method="post" action="/login.php">
                    <div>
                        <label>Username:</label>
                        <input type="text" name="username">
                    </div>
                    <div>
                        <label>Password:</label>
                        <input type="password" name="password">
                    </div>
                    <div>
                        <button type="submit">Login</button>
                    </div>
                </form>
            </body>
            </html>
            """
        
        return "<html><body><h1>PHP Application</h1></body></html>"
    
    def _handle_wordpress_request(self, path):
        return """
        <html>
        <head><title>WordPress Login</title></head>
        <body>
            <h1>WordPress Login</h1>
            <form method="post" action="/wp-login.php">
                <div>
                    <label>Username:</label>
                    <input type="text" name="log">
                </div>
                <div>
                    <label>Password:</label>
                    <input type="password" name="pwd">
                </div>
                <div>
                    <button type="submit">Login</button>
                </div>
            </form>
        </body>
        </html>
        """

# FTP Honeypot
class FTPHoneypot(BaseHoneypot):
    def __init__(self, ports, interaction_level="low"):
        super().__init__("FTP", ports, interaction_level)
        self.commands = {
            "USER": self._handle_user,
            "PASS": self._handle_pass,
            "SYST": self._handle_syst,
            "PWD": self._handle_pwd,
            "TYPE": self._handle_type,
            "CWD": self._handle_cwd,
            "PASV": self._handle_pasv,
            "LIST": self._handle_list,
            "QUIT": self._handle_quit
        }
    
    def run_server(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)
        
        while self.running:
            try:
                client, addr = sock.accept()
                client_thread = threading.Thread(target=self._handle_ftp_client, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                logger.error(f"Error in FTP honeypot: {e}")
        
        sock.close()
    
    def _handle_ftp_client(self, client, addr):
        try:
            client.send(f"220 {self.get_banner()}\r\n".encode())
            
            username = None
            authenticated = False
            
            while self.running:
                try:
                    data = client.recv(1024).decode('utf-8', errors='ignore').strip()
                    if not data:
                        break
                    
                    event_manager.log_event(
                        addr[0], "0.0.0.0", "FTP", 
                        f"command: {data}", 
                        None,
                        "reconnaissance"
                    )
                    
                    command_parts = data.split(' ', 1)
                    command = command_parts[0].upper()
                    arg = command_parts[1] if len(command_parts) > 1 else ""
                    
                    if command == "USER":
                        username = arg
                        response = self._handle_user(arg)
                    elif command == "PASS":
                        if username:
                            event_manager.log_event(
                                addr[0], "0.0.0.0", "FTP", 
                                f"login_attempt", 
                                f"user:{username} pass:{arg}",
                                "credential_harvesting"
                            )
                            
                            # 20% chance of successful login
                            if random.random() < 0.2:
                                authenticated = True
                                response = "230 User logged in, proceed."
                            else:
                                response = "530 Login incorrect."
                        else:
                            response = "503 Login with USER first."
                    else:
                        if command in self.commands:
                            if command in ["QUIT", "SYST"] or authenticated:
                                response = self.commands[command](arg)
                            else:
                                response = "530 Not logged in."
                        else:
                            response = f"502 Command '{command}' not implemented."
                    
                    client.send(f"{response}\r\n".encode())
                    
                    if command == "QUIT":
                        break
                
                except Exception as e:
                    logger.error(f"Error processing FTP command: {e}")
                    break
                
        except Exception as e:
            logger.error(f"Error handling FTP client: {e}")
        finally:
            client.close()
    
    def _handle_user(self, username):
        return "331 Please specify the password."
    
    def _handle_pass(self, password):
        # This is handled in the main client handler
        pass
    
    def _handle_syst(self, arg):
        return "215 UNIX Type: L8"
    
    def _handle_pwd(self, arg):
        return "257 \"/\" is the current directory"
    
    def _handle_type(self, arg):
        return "200 Type set to " + arg
    
    def _handle_cwd(self, arg):
        return "250 Directory successfully changed."
    
    def _handle_pasv(self, arg):
        # Return a fake passive mode address
        ip = "127,0,0,1"
        port = random.randint(10000, 60000)
        p1, p2 = divmod(port, 256)
        return f"227 Entering Passive Mode ({ip},{p1},{p2})."
    
    def _handle_list(self, arg):
        listing = "-rw-r--r--   1 user     group        8192 Jan 01  2023 file1.txt\r\n"
        listing += "-rw-r--r--   1 user     group       16384 Feb 15  2023 file2.txt\r\n"
        listing += "drwxr-xr-x   2 user     group        4096 Mar 10  2023 backup\r\n"
        
        return "150 Here comes the directory listing.\r\n226 Directory send OK."
    
    def _handle_quit(self, arg):
        return "221 Goodbye."

# SMB Honeypot (Simplified)
class SMBHoneypot(BaseHoneypot):
    def __init__(self, ports, interaction_level="medium"):
        super().__init__("SMB", ports, interaction_level)
    
    def run_server(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)
        
        while self.running:
            try:
                client, addr = sock.accept()
                client_thread = threading.Thread(target=self._handle_smb_client, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                logger.error(f"Error in SMB honeypot: {e}")
        
        sock.close()
    
    def _handle_smb_client(self, client, addr):
        try:
            # Log connection
            event_manager.log_event(
                addr[0], "0.0.0.0", "SMB", 
                "connection", 
                None,
                "reconnaissance"
            )
            
            # Receive initial negotiation
            data = client.recv(1024)
            if not data:
                client.close()
                return
            
            # Log SMB packet
            event_manager.log_event(
                addr[0], "0.0.0.0", "SMB",
                "packet", 
                data.hex(),
                "reconnaissance"
            )
            
            # Send a fake SMB response
            # This is a simplified response - in a real implementation, you would parse the SMB packet
            # and respond appropriately based on the SMB protocol
            time.sleep(1)  # Simulate processing time
            
            # Send a generic error response
            error_response = bytes.fromhex("000000a5ff534d4272000000001843c80000000000000000000000000000ffff") + \
                            bytes.fromhex("ffffffff000000000000000000000000000000000000") + \
                            bytes.fromhex("00000000000000000000000000000000")
            
            client.send(error_response)
            time.sleep(0.5)
            client.close()
            
        except Exception as e:
            logger.error(f"Error handling SMB client: {e}")
            client.close()

# TCP Tarpit
class TCPTarpit:
    def __init__(self, ports):
        self.ports = ports
        self.running = False
        self.threads = []
        logger.info(f"Initialized TCP tarpit on ports {ports}")
    
    def start(self):
        self.running = True
        for port in self.ports:
            thread = threading.Thread(target=self.run_tarpit, args=(port,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            logger.info(f"Started TCP tarpit on port {port}")
    
    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1)
        logger.info(f"Stopped TCP tarpit")
    
    def run_tarpit(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)
        
        while self.running:
            try:
                client, addr = sock.accept()
                client_thread = threading.Thread(target=self._handle_tarpit_client, args=(client, addr, port))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                logger.error(f"Error in TCP tarpit: {e}")
        
        sock.close()
    
    def _handle_tarpit_client(self, client, addr, port):
        try:
            service_map = {
                25: "SMTP",
                110: "POP3",
                143: "IMAP",
                587: "SMTP"
            }
            
            service = service_map.get(port, f"TCP:{port}")
            
            event_manager.log_event(
                addr[0], "0.0.0.0", service, 
                "tarpit_connection", 
                None,
                "tarpit"
            )
            
            # Set a very small TCP window size
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
            
            # Send an initial banner extremely slowly
            banner = ""
            if port == 25 or port == 587:
                banner = "220 mail.example.com ESMTP\r\n"
            elif port == 110:
                banner = "+OK POP3 server ready\r\n"
            elif port == 143:
                banner = "* OK IMAP4rev1 Server Ready\r\n"
            else:
                banner = f"Welcome to service on port {port}\r\n"
            
            # Send the banner one byte at a time with delays
            for byte in banner.encode():
                client.send(bytes([byte]))
                time.sleep(1)  # 1 second delay between each byte
            
            # Continue the tarpit by accepting data but responding extremely slowly
            try:
                while self.running:
                    data = client.recv(1)
                    if not data:
                        break
                    
                    # Log received data
                    event_manager.log_event(
                        addr[0], "0.0.0.0", service, 
                        "tarpit_data", 
                        data.hex(),
                        "tarpit"
                    )
                    
                    # Respond with single character and long delay
                    response = b"."
                    time.sleep(5)  # 5 second delay for each response
                    client.send(response)
            except socket.timeout:
                pass
            
        except Exception as e:
            logger.error(f"Error handling tarpit client: {e}")
        finally:
            client.close()

# HTTP Tarpit
class HTTPTarpit:
    def __init__(self, ports):
        self.ports = ports
        self.running = False
        self.threads = []
        logger.info(f"Initialized HTTP tarpit on ports {ports}")
    
    def start(self):
        self.running = True
        for port in self.ports:
            thread = threading.Thread(target=self.run_tarpit, args=(port,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            logger.info(f"Started HTTP tarpit on port {port}")
    
    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1)
        logger.info(f"Stopped HTTP tarpit")
    
    def run_tarpit(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)
        
        while self.running:
            try:
                client, addr = sock.accept()
                client_thread = threading.Thread(target=self._handle_http_tarpit, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                logger.error(f"Error in HTTP tarpit: {e}")
        
        sock.close()
    
    def _handle_http_tarpit(self, client, addr):
        try:
            # Log connection
            event_manager.log_event(
                addr[0], "0.0.0.0", "HTTP", 
                "tarpit_connection", 
                None,
                "tarpit"
            )
            
            # Receive the request
            data = client.recv(4096)
            if not data:
                client.close()
                return
            
            # Log the request
            request = data.decode('utf-8', errors='ignore')
            event_manager.log_event(
                addr[0], "0.0.0.0", "HTTP", 
                "tarpit_request", 
                request,
                "tarpit"
            )
            
            # Send HTTP headers very slowly
            headers = [
                "HTTP/1.1 200 OK",
                "Server: Apache/2.4.41 (Ubuntu)",
                "Content-Type: text/html; charset=UTF-8",
                "Connection: keep-alive",
                # Don't specify content-length to keep connection open
            ]
            
            # Send each header with delay
            for header in headers:
                client.send(f"{header}\r\n".encode())
                time.sleep(1)
            
            # Send an empty line to indicate the end of headers
            client.send(b"\r\n")
            
            # Start sending an "infinite" HTML document
            client.send(b"<html><head><title>Loading...</title></head><body><h1>Loading page, please wait...</h1><div>")
            
            # Send tiny chunks of data with long delays
            counter = 0
            try:
                while self.running and counter < 1000:  # Limit to 1000 iterations
                    time.sleep(2)
                    client.send(f"<!-- Loading data chunk {counter} -->\n".encode())
                    counter += 1
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error handling HTTP tarpit: {e}")
        finally:
            client.close()

# Packet capture and network monitoring
class NetworkMonitor:
    def __init__(self, honeypot_subnet, production_subnets):
        self.honeypot_subnet = honeypot_subnet
        self.production_subnets = production_subnets
        self.running = False
        self.thread = None
        self.alert_counts = {}  # IP -> count
        self.alert_threshold = 5
        logger.info(f"Initialized network monitor for subnet {honeypot_subnet}")
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_traffic)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started network monitor")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Stopped network monitor")
    
    def _monitor_traffic(self):
        try:
            # Use scapy to sniff traffic
            sniff(filter=f"net {self.honeypot_subnet}", prn=self._process_packet, store=0)
        except Exception as e:
            logger.error(f"Error in network monitor: {e}")
    
    def _process_packet(self, packet):
        if not self.running:
            return
        
        try:
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                # Check if traffic is between honeypot and production network
                if self._is_in_subnet(src_ip, self.honeypot_subnet) and any(self._is_in_subnet(dst_ip, subnet) for subnet in self.production_subnets):
                    # This is potentially an attempt to pivot from honeypot to production
                    logger.warning(f"Potential pivot attempt from {src_ip} to {dst_ip}")
                    
                    # Increment alert count
                    self.alert_counts[src_ip] = self.alert_counts.get(src_ip, 0) + 1
                    
                    # Log the event
                    proto = "TCP" if TCP in packet else "UDP" if UDP in packet else "IP"
                    
                    event_manager.log_event(
                        src_ip, dst_ip, proto, 
                        "potential_pivot", 
                        packet.summary(),
                        "lateral_movement"
                    )
                    
                    # Check if threshold exceeded
                    if self.alert_counts.get(src_ip, 0) >= self.alert_threshold:
                        logger.critical(f"ALERT: Threshold exceeded for {src_ip} - potential attack in progress")
                        # Reset counter
                        self.alert_counts[src_ip] = 0
                        
                        # Here you would typically trigger an alert or notification
                        # In a real system, you might integrate with an alerting platform
                        
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def _is_in_subnet(self, ip, subnet):
        try:
            return ipaddress.ip_address(ip) in ipaddress.ip_network(subnet)
        except:
            return False

# Web interface for monitoring
class WebInterface:
    def __init__(self, port=8000):
        self.port = port
        self.app = Flask(__name__)
        self.running = False
        self.thread = None
        
        # Set up routes
        self.app.route("/")(self.index)
        self.app.route("/api/events")(self.api_events)
        self.app.route("/api/stats")(self.api_stats)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started web interface on port {self.port}")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Stopped web interface")
    
    def _run_server(self):
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Error in web interface: {e}")
    
    def index(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deception System Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .dashboard { display: flex; flex-wrap: wrap; }
                .panel { flex: 1; min-width: 300px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
            <script>
                function loadEvents() {
                    fetch('/api/events')
                        .then(response => response.json())
                        .then(data => {
                            const table = document.getElementById('events-table');
                            table.innerHTML = '<tr><th>Time</th><th>Source</th><th>Destination</th><th>Service</th><th>Activity</th><th>Classification</th></tr>';
                            
                            data.forEach(event => {
                                const row = table.insertRow();
                                row.innerHTML = `
                                    <td>${event.timestamp}</td>
                                    <td>${event.source_ip}</td>
                                    <td>${event.destination_ip}</td>
                                    <td>${event.service}</td>
                                    <td>${event.activity}</td>
                                    <td>${event.classification}</td>
                                `;
                            });
                        });
                }
                
                function loadStats() {
                    fetch('/api/stats')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('total-events').textContent = data.total_events;
                            document.getElementById('attack-sources').textContent = data.unique_sources;
                            document.getElementById('top-services').textContent = data.top_services.join(', ');
                            document.getElementById('top-activities').textContent = data.top_activities.join(', ');
                        });
                }
                
                // Load data when page loads
                window.onload = function() {
                    loadEvents();
                    loadStats();
                    
                    // Refresh every 30 seconds
                    setInterval(loadEvents, 30000);
                    setInterval(loadStats, 30000);
                };
            </script>
        </head>
        <body>
            <h1>Deception System Dashboard</h1>
            
            <div class="dashboard">
                <div class="panel">
                    <h2>System Stats</h2>
                    <p>Total Events: <span id="total-events">Loading...</span></p>
                    <p>Unique Attack Sources: <span id="attack-sources">Loading...</span></p>
                    <p>Top Services: <span id="top-services">Loading...</span></p>
                    <p>Top Activities: <span id="top-activities">Loading...</span></p>
                </div>
                
                <div class="panel">
                    <h2>System Status</h2>
                    <p>Honeypots: Active</p>
                    <p>Tarpits: Active</p>
                    <p>Monitoring: Active</p>
                    <p>Last Update: <span id="last-update"></span></p>
                    <script>
                        document.getElementById('last-update').textContent = new Date().toLocaleString();
                    </script>
                </div>
            </div>
            
            <h2>Recent Events</h2>
            <table id="events-table">
                <tr><th>Time</th><th>Source</th><th>Destination</th><th>Service</th><th>Activity</th><th>Classification</th></tr>
                <tr><td colspan="6">Loading...</td></tr>
            </table>
        </body>
        </html>
        """
    
    def api_events(self):
        events = event_manager.get_recent_events(100)
        result = []
        for event in events:
            result.append({
                "id": event[0],
                "timestamp": event[1],
                "source_ip": event[2],
                "destination_ip": event[3],
                "service": event[4],
                "activity": event[5],
                "classification": event[7]
            })
        return jsonify(result)
    
    def api_stats(self):
        events = event_manager.get_recent_events(1000)
        
        # Calculate stats
        sources = set()
        services = {}
        activities = {}
        
        for event in events:
            sources.add(event[2])  # source_ip
            
            service = event[4]
            services[service] = services.get(service, 0) + 1
            
            activity = event[5]
            activities[activity] = activities.get(activity, 0) + 1
        
        # Get top services and activities
        top_services = sorted(services.items(), key=lambda x: x[1], reverse=True)[:5]
        top_activities = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            "total_events": len(events),
            "unique_sources": len(sources),
            "top_services": [s[0] for s in top_services],
            "top_activities": [a[0] for a in top_activities]
        })

# Main Deception System Class
class DeceptionSystem:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Setup database
        self.db_conn = setup_database()
        global event_manager
        event_manager = EventManager(self.db_conn)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        self.honeypots = []
        self.tarpits = []
        
        # Create honeypots
        if self.config["honeypots"]["ssh"]["enabled"]:
            self.honeypots.append(SSHHoneypot(
                self.config["honeypots"]["ssh"]["ports"],
                self.config["honeypots"]["ssh"]["interaction_level"]
            ))
        
        if self.config["honeypots"]["web"]["enabled"]:
            self.honeypots.append(HTTPHoneypot(
                self.config["honeypots"]["web"]["ports"],
                self.config["honeypots"]["web"]["interaction_level"]
            ))
        
        if self.config["honeypots"]["ftp"]["enabled"]:
            self.honeypots.append(FTPHoneypot(
                self.config["honeypots"]["ftp"]["ports"],
                self.config["honeypots"]["ftp"]["interaction_level"]
            ))
        
        if self.config["honeypots"]["smb"]["enabled"]:
            self.honeypots.append(SMBHoneypot(
                self.config["honeypots"]["smb"]["ports"],
                self.config["honeypots"]["smb"]["interaction_level"]
            ))
        
        # Create tarpits
        if self.config["tarpits"]["tcp"]["enabled"]:
            self.tarpits.append(TCPTarpit(
                self.config["tarpits"]["tcp"]["ports"]
            ))
        
        if self.config["tarpits"]["http"]["enabled"]:
            self.tarpits.append(HTTPTarpit(
                self.config["tarpits"]["http"]["ports"]
            ))
        
        # Create network monitor
        self.network_monitor = NetworkMonitor(
            self.config["network"]["honeypot_subnet"],
            self.config["network"]["production_subnets"]
        )
        
        # Create web interface
        self.web_interface = WebInterface()
    
    def start(self):
        logger.info("Starting Deception System...")
        
        # Start honeypots
        for honeypot in self.honeypots:
            honeypot.start()
        
        # Start tarpits
        for tarpit in self.tarpits:
            tarpit.start()
        
        # Start network monitor
        self.network_monitor.start()
        
        # Start web interface
        self.web_interface.start()
        
        logger.info("Deception System started successfully")
    
    def stop(self):
        logger.info("Stopping Deception System...")
        
        # Stop honeypots
        for honeypot in self.honeypots:
            honeypot.stop()
        
        # Stop tarpits
        for tarpit in self.tarpits:
            tarpit.stop()
        
        # Stop network monitor
        self.network_monitor.stop()
        
        # Stop web interface
        self.web_interface.stop()
        
        # Close database connection
        self.db_conn.close()
        
        logger.info("Deception System stopped")
    
    def status(self):
        status = {
            "honeypots": [],
            "tarpits": [],
            "network_monitor": "running" if self.network_monitor.running else "stopped",
            "web_interface": "running" if self.web_interface.running else "stopped"
        }
        
        for honeypot in self.honeypots:
            status["honeypots"].append({
                "service": honeypot.service_name,
                "ports": honeypot.ports,
                "interaction_level": honeypot.interaction_level,
                "status": "running" if honeypot.running else "stopped"
            })
        
        for tarpit in self.tarpits:
            tarpit_type = "HTTP" if isinstance(tarpit, HTTPTarpit) else "TCP"
            status["tarpits"].append({
                "type": tarpit_type,
                "ports": tarpit.ports,
                "status": "running" if tarpit.running else "stopped"
            })
        
        return status

# Main entry point
if __name__ == "__main__":
    # Create and start the deception system
    system = DeceptionSystem()
    
    try:
        system.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down Deception System...")
    
    finally:
        system.stop()
