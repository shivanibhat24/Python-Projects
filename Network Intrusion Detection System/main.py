import argparse
import datetime
import os
import time
import socket
import struct
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
from scapy.all import sniff, IP, TCP, UDP, ICMP
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class NetworkIDS:
    def __init__(self, interface=None, pcap_file=None, log_dir="./logs", 
                 model_path="./model", training_mode=False, threshold=0.8):
        self.interface = interface
        self.pcap_file = pcap_file
        self.log_dir = log_dir
        self.model_path = model_path
        self.training_mode = training_mode
        self.threshold = threshold
        self.ip_flows = defaultdict(list)
        self.flow_stats = defaultdict(dict)
        self.known_ports = {80, 443, 22, 53, 25, 21, 3389, 8080, 8443}
        
        # Create directories if they don't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Initialize or load ML model
        self.model_file = os.path.join(model_path, "ids_model.pkl")
        self.scaler_file = os.path.join(model_path, "scaler.pkl")
        
        if os.path.exists(self.model_file) and not training_mode:
            print("[+] Loading existing model...")
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("[+] Initializing new model...")
            self.model = IsolationForest(contamination=0.01, random_state=42)
            self.scaler = StandardScaler()
    
    def start_capture(self, packet_count=None, timeout=None):
        """Start capturing packets from interface or pcap file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.alert_log = os.path.join(self.log_dir, f"alerts_{timestamp}.log")
        
        # Initialize the log file
        with open(self.alert_log, 'w') as f:
            f.write(f"Network IDS started at {datetime.datetime.now()}\n")
            f.write("Timestamp,Source,Destination,Protocol,Flags,Length,Alert,Score\n")
        
        print(f"[+] Starting packet capture on {self.interface if self.interface else self.pcap_file}")
        print(f"[+] Alerts will be logged to {self.alert_log}")
        
        if self.pcap_file:
            # Read from pcap file
            sniff(offline=self.pcap_file, prn=self.packet_handler, count=packet_count, timeout=timeout)
        else:
            # Sniff from interface
            sniff(iface=self.interface, prn=self.packet_handler, count=packet_count, timeout=timeout)
        
        # After capturing, process any remaining flows
        self.process_flows()
        
        # If in training mode, train the model
        if self.training_mode:
            self.train_model()
    
    def packet_handler(self, packet):
        """Handle each captured packet"""
        if IP in packet:
            ip_src = packet[IP].src
            ip_dst = packet[IP].dst
            protocol = packet[IP].proto
            length = len(packet)
            
            # Extract timestamp
            timestamp = datetime.datetime.now()
            
            # Create flow identifier (source IP, dest IP, protocol)
            flow_id = (ip_src, ip_dst, protocol)
            
            # Basic packet info
            pkt_info = {
                'timestamp': timestamp,
                'length': length,
                'src': ip_src,
                'dst': ip_dst,
                'proto': protocol
            }
            
            # Protocol specific features
            if TCP in packet:
                pkt_info['sport'] = packet[TCP].sport
                pkt_info['dport'] = packet[TCP].dport
                pkt_info['flags'] = packet[TCP].flags
                # Check for SYN flood (many SYN packets from same source)
                if packet[TCP].flags & 0x02:  # SYN flag
                    self.check_syn_flood(ip_src, ip_dst, timestamp)
            elif UDP in packet:
                pkt_info['sport'] = packet[UDP].sport
                pkt_info['dport'] = packet[UDP].dport
                pkt_info['flags'] = 0
            elif ICMP in packet:
                pkt_info['type'] = packet[ICMP].type
                pkt_info['code'] = packet[ICMP].code
                pkt_info['flags'] = 0
                # Check for ICMP flood
                self.check_icmp_flood(ip_src, timestamp)
            
            # Add to flow
            self.ip_flows[flow_id].append(pkt_info)
            
            # Check for unusual ports
            if 'dport' in pkt_info and pkt_info['dport'] not in self.known_ports and pkt_info['dport'] < 1024:
                self.log_alert(
                    timestamp, ip_src, ip_dst, protocol, 
                    pkt_info.get('flags', 0), length,
                    f"Unusual port {pkt_info['dport']} detected", 0.7
                )
            
            # Periodically process flows
            if len(self.ip_flows[flow_id]) >= 100:
                self.process_flow(flow_id)
    
    def check_syn_flood(self, ip_src, ip_dst, timestamp):
        """Check for potential SYN flood attack"""
        key = f"syn_flood_{ip_src}_{ip_dst}"
        current_time = time.time()
        
        if key not in self.flow_stats:
            self.flow_stats[key] = {'count': 1, 'first_seen': current_time, 'last_seen': current_time}
        else:
            stats = self.flow_stats[key]
            stats['count'] += 1
            stats['last_seen'] = current_time
            
            # If we see more than 50 SYN packets in 2 seconds, alert
            time_diff = stats['last_seen'] - stats['first_seen']
            if time_diff <= 2 and stats['count'] > 50:
                self.log_alert(
                    timestamp, ip_src, ip_dst, 'TCP', 'S', 0,
                    f"Possible SYN flood attack: {stats['count']} SYN packets in {time_diff:.2f}s", 0.9
                )
                # Reset counter
                stats['count'] = 0
                stats['first_seen'] = current_time
    
    def check_icmp_flood(self, ip_src, timestamp):
        """Check for potential ICMP flood attack"""
        key = f"icmp_flood_{ip_src}"
        current_time = time.time()
        
        if key not in self.flow_stats:
            self.flow_stats[key] = {'count': 1, 'first_seen': current_time, 'last_seen': current_time}
        else:
            stats = self.flow_stats[key]
            stats['count'] += 1
            stats['last_seen'] = current_time
            
            # If we see more than 30 ICMP packets in 1 second, alert
            time_diff = stats['last_seen'] - stats['first_seen']
            if time_diff <= 1 and stats['count'] > 30:
                self.log_alert(
                    timestamp, ip_src, "multiple", 'ICMP', 0, 0,
                    f"Possible ICMP flood attack: {stats['count']} ICMP packets in {time_diff:.2f}s", 0.9
                )
                # Reset counter
                stats['count'] = 0
                stats['first_seen'] = current_time
    
    def process_flows(self):
        """Process all collected flows"""
        for flow_id in list(self.ip_flows.keys()):
            self.process_flow(flow_id)
    
    def process_flow(self, flow_id):
        """Extract features from a flow and check for anomalies"""
        flow = self.ip_flows[flow_id]
        if len(flow) < 5:  # Ignore very small flows
            return
        
        # Extract features from the flow
        features = self.extract_flow_features(flow)
        
        # Check for anomalies if not in training mode
        if not self.training_mode and hasattr(self, 'model'):
            self.detect_anomalies(flow_id, features)
        
        # Clear the processed flow
        self.ip_flows[flow_id] = []
    
    def extract_flow_features(self, flow):
        """Extract statistical features from a flow"""
        # Basic statistics
        packet_lengths = [pkt['length'] for pkt in flow]
        packet_times = [pkt['timestamp'] for pkt in flow]
        time_diffs = [(packet_times[i] - packet_times[i-1]).total_seconds() 
                      for i in range(1, len(packet_times))]
        
        features = {
            'flow_duration': (packet_times[-1] - packet_times[0]).total_seconds(),
            'num_packets': len(flow),
            'avg_packet_size': np.mean(packet_lengths),
            'std_packet_size': np.std(packet_lengths),
            'min_packet_size': min(packet_lengths),
            'max_packet_size': max(packet_lengths),
            'avg_time_diff': np.mean(time_diffs) if time_diffs else 0,
            'std_time_diff': np.std(time_diffs) if time_diffs else 0,
        }
        
        # Protocol-specific counters
        proto_counts = Counter([pkt.get('proto', 0) for pkt in flow])
        features['tcp_count'] = proto_counts.get(6, 0)  # TCP
        features['udp_count'] = proto_counts.get(17, 0)  # UDP
        features['icmp_count'] = proto_counts.get(1, 0)  # ICMP
        
        # Flag statistics for TCP
        tcp_flags = [pkt.get('flags', 0) for pkt in flow if pkt.get('proto', 0) == 6]
        if tcp_flags:
            features['syn_count'] = sum(1 for f in tcp_flags if f & 0x02)
            features['fin_count'] = sum(1 for f in tcp_flags if f & 0x01)
            features['rst_count'] = sum(1 for f in tcp_flags if f & 0x04)
            features['psh_count'] = sum(1 for f in tcp_flags if f & 0x08)
            features['ack_count'] = sum(1 for f in tcp_flags if f & 0x10)
            features['urg_count'] = sum(1 for f in tcp_flags if f & 0x20)
        else:
            features['syn_count'] = features['fin_count'] = features['rst_count'] = 0
            features['psh_count'] = features['ack_count'] = features['urg_count'] = 0
        
        return features
    
    def detect_anomalies(self, flow_id, features):
        """Detect anomalies in a flow using the trained model"""
        # Convert features to a format suitable for the model
        feature_df = pd.DataFrame([features])
        feature_array = self.scaler.transform(feature_df)
        
        # Predict anomaly score (-1 for anomalies, 1 for normal)
        prediction = self.model.predict(feature_array)
        score = self.model.decision_function(feature_array)[0]
        
        # Normalize score to be between 0 and 1 (1 being most anomalous)
        normalized_score = 1 - (score + 0.5)  # Adjust based on your model's output range
        
        # If anomaly detected, log an alert
        if prediction[0] == -1 or normalized_score > self.threshold:
            src, dst, proto = flow_id
            timestamp = datetime.datetime.now()
            self.log_alert(
                timestamp, src, dst, proto, 
                features.get('syn_count', 0), features.get('avg_packet_size', 0),
                f"Anomalous traffic pattern detected (score: {normalized_score:.3f})", 
                normalized_score
            )
    
    def train_model(self):
        """Train the anomaly detection model on collected flow data"""
        print("[+] Training model on collected flow data...")
        
        # Prepare feature data from all flows
        all_features = []
        for flow_list in self.ip_flows.values():
            if len(flow_list) >= 5:  # Only use flows with enough packets
                features = self.extract_flow_features(flow_list)
                all_features.append(features)
        
        if not all_features:
            print("[!] No sufficient flow data for training")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Handle any NaN values
        df = df.fillna(0)
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(df)
        
        # Train the model
        self.model.fit(scaled_features)
        
        # Save the model and scaler
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"[+] Model trained and saved to {self.model_file}")
    
    def log_alert(self, timestamp, src, dst, proto, flags, length, message, score):
        """Log an alert to the alert file"""
        # Map protocol number to name
        proto_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        proto_name = proto_map.get(proto, proto)
        
        # Create alert line
        alert_line = f"{timestamp},{src},{dst},{proto_name},{flags},{length},{message},{score:.3f}\n"
        
        # Print to console
        print(f"[ALERT] {timestamp} - {message} - {src} -> {dst} ({proto_name})")
        
        # Write to file
        with open(self.alert_log, 'a') as f:
            f.write(alert_line)

def main():
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    parser.add_argument('-i', '--interface', help='Network interface to capture')
    parser.add_argument('-r', '--read', help='Read from pcap file')
    parser.add_argument('-l', '--logdir', default='./logs', help='Directory to store logs')
    parser.add_argument('-m', '--modeldir', default='./model', help='Directory to store/load models')
    parser.add_argument('-t', '--train', action='store_true', help='Train mode - collect data and train model')
    parser.add_argument('-c', '--count', type=int, help='Number of packets to capture')
    parser.add_argument('-d', '--duration', type=int, help='Duration to capture in seconds')
    parser.add_argument('-th', '--threshold', type=float, default=0.8, help='Anomaly threshold (0-1)')
    
    args = parser.parse_args()
    
    if not args.interface and not args.read:
        parser.error("Either --interface or --read must be specified")
    
    ids = NetworkIDS(
        interface=args.interface,
        pcap_file=args.read,
        log_dir=args.logdir,
        model_path=args.modeldir,
        training_mode=args.train,
        threshold=args.threshold
    )
    
    try:
        ids.start_capture(packet_count=args.count, timeout=args.duration)
    except KeyboardInterrupt:
        print("\n[!] Capture interrupted by user")
        ids.process_flows()
        if args.train:
            ids.train_model()
    
    print("[+] IDS operation completed")

if __name__ == "__main__":
    main()
