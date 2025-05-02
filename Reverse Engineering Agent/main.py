import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import serial
import struct
import io
import time
from collections import Counter, defaultdict
import binascii

class ReverseEngineeringAssistant:
    def __init__(self):
        self.raw_data = []
        self.messages = []
        self.patterns = {}
        self.field_candidates = {}
        self.autoencoder = None
        self.freq_analysis = {}
        
    def connect_serial(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        """Connect to a serial port and start capturing data"""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Connected to {port} at {baudrate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False
            
    def capture_data(self, duration=60, chunk_size=1024):
        """Capture raw data from serial port for specified duration in seconds"""
        start_time = time.time()
        self.raw_data = []
        
        print(f"Capturing data for {duration} seconds...")
        
        while time.time() - start_time < duration:
            if self.ser.in_waiting:
                data = self.ser.read(chunk_size)
                if data:
                    self.raw_data.append(data)
                    print(f"Read {len(data)} bytes")
            time.sleep(0.1)
        
        # Concatenate all captured data chunks
        all_data = b''.join(self.raw_data)
        print(f"Captured {len(all_data)} bytes total")
        return all_data
    
    def load_data_from_file(self, filename):
        """Load binary data from a file instead of serial port"""
        try:
            with open(filename, 'rb') as f:
                self.raw_data = [f.read()]
            all_data = b''.join(self.raw_data)
            print(f"Loaded {len(all_data)} bytes from {filename}")
            return all_data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def segment_messages(self, data, methods=None):
        """Try to identify message boundaries in the data stream using various methods"""
        if methods is None:
            methods = ["fixed_length", "delimiters", "timing", "pattern_recognition"]
        
        results = {}
        self.messages = []
        
        if "fixed_length" in methods:
            # Try common fixed lengths: 8, 16, 32, 64, 128 bytes
            lengths = [8, 16, 32, 64, 128]
            for length in lengths:
                segments = [data[i:i+length] for i in range(0, len(data), length)]
                results[f"fixed_{length}"] = segments
        
        if "delimiters" in methods:
            # Try common delimiters: \r\n, \n, 0xFF, etc.
            delimiters = [b'\r\n', b'\n', bytes([0xFF]), bytes([0xAA, 0x55])]
            for delim in delimiters:
                segments = []
                buffer = io.BytesIO(data)
                while True:
                    segment = buffer.readline().rstrip(delim)
                    if not segment:
                        break
                    segments.append(segment)
                if segments:
                    results[f"delim_{delim.hex()}"] = segments
        
        if "pattern_recognition" in methods:
            # Look for repeating patterns in byte sequences
            potential_headers = self.find_potential_headers(data)
            if potential_headers:
                top_header = max(potential_headers, key=potential_headers.get)
                segments = []
                pos = 0
                while pos < len(data):
                    next_pos = data.find(top_header, pos + len(top_header))
                    if next_pos == -1:
                        segments.append(data[pos:])
                        break
                    segments.append(data[pos:next_pos])
                    pos = next_pos
                results["pattern_based"] = segments
        
        # Select the most promising segmentation method
        if results:
            # Choose the method that produces the most consistent message lengths
            best_method = None
            min_variance = float('inf')
            
            for method, segments in results.items():
                if len(segments) > 1:
                    lengths = [len(s) for s in segments]
                    variance = np.var(lengths)
                    if variance < min_variance:
                        min_variance = variance
                        best_method = method
            
            if best_method:
                self.messages = results[best_method]
                print(f"Selected segmentation method: {best_method}")
                print(f"Found {len(self.messages)} potential messages")
                
                # Display statistics
                lengths = [len(m) for m in self.messages]
                print(f"Message length statistics: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
                
                return self.messages
        
        # If no clear segmentation was found, use fixed chunks as fallback
        self.messages = [data[i:i+64] for i in range(0, len(data), 64)]
        print(f"Using default 64-byte chunks: {len(self.messages)} chunks")
        return self.messages
    
    def find_potential_headers(self, data, min_length=2, max_length=8, min_occurrences=3):
        """Find byte sequences that could be message headers"""
        headers = {}
        
        for length in range(min_length, max_length + 1):
            for i in range(len(data) - length):
                pattern = data[i:i+length]
                count = 0
                pos = 0
                
                while True:
                    pos = data.find(pattern, pos)
                    if pos == -1:
                        break
                    count += 1
                    pos += 1
                    
                if count >= min_occurrences:
                    headers[pattern] = count
        
        # Sort by occurrence count and return top candidates
        sorted_headers = sorted(headers.items(), key=lambda x: x[1], reverse=True)[:10]
        return dict(sorted_headers)
    
    def analyze_byte_frequencies(self):
        """Analyze the frequency distribution of bytes in each message position"""
        if not self.messages:
            print("No messages to analyze. Run segment_messages first.")
            return
        
        # Find the maximum message length
        max_length = max(len(msg) for msg in self.messages)
        
        # Initialize frequency counters for each position
        position_frequencies = [Counter() for _ in range(max_length)]
        
        # Count byte frequencies at each position
        for msg in self.messages:
            for i, byte in enumerate(msg):
                position_frequencies[i][byte] += 1
        
        # Calculate entropy for each position
        entropy = []
        for pos, counter in enumerate(position_frequencies):
            total = sum(counter.values())
            if total > 0:
                probs = [count/total for count in counter.values()]
                ent = -sum(p * np.log2(p) for p in probs if p > 0)
                entropy.append(ent)
            else:
                entropy.append(0)
        
        # Store analysis results
        self.freq_analysis = {
            'position_frequencies': position_frequencies,
            'entropy': entropy
        }
        
        # Visualize entropy
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(entropy)), entropy)
        plt.xlabel('Byte Position')
        plt.ylabel('Entropy (bits)')
        plt.title('Byte Entropy by Position')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return self.freq_analysis
    
    def identify_field_candidates(self, entropy_threshold=3.0):
        """Identify potential field boundaries based on entropy analysis"""
        if not self.freq_analysis:
            self.analyze_byte_frequencies()
        
        entropy = self.freq_analysis['entropy']
        
        # Identify potential field boundaries based on entropy changes
        field_boundaries = [0]  # Start of message is always a boundary
        
        for i in range(1, len(entropy) - 1):
            # Look for significant drops in entropy
            if entropy[i-1] > entropy_threshold and entropy[i] < entropy_threshold:
                field_boundaries.append(i)
            # Or significant increases
            elif entropy[i-1] < entropy_threshold and entropy[i] > entropy_threshold:
                field_boundaries.append(i)
        
        field_boundaries.append(len(entropy))  # End of message is always a boundary
        
        # Define fields based on boundaries
        fields = []
        for i in range(len(field_boundaries) - 1):
            start = field_boundaries[i]
            end = field_boundaries[i+1]
            fields.append((start, end))
        
        self.field_candidates = {
            'boundaries': field_boundaries,
            'fields': fields
        }
        
        print(f"Identified {len(fields)} potential fields:")
        for i, (start, end) in enumerate(fields):
            field_length = end - start
            print(f"Field {i+1}: Positions {start}-{end-1} (length: {field_length})")
        
        return self.field_candidates
    
    def train_autoencoder(self, encoding_dim=16):
        """Train an autoencoder to learn compressed representation of messages"""
        if not self.messages:
            print("No messages to analyze. Run segment_messages first.")
            return
        
        # Convert messages to fixed-length arrays for training
        max_length = max(len(msg) for msg in self.messages)
        X = np.zeros((len(self.messages), max_length), dtype=np.float32)
        
        for i, msg in enumerate(self.messages):
            X[i, :len(msg)] = np.frombuffer(msg, dtype=np.uint8)
        
        # Normalize data to 0-1 range
        X = X / 255.0
        
        # Build autoencoder model
        input_data = Input(shape=(max_length,))
        encoded = Dense(128, activation='relu')(input_data)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(max_length, activation='sigmoid')(decoded)
        
        # Full autoencoder
        autoencoder = Model(input_data, decoded)
        
        # Encoder model
        encoder = Model(input_data, encoded)
        
        # Compile model
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        history = autoencoder.fit(
            X, X,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store models
        self.autoencoder = {
            'full_model': autoencoder,
            'encoder': encoder,
            'history': history.history
        }
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Autoencoder Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        
        return self.autoencoder
    
    def cluster_messages(self, n_clusters=5):
        """Cluster messages to identify different message types"""
        if self.autoencoder is None:
            self.train_autoencoder()
        
        # Convert messages to fixed-length arrays
        max_length = max(len(msg) for msg in self.messages)
        X = np.zeros((len(self.messages), max_length), dtype=np.float32)
        
        for i, msg in enumerate(self.messages):
            X[i, :len(msg)] = np.frombuffer(msg, dtype=np.uint8)
        
        # Normalize data
        X = X / 255.0
        
        # Get encoded representation
        encoded_msgs = self.autoencoder['encoder'].predict(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(encoded_msgs)
        
        # Visualize clusters with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(encoded_msgs)
        
        plt.figure(figsize=(10, 8))
        for cluster_id in range(n_clusters):
            plt.scatter(
                X_tsne[clusters == cluster_id, 0],
                X_tsne[clusters == cluster_id, 1],
                label=f'Cluster {cluster_id}'
            )
        plt.legend()
        plt.title('Message Clusters')
        plt.show()
        
        # Analyze each cluster
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_msgs = [self.messages[i] for i in range(len(self.messages)) if clusters[i] == cluster_id]
            lengths = [len(msg) for msg in cluster_msgs]
            
            # Find common bytes at each position
            max_cluster_len = max(lengths)
            common_bytes = []
            
            for pos in range(max_cluster_len):
                pos_bytes = [msg[pos] if pos < len(msg) else None for msg in cluster_msgs]
                pos_bytes = [b for b in pos_bytes if b is not None]
                if pos_bytes:
                    counter = Counter(pos_bytes)
                    most_common = counter.most_common(1)[0]
                    if most_common[1] / len(pos_bytes) > 0.8:  # 80% agreement threshold
                        common_bytes.append(most_common[0])
                    else:
                        common_bytes.append(None)
                else:
                    common_bytes.append(None)
            
            cluster_stats[cluster_id] = {
                'count': len(cluster_msgs),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_length': np.mean(lengths),
                'common_bytes': common_bytes
            }
            
            print(f"\nCluster {cluster_id}: {len(cluster_msgs)} messages")
            print(f"  Length: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
            
            # Show common pattern
            pattern = ''
            for pos, byte in enumerate(common_bytes):
                if byte is not None:
                    pattern += f"{byte:02X} "
                else:
                    pattern += "?? "
                if pos % 16 == 15:
                    pattern += "\n"
            
            print("  Common pattern:")
            print(f"  {pattern}")
        
        return {
            'clusters': clusters,
            'stats': cluster_stats,
            'visualization': X_tsne
        }
    
    def guess_field_types(self):
        """Guess the data types of identified fields"""
        if not self.field_candidates:
            self.identify_field_candidates()
        
        field_types = []
        
        for i, (start, end) in enumerate(self.field_candidates['fields']):
            field_length = end - start
            
            # Extract this field from all messages
            field_values = []
            for msg in self.messages:
                if end <= len(msg):
                    field_values.append(msg[start:end])
            
            # Skip if no values
            if not field_values:
                field_types.append("unknown")
                continue
            
            # Count unique values
            unique_values = set(field_values)
            unique_ratio = len(unique_values) / len(field_values)
            
            # Check if field looks like:
            
            # 1. Constant identifier/header
            if unique_ratio < 0.1:
                field_types.append("constant")
            
            # 2. Counter/sequence number
            elif field_length <= 4 and self.check_if_sequential(field_values):
                field_types.append("counter")
            
            # 3. Timestamp (look for incrementing values with occasional jumps)
            elif field_length in [4, 8] and self.check_if_timestamp(field_values):
                field_types.append("timestamp")
            
            # 4. Length indicator
            elif field_length in [1, 2, 4] and self.check_if_length(field_values):
                field_types.append("length")
            
            # 5. Checksum/CRC
            elif field_length in [1, 2, 4] and start > end - field_length:
                field_types.append("checksum")
            
            # 6. Enumerated value
            elif unique_ratio < 0.3:
                field_types.append("enum")
            
            # 7. ASCII string
            elif self.check_if_ascii(field_values):
                field_types.append("ascii")
            
            # 8. Binary data/payload
            elif field_length > 8:
                field_types.append("binary_data")
                
            # 9. Flags/bitfield
            elif field_length in [1, 2, 4, 8] and self.check_if_bitfield(field_values):
                field_types.append("bitfield")
            
            # Fallback
            else:
                field_types.append("unknown")
        
        # Print results
        print("\nField type analysis:")
        for i, ((start, end), field_type) in enumerate(zip(self.field_candidates['fields'], field_types)):
            field_length = end - start
            print(f"Field {i+1}: Positions {start}-{end-1} (length: {field_length}) → {field_type}")
        
        return dict(zip(range(len(field_types)), field_types))
    
    def check_if_sequential(self, values):
        """Check if values form a sequence (like a counter)"""
        # Convert to integers
        try:
            int_values = []
            for val in values[:100]:  # Check a subset for efficiency
                if len(val) == 1:
                    int_values.append(val[0])
                elif len(val) == 2:
                    int_values.append(int.from_bytes(val, byteorder='big'))
                elif len(val) == 4:
                    int_values.append(int.from_bytes(val, byteorder='big'))
            
            # Check if mostly sequential
            sequential_count = 0
            for i in range(1, len(int_values)):
                if int_values[i] == (int_values[i-1] + 1) % 256:
                    sequential_count += 1
            
            return sequential_count / (len(int_values) - 1) > 0.7
        except:
            return False
    
    def check_if_timestamp(self, values):
        """Check if values look like timestamps"""
        try:
            int_values = []
            for val in values[:100]:
                if len(val) == 4:
                    int_values.append(int.from_bytes(val, byteorder='big'))
                elif len(val) == 8:
                    int_values.append(int.from_bytes(val, byteorder='big'))
            
            # Check if mostly increasing
            increasing_count = 0
            for i in range(1, len(int_values)):
                if int_values[i] > int_values[i-1]:
                    increasing_count += 1
            
            return increasing_count / (len(int_values) - 1) > 0.7
        except:
            return False
    
    def check_if_length(self, values):
        """Check if field might be a length indicator"""
        try:
            # Get message lengths
            msg_lengths = [len(msg) for msg in self.messages]
            
            # Convert field values to integers
            length_values = []
            for val in values:
                if len(val) == 1:
                    length_values.append(val[0])
                elif len(val) == 2:
                    length_values.append(int.from_bytes(val, byteorder='big'))
                elif len(val) == 4:
                    length_values.append(int.from_bytes(val, byteorder='big'))
            
            # Check correlation between field values and message lengths
            matches = 0
            for length_val, msg_len in zip(length_values, msg_lengths):
                # Check if field equals message length or a constant offset
                if length_val == msg_len or abs(length_val - msg_len) < 5:
                    matches += 1
            
            return matches / len(values) > 0.7
        except:
            return False
    
    def check_if_ascii(self, values):
        """Check if field values are printable ASCII"""
        try:
            for val in values[:50]:  # Check a subset
                # Check if mostly ASCII printable
                printable_count = sum(32 <= b <= 126 for b in val)
                if printable_count / len(val) < 0.7:
                    return False
            return True
        except:
            return False
    
    def check_if_bitfield(self, values):
        """Check if field might be a bitfield/flags"""
        try:
            # Convert to integers
            int_values = []
            for val in values:
                if len(val) == 1:
                    int_values.append(val[0])
                elif len(val) == 2:
                    int_values.append(int.from_bytes(val, byteorder='big'))
                elif len(val) == 4:
                    int_values.append(int.from_bytes(val, byteorder='big'))
            
            # Check if only certain bits change
            bit_variations = defaultdict(set)
            for val in int_values:
                for bit in range(8 * len(values[0])):
                    bit_value = (val >> bit) & 1
                    bit_variations[bit].add(bit_value)
            
            # If many bits only have one value, likely a bitfield
            stable_bits = sum(1 for bit, values in bit_variations.items() if len(values) == 1)
            varying_bits = sum(1 for bit, values in bit_variations.items() if len(values) > 1)
            
            return stable_bits > 0 and varying_bits > 0
        except:
            return False
    
    def reconstruct_protocol(self):
        """Attempt to reconstruct the protocol format based on all analysis"""
        if not self.field_candidates:
            self.identify_field_candidates()
        
        field_types = self.guess_field_types()
        
        # Cluster messages to find different message types
        clustering = self.cluster_messages()
        
        # Reconstruct protocol for each cluster
        protocols = {}
        for cluster_id, stats in clustering['stats'].items():
            if stats['count'] < 3:  # Skip small clusters
                continue
                
            # Create protocol description
            protocol = {
                'message_type': f"Message_Type_{cluster_id}",
                'count': stats['count'],
                'typical_length': int(stats['avg_length']),
                'fields': []
            }
            
            # Add fields
            for i, (start, end) in enumerate(self.field_candidates['fields']):
                if start >= stats['max_length']:
                    continue
                    
                field_length = end - start
                field_type = field_types.get(i, "unknown")
                
                # Check if field has a consistent value in this cluster
                common_byte = None
                if start < len(stats['common_bytes']):
                    common_byte = stats['common_bytes'][start]
                
                field = {
                    'position': (start, min(end, stats['max_length'])),
                    'length': min(field_length, stats['max_length'] - start),
                    'type': field_type,
                    'consistent_value': common_byte
                }
                
                protocol['fields'].append(field)
            
            protocols[cluster_id] = protocol
        
        # Print protocol reconstruction
        print("\n==== PROTOCOL RECONSTRUCTION ====\n")
        for msg_type, protocol in protocols.items():
            print(f"Message Type: {protocol['message_type']} ({protocol['count']} messages)")
            print(f"Typical Length: {protocol['typical_length']} bytes")
            print("Fields:")
            
            for i, field in enumerate(protocol['fields']):
                start, end = field['position']
                field_desc = f"  Field {i+1}: Bytes {start}-{end-1} ({field['length']} bytes) - {field['type']}"
                
                if field['consistent_value'] is not None:
                    field_desc += f" - Consistent value: 0x{field['consistent_value']:02X}"
                    
                print(field_desc)
            
            print("\n")
        
        return protocols
    
    def generate_parser_code(self, protocol_name="UnknownProtocol"):
        """Generate Python code to parse messages based on the reconstructed protocol"""
        protocols = self.reconstruct_protocol()
        
        parser_code = f"""
# Generated parser for {protocol_name}
import struct
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

class MessageType(Enum):
"""
        
        # Define message types
        for msg_id, protocol in protocols.items():
            parser_code += f"    {protocol['message_type'].upper()} = {msg_id}\n"
        
        # Define field types
        parser_code += """
class FieldType(Enum):
    CONSTANT = 1
    COUNTER = 2
    TIMESTAMP = 3
    LENGTH = 4
    CHECKSUM = 5
    ENUM = 6
    ASCII = 7
    BINARY_DATA = 8
    BITFIELD = 9
    UNKNOWN = 10
"""
        
        # Message base class
        parser_code += """
@dataclass
class Message:
    raw_data: bytes
    message_type: MessageType
    
    @staticmethod
    def parse(data: bytes) -> Optional['Message']:
        \"\"\"Factory method to parse a message from raw bytes\"\"\"
        if not data:
            return None
            
        # Try to identify the message type
"""
        
        # Add message type detection logic
        for msg_id, protocol in protocols.items():
            # Find a distinctive field for this message type
            distinctive_fields = []
            for field in protocol['fields']:
                if field['consistent_value'] is not None:
                    start, end = field['position']
                    distinctive_fields.append((start, field['consistent_value']))
            
            if distinctive_fields:
                parser_code += f"        # Check for {protocol['message_type']}\n"
                conditions = []
                for pos, value in distinctive_fields[:3]:  # Use up to 3 distinctive fields
                    conditions.append(f"data[{pos}] == {value}")
                
                parser_code += f"        if len(data) >= {protocol['typical_length']} and {' and '.join(conditions)}:\n"
                parser_code += f"            return {protocol['message_type']}Message(data)\n"
                parser_code += "\n"
        
        parser_code += "        # Default fallback\n"
        parser_code += "        return UnknownMessage(data)\n"
        
        # Define message classes for each protocol
        for msg_id, protocol in protocols.items():
            class_name = f"{protocol['message_type']}Message"
            
            parser_code += f"""
@dataclass
class {class_name}(Message):
    # Parsed fields
"""
            
            # Add field definitions
            for i, field in enumerate(protocol['fields']):
                field_name = f"field_{i+1}"
                field_type = "Any"
                
                if field['type'] == "constant":
                    field_type = "int"
                elif field['type'] == "counter":
                    field_type = "int"
                elif field['type'] == "timestamp":
                    field_type = "int"
                elif field['type'] == "length":
                    field_type = "int"
                elif field['type'] == "checksum":
                    field_type = "int"
                elif field['type'] == "enum":
                    field_type = "int"
                elif field['type'] == "ascii":
                    field_type = "str"
                elif field['type'] == "binary_data":
                    field_type = "bytes"
                elif field['type'] == "bitfield":
                    field_type = "int"
                
                parser_code += f"    {field_name}: {field_type} = None\n"
            
            # Constructor
            parser_code += f"""
    def __init__(self, data: bytes):
        super().__init__(data, MessageType.{protocol['message_type'].upper()})
        self._parse()
        
    def _parse(self):
        \"\"\"Parse the message fields\"\"\"
"""
            
            # Add parsing logic for each field
            for i, field in enumerate(protocol['fields']):
                start, end = field['position']
                field_name = f"field_{i+1}"
                
                parser_code += f"        # Field {i+1}: {field['type']} at positions {start}-{end-1}\n"
                
                if field['type'] == "ascii":
                    parser_code += f"        try:\n"
                    parser_code += f"            self.{field_name} = self.raw_data[{start}:{end}].decode('ascii').strip()\n"
                    parser_code += f"        except UnicodeDecodeError:\n"
                    parser_code += f"            self.{field_name} = None\n"
                elif field['type'] in ["constant", "counter", "timestamp", "length", "checksum", "enum", "bitfield"]:
                    if field['length'] == 1:
                        parser_code += f"        self.{field_name} = self.raw_data[{start}] if {start} < len(self.raw_data) else None\n"
                    elif field['length'] == 2:
                        parser_code += f"        self.{field_name} = int.from_bytes(self.raw_data[{start}:{end}], byteorder='big') if {end} <= len(self.raw_data) else None\n"
                    elif field['length'] == 4:
                        parser_code += f"        self.{field_name} = int.from_bytes(self.raw_data[{start}:{end}], byteorder='big') if {end} <= len(self.raw_data) else None\n"
                    else:
                        parser_code += f"        self.{field_name} = int.from_bytes(self.raw_data[{start}:{end}], byteorder='big') if {end} <= len(self.raw_data) else None\n"
                else:  # binary_data or unknown
                    parser_code += f"        self.{field_name} = self.raw_data[{start}:{end}] if {end} <= len(self.raw_data) else None\n"
            
            # Add any field-specific methods
            for i, field in enumerate(protocol['fields']):
                field_name = f"field_{i+1}"
                
                if field['type'] == "bitfield":
                    parser_code += f"""
    def get_{field_name}_bit(self, bit_position: int) -> bool:
        \"\"\"Get specific bit from {field_name}\"\"\"
        if self.{field_name} is None:
            return False
        return bool((self.{field_name} >> bit_position) & 1)
"""
                elif field['type'] == "timestamp":
                    parser_code += f"""
    def get_{field_name}_as_datetime(self) -> Optional[datetime]:
        \"\"\"Convert {field_name} to datetime if it's a timestamp\"\"\"
        import datetime
        if self.{field_name} is None:
            return None
        try:
            # This is a guess - might need to be adjusted based on actual timestamp format
            return datetime.datetime.fromtimestamp(self.{field_name})
        except:
            return None
"""
        
        # Add UnknownMessage class
        parser_code += """
@dataclass
class UnknownMessage(Message):
    def __init__(self, data: bytes):
        super().__init__(data, MessageType.UNKNOWN)
"""
        
        # Utility function for using the parser
        parser_code += """
def parse_serial_stream(data: bytes) -> List[Message]:
    \"\"\"Parse a stream of bytes into individual messages\"\"\"
    messages = []
    
    # This is a very basic implementation - in reality you'll need
    # to implement proper message segmentation based on the protocol
    offset = 0
    while offset < len(data):
        # Try different message lengths (adjust based on your protocol)
        for length in [32, 64, 128]:
            if offset + length <= len(data):
                message_data = data[offset:offset+length]
                message = Message.parse(message_data)
                if isinstance(message, UnknownMessage):
                    continue
                messages.append(message)
                offset += length
                break
        else:
            # No valid message found, skip one byte
            offset += 1
    
    return messages

def print_message(message: Message) -> None:
    \"\"\"Pretty print a parsed message\"\"\"
    print(f"Message Type: {message.message_type}")
    print(f"Raw Data: {message.raw_data.hex()}")
    
    for field_name, field_value in vars(message).items():
        if field_name not in ['raw_data', 'message_type']:
            if isinstance(field_value, bytes):
                print(f"  {field_name}: {field_value.hex()}")
            else:
                print(f"  {field_name}: {field_value}")
    print()

# Example usage:
if __name__ == "__main__":
    # Example: Read from a file
    with open("captured_data.bin", "rb") as f:
        data = f.read()
    
    # Parse messages
    messages = parse_serial_stream(data)
    
    # Print results
    print(f"Found {len(messages)} messages")
    for msg in messages:
        print_message(msg)
"""
        
        return parser_code
    
    def extract_relevant_data(self, field_indices=None):
        """Extract relevant data for visualization and further analysis"""
        if not self.messages:
            print("No messages to analyze. Run segment_messages first.")
            return
        
        if not field_indices:
            # If no specific fields are selected, try to find interesting ones
            if not self.field_candidates:
                self.identify_field_candidates()
            
            field_types = self.guess_field_types()
            field_indices = []
            
            for i, field_type in field_types.items():
                if field_type in ["counter", "timestamp", "enum", "length"]:
                    field_indices.append(i)
        
        if not field_indices:
            print("No relevant fields identified for extraction.")
            return
        
        # Extract data from selected fields
        extracted_data = []
        field_positions = [self.field_candidates['fields'][i] for i in field_indices]
        
        for msg in self.messages:
            row = {}
            for i, (start, end) in zip(field_indices, field_positions):
                if end <= len(msg):
                    field_data = msg[start:end]
                    
                    # Convert to appropriate type based on length
                    if len(field_data) == 1:
                        value = field_data[0]
                    elif len(field_data) == 2:
                        value = int.from_bytes(field_data, byteorder='big')
                    elif len(field_data) == 4:
                        value = int.from_bytes(field_data, byteorder='big')
                    else:
                        value = field_data.hex()
                    
                    row[f"field_{i}"] = value
            
            if row:
                extracted_data.append(row)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(extracted_data)
        
        # Print summary
        print(f"Extracted {len(df)} records with {len(field_indices)} fields")
        print(df.head())
        
        # Plot data if numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(12, 8))
            
            # Scatter plot of first two numeric fields
            plt.subplot(2, 2, 1)
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title('Scatter Plot')
            
            # Time series of first numeric field
            plt.subplot(2, 2, 2)
            plt.plot(df.index, df[numeric_cols[0]])
            plt.xlabel('Message Index')
            plt.ylabel(numeric_cols[0])
            plt.title('Time Series')
            
            # Histogram of first numeric field
            plt.subplot(2, 2, 3)
            plt.hist(df[numeric_cols[0]], bins=30)
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
            plt.title('Histogram')
            
            # Correlation heatmap
            if len(numeric_cols) > 2:
                plt.subplot(2, 2, 4)
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            
            plt.tight_layout()
            plt.show()
        
        return df
    
    def visualize_messages(self, num_messages=10):
        """Visualize message structure in a binary view"""
        if not self.messages:
            print("No messages to analyze. Run segment_messages first.")
            return
        
        # Determine max message length for display
        max_display_len = min(max(len(msg) for msg in self.messages), 64)
        
        # Select a sample of messages
        sample = self.messages[:num_messages]
        
        # Create a binary visualization
        fig, ax = plt.subplots(figsize=(15, num_messages * 0.4))
        
        # Prepare data for heatmap
        data = np.zeros((len(sample), max_display_len))
        
        for i, msg in enumerate(sample):
            for j in range(min(len(msg), max_display_len)):
                data[i, j] = msg[j]
        
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, max_display_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sample), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        
        # Add labels
        ax.set_xticks(np.arange(0, max_display_len, 4))
        ax.set_yticks(np.arange(0, len(sample)))
        ax.set_xticklabels([f"{i}" for i in range(0, max_display_len, 4)])
        ax.set_yticklabels([f"Msg {i}" for i in range(len(sample))])
        
        # Title and labels
        ax.set_title("Binary Message Visualization")
        ax.set_xlabel("Byte Position")
        ax.set_ylabel("Message")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Byte Value")
        
        # Add field boundaries if available
        if self.field_candidates:
            for start, end in self.field_candidates['fields']:
                if start < max_display_len:
                    ax.axvline(x=start - 0.5, color='red', linestyle='--')
                if end < max_display_len:
                    ax.axvline(x=end - 0.5, color='red', linestyle='--')
        
        plt.tight_layout()
        plt.show()
        
        # Print hex view of sample messages
        print("Hex view of sample messages:")
        for i, msg in enumerate(sample):
            hex_str = ' '.join(f"{b:02X}" for b in msg[:max_display_len])
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in msg[:max_display_len])
            print(f"Msg {i}: {hex_str}")
            print(f"ASCII: {ascii_str}")
            print("-" * 80)

# Example usage
if __name__ == "__main__":
    # Create the assistant
    rea = ReverseEngineeringAssistant()
    
    # Either load data from file
    data = rea.load_data_from_file("sample_data.bin")
    
    # Or capture from serial port
    # rea.connect_serial(port='COM3', baudrate=115200)
    # data = rea.capture_data(duration=30)
    
    # Segment into messages
    messages = rea.segment_messages(data)
    
    # Visualize messages
    rea.visualize_messages(num_messages=20)
    
    # Analyze byte frequencies
    rea.analyze_byte_frequencies()
    
    # Identify field candidates
    rea.identify_field_candidates()
    
    # Train autoencoder
    rea.train_autoencoder()
    
    # Cluster messages
    rea.cluster_messages()
    
    # Guess field types
    rea.guess_field_types()
    
    # Reconstruct protocol
    protocol = rea.reconstruct_protocol()
    
    # Generate parser code
    parser_code = rea.generate_parser_code("MyUnknownProtocol")
    
    with open("protocol_parser.py", "w") as f:
        f.write(parser_code)
    
    print("Parser code generated and saved to protocol_parser.py")
