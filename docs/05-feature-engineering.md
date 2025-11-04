# Feature Engineering cho IDS

## Giới thiệu

**Feature Engineering** là quá trình chuyển đổi raw network traffic thành các features có ý nghĩa cho ML model. Đây là bước quan trọng nhất quyết định performance của IDS.

```
Raw Packet → Feature Extraction → ML Model → Prediction
```

---

## NSL-KDD Features Chi tiết

### 1. Basic Features từ Network Packets

#### Duration
```python
# Thời gian connection tồn tại
duration = timestamp_last_packet - timestamp_first_packet

# Normal web browsing:
duration = 5.2 seconds  # User đọc trang, click link

# DoS attack:
duration = 0.0 seconds  # SYN flood, không có data transfer

# Download file:
duration = 120 seconds  # Connection dài để download
```

#### Protocol Type
```python
protocol_type = {
    'tcp': 0,   # Transmission Control Protocol
    'udp': 1,   # User Datagram Protocol
    'icmp': 2,  # Internet Control Message Protocol
}

# Example:
# HTTP traffic → tcp
# DNS query → udp
# Ping → icmp
```

**Attack patterns:**
- ICMP flood (DoS) → protocol_type = icmp
- SYN flood (DoS) → protocol_type = tcp
- UDP flood (DoS) → protocol_type = udp

#### Service
```python
# Port → Service mapping
service_map = {
    20: 'ftp_data',
    21: 'ftp',
    22: 'ssh',
    23: 'telnet',
    25: 'smtp',
    53: 'domain',      # DNS
    80: 'http',
    443: 'http',       # HTTPS (NSL-KDD treats as http)
    3306: 'sql_net',   # MySQL
    # ... 70 services total
}

# Extract from packet:
dest_port = packet.tcp.dstport
service = service_map.get(dest_port, 'other')
```

#### Connection Flag
```python
# TCP connection states
flags = {
    'SF': 'Normal establishment and termination',
        # SYN → SYN-ACK → ACK → ... → FIN → ACK

    'S0': 'Connection attempt rejected',
        # SYN → RST

    'S1': 'Connection established but not terminated',
        # SYN → SYN-ACK → ACK (no FIN)

    'S2': 'Connection established and close attempt',
        # SYN → SYN-ACK → ACK → FIN (no FIN-ACK)

    'S3': 'Connection established, closed by originator',

    'REJ': 'Connection attempt rejected',
        # SYN → RST-ACK

    'RSTO': 'Connection reset by originator',
    'RSTR': 'Connection reset by responder',
    'SH': 'SYN seen, no response',
    'OTH': 'Other',
}
```

**Attack detection:**
```python
# SYN Flood:
if flag == 'S0' and count > 100:
    # Nhiều connection bị reject → DoS attack

# Port Scan:
if flag == 'REJ' and diff_srv_rate > 0.9:
    # Scan nhiều ports, bị reject → Probe attack
```

#### Src/Dst Bytes
```python
src_bytes = sum(len(packet.payload) for packet in connection
                if packet.src == source_ip)

dst_bytes = sum(len(packet.payload) for packet in connection
                if packet.dst == dest_ip)

# Normal HTTP GET:
# src_bytes = 200 (request header)
# dst_bytes = 5000 (HTML response)

# DoS SYN Flood:
# src_bytes = 0 (no data)
# dst_bytes = 0 (no response)

# Data exfiltration (R2L):
# src_bytes = 500000 (uploading stolen data)
# dst_bytes = 100 (small ack)
```

---

### 2. Content Features

#### Hot Indicators
```python
# "Hot" = security-sensitive operations
hot_indicators = [
    '/etc/passwd',
    '/etc/shadow',
    'root',
    'admin',
    'su',
    'sudo',
]

hot = count_occurrences(packet_payload, hot_indicators)

# Example:
# Request: GET /files/document.pdf → hot = 0
# Attack:  GET /../../../../etc/passwd → hot = 1 (passwd detected!)
```

#### Failed Logins
```python
num_failed_logins = count_failed_auth_attempts(connection)

# HTTP 401 responses
# SSH authentication failures
# FTP login rejections

# Normal user:
# num_failed_logins = 0 or 1 (typo)

# Brute force attack (R2L):
# num_failed_logins = 50+ (trying many passwords)
```

#### Logged In
```python
logged_in = 1 if authentication_successful else 0

# Distinguish:
# - Pre-authentication attacks (R2L): logged_in = 0
# - Post-authentication attacks (U2R): logged_in = 1
```

#### Root Shell
```python
root_shell = 1 if shell_access and uid == 0 else 0

# U2R attack indicator:
# Normal user (uid=1000) → root shell (uid=0)
# Privilege escalation!
```

#### File Creations
```python
num_file_creations = count_file_operations(connection, ['CREATE', 'WRITE'])

# Normal:
# num_file_creations = 0-2 (save document)

# U2R attack:
# num_file_creations = 10+ (installing backdoor, rootkit)
```

---

### 3. Time-based Features (2-second window)

#### Count
```python
# Connections to the same destination IP trong 2 giây qua
count = len([conn for conn in last_2_seconds
             if conn.dst_ip == current_connection.dst_ip])

# Normal browsing:
count = 2-5 (load page + CSS/JS/images)

# DoS attack:
count = 500+ (flood attack!)
```

**Computation:**
```python
from collections import deque
from datetime import datetime, timedelta

connection_history = deque(maxlen=10000)

def compute_count(current_conn):
    cutoff_time = current_conn.timestamp - timedelta(seconds=2)

    count = sum(1 for conn in connection_history
                if conn.dst_ip == current_conn.dst_ip
                and conn.timestamp >= cutoff_time)

    return count
```

#### Service Count
```python
srv_count = len([conn for conn in last_2_seconds
                 if conn.service == current_connection.service])

# Example:
# User browsing website:
# srv_count = 5 (all HTTP requests)

# Port scanning:
# srv_count = 1 (each port = different service)
```

#### Error Rates
```python
# SYN Error Rate
connections_last_2s = get_connections_last_2_seconds()
syn_errors = [conn for conn in connections_last_2s
              if conn.flag in ['S0', 'S1', 'S2', 'S3']]

serror_rate = len(syn_errors) / len(connections_last_2s)

# Normal traffic:
# serror_rate = 0.0 - 0.1

# SYN Flood (DoS):
# serror_rate = 0.9 - 1.0 (almost all SYN errors!)
```

```python
# REJ Error Rate
rej_errors = [conn for conn in connections_last_2s
              if conn.flag in ['REJ', 'RSTR', 'RSTO']]

rerror_rate = len(rej_errors) / len(connections_last_2s)

# Normal:
# rerror_rate = 0.0

# Port scan (Probe):
# rerror_rate = 0.8+ (scanning closed ports → REJ)
```

#### Same/Different Service Rate
```python
same_srv = [conn for conn in connections_last_2s
            if conn.service == current_connection.service]

same_srv_rate = len(same_srv) / count

diff_srv_rate = 1 - same_srv_rate

# Normal user browsing:
# same_srv_rate = 1.0 (all HTTP)
# diff_srv_rate = 0.0

# Port scan:
# same_srv_rate = 0.1 (scan nhiều services)
# diff_srv_rate = 0.9
```

---

### 4. Host-based Features (100-connection window)

#### Destination Host Count
```python
# Số connections đến cùng dest host trong 100 connections gần nhất
dst_host_count = len([conn for conn in last_100_connections
                      if conn.dst_ip == current_connection.dst_ip])

# Normal:
# dst_host_count = 10-50

# Targeted attack:
# dst_host_count = 100 (all connections to same victim!)
```

#### Destination Host Service Count
```python
dst_host_srv_count = len([conn for conn in last_100_connections
                          if conn.dst_ip == current_connection.dst_ip
                          and conn.service == current_connection.service])

# Example:
# Multiple users accessing same web server:
# dst_host_srv_count = 80 (80/100 connections to same host:service)
```

#### Same Source Port Rate
```python
same_src_port = [conn for conn in last_100_connections
                 if conn.src_port == current_connection.src_port]

dst_host_same_src_port_rate = len(same_src_port) / dst_host_count

# Normal:
# src port thay đổi mỗi connection
# dst_host_same_src_port_rate = 0.01

# Some attacks:
# Reuse same source port
# dst_host_same_src_port_rate = 0.8+
```

---

## Feature Engineering từ Raw Packets

### Extracting từ PCAP file

```python
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from datetime import datetime

def extract_features_from_pcap(pcap_file):
    packets = rdpcap(pcap_file)

    # Group packets into connections
    connections = {}

    for packet in packets:
        if IP in packet:
            # Create connection key
            conn_key = (
                packet[IP].src,
                packet[IP].dst,
                packet[TCP].sport if TCP in packet else 0,
                packet[TCP].dport if TCP in packet else 0,
            )

            if conn_key not in connections:
                connections[conn_key] = {
                    'packets': [],
                    'start_time': packet.time,
                }

            connections[conn_key]['packets'].append(packet)
            connections[conn_key]['end_time'] = packet.time

    # Extract features for each connection
    features_list = []
    for conn_key, conn_data in connections.items():
        features = extract_connection_features(conn_key, conn_data)
        features_list.append(features)

    return features_list

def extract_connection_features(conn_key, conn_data):
    src_ip, dst_ip, src_port, dst_port = conn_key
    packets = conn_data['packets']

    # Basic features
    features = {}

    # Duration
    features['duration'] = conn_data['end_time'] - conn_data['start_time']

    # Protocol
    if TCP in packets[0]:
        features['protocol_type'] = 'tcp'
    elif UDP in packets[0]:
        features['protocol_type'] = 'udp'
    elif ICMP in packets[0]:
        features['protocol_type'] = 'icmp'

    # Service (from port)
    features['service'] = get_service_from_port(dst_port)

    # Flag
    if TCP in packets[0]:
        features['flag'] = get_tcp_flag_status(packets)

    # Bytes
    features['src_bytes'] = sum(len(p) for p in packets
                                if p[IP].src == src_ip)
    features['dst_bytes'] = sum(len(p) for p in packets
                                if p[IP].dst == dst_ip)

    # ... more features

    return features

def get_tcp_flag_status(packets):
    """Determine connection status from TCP flags"""

    has_syn = any(p[TCP].flags & 0x02 for p in packets if TCP in p)
    has_syn_ack = any(p[TCP].flags & 0x12 for p in packets if TCP in p)
    has_fin = any(p[TCP].flags & 0x01 for p in packets if TCP in p)
    has_rst = any(p[TCP].flags & 0x04 for p in packets if TCP in p)

    if has_syn and has_syn_ack and has_fin:
        return 'SF'  # Normal
    elif has_syn and not has_syn_ack:
        return 'S0'  # No response
    elif has_rst:
        return 'REJ'  # Rejected
    else:
        return 'OTH'
```

---

## Feature Selection

### Remove Low-Variance Features

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Remove features where all values are almost the same
# Example: 'land' (almost always 0) → low variance → remove
```

### Feature Importance từ Random Forest

```python
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 20 features
top_features = [feature_names[i] for i in indices[:20]]

# Use only top features
X_reduced = X[:, indices[:20]]
```

### Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = X.corr()

# Visualize
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.show()

# Remove highly correlated features (>0.95)
correlated_features = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            correlated_features.append(corr_matrix.columns[i])

X_reduced = X.drop(correlated_features, axis=1)
```

---

## Feature Transformation

### Log Transformation

```python
# For skewed features (many small values, few large values)
X['duration_log'] = np.log1p(X['duration'])
X['src_bytes_log'] = np.log1p(X['src_bytes'])

# Before: [0, 1, 5, 100, 10000]
# After:  [0, 0.69, 1.79, 4.62, 9.21]
```

### Binning

```python
# Convert continuous to categorical
X['duration_bin'] = pd.cut(X['duration'],
                           bins=[0, 1, 10, 100, np.inf],
                           labels=['very_short', 'short', 'medium', 'long'])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['src_bytes', 'dst_bytes']])

# Creates:
# src_bytes, dst_bytes, src_bytes^2, src_bytes*dst_bytes, dst_bytes^2
```

---

## Real-time Feature Extraction

### Sliding Window Implementation

```python
from collections import deque
from datetime import datetime, timedelta

class FeatureExtractor:
    def __init__(self):
        # Store last 100 connections
        self.connection_history = deque(maxlen=100)

        # Store connections in last 2 seconds
        self.time_window_2s = deque(maxlen=1000)

    def extract_features(self, connection):
        """Extract all 41 features from a connection"""

        features = {}

        # Basic features
        features['duration'] = connection.duration
        features['protocol_type'] = self.encode_protocol(connection.protocol)
        features['service'] = self.encode_service(connection.service)
        features['flag'] = self.encode_flag(connection.flag)
        features['src_bytes'] = connection.src_bytes
        features['dst_bytes'] = connection.dst_bytes
        # ... more basic features

        # Time-based features
        features['count'] = self.compute_count(connection)
        features['srv_count'] = self.compute_srv_count(connection)
        features['serror_rate'] = self.compute_serror_rate(connection)
        features['same_srv_rate'] = self.compute_same_srv_rate(connection)
        # ... more time features

        # Host-based features
        features['dst_host_count'] = self.compute_dst_host_count(connection)
        # ... more host features

        # Update history
        self.connection_history.append(connection)
        self.update_time_window(connection)

        return features

    def compute_count(self, current_conn):
        """Connections to same dest IP in last 2 seconds"""
        cutoff = current_conn.timestamp - timedelta(seconds=2)

        count = sum(1 for conn in self.time_window_2s
                    if conn.dst_ip == current_conn.dst_ip
                    and conn.timestamp >= cutoff)

        return count

    def compute_serror_rate(self, current_conn):
        """SYN error rate in last 2 seconds"""
        cutoff = current_conn.timestamp - timedelta(seconds=2)

        recent_conns = [conn for conn in self.time_window_2s
                        if conn.timestamp >= cutoff]

        if len(recent_conns) == 0:
            return 0.0

        syn_errors = sum(1 for conn in recent_conns
                         if conn.flag in ['S0', 'S1', 'S2', 'S3'])

        return syn_errors / len(recent_conns)

    def update_time_window(self, connection):
        """Update 2-second sliding window"""
        self.time_window_2s.append(connection)

        # Remove old connections
        cutoff = connection.timestamp - timedelta(seconds=2)
        while (self.time_window_2s and
               self.time_window_2s[0].timestamp < cutoff):
            self.time_window_2s.popleft()
```

---

## Feature Engineering Best Practices

### 1. Domain Knowledge

```python
# Good feature engineering requires understanding the domain

# Bad feature:
features['random_value'] = np.random.rand()  # No meaning!

# Good feature:
features['bytes_ratio'] = src_bytes / (dst_bytes + 1)
# High ratio = uploading data (potential data exfiltration)
```

### 2. Handle Missing Values

```python
# Fill missing values
X['src_bytes'].fillna(0, inplace=True)

# Or use median
X['duration'].fillna(X['duration'].median(), inplace=True)
```

### 3. Feature Scaling

```python
# Always scale before training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for prediction
joblib.dump(scaler, 'scaler.pkl')
```

### 4. Consistent Encoding

```python
# Use same encoder for train and test
encoder = LabelEncoder()
encoder.fit(train_df['service'])

# Train
train_df['service'] = encoder.transform(train_df['service'])

# Test (use same encoder!)
test_df['service'] = encoder.transform(test_df['service'])
```

---

## Summary: Feature Importance Ranking

Based on Random Forest analysis:

| Rank | Feature | Importance | What it detects |
|------|---------|------------|-----------------|
| 1 | src_bytes | 0.152 | Data transfer patterns |
| 2 | dst_bytes | 0.134 | Response sizes |
| 3 | count | 0.099 | Connection frequency (DoS) |
| 4 | srv_count | 0.086 | Service-specific patterns |
| 5 | dst_host_srv_count | 0.075 | Targeted attacks |
| 6 | serror_rate | 0.063 | SYN floods |
| 7 | dst_host_same_srv_rate | 0.052 | Attack focus |
| 8 | same_srv_rate | 0.049 | Scanning behavior |
| 9 | service | 0.042 | Service exploitation |
| 10 | protocol_type | 0.040 | Protocol-based attacks |

---

**Next**: [06 - Model Training](06-model-training.md)
