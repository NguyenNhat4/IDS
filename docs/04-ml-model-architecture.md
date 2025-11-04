# ML Model Architecture cho IDS

## Tổng quan Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    IDS ML Pipeline                            │
└──────────────────────────────────────────────────────────────┘

1. Data Collection
   ┌─────────────────┐
   │ Network Traffic │
   │  (pcap files)   │
   └────────┬────────┘
            │
            ▼
2. Feature Extraction
   ┌─────────────────────────┐
   │ Extract 41 features:    │
   │ - Protocol type         │
   │ - Service               │
   │ - Connection duration   │
   │ - Bytes transferred     │
   │ - Error rates           │
   │ - Connection counts     │
   └────────┬────────────────┘
            │
            ▼
3. Data Preprocessing
   ┌─────────────────────────┐
   │ - Handle missing values │
   │ - Encode categorical    │
   │ - Normalize numerical   │
   │ - Balance classes       │
   └────────┬────────────────┘
            │
            ▼
4. Feature Selection
   ┌─────────────────────────┐
   │ - Remove low variance   │
   │ - Feature importance    │
   │ - Correlation analysis  │
   └────────┬────────────────┘
            │
            ▼
5. Model Training
   ┌─────────────────────────┐
   │ - Split train/test      │
   │ - Train classifier      │
   │ - Cross-validation      │
   │ - Hyperparameter tuning │
   └────────┬────────────────┘
            │
            ▼
6. Model Evaluation
   ┌─────────────────────────┐
   │ - Accuracy, Precision   │
   │ - Recall, F1-Score      │
   │ - Confusion Matrix      │
   │ - ROC-AUC               │
   └────────┬────────────────┘
            │
            ▼
7. Deployment
   ┌─────────────────────────┐
   │ - Save model (pickle)   │
   │ - Integrate to FastAPI  │
   │ - Real-time prediction  │
   └─────────────────────────┘
```

---

## NSL-KDD Dataset

### Dataset Structure

**Files:**
- `KDDTrain+.txt` - Training data (125,973 records)
- `KDDTest+.txt` - Test data (22,544 records)
- `Field Names.txt` - Feature names

**Format:**
```csv
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal,20
```

### 41 Features

#### 1. Basic Features (9 features)
```python
basic_features = {
    'duration': int,           # Length of connection (seconds)
    'protocol_type': str,      # tcp, udp, icmp
    'service': str,            # http, ftp, smtp, etc. (70 services)
    'flag': str,               # Status (SF, S0, REJ, etc.)
    'src_bytes': int,          # Bytes from source to dest
    'dst_bytes': int,          # Bytes from dest to source
    'land': int,               # 1 if src & dest IP/port are same
    'wrong_fragment': int,     # Number of wrong fragments
    'urgent': int,             # Number of urgent packets
}
```

**Example:**
```python
{
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',           # SF = Normal connection termination
    'src_bytes': 181,
    'dst_bytes': 5450,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0
}
```

#### 2. Content Features (13 features)
```python
content_features = {
    'hot': int,                      # Number of "hot" indicators
    'num_failed_logins': int,        # Failed login attempts
    'logged_in': int,                # 1 if successfully logged in
    'num_compromised': int,          # Number of compromised conditions
    'root_shell': int,               # 1 if root shell obtained
    'su_attempted': int,             # 1 if su root attempted
    'num_root': int,                 # Number of root accesses
    'num_file_creations': int,       # Number of file creation operations
    'num_shells': int,               # Number of shell prompts
    'num_access_files': int,         # Number of access control files
    'num_outbound_cmds': int,        # Number of outbound commands
    'is_host_login': int,            # 1 if host-based login
    'is_guest_login': int,           # 1 if guest login
}
```

**Use case:**
- `num_failed_logins` > 3 → Likely brute force (R2L)
- `root_shell` = 1 → Privilege escalation (U2R)
- `num_file_creations` > 0 → Suspicious activity

#### 3. Time-based Traffic Features (9 features)
Features computed using a 2-second time window:

```python
time_features = {
    'count': int,                    # Connections to same host (last 2s)
    'srv_count': int,                # Connections to same service (last 2s)
    'serror_rate': float,            # % connections with SYN errors
    'srv_serror_rate': float,        # % connections with SYN errors (same service)
    'rerror_rate': float,            # % connections with REJ errors
    'srv_rerror_rate': float,        # % connections with REJ errors (same service)
    'same_srv_rate': float,          # % connections to same service
    'diff_srv_rate': float,          # % connections to different services
    'srv_diff_host_rate': float,     # % connections to different hosts (same service)
}
```

**Example - Normal traffic:**
```python
{
    'count': 5,                   # 5 connections in 2s (reasonable)
    'srv_count': 5,
    'serror_rate': 0.0,           # No errors
    'same_srv_rate': 1.0,         # All to same service
    'diff_srv_rate': 0.0,
}
```

**Example - DoS attack:**
```python
{
    'count': 511,                 # 511 connections in 2s! (attack!)
    'srv_count': 511,
    'serror_rate': 0.99,          # 99% errors! (SYN flood)
    'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0,
}
```

**Example - Port scan (Probe):**
```python
{
    'count': 100,
    'srv_count': 10,
    'diff_srv_rate': 0.9,         # 90% to different services (scanning!)
}
```

#### 4. Host-based Traffic Features (10 features)
Features computed using a 100-connection window:

```python
host_features = {
    'dst_host_count': int,                # Connections to dest host (last 100)
    'dst_host_srv_count': int,            # Same service to dest (last 100)
    'dst_host_same_srv_rate': float,      # % same service
    'dst_host_diff_srv_rate': float,      # % diff service
    'dst_host_same_src_port_rate': float, # % same source port
    'dst_host_srv_diff_host_rate': float, # % diff source hosts
    'dst_host_serror_rate': float,        # % SYN errors
    'dst_host_srv_serror_rate': float,    # % SYN errors (same service)
    'dst_host_rerror_rate': float,        # % REJ errors
    'dst_host_srv_rerror_rate': float,    # % REJ errors (same service)
}
```

### Labels (Target)

**5 Classes:**
```python
labels = {
    0: 'normal',
    1: 'DoS',      # Denial of Service
    2: 'Probe',    # Surveillance/Probing
    3: 'R2L',      # Remote to Local
    4: 'U2R',      # User to Root
}
```

**Attack subtypes in NSL-KDD:**
```python
DoS_attacks = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
Probe_attacks = ['ipsweep', 'nmap', 'portsweep', 'satan']
R2L_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop',
               'phf', 'spy', 'warezclient', 'warezmaster']
U2R_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
```

---

## Model Selection

### Comparison of Algorithms

| Algorithm | Accuracy | Training Time | Prediction Time | Pros | Cons |
|-----------|----------|---------------|-----------------|------|------|
| **Random Forest** | ~99% | Medium | Fast | High accuracy, interpretable | Memory intensive |
| **Decision Tree** | ~95% | Fast | Very Fast | Simple, interpretable | Overfitting |
| **SVM** | ~98% | Slow | Medium | Good with high-dim data | Slow on large datasets |
| **Naive Bayes** | ~90% | Very Fast | Very Fast | Simple, fast | Assumes independence |
| **Neural Network** | ~99% | Slow | Fast | Very high accuracy | Black box, needs GPU |
| **KNN** | ~94% | Fast | Very Slow | Simple | Slow prediction |

**Chọn Random Forest vì:**
1. Accuracy cao (~99%)
2. Prediction time nhanh (real-time)
3. Feature importance (giải thích được)
4. Không cần GPU
5. Robust với overfitting

---

## Random Forest Classifier

### Concept

```
Random Forest = Ensemble of Decision Trees

         ┌──── Decision Tree 1 ────┐
         │   duration > 5?         │
         │   /           \         │
         │ count>10?    Normal     │
         │  /    \                 │
         │ DoS  Probe              │
         └─────────────────────────┘
                  │
                  │ Vote
                  ▼
         ┌──── Decision Tree 2 ────┐
         │   src_bytes > 1000?     │
         │   /               \     │
         │ service='http'?  R2L    │
         │  /          \           │
         │ Normal    Probe         │
         └─────────────────────────┘
                  │
                  │ Vote
                  ▼
         ┌──── Decision Tree 3 ────┐
         │   serror_rate > 0.5?    │
         │   /                 \   │
         │ DoS             Normal  │
         └─────────────────────────┘

Final Prediction = Majority Vote
Tree 1: DoS
Tree 2: Probe
Tree 3: DoS
→ Result: DoS (2 votes)
```

### Hyperparameters

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=20,            # Max depth of each tree
    min_samples_split=10,    # Min samples to split node
    min_samples_leaf=4,      # Min samples in leaf node
    max_features='sqrt',     # Features to consider per split
    random_state=42,         # Reproducibility
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle imbalanced data
)
```

**Parameter tuning:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## Data Preprocessing

### 1. Load Data

```python
import pandas as pd

# Column names
columns = ['duration', 'protocol_type', 'service', 'flag',
           'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
           'urgent', 'hot', 'num_failed_logins', 'logged_in',
           # ... all 41 features ...
           'label', 'difficulty']

# Load dataset
train_df = pd.read_csv('KDDTrain+.txt', names=columns)
test_df = pd.read_csv('KDDTest+.txt', names=columns)

print(train_df.shape)  # (125973, 43)
print(train_df.head())
```

### 2. Encode Labels

```python
# Map attack types to categories
attack_mapping = {
    'normal': 'normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
    'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe',
    'portsweep': 'Probe', 'satan': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L',
    'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R',
    'perl': 'U2R', 'rootkit': 'U2R'
}

train_df['category'] = train_df['label'].map(attack_mapping)

# Encode to numbers
label_encoder = {
    'normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4
}
train_df['label_encoded'] = train_df['category'].map(label_encoder)
```

### 3. Handle Categorical Features

```python
from sklearn.preprocessing import LabelEncoder

# Categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']

# Encode each
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

# protocol_type: tcp=2, udp=1, icmp=0
# service: http=25, ftp=15, smtp=45, ...
# flag: SF=10, S0=8, REJ=7, ...
```

### 4. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Select numerical columns
numerical_cols = ['duration', 'src_bytes', 'dst_bytes',
                  'count', 'srv_count', ...]

# Standardize (mean=0, std=1)
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Before:
# duration: [0, 1, 5, 100, 10000]
# After:
# duration: [-0.5, -0.4, -0.3, 0.2, 2.5]
```

### 5. Handle Class Imbalance

**Problem:**
```python
train_df['category'].value_counts()
# normal        67343  (53%)
# DoS           45927  (36%)
# Probe         11656  (9%)
# R2L            995   (1%)  ← Imbalanced!
# U2R             52   (0%)  ← Very imbalanced!
```

**Solution 1: SMOTE (Synthetic Minority Over-sampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# After SMOTE:
# All classes have ~67,000 samples
```

**Solution 2: Class Weights**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
# {0: 0.5, 1: 0.7, 2: 2.8, 3: 32.0, 4: 600.0}
# U2R gets 600x weight!

model = RandomForestClassifier(class_weight='balanced')
```

---

## Training Pipeline

### Complete Training Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load data
columns = [...]  # 41 feature names + label
train_df = pd.read_csv('KDDTrain+.txt', names=columns)

# 2. Preprocessing
# Map labels
attack_mapping = {...}
train_df['category'] = train_df['label'].map(attack_mapping)

# Encode labels
label_map = {'normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
train_df['label_encoded'] = train_df['category'].map(label_map)

# Encode categorical
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

# 3. Feature selection
X = train_df.drop(['label', 'category', 'label_encoded', 'difficulty'], axis=1)
y = train_df['label_encoded']

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred,
                          target_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R']))

# 8. Save model
joblib.dump(model, 'ids_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved!")
```

---

## Model Evaluation

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
            yticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

**Example output:**
```
              Predicted
           Normal  DoS  Probe  R2L  U2R
Actual
Normal     13200   50    15     2    0
DoS           30  9100   10     0    0
Probe         20    15  2280    0    0
R2L            5     0     2   185   3
U2R            0     0     0     2    8
```

### Performance Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.4f}")   # 0.9890
print(f"Precision: {precision:.4f}")  # 0.9885
print(f"Recall:    {recall:.4f}")     # 0.9890
print(f"F1-Score:  {f1:.4f}")         # 0.9887
```

### Feature Importance

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

**Top 10 features:**
```
feature                          importance
src_bytes                        0.1523
dst_bytes                        0.1342
count                            0.0987
srv_count                        0.0856
dst_host_srv_count               0.0745
serror_rate                      0.0632
dst_host_same_srv_rate           0.0521
same_srv_rate                    0.0487
service                          0.0423
protocol_type                    0.0398
```

---

**Next**: [05 - Feature Engineering](05-feature-engineering.md)
