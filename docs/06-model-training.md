# Model Training Guide

## Tổng quan

File này hướng dẫn chi tiết cách train ML model cho IDS từ đầu đến cuối.

---

## Step 1: Dataset Preparation

### Download NSL-KDD Dataset

```bash
# Create dataset directory
mkdir -p ml/dataset

# Download NSL-KDD
cd ml/dataset

# Option 1: Download từ UNB
wget https://www.unb.ca/cic/datasets/nsl.html

# Option 2: Download từ Kaggle
# https://www.kaggle.com/datasets/hassan06/nslkdd

# Files needed:
# - KDDTrain+.txt
# - KDDTest+.txt
```

### Dataset Structure

```
ml/dataset/
├── KDDTrain+.txt      # 125,973 records
├── KDDTest+.txt       # 22,544 records
└── Field Names.txt    # Column names
```

---

## Step 2: Exploratory Data Analysis (EDA)

### Load and Inspect Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Column names
columns = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Load data
train_df = pd.read_csv('ml/dataset/KDDTrain+.txt', names=columns)
test_df = pd.read_csv('ml/dataset/KDDTest+.txt', names=columns)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Display first rows
train_df.head()
```

### Check Data Distribution

```python
# Check for missing values
print(train_df.isnull().sum())
# NSL-KDD has no missing values!

# Data types
print(train_df.dtypes)

# Statistical summary
print(train_df.describe())
```

### Analyze Labels

```python
# Attack types distribution
print(train_df['label'].value_counts())

# Output:
# normal            67343
# neptune           41214  (DoS)
# satan              3633  (Probe)
# ipsweep            3599  (Probe)
# portsweep          2931  (Probe)
# smurf              2646  (DoS)
# nmap               1493  (Probe)
# back                956  (DoS)
# teardrop            892  (DoS)
# warezclient         890  (R2L)
# pod                 201  (DoS)
# guess_passwd         53  (R2L)
# buffer_overflow      30  (U2R)
# ...

# Visualize
plt.figure(figsize=(12, 6))
train_df['label'].value_counts().head(15).plot(kind='bar')
plt.title('Top 15 Attack Types')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### Map to Categories

```python
# Map specific attacks to categories
attack_mapping = {
    'normal': 'Normal',

    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
    'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS',
    'mailbomb': 'DoS',

    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',

    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',

    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R',
    'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

train_df['category'] = train_df['label'].map(attack_mapping)
test_df['category'] = test_df['label'].map(attack_mapping)

# Category distribution
print(train_df['category'].value_counts())

# Visualize
plt.figure(figsize=(10, 6))
train_df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Attack Categories Distribution')
plt.ylabel('')
plt.show()
```

### Feature Correlation Analysis

```python
# Select numerical features only
numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('difficulty')  # Remove non-feature column

# Correlation matrix
corr_matrix = train_df[numerical_features].corr()

# Heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Find highly correlated pairs
threshold = 0.9
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print("Highly correlated features:")
for feat1, feat2, corr in high_corr_pairs:
    print(f"{feat1} <-> {feat2}: {corr:.3f}")
```

---

## Step 3: Data Preprocessing

### Complete Preprocessing Script

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_data(train_df, test_df):
    """
    Preprocess NSL-KDD dataset
    """

    # 1. Map labels to categories
    attack_mapping = {...}  # As defined above
    train_df['category'] = train_df['label'].map(attack_mapping)
    test_df['category'] = test_df['label'].map(attack_mapping)

    # 2. Encode target variable
    label_map = {
        'Normal': 0,
        'DoS': 1,
        'Probe': 2,
        'R2L': 3,
        'U2R': 4
    }

    train_df['label_encoded'] = train_df['category'].map(label_map)
    test_df['label_encoded'] = test_df['category'].map(label_map)

    # 3. Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()

        # Fit on combined data to handle all values
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)

        # Transform
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

        encoders[col] = le

    # Save encoders
    joblib.dump(encoders, 'ml/trained_models/encoders.pkl')

    # 4. Select features
    feature_cols = [col for col in train_df.columns
                    if col not in ['label', 'category', 'label_encoded', 'difficulty']]

    X_train = train_df[feature_cols]
    y_train = train_df['label_encoded']

    X_test = test_df[feature_cols]
    y_test = test_df['label_encoded']

    # 5. Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, 'ml/trained_models/scaler.pkl')

    # Save feature names
    joblib.dump(feature_cols, 'ml/trained_models/feature_names.pkl')

    return X_train_scaled, y_train, X_test_scaled, y_test

# Run preprocessing
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
```

---

## Step 4: Model Training

### Train Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

# Initialize model
model = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=20,           # Max tree depth
    min_samples_split=10,   # Min samples to split
    min_samples_leaf=4,     # Min samples in leaf
    max_features='sqrt',    # Features per split
    class_weight='balanced', # Handle imbalanced data
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbose=1               # Show progress
)

# Train
print("Training Random Forest...")
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# Save model
joblib.dump(model, 'ml/trained_models/ids_model.pkl')
print("Model saved to ml/trained_models/ids_model.pkl")
```

---

## Step 5: Model Evaluation

### Predictions

```python
# Predict on test set
print("Making predictions...")
start_time = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"Prediction completed in {prediction_time:.2f} seconds")
print(f"Average prediction time per sample: {prediction_time/len(X_test)*1000:.2f} ms")
```

### Classification Report

```python
# Detailed metrics
target_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:")
print(report)

# Example output:
#               precision    recall  f1-score   support
#
#       Normal       0.99      0.99      0.99      9711
#          DoS       0.99      0.99      0.99      7458
#        Probe       0.97      0.94      0.96      2421
#          R2L       0.92      0.88      0.90       2754
#          U2R       0.85      0.82      0.84       200
#
#     accuracy                           0.98     22544
#    macro avg       0.94      0.92      0.93     22544
# weighted avg       0.98      0.98      0.98     22544
```

### Confusion Matrix

```python
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('ml/confusion_matrix.png')
plt.show()

print("\nConfusion Matrix:")
print(cm)
```

### Accuracy Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
```

### Per-Class Metrics

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=[0, 1, 2, 3, 4]
)

metrics_df = pd.DataFrame({
    'Class': target_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\nPer-Class Metrics:")
print(metrics_df)

# Visualize
metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(12, 6))
plt.title('Per-Class Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('ml/per_class_metrics.png')
plt.show()
```

### ROC-AUC Curves

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
y_pred_proba = model.predict_proba(X_test)

# Compute ROC curve for each class
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(target_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2,
             label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Multi-class')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ml/roc_curves.png')
plt.show()
```

---

## Step 6: Feature Importance Analysis

```python
# Get feature importance
feature_names = joblib.load('ml/trained_models/feature_names.pkl')
importances = model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Display top 20
print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20))

# Visualize
plt.figure(figsize=(12, 10))
top_20 = feature_importance_df.head(20)
plt.barh(range(len(top_20)), top_20['Importance'])
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ml/feature_importance.png')
plt.show()
```

---

## Step 7: Hyperparameter Tuning (Optional)

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2']
}

# Grid search with cross-validation
print("Starting Grid Search...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# Best score
print(f"\nBest Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# Train with best parameters
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'ml/trained_models/ids_model_tuned.pkl')
```

---

## Complete Training Script

```python
# ml/train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Column names
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_MAPPING = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R'
}

LABEL_MAP = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
TARGET_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv('ml/dataset/KDDTrain+.txt', names=COLUMNS)
    test_df = pd.read_csv('ml/dataset/KDDTest+.txt', names=COLUMNS)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def preprocess(train_df, test_df):
    print("Preprocessing data...")

    # Map labels
    train_df['category'] = train_df['label'].map(ATTACK_MAPPING)
    test_df['category'] = test_df['label'].map(ATTACK_MAPPING)

    train_df['label_encoded'] = train_df['category'].map(LABEL_MAP)
    test_df['label_encoded'] = test_df['category'].map(LABEL_MAP)

    # Encode categorical
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    # Features
    feature_cols = [c for c in COLUMNS if c not in ['label', 'difficulty']]
    X_train = train_df[feature_cols]
    y_train = train_df['label_encoded']
    X_test = test_df[feature_cols]
    y_test = test_df['label_encoded']

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save artifacts
    os.makedirs('ml/trained_models', exist_ok=True)
    joblib.dump(encoders, 'ml/trained_models/encoders.pkl')
    joblib.dump(scaler, 'ml/trained_models/scaler.pkl')
    joblib.dump(feature_cols, 'ml/trained_models/feature_names.pkl')

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'ml/trained_models/ids_model.pkl')
    print("Model saved!")
    return model

def evaluate(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

if __name__ == '__main__':
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = preprocess(train_df, test_df)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    print("\nTraining complete!")
```

**Run training:**
```bash
python ml/train.py
```

---

**Next**: [07 - Deployment](07-deployment.md)
