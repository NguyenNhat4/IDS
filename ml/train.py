"""
IDS ML Model Training Script
Trains a Random Forest classifier on NSL-KDD dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import sys

# Column names for NSL-KDD dataset
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

# Attack type mapping
ATTACK_MAPPING = {
    'normal': 'Normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
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
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R'
}

LABEL_MAP = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
TARGET_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def load_data():
    """Load NSL-KDD datasets"""
    print("=" * 60)
    print("Loading NSL-KDD Dataset...")
    print("=" * 60)

    dataset_dir = 'ml/dataset'

    train_file = os.path.join(dataset_dir, 'KDDTrain+.txt')
    test_file = os.path.join(dataset_dir, 'KDDTest+.txt')

    if not os.path.exists(train_file):
        print(f"\n❌ Error: Training file not found: {train_file}")
        print("\n📥 Please download NSL-KDD dataset:")
        print("   1. Visit: https://www.unb.ca/cic/datasets/nsl.html")
        print("   2. Download KDDTrain+.txt and KDDTest+.txt")
        print(f"   3. Place files in: {os.path.abspath(dataset_dir)}/")
        sys.exit(1)

    train_df = pd.read_csv(train_file, names=COLUMNS)
    test_df = pd.read_csv(test_file, names=COLUMNS)

    print(f"✓ Train set loaded: {train_df.shape}")
    print(f"✓ Test set loaded:  {test_df.shape}")

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess datasets"""
    print("\n" + "=" * 60)
    print("Preprocessing Data...")
    print("=" * 60)

    # Map labels to categories
    print("\n1. Mapping attack types to categories...")
    train_df['category'] = train_df['label'].map(ATTACK_MAPPING)
    test_df['category'] = test_df['label'].map(ATTACK_MAPPING)

    # Handle unknown labels
    train_df['category'].fillna('Unknown', inplace=True)
    test_df['category'].fillna('Unknown', inplace=True)

    # Remove unknown labels
    train_df = train_df[train_df['category'] != 'Unknown']
    test_df = test_df[test_df['category'] != 'Unknown']

    print("   Category distribution (train):")
    print(train_df['category'].value_counts())

    # Encode labels
    print("\n2. Encoding labels...")
    train_df['label_encoded'] = train_df['category'].map(LABEL_MAP)
    test_df['label_encoded'] = test_df['category'].map(LABEL_MAP)

    # Encode categorical features
    print("\n3. Encoding categorical features...")
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        # Transform
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
        print(f"   - {col}: {len(le.classes_)} unique values")

    # Select features
    print("\n4. Selecting features...")
    feature_cols = [c for c in COLUMNS if c not in ['label', 'difficulty']]
    X_train = train_df[feature_cols]
    y_train = train_df['label_encoded']
    X_test = test_df[feature_cols]
    y_test = test_df['label_encoded']

    print(f"   Features: {len(feature_cols)}")

    # Scale features
    print("\n5. Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("   ✓ Features standardized (mean=0, std=1)")

    # Save preprocessing artifacts
    print("\n6. Saving preprocessing artifacts...")
    output_dir = 'ml/trained_models'
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'feature_names.pkl'))
    print(f"   ✓ Saved to {output_dir}/")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "=" * 60)
    print("Training Random Forest Model...")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\nModel Configuration:")
    print(f"  - Estimators: {model.n_estimators}")
    print(f"  - Max Depth: {model.max_depth}")
    print(f"  - Class Weight: {model.class_weight}")

    print("\n🚀 Training started...")
    model.fit(X_train, y_train)
    print("✓ Training completed!")

    # Save model
    model_path = 'ml/trained_models/ids_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)

    # Predictions
    print("\n📊 Making predictions on test set...")
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Classification Report
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=4))

    # Confusion Matrix
    print("\n" + "=" * 60)
    print("Confusion Matrix:")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print("\n        ", "  ".join(f"{name:>8}" for name in TARGET_NAMES))
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i]:>8}", "  ".join(f"{val:>8}" for val in row))

    # Feature Importance
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features:")
    print("=" * 60)

    feature_names = joblib.load('ml/trained_models/feature_names.pkl')
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    for i, idx in enumerate(indices, 1):
        print(f"{i:2}. {feature_names[idx]:30} {importances[idx]:.4f}")

    return accuracy


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("   IDS ML MODEL TRAINING")
    print("   Random Forest Classifier on NSL-KDD Dataset")
    print("=" * 60)

    try:
        # Load data
        train_df, test_df = load_data()

        # Preprocess
        X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

        # Train
        model = train_model(X_train, y_train)

        # Evaluate
        accuracy = evaluate_model(model, X_test, y_test)

        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n🎯 Final Accuracy: {accuracy*100:.2f}%")
        print(f"\n📁 Model saved in: ml/trained_models/")
        print(f"\n🚀 Ready to use! Start backend:")
        print(f"   python backend/main.py")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
