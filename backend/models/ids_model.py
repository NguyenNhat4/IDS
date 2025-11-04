import joblib
import numpy as np
from pathlib import Path
import os


class IDSModel:
    """
    IDS Machine Learning Model
    Loads trained Random Forest model and provides prediction interface
    """

    def __init__(self):
        # Get model directory
        current_dir = Path(__file__).parent.parent.parent
        model_dir = current_dir / "ml" / "trained_models"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                "Please train the model first by running: python ml/train.py"
            )

        # Load model
        model_path = model_dir / "ids_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please train the model first by running: python ml/train.py"
            )

        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")

        # Load scaler
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded")
        else:
            self.scaler = None
            print("⚠ Scaler not found, will use raw features")

        # Load encoders
        encoders_path = model_dir / "encoders.pkl"
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
            print(f"✓ Encoders loaded")
        else:
            self.encoders = None
            print("⚠ Encoders not found, will use raw categorical values")

        # Load feature names
        feature_names_path = model_dir / "feature_names.pkl"
        if feature_names_path.exists():
            self.feature_names = joblib.load(feature_names_path)
            print(f"✓ Feature names loaded ({len(self.feature_names)} features)")
        else:
            # Default feature names from NSL-KDD
            self.feature_names = self._get_default_feature_names()
            print(f"⚠ Using default feature names")

        # Label mapping
        self.label_map = {
            0: 'Normal',
            1: 'DoS',
            2: 'Probe',
            3: 'R2L',
            4: 'U2R'
        }

        print(f"✓ IDS Model initialized successfully")

    def _get_default_feature_names(self):
        """Get default NSL-KDD feature names"""
        return [
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
            'dst_host_srv_rerror_rate'
        ]

    def preprocess_features(self, features: dict):
        """
        Preprocess input features for prediction

        Args:
            features: Dictionary of feature values

        Returns:
            numpy array ready for model prediction
        """
        # Encode categorical features if encoders available
        if self.encoders:
            if 'protocol_type' in features and isinstance(features['protocol_type'], str):
                try:
                    features['protocol_type'] = self.encoders['protocol_type'].transform(
                        [features['protocol_type']]
                    )[0]
                except:
                    # Default to tcp (most common)
                    features['protocol_type'] = 0

            if 'service' in features and isinstance(features['service'], str):
                try:
                    features['service'] = self.encoders['service'].transform(
                        [features['service']]
                    )[0]
                except:
                    # Default to http
                    features['service'] = 0

            if 'flag' in features and isinstance(features['flag'], str):
                try:
                    features['flag'] = self.encoders['flag'].transform(
                        [features['flag']]
                    )[0]
                except:
                    # Default to SF
                    features['flag'] = 0

        # Create feature vector in correct order
        feature_vector = []
        for name in self.feature_names:
            value = features.get(name, 0)
            # Convert to float
            try:
                value = float(value)
            except:
                value = 0.0
            feature_vector.append(value)

        # Convert to numpy array
        feature_vector = np.array([feature_vector])

        # Scale if scaler available
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)

        return feature_vector

    def predict(self, features: dict):
        """
        Predict attack type from features

        Args:
            features: Dictionary of connection features

        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess
            X = self.preprocess_features(features)

            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Get label
            label = self.label_map.get(prediction, 'Unknown')

            # Get confidence
            confidence = float(probabilities[prediction])

            # All class probabilities
            class_probabilities = {
                self.label_map[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }

            return {
                'prediction': label,
                'prediction_id': int(prediction),
                'confidence': confidence,
                'probabilities': class_probabilities,
                'is_attack': label != 'Normal'
            }

        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def get_model_info(self):
        """Get model information"""
        info = {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'classes': list(self.label_map.values()),
            'n_classes': len(self.label_map)
        }

        # Add Random Forest specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth

        return info


# Global model instance (singleton)
_model_instance = None


def get_model():
    """Get or create model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = IDSModel()
    return _model_instance
