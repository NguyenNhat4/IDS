from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import get_model

router = APIRouter()


class ConnectionFeatures(BaseModel):
    """
    Network connection features for IDS detection
    Based on NSL-KDD dataset (41 features)
    """
    # Basic features
    duration: float = Field(0, description="Connection duration in seconds")
    protocol_type: str = Field("tcp", description="Protocol type: tcp, udp, icmp")
    service: str = Field("http", description="Network service: http, ftp, smtp, etc.")
    flag: str = Field("SF", description="Connection status flag")
    src_bytes: int = Field(0, description="Bytes sent from source to destination")
    dst_bytes: int = Field(0, description="Bytes sent from destination to source")
    land: int = Field(0, description="1 if connection is from/to same host/port")
    wrong_fragment: int = Field(0, description="Number of wrong fragments")
    urgent: int = Field(0, description="Number of urgent packets")

    # Content features
    hot: int = Field(0, description="Number of hot indicators")
    num_failed_logins: int = Field(0, description="Number of failed login attempts")
    logged_in: int = Field(0, description="1 if successfully logged in, 0 otherwise")
    num_compromised: int = Field(0, description="Number of compromised conditions")
    root_shell: int = Field(0, description="1 if root shell is obtained")
    su_attempted: int = Field(0, description="1 if su root command attempted")
    num_root: int = Field(0, description="Number of root accesses")
    num_file_creations: int = Field(0, description="Number of file creation operations")
    num_shells: int = Field(0, description="Number of shell prompts")
    num_access_files: int = Field(0, description="Number of operations on access control files")
    num_outbound_cmds: int = Field(0, description="Number of outbound commands in FTP session")
    is_host_login: int = Field(0, description="1 if login belongs to host list")
    is_guest_login: int = Field(0, description="1 if login is guest")

    # Time-based traffic features (computed using 2-second time window)
    count: int = Field(0, description="Connections to same host in past 2 seconds")
    srv_count: int = Field(0, description="Connections to same service in past 2 seconds")
    serror_rate: float = Field(0.0, description="% of connections with SYN errors")
    srv_serror_rate: float = Field(0.0, description="% of connections with SYN errors (same service)")
    rerror_rate: float = Field(0.0, description="% of connections with REJ errors")
    srv_rerror_rate: float = Field(0.0, description="% of connections with REJ errors (same service)")
    same_srv_rate: float = Field(0.0, description="% of connections to same service")
    diff_srv_rate: float = Field(0.0, description="% of connections to different services")
    srv_diff_host_rate: float = Field(0.0, description="% of connections to different hosts")

    # Host-based traffic features (computed using 100-connection window)
    dst_host_count: int = Field(0, description="Count of connections with same dest host")
    dst_host_srv_count: int = Field(0, description="Count of connections with same dest host/service")
    dst_host_same_srv_rate: float = Field(0.0, description="% same service")
    dst_host_diff_srv_rate: float = Field(0.0, description="% different services")
    dst_host_same_src_port_rate: float = Field(0.0, description="% same source port")
    dst_host_srv_diff_host_rate: float = Field(0.0, description="% different source hosts")
    dst_host_serror_rate: float = Field(0.0, description="% connections with SYN errors")
    dst_host_srv_serror_rate: float = Field(0.0, description="% connections with SYN errors (same service)")
    dst_host_rerror_rate: float = Field(0.0, description="% connections with REJ errors")
    dst_host_srv_rerror_rate: float = Field(0.0, description="% connections with REJ errors (same service)")

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 200,
                "dst_bytes": 5000,
                "count": 5,
                "srv_count": 5,
                "serror_rate": 0.0,
                "same_srv_rate": 1.0
            }
        }


@router.post("/predict")
async def predict_attack(features: ConnectionFeatures):
    """
    Detect if a network connection is an attack

    Args:
        features: Connection features

    Returns:
        Detection result with prediction, confidence, and probabilities
    """
    try:
        start_time = time.time()

        # Get model
        model = get_model()

        # Convert to dict
        features_dict = features.dict()

        # Predict
        result = model.predict(features_dict)

        # Add timing information
        result['prediction_time_ms'] = (time.time() - start_time) * 1000

        # Add input summary
        result['input_summary'] = {
            'protocol': features.protocol_type,
            'service': features.service,
            'src_bytes': features.src_bytes,
            'dst_bytes': features.dst_bytes,
            'count': features.count
        }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/stats")
async def get_model_stats():
    """
    Get model statistics and information

    Returns:
        Model metadata including type, features, and configuration
    """
    try:
        model = get_model()
        info = model.get_model_info()

        return {
            'status': 'operational',
            'model': info,
            'supported_attacks': ['DoS', 'Probe', 'R2L', 'U2R'],
            'features_count': info['n_features'],
            'classes_count': info['n_classes']
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@router.post("/batch_predict")
async def batch_predict(connections: list[ConnectionFeatures]):
    """
    Predict multiple connections at once

    Args:
        connections: List of connection features

    Returns:
        List of predictions
    """
    try:
        model = get_model()
        results = []

        for features in connections:
            features_dict = features.dict()
            result = model.predict(features_dict)
            results.append(result)

        return {
            'total': len(results),
            'predictions': results,
            'summary': {
                'attacks_detected': sum(1 for r in results if r['is_attack']),
                'normal_traffic': sum(1 for r in results if not r['is_attack']),
                'attack_types': {
                    attack_type: sum(1 for r in results if r['prediction'] == attack_type)
                    for attack_type in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
                }
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )
