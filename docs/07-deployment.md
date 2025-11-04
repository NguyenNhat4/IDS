# Deployment Guide - IDS Web Application

## Tổng quan

File này hướng dẫn deploy ML model vào web application sử dụng FastAPI backend và HTML/CSS/JS frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Frontend (HTML/CSS/JS)                          │  │
│  │  - Dashboard                                     │  │
│  │  - Attack Simulator                              │  │
│  │  - Real-time Visualization                       │  │
│  └──────────────────┬───────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────┘
                      │ HTTP Requests (fetch API)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  REST API Endpoints                              │  │
│  │  - /predict (detect attack)                      │  │
│  │  - /simulate (simulate attacks)                  │  │
│  │  - /stats (get statistics)                       │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  ML Model                                        │  │
│  │  - Load trained model (pkl)                      │  │
│  │  - Feature extraction                            │  │
│  │  - Prediction                                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Backend Setup (FastAPI)

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic python-multipart
```

### Create requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
```

---

## FastAPI Application Structure

### Main Application (backend/main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes import detection, attack_simulator

# Initialize FastAPI app
app = FastAPI(
    title="IDS Application",
    description="Intrusion Detection System with ML",
    version="1.0.0"
)

# CORS middleware (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Include routers
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(attack_simulator.router, prefix="/api", tags=["Simulator"])

@app.get("/")
async def root():
    return {"message": "IDS Application API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### Model Loader (backend/models/ids_model.py)

```python
import joblib
import numpy as np
from pathlib import Path

class IDSModel:
    def __init__(self):
        model_dir = Path("ml/trained_models")

        # Load model
        self.model = joblib.load(model_dir / "ids_model.pkl")

        # Load scaler
        self.scaler = joblib.load(model_dir / "scaler.pkl")

        # Load encoders
        self.encoders = joblib.load(model_dir / "encoders.pkl")

        # Load feature names
        self.feature_names = joblib.load(model_dir / "feature_names.pkl")

        # Label mapping
        self.label_map = {
            0: 'Normal',
            1: 'DoS',
            2: 'Probe',
            3: 'R2L',
            4: 'U2R'
        }

    def preprocess_features(self, features: dict):
        """
        Preprocess input features for prediction
        """
        # Encode categorical features
        if 'protocol_type' in features:
            features['protocol_type'] = self.encoders['protocol_type'].transform(
                [features['protocol_type']]
            )[0]

        if 'service' in features:
            features['service'] = self.encoders['service'].transform(
                [features['service']]
            )[0]

        if 'flag' in features:
            features['flag'] = self.encoders['flag'].transform(
                [features['flag']]
            )[0]

        # Create feature vector in correct order
        feature_vector = [features.get(name, 0) for name in self.feature_names]

        # Scale
        feature_vector = self.scaler.transform([feature_vector])

        return feature_vector

    def predict(self, features: dict):
        """
        Predict attack type from features
        """
        # Preprocess
        X = self.preprocess_features(features)

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get label
        label = self.label_map[prediction]

        # Get confidence
        confidence = float(probabilities[prediction])

        # All class probabilities
        class_probabilities = {
            self.label_map[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'is_attack': label != 'Normal'
        }

# Global model instance
ids_model = IDSModel()
```

### Detection Router (backend/routes/detection.py)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from models.ids_model import ids_model
import time

router = APIRouter()

class ConnectionFeatures(BaseModel):
    duration: float = 0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: int = 0
    dst_bytes: int = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0

@router.post("/predict")
async def predict_attack(features: ConnectionFeatures):
    """
    Detect if connection is an attack
    """
    try:
        start_time = time.time()

        # Convert to dict
        features_dict = features.dict()

        # Predict
        result = ids_model.predict(features_dict)

        # Add timing
        result['prediction_time_ms'] = (time.time() - start_time) * 1000

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_model_stats():
    """
    Get model statistics
    """
    return {
        'model_type': 'Random Forest',
        'n_estimators': ids_model.model.n_estimators,
        'n_features': len(ids_model.feature_names),
        'classes': list(ids_model.label_map.values()),
    }
```

### Attack Simulator Router (backend/routes/attack_simulator.py)

```python
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import random

router = APIRouter()

class SimulatedConnection(BaseModel):
    type: str
    features: dict
    description: str

@router.get("/simulate/{attack_type}")
async def simulate_attack(attack_type: str):
    """
    Generate simulated attack features
    """

    if attack_type == "dos":
        return {
            'type': 'DoS',
            'description': 'SYN Flood Attack - Overwhelming server with connection requests',
            'features': {
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'S0',  # SYN, no response
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 511,  # Very high!
                'srv_count': 511,
                'serror_rate': 0.99,  # 99% errors
                'same_srv_rate': 1.0,
                'dst_host_count': 255,
                'dst_host_srv_count': 255,
            }
        }

    elif attack_type == "probe":
        return {
            'type': 'Probe',
            'description': 'Port Scan - Scanning for open ports',
            'features': {
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'private',
                'flag': 'REJ',  # Rejected
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 100,
                'srv_count': 10,
                'diff_srv_rate': 0.9,  # Scanning different services
                'rerror_rate': 0.8,  # Many rejections
                'dst_host_diff_srv_rate': 0.9,
            }
        }

    elif attack_type == "r2l":
        return {
            'type': 'R2L',
            'description': 'Brute Force Login - Attempting multiple passwords',
            'features': {
                'duration': 5,
                'protocol_type': 'tcp',
                'service': 'ftp',
                'flag': 'SF',
                'src_bytes': 100,
                'dst_bytes': 200,
                'num_failed_logins': 5,  # Failed login attempts
                'logged_in': 0,  # Not logged in
                'count': 10,
                'srv_count': 10,
                'same_srv_rate': 1.0,
            }
        }

    elif attack_type == "u2r":
        return {
            'type': 'U2R',
            'description': 'Buffer Overflow - Attempting privilege escalation',
            'features': {
                'duration': 10,
                'protocol_type': 'tcp',
                'service': 'telnet',
                'flag': 'SF',
                'src_bytes': 500,
                'dst_bytes': 1000,
                'logged_in': 1,  # Already logged in
                'num_root': 1,  # Root access
                'num_file_creations': 3,  # Creating files
                'root_shell': 1,  # Got root shell!
            }
        }

    elif attack_type == "normal":
        return {
            'type': 'Normal',
            'description': 'Normal Web Browsing',
            'features': {
                'duration': 5,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': 200,
                'dst_bytes': 5000,
                'logged_in': 1,
                'count': 5,
                'srv_count': 5,
                'serror_rate': 0.0,
                'same_srv_rate': 1.0,
            }
        }

    else:
        return {'error': 'Unknown attack type'}

@router.get("/simulate/random")
async def simulate_random():
    """
    Generate random simulated connections
    """
    types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    selected = random.choice(types)
    return await simulate_attack(selected)
```

---

## Frontend Setup

### HTML Structure (frontend/index.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IDS Application - ML-based Intrusion Detection</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>🛡️ IDS Application</h1>
            <p>Machine Learning-based Intrusion Detection System</p>
        </header>

        <!-- Attack Simulator -->
        <section class="simulator-panel">
            <h2>Attack Simulator</h2>
            <p>Simulate different types of network attacks and see how the ML model detects them.</p>

            <div class="button-group">
                <button class="btn btn-normal" onclick="simulateAttack('normal')">
                    Normal Traffic
                </button>
                <button class="btn btn-dos" onclick="simulateAttack('dos')">
                    DoS Attack
                </button>
                <button class="btn btn-probe" onclick="simulateAttack('probe')">
                    Probe Attack
                </button>
                <button class="btn btn-r2l" onclick="simulateAttack('r2l')">
                    R2L Attack
                </button>
                <button class="btn btn-u2r" onclick="simulateAttack('u2r')">
                    U2R Attack
                </button>
            </div>
        </section>

        <!-- Detection Results -->
        <section class="results-panel">
            <h2>Detection Results</h2>
            <div id="results">
                <p class="placeholder">Simulate an attack to see detection results...</p>
            </div>
        </section>

        <!-- Statistics -->
        <section class="stats-panel">
            <h2>Statistics</h2>
            <div id="stats">
                <div class="stat-item">
                    <span class="stat-label">Total Requests:</span>
                    <span class="stat-value" id="total-requests">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Attacks Detected:</span>
                    <span class="stat-value" id="attacks-detected">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Normal Traffic:</span>
                    <span class="stat-value" id="normal-traffic">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Detection Rate:</span>
                    <span class="stat-value" id="detection-rate">0%</span>
                </div>
            </div>
        </section>

        <!-- Logs -->
        <section class="logs-panel">
            <h2>Activity Logs</h2>
            <div id="logs" class="logs-container"></div>
        </section>
    </div>

    <script src="js/app.js"></script>
</body>
</html>
```

### CSS Styles (frontend/css/style.css)

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    background: white;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

header h1 {
    color: #333;
    font-size: 2.5em;
    margin-bottom: 10px;
}

header p {
    color: #666;
    font-size: 1.1em;
}

section {
    background: white;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

section h2 {
    color: #333;
    margin-bottom: 15px;
    font-size: 1.5em;
}

.button-group {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.btn {
    flex: 1;
    min-width: 150px;
    padding: 15px 25px;
    border: none;
    border-radius: 5px;
    font-size: 1em;
    font-weight: bold;
    color: white;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.btn-normal { background: #10b981; }
.btn-dos { background: #ef4444; }
.btn-probe { background: #f59e0b; }
.btn-r2l { background: #8b5cf6; }
.btn-u2r { background: #ec4899; }

#results {
    min-height: 200px;
}

.result-card {
    border-left: 4px solid #667eea;
    padding: 20px;
    background: #f9fafb;
    border-radius: 5px;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.result-title {
    font-size: 1.3em;
    font-weight: bold;
}

.badge {
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: bold;
}

.badge-attack {
    background: #fee2e2;
    color: #dc2626;
}

.badge-normal {
    background: #d1fae5;
    color: #059669;
}

.confidence-bar {
    width: 100%;
    height: 30px;
    background: #e5e7eb;
    border-radius: 15px;
    overflow: hidden;
    margin: 10px 0;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

.probabilities {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 15px;
}

.prob-item {
    padding: 10px;
    background: white;
    border-radius: 5px;
    text-align: center;
}

.prob-label {
    font-size: 0.9em;
    color: #666;
}

.prob-value {
    font-size: 1.2em;
    font-weight: bold;
    color: #333;
}

#stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.stat-item {
    padding: 15px;
    background: #f9fafb;
    border-radius: 5px;
    display: flex;
    flex-direction: column;
}

.stat-label {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 5px;
}

.stat-value {
    font-size: 2em;
    font-weight: bold;
    color: #667eea;
}

.logs-container {
    max-height: 300px;
    overflow-y: auto;
    background: #f9fafb;
    padding: 15px;
    border-radius: 5px;
}

.log-entry {
    padding: 10px;
    margin-bottom: 10px;
    background: white;
    border-radius: 5px;
    border-left: 3px solid #667eea;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
}

.log-time {
    color: #666;
    margin-right: 10px;
}

.log-attack {
    border-left-color: #dc2626;
}

.log-normal {
    border-left-color: #059669;
}

.placeholder {
    text-align: center;
    color: #9ca3af;
    padding: 50px;
    font-size: 1.1em;
}
```

### JavaScript Logic (frontend/js/app.js)

```javascript
const API_BASE = 'http://localhost:8000/api';

let totalRequests = 0;
let attacksDetected = 0;
let normalTraffic = 0;

async function simulateAttack(attackType) {
    try {
        // Get simulated features
        const response = await fetch(`${API_BASE}/simulate/${attackType}`);
        const data = await response.json();

        // Display connection info
        console.log('Simulated:', data);

        // Detect
        const result = await detectAttack(data.features);

        // Update UI
        displayResult(result, data.description);
        updateStats(result);
        addLog(attackType, result);

    } catch (error) {
        console.error('Error:', error);
        alert('Error simulating attack. Make sure backend is running!');
    }
}

async function detectAttack(features) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(features)
    });

    return await response.json();
}

function displayResult(result, description) {
    const resultsDiv = document.getElementById('results');

    const isAttack = result.prediction !== 'Normal';
    const badgeClass = isAttack ? 'badge-attack' : 'badge-normal';

    const html = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-title">${result.prediction}</div>
                <div class="badge ${badgeClass}">
                    ${isAttack ? '⚠️ ATTACK DETECTED' : '✓ SAFE'}
                </div>
            </div>

            <p style="color: #666; margin-bottom: 15px;">${description}</p>

            <div>
                <strong>Confidence:</strong>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%">
                        ${(result.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            <div>
                <strong>Class Probabilities:</strong>
                <div class="probabilities">
                    ${Object.entries(result.probabilities).map(([label, prob]) => `
                        <div class="prob-item">
                            <div class="prob-label">${label}</div>
                            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
                Prediction time: ${result.prediction_time_ms.toFixed(2)} ms
            </p>
        </div>
    `;

    resultsDiv.innerHTML = html;
}

function updateStats(result) {
    totalRequests++;

    if (result.prediction !== 'Normal') {
        attacksDetected++;
    } else {
        normalTraffic++;
    }

    document.getElementById('total-requests').textContent = totalRequests;
    document.getElementById('attacks-detected').textContent = attacksDetected;
    document.getElementById('normal-traffic').textContent = normalTraffic;

    const detectionRate = totalRequests > 0 ? (attacksDetected / totalRequests * 100).toFixed(1) : 0;
    document.getElementById('detection-rate').textContent = detectionRate + '%';
}

function addLog(attackType, result) {
    const logsDiv = document.getElementById('logs');
    const timestamp = new Date().toLocaleTimeString();

    const isAttack = result.prediction !== 'Normal';
    const logClass = isAttack ? 'log-attack' : 'log-normal';

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${logClass}`;
    logEntry.innerHTML = `
        <span class="log-time">[${timestamp}]</span>
        <strong>${attackType.toUpperCase()}</strong> simulated →
        Detected as: <strong>${result.prediction}</strong>
        (${(result.confidence * 100).toFixed(1)}% confidence)
    `;

    logsDiv.insertBefore(logEntry, logsDiv.firstChild);

    // Keep only last 20 logs
    while (logsDiv.children.length > 20) {
        logsDiv.removeChild(logsDiv.lastChild);
    }
}
```

---

## Running the Application

### 1. Train Model (if not done)

```bash
python ml/train.py
```

### 2. Start Backend

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run server
cd backend
python main.py

# Or with uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Backend will be available at: http://localhost:8000**

### 3. Open Frontend

**Option 1: Serve with Python**
```bash
cd frontend
python -m http.server 8080
```

**Option 2: Open directly**
- Open `frontend/index.html` in browser

**Frontend will be available at: http://localhost:8080**

---

## Testing the Application

### 1. Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# Simulate DoS attack
curl http://localhost:8000/api/simulate/dos

# Predict
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0", "count": 511, "serror_rate": 0.99}'
```

### 2. Test Frontend

1. Open http://localhost:8080
2. Click "DoS Attack" button
3. See detection result
4. Try other attack types
5. Check statistics and logs

---

## Deployment to Production

### Using Docker

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run:**
```bash
docker build -t ids-app .
docker run -p 8000:8000 ids-app
```

---

**Next**: Check [README.md](../README.md) for complete setup instructions!
