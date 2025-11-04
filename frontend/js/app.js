// API Configuration
const API_BASE = 'http://localhost:8000/api';

// Statistics
let totalRequests = 0;
let attacksDetected = 0;
let normalTraffic = 0;

/**
 * Simulate an attack
 */
async function simulateAttack(attackType) {
    try {
        addLog('system', `Simulating ${attackType.toUpperCase()} attack...`);

        // Get simulated features from backend
        const response = await fetch(`${API_BASE}/simulate/${attackType}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Simulated connection:', data);

        // Detect using ML model
        const result = await detectAttack(data.features);

        // Update UI
        displayResult(result, data.description, data.explanation, attackType);
        updateStats(result);
        addLog(attackType, `${attackType.toUpperCase()} detected as: ${result.prediction} (${(result.confidence * 100).toFixed(1)}% confidence)`, result.is_attack);

    } catch (error) {
        console.error('Error:', error);
        addLog('error', `Error: ${error.message}`);

        // Show user-friendly error
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `
            <div class="placeholder">
                <p style="color: #dc2626;">⚠️ Error: ${error.message}</p>
                <p class="hint">Make sure the backend server is running at http://localhost:8000</p>
                <p class="hint">Run: <code>python backend/main.py</code></p>
            </div>
        `;
    }
}

/**
 * Detect attack using ML model
 */
async function detectAttack(features) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(features)
    });

    if (!response.ok) {
        throw new Error(`Detection failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Display detection result
 */
function displayResult(result, description, explanation, attackType) {
    const resultsDiv = document.getElementById('results');

    const isAttack = result.prediction !== 'Normal';
    const badgeClass = isAttack ? 'badge-attack' : 'badge-normal';
    const badgeText = isAttack ? '⚠️ ATTACK DETECTED' : '✓ SAFE';
    const borderColor = isAttack ? '#dc2626' : '#059669';

    // Color based on prediction
    const predictionColors = {
        'Normal': '#059669',
        'DoS': '#dc2626',
        'Probe': '#f59e0b',
        'R2L': '#8b5cf6',
        'U2R': '#ec4899'
    };

    const html = `
        <div class="result-card" style="border-left-color: ${borderColor}">
            <div class="result-header">
                <div class="result-title" style="color: ${predictionColors[result.prediction]}">
                    ${result.prediction}
                </div>
                <div class="badge ${badgeClass}">
                    ${badgeText}
                </div>
            </div>

            <p style="color: #666; margin-bottom: 15px; font-size: 1.1em;">
                <strong>Simulated Attack:</strong> ${description}
            </p>

            <div class="explanation">
                <strong>🔍 Analysis:</strong> ${explanation}
            </div>

            <div style="margin: 20px 0;">
                <strong>Confidence Level:</strong>
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
                        <div class="prob-item" ${label === result.prediction ? 'style="background: #e0e7ff; border: 2px solid #667eea;"' : ''}>
                            <div class="prob-label">${label}</div>
                            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
                <p style="color: #666; font-size: 0.95em; margin-bottom: 8px;">
                    <strong>⚡ Prediction Time:</strong> ${result.prediction_time_ms.toFixed(2)} ms
                </p>
                <p style="color: #666; font-size: 0.95em;">
                    <strong>📊 Feature Summary:</strong>
                    Protocol: ${result.input_summary.protocol} |
                    Service: ${result.input_summary.service} |
                    Bytes: ${result.input_summary.src_bytes}→${result.input_summary.dst_bytes} |
                    Connections: ${result.input_summary.count}
                </p>
            </div>
        </div>
    `;

    resultsDiv.innerHTML = html;
}

/**
 * Update statistics
 */
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

    const detectionRate = totalRequests > 0
        ? ((attacksDetected / totalRequests) * 100).toFixed(1)
        : 0;

    document.getElementById('detection-rate').textContent = detectionRate + '%';
}

/**
 * Add log entry
 */
function addLog(type, message, isAttack = null) {
    const logsDiv = document.getElementById('logs');
    const timestamp = new Date().toLocaleTimeString();

    let logClass = 'log-entry';
    if (isAttack === true) {
        logClass += ' log-attack';
    } else if (isAttack === false) {
        logClass += ' log-normal';
    }

    const logEntry = document.createElement('div');
    logEntry.className = logClass;

    const icon = type === 'error' ? '❌' :
                 type === 'system' ? '🔧' :
                 isAttack ? '⚠️' : '✓';

    logEntry.innerHTML = `
        <span class="log-time">[${timestamp}]</span>
        <span>${icon} ${message}</span>
    `;

    logsDiv.insertBefore(logEntry, logsDiv.firstChild);

    // Keep only last 50 logs
    while (logsDiv.children.length > 50) {
        logsDiv.removeChild(logsDiv.lastChild);
    }

    // Auto-scroll to top
    logsDiv.scrollTop = 0;
}

/**
 * Clear logs
 */
function clearLogs() {
    const logsDiv = document.getElementById('logs');
    logsDiv.innerHTML = `
        <div class="log-entry">
            <span class="log-time">[System]</span>
            <span>Logs cleared.</span>
        </div>
    `;
    addLog('system', 'Logs have been cleared.');
}

/**
 * Check backend connection on page load
 */
async function checkBackend() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            const data = await response.json();
            addLog('system', `Connected to backend. Model: ${data.model.model_type} with ${data.model.n_estimators} trees`);
            console.log('Backend stats:', data);
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        addLog('error', 'Cannot connect to backend. Please start the server: python backend/main.py');
        console.error('Backend connection error:', error);
    }
}

// Check backend on page load
window.addEventListener('load', () => {
    checkBackend();
    addLog('system', 'IDS Application frontend loaded successfully');
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey) {
        switch(e.key) {
            case '1':
                e.preventDefault();
                simulateAttack('normal');
                break;
            case '2':
                e.preventDefault();
                simulateAttack('dos');
                break;
            case '3':
                e.preventDefault();
                simulateAttack('probe');
                break;
            case '4':
                e.preventDefault();
                simulateAttack('r2l');
                break;
            case '5':
                e.preventDefault();
                simulateAttack('u2r');
                break;
        }
    }
});

console.log('%cIDS Application', 'color: #667eea; font-size: 24px; font-weight: bold;');
console.log('%cKeyboard Shortcuts:', 'color: #667eea; font-size: 14px;');
console.log('Ctrl+1: Normal Traffic');
console.log('Ctrl+2: DoS Attack');
console.log('Ctrl+3: Probe Attack');
console.log('Ctrl+4: R2L Attack');
console.log('Ctrl+5: U2R Attack');
