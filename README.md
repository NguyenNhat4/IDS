# IDS Application - Machine Learning-based Intrusion Detection System

![IDS Application](https://img.shields.io/badge/ML-Random_Forest-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

Dự án demo IDS (Intrusion Detection System) sử dụng Machine Learning để phát hiện các cuộc tấn công mạng. Được xây dựng cho môn học Machine Learning với mục đích educational.

## 🎯 Features

- **ML Model**: Random Forest Classifier với độ chính xác ~99%
- **Dataset**: NSL-KDD (125,973 training samples, 22,544 test samples)
- **Attack Types**: DoS, Probe, R2L, U2R
- **Web Interface**: Real-time attack simulation và detection
- **Educational**: Chi tiết documentation về từng loại attack
- **API**: RESTful API với FastAPI

## 📁 Project Structure

```
IDS-Application/
├── docs/                          # Educational documentation
│   ├── 01-introduction.md         # Giới thiệu IDS và ML
│   ├── 02-attack-types.md         # Chi tiết các loại tấn công
│   ├── 03-how-websites-get-attacked.md  # Cách website bị tấn công
│   ├── 04-ml-model-architecture.md      # Kiến trúc ML model
│   ├── 05-feature-engineering.md        # Feature engineering
│   ├── 06-model-training.md             # Hướng dẫn training
│   └── 07-deployment.md                 # Deployment guide
├── backend/                       # FastAPI backend
│   ├── main.py                    # Main application
│   ├── models/                    # ML model loader
│   │   └── ids_model.py
│   └── routes/                    # API routes
│       ├── detection.py           # Detection endpoint
│       └── attack_simulator.py    # Attack simulator
├── frontend/                      # Web interface
│   ├── index.html                 # Main page
│   ├── css/style.css              # Styles
│   └── js/app.js                  # JavaScript logic
├── ml/                            # Machine Learning
│   ├── dataset/                   # NSL-KDD dataset (place here)
│   ├── trained_models/            # Saved models
│   └── train.py                   # Training script
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd IDS-Application
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Download Dataset

Download NSL-KDD dataset:
1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download `KDDTrain+.txt` and `KDDTest+.txt`
3. Place files in `ml/dataset/` folder

**Hoặc sử dụng link trực tiếp:**
```bash
mkdir -p ml/dataset
cd ml/dataset

# Download (Linux/Mac with wget):
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt

cd ../..
```

### 4. Train Model

```bash
python ml/train.py
```

Expected output:
```
====================================================
   IDS ML MODEL TRAINING
   Random Forest Classifier on NSL-KDD Dataset
====================================================

Loading NSL-KDD Dataset...
✓ Train set loaded: (125973, 43)
✓ Test set loaded:  (22544, 43)

...

✅ TRAINING COMPLETED SUCCESSFULLY!
🎯 Final Accuracy: 98.85%
```

Training time: ~2-5 minutes (depends on CPU)

### 5. Start Backend

```bash
python backend/main.py
```

Backend will run at: **http://localhost:8000**

API Docs: **http://localhost:8000/api/docs**

### 6. Open Frontend

Open browser and navigate to:
**http://localhost:8000**

Or serve frontend separately:
```bash
cd frontend
python -m http.server 8080
```

Then open: **http://localhost:8080**

## 🎮 Usage

### Web Interface

1. Click attack type buttons to simulate attacks:
   - **Normal Traffic** - Legitimate web browsing
   - **DoS Attack** - SYN Flood attack
   - **Probe Attack** - Port scanning
   - **R2L Attack** - Brute force login
   - **U2R Attack** - Buffer overflow / privilege escalation

2. View real-time detection results
3. Check statistics and activity logs

### API Usage

**Simulate Attack:**
```bash
curl http://localhost:8000/api/simulate/dos
```

**Detect Attack:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 511,
    "serror_rate": 0.99
  }'
```

Response:
```json
{
  "prediction": "DoS",
  "confidence": 0.987,
  "probabilities": {
    "Normal": 0.002,
    "DoS": 0.987,
    "Probe": 0.008,
    "R2L": 0.002,
    "U2R": 0.001
  },
  "is_attack": true,
  "prediction_time_ms": 15.3
}
```

## 📊 Model Performance

### Metrics (NSL-KDD Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.99 | 0.99 | 0.99 | 9,711 |
| DoS | 0.99 | 0.99 | 0.99 | 7,458 |
| Probe | 0.97 | 0.94 | 0.96 | 2,421 |
| R2L | 0.92 | 0.88 | 0.90 | 2,754 |
| U2R | 0.85 | 0.82 | 0.84 | 200 |

**Overall Accuracy: 98.85%**

### Top 10 Important Features

1. `src_bytes` - 15.2%
2. `dst_bytes` - 13.4%
3. `count` - 9.9%
4. `srv_count` - 8.6%
5. `dst_host_srv_count` - 7.5%
6. `serror_rate` - 6.3%
7. `dst_host_same_srv_rate` - 5.2%
8. `same_srv_rate` - 4.9%
9. `service` - 4.2%
10. `protocol_type` - 4.0%

## 🛡️ Attack Types

### 1. DoS (Denial of Service)
- **Mục đích**: Làm hệ thống không thể phục vụ
- **Techniques**: SYN Flood, Ping Flood, Smurf
- **Indicators**: count > 500, serror_rate > 90%

### 2. Probe (Reconnaissance)
- **Mục đích**: Thu thập thông tin hệ thống
- **Techniques**: Port Scan, OS Fingerprinting
- **Indicators**: diff_srv_rate > 80%, rerror_rate > 70%

### 3. R2L (Remote to Local)
- **Mục đích**: Truy cập trái phép từ xa
- **Techniques**: Brute Force, SQL Injection
- **Indicators**: num_failed_logins > 3, logged_in = 0

### 4. U2R (User to Root)
- **Mục đích**: Leo thang đặc quyền
- **Techniques**: Buffer Overflow, Rootkit
- **Indicators**: root_shell = 1, num_file_creations > 0

## 📚 Educational Documentation

Xem thư mục `docs/` để học về:
- Intrusion Detection Systems
- Machine Learning cho security
- Network attack patterns
- Feature engineering
- Model training và deployment

## 🔧 Development

### Run in Development Mode

```bash
# Backend with auto-reload
uvicorn backend.main:app --reload --port 8000

# Or
python backend/main.py
```

### Project Dependencies

- **Backend**: FastAPI, Uvicorn
- **ML**: scikit-learn, pandas, numpy
- **Frontend**: Vanilla JavaScript (no framework)

## 📈 Future Improvements

- [ ] Add more attack types (XSS, CSRF, SQL Injection)
- [ ] Real-time network traffic monitoring
- [ ] Deep Learning models (LSTM, CNN)
- [ ] Docker deployment
- [ ] Database for logging
- [ ] User authentication
- [ ] Dashboard with charts (Chart.js)

## 🎓 Educational Purpose

Dự án này được xây dựng cho mục đích học tập và demo. **Không sử dụng cho production** mà không có các biện pháp bảo mật bổ sung.

## 📝 License

MIT License - Free to use for educational purposes

## 👨‍💻 Author

- Project for Machine Learning course
- Using NSL-KDD dataset
- Built with FastAPI + scikit-learn

## 🙏 Acknowledgments

- NSL-KDD Dataset: University of New Brunswick
- scikit-learn documentation
- FastAPI framework
- Random Forest algorithm

## 📞 Support

Nếu gặp vấn đề:
1. Check backend đang chạy: http://localhost:8000/health
2. Check dataset đã download chưa: `ml/dataset/KDDTrain+.txt`
3. Check model đã train chưa: `ml/trained_models/ids_model.pkl`
4. Xem logs trong terminal

---

**Happy Learning! 🎓🛡️**
#   I D S  
 