# Quick Start Guide - IDS Application

## Hướng dẫn nhanh (5 phút)

### Bước 1: Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt
venv\Scripts\activate   # Windows
# hoặc
source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Download dataset

**Option A: Tự động (Linux/Mac)**
```bash
mkdir -p ml/dataset
cd ml/dataset
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
cd ../..
```

**Option B: Thủ công (Windows)**
1. Tạo folder `ml/dataset/`
2. Download 2 files:
   - KDDTrain+.txt: https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
   - KDDTest+.txt: https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
3. Đặt vào folder `ml/dataset/`

### Bước 3: Train model

```bash
python ml/train.py
```

Chờ 2-5 phút để training hoàn tất.

### Bước 4: Chạy ứng dụng

```bash
python backend/main.py
```

### Bước 5: Mở trình duyệt

Truy cập: **http://localhost:8000**

## ✅ Checklist

- [ ] Python 3.8+ đã cài
- [ ] Dependencies đã cài (`pip install -r requirements.txt`)
- [ ] Dataset đã download (2 files .txt trong `ml/dataset/`)
- [ ] Model đã train (`ml/trained_models/ids_model.pkl` tồn tại)
- [ ] Backend đang chạy (http://localhost:8000)
- [ ] Frontend mở được trong browser

## 🎯 Demo Flow

1. Click "DoS Attack" → Thấy model detect là DoS với confidence ~99%
2. Click "Normal Traffic" → Thấy model detect là Normal
3. Thử các attack types khác
4. Xem Statistics và Logs

## ❓ Troubleshooting

### Lỗi: "Model file not found"
→ Chạy `python ml/train.py` để train model

### Lỗi: "Dataset file not found"
→ Download dataset vào `ml/dataset/`

### Backend không chạy được
→ Check port 8000 đã bị chiếm chưa: `netstat -ano | findstr :8000`

### Frontend không kết nối backend
→ Check backend đang chạy: http://localhost:8000/health

## 📚 Tiếp theo

- Đọc docs/ để hiểu về attack types
- Xem API docs: http://localhost:8000/api/docs
- Thử modify ML model trong `ml/train.py`
- Customize frontend trong `frontend/`

Good luck! 🚀
