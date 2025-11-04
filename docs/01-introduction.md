# Giới thiệu về IDS và Machine Learning

## IDS (Intrusion Detection System) là gì?

**Intrusion Detection System (Hệ thống phát hiện xâm nhập)** là một công cụ bảo mật được sử dụng để giám sát lưu lượng mạng hoặc hoạt động hệ thống nhằm phát hiện các hoạt động đáng ngờ hoặc vi phạm chính sách bảo mật.

### Phân loại IDS

#### 1. **Network-based IDS (NIDS)**
- Giám sát lưu lượng mạng
- Phát hiện tấn công dựa trên network packets
- Ví dụ: Snort, Suricata

#### 2. **Host-based IDS (HIDS)**
- Giám sát hoạt động trên một máy chủ cụ thể
- Phát hiện thay đổi file, process, system calls
- Ví dụ: OSSEC, Tripwire

### Phương pháp phát hiện

#### 1. **Signature-based Detection**
- Sử dụng các "chữ ký" (signatures) đã biết của các tấn công
- **Ưu điểm**: Độ chính xác cao với các tấn công đã biết
- **Nhược điểm**: Không phát hiện được tấn công mới (zero-day attacks)

#### 2. **Anomaly-based Detection** ⭐ (Sử dụng ML)
- Học hành vi bình thường của hệ thống
- Phát hiện bất thường (anomaly) so với baseline
- **Ưu điểm**: Có thể phát hiện tấn công mới
- **Nhược điểm**: Tỷ lệ false positive cao hơn

## Tại sao sử dụng Machine Learning cho IDS?

### 1. **Khả năng học và thích nghi**
Machine Learning có thể:
- Học từ dữ liệu lịch sử
- Nhận diện patterns phức tạp
- Thích nghi với các threat mới

### 2. **Xử lý Big Data**
- Phân tích hàng triệu requests/second
- Tự động hóa việc phát hiện
- Giảm công việc thủ công

### 3. **Phát hiện Zero-Day Attacks**
- Không cần biết trước signature
- Dựa vào hành vi bất thường
- Liên tục cải thiện qua thời gian

## Các thuật toán ML phổ biến cho IDS

### 1. **Supervised Learning**
```
Training Data: [Features] → [Label: Normal/Attack]
```

**Thuật toán:**
- Decision Trees
- Random Forest ⭐ (Phổ biến nhất)
- Support Vector Machine (SVM)
- Neural Networks
- Naive Bayes

**Ưu điểm:**
- Độ chính xác cao
- Phân loại rõ ràng từng loại tấn công

**Nhược điểm:**
- Cần labeled data (tốn thời gian)
- Khó phát hiện tấn công hoàn toàn mới

### 2. **Unsupervised Learning**
```
Training Data: [Features] → Tự tìm clusters
```

**Thuật toán:**
- K-Means Clustering
- DBSCAN
- Isolation Forest
- Autoencoders

**Ưu điểm:**
- Không cần labeled data
- Phát hiện anomalies tốt

**Nhược điểm:**
- Khó diễn giải kết quả
- False positive rate cao

### 3. **Deep Learning**
```
Training Data: Raw packets → Neural Network → Classification
```

**Thuật toán:**
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)
- Autoencoders

**Ưu điểm:**
- Tự động feature extraction
- Độ chính xác rất cao với dữ liệu lớn

**Nhược điểm:**
- Cần computational resources lớn
- Khó giải thích (black box)

## Kiến trúc IDS với ML

```
┌─────────────────┐
│  Network Traffic │
│    (Packets)     │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Extraction  │ ← Chuyển raw data thành features
│  - Packet size      │
│  - Protocol type    │
│  - Connection time  │
│  - Flags, ports...  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │ ← Normalize, scale data
│  - Normalization    │
│  - Encoding         │
│  - Feature scaling  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ML Model          │ ← Trained model
│  (Classification)   │
│  - Random Forest    │
│  - Neural Network   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Prediction        │
│  - Normal (0)       │
│  - Attack (1)       │
│  - Attack Type      │
└─────────────────────┘
```

## Metrics đánh giá IDS

### 1. **Confusion Matrix**
```
                  Predicted
               Normal  Attack
Actual Normal    TN      FP
       Attack    FN      TP
```

- **TP (True Positive)**: Phát hiện đúng attack
- **TN (True Normal)**: Phát hiện đúng normal
- **FP (False Positive)**: Báo nhầm attack (thực ra normal)
- **FN (False Negative)**: Bỏ sót attack (nguy hiểm!)

### 2. **Performance Metrics**

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- Độ chính xác tổng thể

**Precision** = TP / (TP + FP)
- Trong các cảnh báo attack, bao nhiêu % là thật

**Recall (Detection Rate)** = TP / (TP + FN)
- Trong các attack thật, phát hiện được bao nhiêu %

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Cân bằng giữa Precision và Recall

**False Positive Rate** = FP / (FP + TN)
- Tỷ lệ báo nhầm (cần thấp!)

## Challenges trong IDS với ML

### 1. **Imbalanced Dataset**
- Normal traffic >> Attack traffic
- Model có thể bị bias về class Normal
- **Giải pháp**: SMOTE, class weights, undersampling

### 2. **Feature Selection**
- Quá nhiều features → overfitting
- Quá ít features → underfitting
- **Giải pháp**: Feature importance, PCA, domain knowledge

### 3. **Real-time Processing**
- Cần prediction nhanh (< 100ms)
- Model không được quá phức tạp
- **Giải pháp**: Model optimization, caching

### 4. **Adversarial Attacks**
- Attackers có thể manipulate để bypass ML model
- **Giải pháp**: Adversarial training, ensemble models

## Dataset phổ biến cho IDS

### 1. **NSL-KDD** (Sử dụng trong project này)
- Cải tiến từ KDD Cup 99
- 125,973 training records
- 22,544 test records
- 41 features
- 5 classes: Normal, DoS, Probe, R2L, U2R

### 2. **CICIDS2017**
- Dataset hiện đại hơn
- Bao gồm nhiều loại tấn công mới
- 80+ features
- Real-world traffic patterns

### 3. **UNSW-NB15**
- 49 features
- 9 attack categories
- Modern attack types

## Trong project này

Chúng ta sẽ xây dựng một **Web-based IDS Demo** với:

1. **ML Model**: Random Forest classifier
2. **Dataset**: NSL-KDD
3. **Features**: Network traffic features (41 features)
4. **Attack Types**: DoS, Probe, R2L, U2R
5. **Web Interface**:
   - Simulate attacks
   - Real-time detection
   - Visualization
   - Educational explanations

## Tài liệu tham khảo

1. [NIST Guide to IDS](https://csrc.nist.gov/publications/detail/sp/800-94/final)
2. [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
3. [Scikit-learn Documentation](https://scikit-learn.org/)
4. [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Next**: [02 - Attack Types](02-attack-types.md)
