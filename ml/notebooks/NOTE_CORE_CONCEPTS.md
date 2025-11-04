# Core Concepts: Intrusion Detection System using SVM & XGBoost

## Tổng Quan (Overview)

Notebook này triển khai một hệ thống phát hiện xâm nhập (IDS - Intrusion Detection System) sử dụng thuật toán Support Vector Machine (SVM) đạt độ chính xác 83%. Hệ thống phân loại network traffic thành hai loại: **normal** (bình thường) và **intrusion** (xâm nhập).

## 1. Dataset: KDD Cup 1999

### Đặc điểm Dataset
- **Training set**: 125,973 mẫu
- **Test set**: 22,543 mẫu
- **Features**: 42 thuộc tính mô tả network connections
- **Target classes**: Binary classification (normal vs attack)

### Các loại thuộc tính
1. **Basic features**: duration, protocol_type, service, flag
2. **Content features**: src_bytes, dst_bytes, wrong_fragment, urgent, hot
3. **Time-based traffic features**: count, srv_count, error rates
4. **Host-based traffic features**: dst_host_count, dst_host_srv_count

### Các loại tấn công trong dataset
- DoS: neptune, smurf, pod, teardrop, back, land
- Probe: satan, ipsweep, nmap, portsweep
- R2L: warezclient, guess_passwd, warezmaster, imap, ftp_write, multihop, phf, spy
- U2R: buffer_overflow, rootkit, loadmodule, perl

## 2. Data Preprocessing Pipeline

### 2.1. Column Renaming
Thay thế numeric column names bằng tên có ý nghĩa để dễ hiểu và xử lý.

```python
# Từ: [0, 1, 2, 3, ...]
# Thành: ['duration', 'protocol_type', 'service', 'flag', ...]
```

### 2.2. Binary Target Conversion
Chuyển đổi multi-class thành binary classification:

```python
# normal → 0 (bình thường)
# tất cả các loại attack → 1 (xâm nhập)
train['class'] = (train['class'] != "normal") * 1.0
```

**Lý do**: Mục tiêu là phát hiện có xâm nhập hay không, chứ không phân loại chi tiết loại tấn công.

### 2.3. Categorical Encoding

#### One-Hot Encoding
Chuyển đổi categorical features thành numerical:
- `protocol_type`: tcp, udp, icmp
- `service`: 70 loại services khác nhau
- `flag`: 11 connection status flags

**Kỹ thuật quan trọng**:
```python
# Kết hợp train + test trước khi encoding
whole = pd.concat([test.assign(ind="test"), train.assign(ind="train")])

# One-hot encoding với drop_first=True để tránh multicollinearity
one_hot = pd.get_dummies(whole[col], drop_first=True)

# Tách lại train và test
test = whole[whole["ind"].eq("test")].drop(columns="ind")
train = whole[whole["ind"].eq("train")].drop(columns="ind")
```

**Lợi ích**: Đảm bảo train và test có cùng số features sau encoding, tránh lỗi khi predict.

### 2.4. Feature Scaling

#### MinMaxScaler
Chuẩn hóa tất cả features về scale [0, 1]:

```python
ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```

**Tại sao cần scaling cho SVM?**
- SVM sử dụng distance-based calculation
- Features có scale khác nhau sẽ tạo bias
- Features có giá trị lớn sẽ chi phối decision boundary
- Giúp gradient descent converge nhanh hơn

### 2.5. Dimensionality Reduction: PCA

#### Principal Component Analysis
Giảm số chiều từ 100+ features xuống còn 12 principal components:

```python
pca = PCA(n_components=12)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

**Phân tích explained variance**:
- Component 1: 41.84% variance
- Component 2: 15.88% variance
- Component 3: 10.81% variance
- 12 components giải thích ~95% tổng variance

**Lợi ích của PCA**:
1. **Tránh curse of dimensionality**: SVM hoạt động kém với high-dimensional data
2. **Giảm overfitting**: Ít features = ít noise
3. **Tăng tốc training**: Ít dimensions = tính toán nhanh hơn
4. **Giữ được thông tin quan trọng**: 12 components vẫn giữ 95% variance

## 3. Support Vector Machine (SVM)

### 3.1. SVM Concept

SVM tìm hyperplane tối ưu để phân tách hai classes với margin lớn nhất:

```
maximize: margin = 2/||w||
subject to: y_i(w·x_i + b) ≥ 1
```

**Support vectors**: Các điểm dữ liệu nằm trên margin boundary, quyết định vị trí hyperplane.

### 3.2. Kernel Trick

Kernel function ánh xạ data lên không gian chiều cao hơn để tạo linear separability:

#### a) Linear Kernel
```python
K(x, x') = x · x'
```
- Đơn giản nhất, nhanh nhất
- Chỉ phù hợp với linearly separable data
- Trong notebook: bị interrupt vì train quá lâu

#### b) Polynomial Kernel
```python
K(x, x') = (γ·x·x' + c₀)^d
```
- Results: **Accuracy 75%**
- Precision: 92% (attack), 65% (normal)
- Recall: 62% (attack), 93% (normal)

#### c) RBF Kernel (Radial Basis Function)
```python
K(x, x') = exp(-γ·||x - x'||²)
```
- Most popular kernel cho non-linear problems
- Results: **Accuracy 75%** (base model)
- Sau tuning: **Accuracy 81%**

**Tại sao RBF kernel hiệu quả?**
- Network intrusion data thường non-linear và complex
- RBF có thể model complex decision boundaries
- Linh hoạt với parameter γ (gamma)

### 3.3. Hyperparameter Tuning

#### Parameters chính của SVM:

1. **C (Regularization parameter)**
   - Cân bằng giữa maximize margin và minimize error
   - C lớn: hard margin (ít misclassification, dễ overfit)
   - C nhỏ: soft margin (chấp nhận errors, generalize tốt hơn)

2. **γ (Gamma - cho RBF kernel)**
   - Định nghĩa influence radius của support vectors
   - γ lớn: decision boundary phức tạp hơn (có thể overfit)
   - γ nhỏ: decision boundary mượt hơn (có thể underfit)

3. **degree (cho Polynomial kernel)**
   - Bậc của polynomial function
   - Degree cao: complex boundary nhưng dễ overfit

#### Grid Search
Systematic search qua tất cả combinations:

```python
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
}
```

**Best params**: C=1000, gamma=1, kernel='rbf'
**Result**: Accuracy 81%, Precision 97%, Recall 68%

#### Randomized Search
Random sampling từ continuous distributions:

```python
svm_dist = {
    "C": scipy.stats.expon(scale=100),
    "gamma": scipy.stats.expon(scale=0.01),
    "kernel": ["rbf"]
}
```

**Best params**: C=1044.612, gamma=0.459, kernel='rbf'
**Result**: Accuracy 81%, F1-score cân bằng hơn

**Grid Search vs Random Search**:
- Grid: exhaustive, đảm bảo tìm được best trong grid, nhưng chậm
- Random: faster, có thể tìm được params tốt hơn ngoài grid, nhưng không deterministic

### 3.4. Stratified K-Fold Cross Validation

```python
cv = StratifiedKFold(n_splits=5)
```

**Tại sao dùng Stratified?**
- Dataset có class imbalance (9711 normal vs 12832 attacks)
- Stratified đảm bảo mỗi fold có tỷ lệ classes giống nhau
- Evaluation metrics reliable hơn

## 4. Model Evaluation

### 4.1. Metrics Explained

#### Confusion Matrix cho Intrusion Detection:
```
                Predicted
              Normal  Attack
Actual Normal   TN      FP    (False Alarm)
       Attack   FN      TP    (Missed Attack)
```

#### Precision (Độ chính xác)
```
Precision = TP / (TP + FP)
```
- Attack Precision 97%: Khi model dự đoán attack, đúng 97% cases
- Quan trọng khi muốn giảm false alarms

#### Recall (Độ nhạy)
```
Recall = TP / (TP + FN)
```
- Attack Recall 68-72%: Phát hiện được 68-72% actual attacks
- Quan trọng khi không muốn miss attacks

#### F1-Score (Harmonic mean)
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
- Cân bằng giữa Precision và Recall
- Best model: F1 = 0.81 cho cả hai classes

### 4.2. ROC-AUC Analysis

**ROC Curve** (Receiver Operating Characteristic):
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Cho thấy trade-off giữa sensitivity và specificity

**AUC** (Area Under Curve):
- Perfect classifier: AUC = 1.0
- Random classifier: AUC = 0.5
- Best model đạt AUC cao (từ visualization)

### 4.3. Model Comparison

| Model | Accuracy | Attack Precision | Attack Recall | F1-Score |
|-------|----------|------------------|---------------|----------|
| Poly Kernel | 75% | 92% | 62% | 0.74 |
| RBF Base | 75% | 92% | 62% | 0.74 |
| Grid Search | 81% | 97% | 68% | 0.80 |
| Random Search | 81% | 93% | 72% | 0.81 |

**Best Model**: Random Search SVM (C=1044.612, gamma=0.459)
- Balanced precision-recall
- Highest F1-score
- More generalizable

## 5. XGBoost Integration

```python
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
```

Notebook có train XGBoost nhưng results tương tự Random Search SVM. XGBoost là:
- Ensemble learning method (gradient boosting)
- Thường outperform SVM với large datasets
- Có thể so sánh với SVM cho production deployment

## 6. Key Takeaways

### Best Practices áp dụng:
1. **Binary classification** đơn giản hơn multi-class cho detection task
2. **Combine train-test** khi encoding categorical để consistency
3. **Feature scaling** bắt buộc cho SVM
4. **PCA** giảm dimensions hiệu quả cho high-dimensional data
5. **Hyperparameter tuning** cải thiện performance đáng kể (75% → 81%)
6. **Multiple kernels testing** để tìm best fit cho data
7. **Balanced metrics** (F1) quan trọng hơn accuracy cho security applications

### Trade-offs trong IDS:
- **High Precision**: Ít false alarms, nhưng miss một số attacks
- **High Recall**: Catch nhiều attacks, nhưng nhiều false alarms
- **Optimal**: Cân bằng với F1-score cao (~0.81)

### Production Considerations:
1. Model phải balance giữa detection rate và false alarm rate
2. 81% accuracy là khá tốt nhưng vẫn có 19% errors
3. Nên ensemble nhiều models hoặc có human verification layer
4. Regular retraining với new attack patterns
5. Real-time inference speed cũng quan trọng (SVM với PCA khá nhanh)

## 7. Mathematical Intuition

### SVM Optimization Problem:
```
Primal form:
min (1/2)||w||² + C·Σξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

Dual form:
max Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
subject to: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
```

### PCA Transformation:
```
1. Compute covariance matrix: Σ = (1/n)X^T·X
2. Eigendecomposition: Σ·v = λ·v
3. Sort eigenvectors by eigenvalues
4. Project data: Z = X·V_k (k components)
```

## Kết Luận

Notebook này demonstrate một complete ML pipeline cho intrusion detection:
- Proper data preprocessing và feature engineering
- Systematic model selection và tuning
- Comprehensive evaluation với multiple metrics
- Trade-off analysis cho production deployment

Best model (Random Search SVM với RBF kernel) đạt 81% accuracy với balanced precision-recall, phù hợp cho real-world IDS applications.