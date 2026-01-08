# News Classification - Phân loại tin tức Tiếng Việt

Dự án phân loại tin tức tiếng Việt tự động sử dụng Machine Learning với nhiều mô hình: Logistic Regression, Naive Bayes, SVM, PhoBERT và Ensemble.

## Cấu trúc dự án

```
news-classification/
├── crawler/                    # Phần crawl dữ liệu
│   ├── src/                    # Scrapy project source code
│   │   ├── spiders/           # Spiders
│   │   ├── pipelines.py       # Data pipelines
│   │   └── settings.py        # Scrapy settings
│   ├── run_crawler.py         # Script chạy crawler
│   └── test_crawler.py        # Script test crawler
│
├── models/                     # Các mô hình ML
│   ├── utils.py               # Utilities chung (text preprocessing)
│   ├── ensemble.py            # Ensemble model
│   ├── logistic_regression/   # Logistic Regression model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── naive_bayes/           # Naive Bayes model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── svm/                   # SVM model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── phobert/               # PhoBERT model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── ensemble/              # Ensemble utilities
│   │   └── predict.py
│   └── saved/                 # Thư mục lưu models đã train
│       ├── logistic_regression/
│       ├── naive_bayes/
│       ├── svm/
│       ├── phobert/
│       └── ensemble/
│
├── api/                       # API Interface
│   ├── app.py                # Flask API server
│   └── requirements.txt      # API dependencies
│
├── data/                      # Dữ liệu
│   ├── raw/                  # Dữ liệu thô từ crawler
│   └── processed/            # Dữ liệu đã xử lý để train
│
├── scripts/                   # Scripts tiện ích
│   ├── view_parquet.py       # Xem thông tin file Parquet
│   └── merge_data.py         # Gộp nhiều file thành 1
│
├── requirements.txt           # Dependencies
└── README.md                  # File này
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Crawl dữ liệu

```bash
# Vào thư mục crawler
cd crawler

# Chạy crawler (mặc định 25 trang mỗi category)
python run_crawler.py

# Hoặc chỉ định số trang
python run_crawler.py --max-pages 30

# Chạy category cụ thể
python run_crawler.py --categories the-gioi,thoi-su --max-pages 25
```

### 2. Xem dữ liệu

```bash
# Xem file Parquet mới nhất
python scripts/view_parquet.py

# Xem file cụ thể
python scripts/view_parquet.py data/raw/dantri_20260107_023016.parquet
```

### 3. Gộp dữ liệu

```bash
# Gộp tất cả file Parquet thành 1 file
python scripts/merge_data.py

# Gộp thành CSV
python scripts/merge_data.py --format csv
```

## Categories

- Thế giới (`the-gioi`)
- Thời sự (`thoi-su`)
- Pháp luật (`phap-luat`)
- Sức khỏe (`suc-khoe`)
- Đời sống (`doi-song`)
- Du lịch (`du-lich`)
- Kinh doanh (`kinh-doanh`)
- Bất động sản (`bat-dong-san`)
- Thể thao (`the-thao`)
- Giải trí (`giai-tri`)
- Giáo dục (`giao-duc`)
- Công nghệ (`cong-nghe`)

## Dữ liệu

Dữ liệu được lưu dạng Parquet trong `data/raw/` với các trường:
- `title`: Tiêu đề bài báo
- `content`: Nội dung bài báo
- `category`: Chủ đề (Thế giới, Thời sự, ...)
- `url`: URL bài viết
- `source`: Nguồn (dantri)

## Workflow đầy đủ

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Crawl dữ liệu

```bash
cd crawler
python run_crawler.py --max-pages 25
```

### Bước 3: Gộp dữ liệu

```bash
python scripts/merge_data.py
```

### Bước 4: Train models

```bash
# Bắt đầu với Logistic Regression (nhanh nhất)
python models/logistic_regression/train.py --sample-size 10000

# Train thêm các models khác (tùy chọn)
python models/naive_bayes/train.py --sample-size 10000
python models/svm/train.py --sample-size 10000
```

### Bước 5: Test models

```bash
# Test với Logistic Regression
python models/logistic_regression/predict.py --title "Test" --content "Test content"

# Test với Ensemble (nếu đã train nhiều models)
python models/ensemble/predict.py --title "Test" --content "Test content" --show-all
```

### Bước 6: Chạy API (tùy chọn)

```bash
python api/app.py
```

Truy cập `http://localhost:5000` để sử dụng web interface.

## Models

Dự án đã implement 5 mô hình:

1. **Logistic Regression** - Nhanh, phù hợp cho baseline
2. **Naive Bayes** - Nhanh, hiệu quả với text classification
3. **SVM** - Mạnh mẽ với kernel linear hoặc RBF
4. **PhoBERT** - Transformer-based model cho tiếng Việt (cần GPU để train nhanh)
5. **Ensemble** - Kết hợp tất cả models để tăng độ chính xác

### Train Models

#### 1. Logistic Regression (Khuyến nghị bắt đầu)

```bash
# Train với sample nhỏ để test nhanh
python models/logistic_regression/train.py --sample-size 10000

# Train với toàn bộ dữ liệu
python models/logistic_regression/train.py
```

#### 2. Naive Bayes

```bash
python models/naive_bayes/train.py --sample-size 10000
```

#### 3. SVM

```bash
python models/svm/train.py --sample-size 10000
```

#### 4. PhoBERT (Chậm, cần GPU)

```bash
# Test với sample nhỏ trước
python models/phobert/train.py --sample-size 5000 --batch-size 8 --num-epochs 2

# Train đầy đủ
python models/phobert/train.py --batch-size 16 --num-epochs 5
```

### Predict với Models

#### Predict với từng model riêng lẻ

```bash
# Logistic Regression
python models/logistic_regression/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..."

# Naive Bayes
python models/naive_bayes/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..."

# SVM
python models/svm/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..."

# PhoBERT
python models/phobert/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..."
```

#### Predict với Ensemble Model

```bash
# Predict với ensemble (tự động kết hợp tất cả models đã train)
python models/ensemble/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..."

# Hiển thị kết quả từ tất cả models + ensemble
python models/ensemble/predict.py --title "Bóng đá Việt Nam thắng" --content "Đội tuyển Việt Nam..." --show-all

# Hiển thị probability
python models/ensemble/predict.py --title "Bóng đá" --content "Nội dung" --show-proba
```

## API Interface

### Chạy API Server

```bash
# Cài đặt dependencies (nếu chưa có)
pip install flask flask-cors

# Chạy API
python api/app.py
```

API sẽ chạy tại: `http://localhost:5000`

### API Endpoints

1. **GET /** - Web interface để test
2. **GET /api/health** - Kiểm tra trạng thái API và models
3. **POST /api/predict** - Predict category
   ```json
   {
     "title": "Tiêu đề bài viết",
     "content": "Nội dung bài viết",
     "model": "ensemble"  // optional: logistic_regression, naive_bayes, svm, phobert, ensemble
   }
   ```
4. **POST /api/predict_all** - Predict với tất cả models và so sánh

### Ví dụ sử dụng API

```bash
# Sử dụng curl
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Bóng đá Việt Nam thắng\", \"content\": \"Đội tuyển Việt Nam...\"}"

# Hoặc mở trình duyệt
# http://localhost:5000
```



