# News Classification - Phân loại tin tức Tiếng Việt

Dự án phân loại tin tức tiếng Việt tự động sử dụng Machine Learning.

## Cấu trúc dự án

```
news-classification/
├── crawler/              # Phần crawl dữ liệu
│   ├── src/              # Scrapy project source code
│   ├── run_crawler.py    # Script chạy crawler
│   ├── test_crawler.py   # Script test crawler
│   └── scrapy.cfg        # Cấu hình Scrapy
│
├── models/               # Các mô hình ML
│   ├── phobert_model.py
│   ├── svm_model.py
│   ├── naive_bayes_model.py
│   ├── logistic_regression_model.py
│   └── ensemble.py
│
├── data/                 # Dữ liệu
│   ├── raw/             # Dữ liệu thô từ crawler
│   └── processed/       # Dữ liệu đã xử lý để train
│
├── scripts/              # Scripts tiện ích
│   ├── view_parquet.py  # Xem thông tin file Parquet
│   └── merge_data.py    # Gộp nhiều file thành 1
│
├── notebooks/            # Jupyter notebooks (tùy chọn)
│
├── requirements.txt      # Dependencies
└── README.md            # File này
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

## Models (Sắp tới)

Các mô hình ML sẽ được thêm vào thư mục `models/`:
- PhoBERT (Transformer-based)
- SVM (Traditional ML)
- Naive Bayes (Traditional ML)
- Logistic Regression (Traditional ML)
- Ensemble (Kết hợp các mô hình)

## License

MIT

