"""
Script để train Logistic Regression model
"""
import os
import sys
import argparse
import pandas as pd

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.logistic_regression.model import NewsLogisticRegression


def main():
    parser = argparse.ArgumentParser(description='Train Logistic Regression model')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/news_dataset.parquet',
        help='Đường dẫn đến file dữ liệu (mặc định: data/processed/news_dataset.parquet)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Tỷ lệ test set (mặc định: 0.2)'
    )
    parser.add_argument(
        '--validation-size',
        type=float,
        default=0.1,
        help='Tỷ lệ validation set (mặc định: 0.1)'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=10000,
        help='Số lượng features tối đa cho TF-IDF (mặc định: 10000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/saved/logistic_regression',
        help='Thư mục lưu model (mặc định: models/saved/logistic_regression)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Số lượng mẫu để train (None = tất cả, dùng để test nhanh)'
    )
    
    args = parser.parse_args()
    
    # Đọc dữ liệu
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data
    print(f"Đang đọc dữ liệu từ {data_path}...")
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Đã đọc {len(df)} mẫu")
    
    # Sample nếu cần (để test nhanh)
    if args.sample_size and args.sample_size < len(df):
        print(f"Đang sample {args.sample_size} mẫu để test nhanh...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"Sau khi sample: {len(df)} mẫu")
    
    # Kiểm tra dữ liệu
    print("\n=== Thống kê dữ liệu ===")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số categories: {df['category'].nunique()}")
    print("\nPhân bố theo category:")
    print(df['category'].value_counts())
    
    # Kiểm tra missing values
    missing = df[['title', 'content', 'category']].isnull().sum()
    if missing.sum() > 0:
        print(f"\nCảnh báo: Có {missing.sum()} giá trị thiếu")
        print(missing)
        # Loại bỏ rows có missing values
        df = df.dropna(subset=['title', 'content', 'category'])
        print(f"Sau khi loại bỏ missing: {len(df)} mẫu")
    
    # Tạo model
    print("\n=== Khởi tạo model ===")
    model = NewsLogisticRegression(
        max_features=args.max_features,
        random_state=42
    )
    
    # Prepare data
    print("\n=== Chuẩn bị dữ liệu ===")
    X, y, df_processed = model.prepare_data(df)
    
    print(f"Số lượng features sau vectorization: {model.max_features}")
    print(f"Số lượng categories: {len(model.label_encoder)}")
    print(f"Categories: {list(model.label_encoder.keys())}")
    
    # Train
    print("\n=== Bắt đầu training ===")
    results = model.train(
        X, y,
        test_size=args.test_size,
        validation_size=args.validation_size
    )
    
    # Lưu model
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    print(f"\n=== Lưu model ===")
    model.save(output_dir)
    
    print("\n=== Hoàn tất ===")
    print("Model đã được lưu tại:", output_dir)
    print("\nĐể sử dụng model, chạy:")
    print("  python models/logistic_regression/predict.py")


if __name__ == '__main__':
    main()

