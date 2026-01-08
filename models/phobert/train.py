"""
Script để train PhoBERT model
"""
import os
import sys
import argparse
import pandas as pd
import torch

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.phobert.model import NewsPhoBERT


def main():
    parser = argparse.ArgumentParser(description='Train PhoBERT model')
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
        '--model-name',
        type=str,
        default='vinai/phobert-base',
        help='Tên model PhoBERT (mặc định: vinai/phobert-base)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Độ dài tối đa của sequence (mặc định: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (mặc định: 16, giảm nếu hết memory)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Số epochs (mặc định: 3)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (mặc định: 2e-5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/saved/phobert',
        help='Thư mục lưu model (mặc định: models/saved/phobert)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Số lượng mẫu để train (None = tất cả, dùng để test nhanh)'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("Cảnh báo: Không có GPU, training sẽ chậm hơn đáng kể!")
        print("Khuyến nghị: Giảm batch_size và sample_size để train nhanh hơn")
    
    # Đọc dữ liệu
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data
    print(f"\nĐang đọc dữ liệu từ {data_path}...")
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
    model = NewsPhoBERT(
        model_name=args.model_name,
        max_length=args.max_length,
        random_state=42
    )
    
    # Prepare data
    print("\n=== Chuẩn bị dữ liệu ===")
    X, y, df_processed = model.prepare_data(df)
    
    print(f"Model: {model.model_name}")
    print(f"Max length: {model.max_length}")
    print(f"Số lượng categories: {len(model.label_encoder)}")
    print(f"Categories: {list(model.label_encoder.keys())}")
    
    # Train
    print("\n=== Bắt đầu training ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("Lưu ý: Training có thể mất nhiều thời gian, đặc biệt nếu không có GPU")
    
    results = model.train(
        X, y,
        test_size=args.test_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Lưu model
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    print(f"\n=== Lưu model ===")
    model.save(output_dir)
    
    print("\n=== Hoàn tất ===")
    print("Model đã được lưu tại:", output_dir)
    print("\nĐể sử dụng model, chạy:")
    print("  python models/phobert/predict.py")


if __name__ == '__main__':
    main()

