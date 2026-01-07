"""
Script để gộp nhiều file Parquet từ crawler thành 1 file Parquet/CSV duy nhất
"""
import pandas as pd
import glob
import os
from pathlib import Path


def merge_parquet_files(input_dir=None, output_file=None, output_format='parquet'):
    """
    Gộp tất cả file Parquet trong thư mục thành 1 file Parquet/CSV
    
    Args:
        input_dir: Thư mục chứa các file Parquet
        output_file: File output
        output_format: 'parquet' hoặc 'csv'
    """
    # Đường dẫn mặc định từ root project
    project_root = os.path.dirname(os.path.dirname(__file__))
    if input_dir is None:
        input_dir = os.path.join(project_root, 'data', 'raw')
    if output_file is None:
        output_file = os.path.join(project_root, 'data', 'processed', 'news_dataset.parquet')
    
    # Tìm tất cả file Parquet
    parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))
    
    if not parquet_files:
        print(f"Không tìm thấy file Parquet nào trong {input_dir}")
        return None
    
    print(f"Tìm thấy {len(parquet_files)} file Parquet")
    
    # Đọc và gộp tất cả dữ liệu
    dfs = []
    total_articles = 0
    
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
            total_articles += len(df)
            print(f"  Đã đọc {len(df)} mẫu từ {os.path.basename(parquet_file)}")
        except Exception as e:
            print(f"  Lỗi khi đọc {parquet_file}: {e}")
    
    if not dfs:
        print("Không có dữ liệu để gộp")
        return None
    
    # Gộp tất cả DataFrame
    df = pd.concat(dfs, ignore_index=True)
    
    # Loại bỏ duplicate dựa trên URL
    initial_count = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"\nĐã loại bỏ {removed} bài trùng lặp")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lưu file
    if output_format == 'parquet':
        df.to_parquet(
            output_file,
            index=False,
            compression='snappy',
            engine='pyarrow'
        )
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nĐã gộp {len(df)} mẫu vào {output_file} ({file_size:.2f} MB)")
    else:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nĐã gộp {len(df)} mẫu vào {output_file} ({file_size:.2f} MB)")
    
    print(f"\nPhân bố theo category:")
    print(df['category'].value_counts())
    
    # Thống kê
    print(f"\nThống kê:")
    print(f"  - Tổng số bài: {len(df)}")
    print(f"  - Số category: {df['category'].nunique()}")
    print(f"  - Category ít nhất: {df['category'].value_counts().min()} mẫu")
    print(f"  - Category nhiều nhất: {df['category'].value_counts().max()} mẫu")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gộp dữ liệu từ nhiều file Parquet thành 1 file')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Thư mục chứa file Parquet (mặc định: data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/news_dataset.parquet',
        help='File output (mặc định: data/processed/news_dataset.parquet)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Format output: parquet hoặc csv (mặc định: parquet)'
    )
    
    args = parser.parse_args()
    
    merge_parquet_files(args.input_dir, args.output, args.format)
