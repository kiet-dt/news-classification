import pandas as pd
import sys
import os
import glob


def view_parquet(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy file: {filepath}")
        return
    
    # Đọc file
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return
    
    print("=" * 60)
    print(f"THÔNG TIN FILE: {os.path.basename(filepath)}")
    print("=" * 60)
    
    # Thông tin cơ bản
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\nKích thước file: {file_size:.2f} MB")
    print(f"Tổng số bài viết: {len(df):,}")
    print(f"Số cột: {len(df.columns)}")
    
    # Các cột
    print(f"\nCác cột:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Phân bố category
    if 'category' in df.columns:
        print(f"\nPhân bố theo category:")
        print("-" * 60)
        for cat, count in df['category'].value_counts().items():
            percentage = count / len(df) * 100
            print(f"  {cat:20s}: {count:5,} bài ({percentage:5.1f}%)")
    
    # Xem vài dòng đầu
    print(f"\n5 dòng đầu tiên:")
    print("-" * 60)
    for idx, row in df.head(5).iterrows():
        title = str(row['title'])[:60] if pd.notna(row['title']) else 'N/A'
        category = str(row['category']) if pd.notna(row['category']) else 'N/A'
        print(f"\n[{idx+1}] {title}...")
        print(f"    Category: {category}")
        if 'url' in df.columns:
            url = str(row['url'])[:80] if pd.notna(row['url']) else 'N/A'
            print(f"    URL: {url}")
    
    # Thống kê
    print(f"\nThống kê:")
    print("-" * 60)
    
    # Thống kê title
    if 'title' in df.columns:
        avg_title_len = df['title'].str.len().mean()
        min_title_len = df['title'].str.len().min()
        max_title_len = df['title'].str.len().max()
        print(f"  Title:")
        print(f"    - Độ dài trung bình: {avg_title_len:.0f} ký tự")
        print(f"    - Ngắn nhất: {min_title_len:,} ký tự")
        print(f"    - Dài nhất: {max_title_len:,} ký tự")
    
    # Thống kê content
    if 'content' in df.columns:
        avg_content_len = df['content'].str.len().mean()
        min_content_len = df['content'].str.len().min()
        max_content_len = df['content'].str.len().max()
        print(f"  Content:")
        print(f"    - Độ dài trung bình: {avg_content_len:.0f} ký tự")
        print(f"    - Ngắn nhất: {min_content_len:,} ký tự")
        print(f"    - Dài nhất: {max_content_len:,} ký tự")
    
    # Thống kê category
    if 'category' in df.columns:
        unique_categories = df['category'].nunique()
        most_common_category = df['category'].mode()[0] if len(df['category'].mode()) > 0 else 'N/A'
        print(f"  Category:")
        print(f"    - Số category unique: {unique_categories}")
        print(f"    - Category phổ biến nhất: {most_common_category}")
    
    # Thống kê URL
    if 'url' in df.columns:
        unique_urls = df['url'].nunique()
        avg_url_len = df['url'].str.len().mean()
        min_url_len = df['url'].str.len().min()
        max_url_len = df['url'].str.len().max()
        print(f"  URL:")
        print(f"    - Số URL unique: {unique_urls:,}")
        print(f"    - Độ dài trung bình: {avg_url_len:.0f} ký tự")
        print(f"    - Ngắn nhất: {min_url_len:,} ký tự")
        print(f"    - Dài nhất: {max_url_len:,} ký tự")
    
    # Thống kê source
    if 'source' in df.columns:
        unique_sources = df['source'].nunique()
        print(f"  Source:")
        print(f"    - Số source unique: {unique_sources}")
        if unique_sources > 0:
            print(f"    - Phân bố source:")
            for source, count in df['source'].value_counts().items():
                percentage = count / len(df) * 100
                print(f"      + {source}: {count:,} bài ({percentage:.1f}%)")
    
    # Kiểm tra missing values
    print(f"\nKiểm tra dữ liệu thiếu:")
    print("-" * 60)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing.items():
            if count > 0:
                print(f"  - {col}: {count} giá trị thiếu")
    else:
        print("Không có dữ liệu thiếu")
    
    # Kiểm tra duplicate
    if 'url' in df.columns:
        duplicates = df.duplicated(subset=['url']).sum()
        print(f"\nKiểm tra trùng lặp:")
        print("-" * 60)
        if duplicates > 0:
            print(f"Có {duplicates} bài viết trùng lặp (dựa trên URL)")
        else:
            print("Không có bài viết trùng lặp")


def list_parquet_files():
    """Liệt kê tất cả file Parquet trong data/raw"""
    # Đường dẫn từ scripts/ lên root rồi vào data/raw
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if not files:
        print("Không tìm thấy file Parquet nào trong data/raw/")
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    print("Các file Parquet có sẵn:")
    print("-" * 60)
    for i, filepath in enumerate(files, 1):
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        mtime = os.path.getmtime(filepath)
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{i}] {os.path.basename(filepath)}")
        print(f"Kích thước: {file_size:.2f} MB | Sửa đổi: {mod_time}")
    
    return files


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Tìm file Parquet mới nhất
        files = list_parquet_files()
        if files:
            filepath = files[0]
            print(f"\nTự động chọn file mới nhất: {os.path.basename(filepath)}\n")
        else:
            sys.exit(1)
    
    view_parquet(filepath)
    
    print("\n" + "=" * 60)
    print("Tip: Để xem file khác, chạy:")
    print(f"python view_parquet.py <tên_file>")
    print("=" * 60)

