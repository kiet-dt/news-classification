"""
Script để chạy crawler Dân trí
"""
import subprocess
import sys
import os


def run_crawler(categories=None, max_pages=25):
    """
    Chạy crawler Dân trí
    
    Args:
        categories: Danh sách category cách nhau bởi dấu phẩy
                   Ví dụ: 'the-thao,giai-tri,kinh-doanh'
                   Mặc định: crawl tất cả category chính
        max_pages: Số trang tối đa mỗi category (mặc định: 25)
    """
    
    # Tạo thư mục data nếu chưa có (từ root project)
    os.makedirs('../data/raw', exist_ok=True)
    
    # Xây dựng lệnh scrapy (chạy từ trong crawler/)
    cmd = ['scrapy', 'crawl', 'dantri']
    
    if categories:
        cmd.extend(['-a', f'categories={categories}'])
    
    cmd.extend(['-a', f'max_pages={max_pages}'])
    
    # Chạy crawler
    print(f"Bắt đầu crawl Dân trí...")
    print(f"Categories: {categories or 'Tất cả'}")
    print(f"Max pages: {max_pages}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("Crawl hoàn tất!")
        print(f"Kiểm tra dữ liệu trong thư mục: ../data/raw/")
    except subprocess.CalledProcessError as e:
        print(f"\nLỗi khi chạy crawler: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCrawler bị dừng bởi người dùng")
        sys.exit(0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawl tin tức từ Dân trí')
    parser.add_argument(
        '--categories',
        type=str,
        default=None,
        help='Danh sách category cách nhau bởi dấu phẩy (ví dụ: the-thao,giai-tri)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=25,
        help='Số trang tối đa mỗi category (mặc định: 25)'
    )
    
    args = parser.parse_args()
    
    run_crawler(
        categories=args.categories,
        max_pages=args.max_pages
    )

