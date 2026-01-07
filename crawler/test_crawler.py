"""
Script test crawler với số lượng nhỏ để kiểm tra
"""
import subprocess
import sys


def test_crawler():
    """Test crawler với 1 category, 2 trang"""
    print("Bắt đầu test crawler...")
    print("=" * 50)
    
    # Test với 1 category, 2 trang
    cmd = [
        'scrapy', 'crawl', 'dantri',
        '-a', 'categories=the-thao',
        '-a', 'max_pages=2'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("Test thành công!")
        print("Kiểm tra file trong ../data/raw/ để xem kết quả")
    except subprocess.CalledProcessError as e:
        print(f"\nTest thất bại: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest bị dừng")
        sys.exit(0)


if __name__ == '__main__':
    test_crawler()

