"""
Script để predict category cho tin tức mới
"""
import os
import sys
import argparse

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.svm.model import NewsSVM


def main():
    parser = argparse.ArgumentParser(description='Predict category cho tin tức')
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Tiêu đề bài viết'
    )
    parser.add_argument(
        '--content',
        type=str,
        default=None,
        help='Nội dung bài viết (tùy chọn)'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text đầy đủ (title + content)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/saved/svm',
        help='Thư mục chứa model (mặc định: models/saved/svm)'
    )
    parser.add_argument(
        '--show-proba',
        action='store_true',
        help='Hiển thị probability cho tất cả categories'
    )
    
    args = parser.parse_args()
    
    # Load model
    model_dir = os.path.join(project_root, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    print(f"Đang load model từ {model_dir}...")
    model = NewsSVM()
    try:
        model.load(model_dir)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        print("Hãy chạy models/svm/train.py trước để train model")
        sys.exit(1)
    
    # Lấy input
    if args.text:
        text = args.text
    elif args.title:
        if args.content:
            text = (args.title, args.content)
        else:
            text = args.title
    else:
        # Interactive mode
        print("\n=== Chế độ nhập liệu ===")
        title = input("Nhập tiêu đề (hoặc Enter để bỏ qua): ").strip()
        content = input("Nhập nội dung (hoặc Enter để bỏ qua): ").strip()
        
        if title and content:
            text = (title, content)
        elif title:
            text = title
        elif content:
            text = content
        else:
            print("Lỗi: Cần nhập ít nhất title hoặc content")
            sys.exit(1)
    
    # Predict
    print("\n=== Kết quả dự đoán ===")
    if args.show_proba:
        category, proba_dict = model.predict(text, return_proba=True)
        print(f"\nCategory dự đoán: {category}")
        print(f"\nProbability cho tất cả categories:")
        for cat, prob in proba_dict.items():
            print(f"  {cat}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        category = model.predict(text, return_proba=False)
        print(f"Category dự đoán: {category}")
        
        # Hiển thị top 3
        _, proba_dict = model.predict(text, return_proba=True)
        print(f"\nTop 3 categories:")
        for i, (cat, prob) in enumerate(list(proba_dict.items())[:3], 1):
            print(f"  {i}. {cat}: {prob:.4f} ({prob*100:.2f}%)")


if __name__ == '__main__':
    main()

