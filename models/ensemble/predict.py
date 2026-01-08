"""
Script để predict với Ensemble model
"""
import os
import sys
import argparse
import json

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.ensemble import NewsEnsemble
from models.logistic_regression.model import NewsLogisticRegression
from models.naive_bayes.model import NewsNaiveBayes
from models.svm.model import NewsSVM
from models.phobert.model import NewsPhoBERT


def load_all_models():
    """Load tất cả các mô hình đã train"""
    models = {}
    
    # Load Logistic Regression
    try:
        lr_model = NewsLogisticRegression()
        lr_model.load('models/saved/logistic_regression')
        models['logistic_regression'] = lr_model
        print("✓ Đã load Logistic Regression")
    except Exception as e:
        print(f"✗ Không thể load Logistic Regression: {e}")
    
    # Load Naive Bayes
    try:
        nb_model = NewsNaiveBayes()
        nb_model.load('models/saved/naive_bayes')
        models['naive_bayes'] = nb_model
        print("✓ Đã load Naive Bayes")
    except Exception as e:
        print(f"✗ Không thể load Naive Bayes: {e}")
    
    # Load SVM
    try:
        svm_model = NewsSVM()
        svm_model.load('models/saved/svm')
        models['svm'] = svm_model
        print("✓ Đã load SVM")
    except Exception as e:
        print(f"✗ Không thể load SVM: {e}")
    
    # Load PhoBERT
    try:
        phobert_model = NewsPhoBERT()
        phobert_model.load('models/saved/phobert')
        models['phobert'] = phobert_model
        print("✓ Đã load PhoBERT")
    except Exception as e:
        print(f"✗ Không thể load PhoBERT: {e}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Predict với Ensemble model')
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
        '--method',
        type=str,
        default='weighted_average',
        choices=['voting', 'weighted_voting', 'average', 'weighted_average'],
        help='Phương pháp ensemble (mặc định: weighted_average)'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Hiển thị kết quả từ tất cả models riêng lẻ'
    )
    parser.add_argument(
        '--show-proba',
        action='store_true',
        help='Hiển thị probability'
    )
    
    args = parser.parse_args()
    
    # Load tất cả models
    print("Đang load các mô hình...")
    models = load_all_models()
    
    if not models:
        print("Lỗi: Không có mô hình nào được load thành công")
        print("Hãy train các mô hình trước:")
        print("  python models/logistic_regression/train.py")
        print("  python models/naive_bayes/train.py")
        print("  python models/svm/train.py")
        print("  python models/phobert/train.py")
        sys.exit(1)
    
    # Tạo ensemble
    # Trọng số có thể điều chỉnh dựa trên performance của từng model
    weights = {
        'logistic_regression': 1.0,
        'naive_bayes': 0.9,
        'svm': 1.0,
        'phobert': 1.2  # PhoBERT thường tốt hơn nên trọng số cao hơn
    }
    
    ensemble = NewsEnsemble(models=models, weights=weights, method=args.method)
    
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
    print("\n" + "="*60)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("="*60)
    
    if args.show_all:
        # Hiển thị kết quả từ tất cả models
        results = ensemble.predict_all_models(text)
        
        print("\n--- Kết quả từ từng mô hình ---")
        for name, result in results['individual_predictions'].items():
            if 'error' in result:
                print(f"\n{name}: Lỗi - {result['error']}")
            else:
                print(f"\n{name}:")
                print(f"  Category: {result['category']}")
                if args.show_proba:
                    print(f"  Top 3 probabilities:")
                    for i, (cat, prob) in enumerate(list(result['probability'].items())[:3], 1):
                        print(f"    {i}. {cat}: {prob:.4f} ({prob*100:.2f}%)")
        
        print("\n--- Kết quả Ensemble ---")
        if 'ensemble_error' in results:
            print(f"Lỗi: {results['ensemble_error']}")
        else:
            print(f"Category dự đoán: {results['ensemble_prediction']}")
            if args.show_proba:
                print(f"\nProbability (Ensemble):")
                for cat, prob in list(results['ensemble_probability'].items())[:5]:
                    print(f"  {cat}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        # Chỉ hiển thị kết quả ensemble
        if args.show_proba:
            category, proba_dict = ensemble.predict(text, return_proba=True)
            print(f"\nCategory dự đoán (Ensemble): {category}")
            print(f"\nTop 5 probabilities:")
            for i, (cat, prob) in enumerate(list(proba_dict.items())[:5], 1):
                print(f"  {i}. {cat}: {prob:.4f} ({prob*100:.2f}%)")
        else:
            category = ensemble.predict(text, return_proba=False)
            print(f"\nCategory dự đoán (Ensemble): {category}")


if __name__ == '__main__':
    main()

