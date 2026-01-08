"""
API Interface cho News Classification
Sử dụng Flask để tạo REST API
"""
import os
import sys
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.ensemble import NewsEnsemble
from models.logistic_regression.model import NewsLogisticRegression
from models.naive_bayes.model import NewsNaiveBayes
from models.svm.model import NewsSVM
from models.phobert.model import NewsPhoBERT

app = Flask(__name__)
CORS(app)  # Cho phép CORS để có thể gọi từ frontend

# Global variables để cache models
models_cache = {}
ensemble_cache = None


def load_all_models():
    """Load tất cả các mô hình đã train"""
    global models_cache
    
    if models_cache:
        return models_cache
    
    models = {}
    
    # Load Logistic Regression
    try:
        lr_model = NewsLogisticRegression()
        lr_model.load('models/saved/logistic_regression')
        models['logistic_regression'] = lr_model
    except Exception as e:
        print(f"Không thể load Logistic Regression: {e}")
    
    # Load Naive Bayes
    try:
        nb_model = NewsNaiveBayes()
        nb_model.load('models/saved/naive_bayes')
        models['naive_bayes'] = nb_model
    except Exception as e:
        print(f"Không thể load Naive Bayes: {e}")
    
    # Load SVM
    try:
        svm_model = NewsSVM()
        svm_model.load('models/saved/svm')
        models['svm'] = svm_model
    except Exception as e:
        print(f"Không thể load SVM: {e}")
    
    # Load PhoBERT
    try:
        phobert_model = NewsPhoBERT()
        phobert_model.load('models/saved/phobert')
        models['phobert'] = phobert_model
    except Exception as e:
        print(f"Không thể load PhoBERT: {e}")
    
    models_cache = models
    return models


def get_ensemble():
    """Lấy hoặc tạo ensemble model"""
    global ensemble_cache
    
    if ensemble_cache:
        return ensemble_cache
    
    models = load_all_models()
    
    if not models:
        return None
    
    # Trọng số cho ensemble
    weights = {
        'logistic_regression': 1.0,
        'naive_bayes': 0.9,
        'svm': 1.0,
        'phobert': 1.2
    }
    
    ensemble = NewsEnsemble(models=models, weights=weights, method='weighted_average')
    ensemble_cache = ensemble
    return ensemble


@app.route('/')
def index():
    """Trang chủ với form để test"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Classification API</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            form {
                margin-top: 20px;
            }
            label {
                display: block;
                margin-top: 15px;
                font-weight: bold;
                color: #555;
            }
            input[type="text"], textarea {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }
            textarea {
                height: 150px;
                resize: vertical;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 20px;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }
            .result h3 {
                margin-top: 0;
                color: #333;
            }
            .probability {
                margin: 10px 0;
                padding: 8px;
                background: white;
                border-radius: 3px;
            }
            .error {
                color: red;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📰 News Classification API</h1>
            <p style="text-align: center; color: #666;">Phân loại tin tức tiếng Việt tự động</p>
            
            <form id="predictForm">
                <label for="title">Tiêu đề bài viết:</label>
                <input type="text" id="title" name="title" placeholder="Nhập tiêu đề..." required>
                
                <label for="content">Nội dung bài viết:</label>
                <textarea id="content" name="content" placeholder="Nhập nội dung..."></textarea>
                
                <button type="submit">Phân loại</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('predictForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const title = document.getElementById('title').value;
                const content = document.getElementById('content').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<p>Đang xử lý...</p>';
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            title: title,
                            content: content
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = '<div class="error">Lỗi: ' + data.error + '</div>';
                    } else {
                        let html = '<div class="result">';
                        html += '<h3>Kết quả dự đoán:</h3>';
                        html += '<p><strong>Category:</strong> ' + data.category + '</p>';
                        
                        if (data.probability) {
                            html += '<h4>Top 5 categories:</h4>';
                            for (let i = 0; i < Math.min(5, data.probability.length); i++) {
                                const item = data.probability[i];
                                html += '<div class="probability">';
                                html += '<strong>' + item.category + ':</strong> ';
                                html += (item.probability * 100).toFixed(2) + '%';
                                html += '</div>';
                            }
                        }
                        
                        html += '</div>';
                        resultDiv.innerHTML = html;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="error">Lỗi: ' + error.message + '</div>';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/api/health', methods=['GET'])
def health():
    """Kiểm tra trạng thái API và models"""
    models = load_all_models()
    
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'total_models': len(models)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict category cho text"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'Không có dữ liệu'}), 400
        
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title and not content:
            return jsonify({'error': 'Cần có ít nhất title hoặc content'}), 400
        
        # Tạo text input
        if title and content:
            text = (title, content)
        elif title:
            text = title
        else:
            text = content
        
        # Lấy model type (mặc định là ensemble)
        model_type = data.get('model', 'ensemble')
        
        if model_type == 'ensemble':
            ensemble = get_ensemble()
            if not ensemble:
                return jsonify({'error': 'Không có mô hình nào được load'}), 500
            
            category, proba_dict = ensemble.predict(text, return_proba=True)
        else:
            # Predict với model cụ thể
            models = load_all_models()
            if model_type not in models:
                return jsonify({'error': f'Model {model_type} không tồn tại'}), 400
            
            model = models[model_type]
            category, proba_dict = model.predict(text, return_proba=True)
        
        # Format probability
        proba_list = [
            {'category': cat, 'probability': prob}
            for cat, prob in list(proba_dict.items())[:5]
        ]
        
        return jsonify({
            'category': category,
            'probability': proba_list,
            'model_used': model_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    """Predict với tất cả models và so sánh"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'Không có dữ liệu'}), 400
        
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title and not content:
            return jsonify({'error': 'Cần có ít nhất title hoặc content'}), 400
        
        # Tạo text input
        if title and content:
            text = (title, content)
        elif title:
            text = title
        else:
            text = content
        
        ensemble = get_ensemble()
        if not ensemble:
            return jsonify({'error': 'Không có mô hình nào được load'}), 500
        
        results = ensemble.predict_all_models(text)
        
        # Format kết quả
        formatted_results = {
            'individual_predictions': {},
            'ensemble_prediction': results.get('ensemble_prediction'),
            'ensemble_probability': []
        }
        
        # Format individual predictions
        for name, result in results.get('individual_predictions', {}).items():
            if 'error' in result:
                formatted_results['individual_predictions'][name] = {'error': result['error']}
            else:
                formatted_results['individual_predictions'][name] = {
                    'category': result['category'],
                    'top_probabilities': [
                        {'category': cat, 'probability': prob}
                        for cat, prob in list(result['probability'].items())[:3]
                    ]
                }
        
        # Format ensemble probability
        if results.get('ensemble_probability'):
            formatted_results['ensemble_probability'] = [
                {'category': cat, 'probability': prob}
                for cat, prob in list(results['ensemble_probability'].items())[:5]
            ]
        
        return jsonify(formatted_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("News Classification API")
    print("="*60)
    print("Đang load các mô hình...")
    
    models = load_all_models()
    print(f"Đã load {len(models)} mô hình: {list(models.keys())}")
    
    if models:
        ensemble = get_ensemble()
        print("Ensemble model đã sẵn sàng")
    else:
        print("Cảnh báo: Không có mô hình nào được load")
    
    print("\nAPI đang chạy tại: http://localhost:5000")
    print("Truy cập http://localhost:5000 để sử dụng web interface")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

