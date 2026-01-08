"""
Ensemble Model - Kết hợp các mô hình để tăng độ chính xác
"""
import os
import sys
import pickle
import numpy as np
from collections import Counter

# Thêm root project vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import VietnameseTextPreprocessor


class NewsEnsemble:
    """Ensemble model kết hợp nhiều mô hình"""
    
    def __init__(self, models=None, weights=None, method='weighted_voting'):
        """
        Args:
            models: Dict chứa các mô hình đã train
                   Format: {'model_name': model_instance}
            weights: Dict chứa trọng số cho mỗi mô hình
                    Format: {'model_name': weight}
                    Nếu None, sẽ dùng trọng số đều
            method: Phương pháp ensemble
                   - 'voting': Hard voting (mỗi model vote 1 phiếu)
                   - 'weighted_voting': Weighted voting (theo trọng số)
                   - 'average': Average probability
                   - 'weighted_average': Weighted average probability
        """
        self.models = models or {}
        self.weights = weights or {}
        self.method = method
        self.label_decoder = None  # Sẽ lấy từ model đầu tiên
        
        # Nếu không có weights, dùng trọng số đều
        if not self.weights and self.models:
            self.weights = {name: 1.0 for name in self.models.keys()}
    
    def add_model(self, name, model, weight=1.0):
        """Thêm một mô hình vào ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
        # Lấy label_decoder từ model đầu tiên
        if self.label_decoder is None:
            if hasattr(model, 'label_decoder'):
                self.label_decoder = model.label_decoder
    
    def predict(self, text, return_proba=False):
        """
        Predict category cho text
        
        Args:
            text: Text cần predict
            return_proba: Có trả về probability không
        
        Returns:
            Category name (và probability nếu return_proba=True)
        """
        if not self.models:
            raise ValueError("Chưa có mô hình nào trong ensemble. Hãy thêm mô hình trước.")
        
        # Predict với tất cả models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if return_proba:
                    category, proba_dict = model.predict(text, return_proba=True)
                    predictions[name] = category
                    probabilities[name] = proba_dict
                else:
                    category = model.predict(text, return_proba=False)
                    predictions[name] = category
            except Exception as e:
                print(f"Lỗi khi predict với {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("Không có mô hình nào predict thành công")
        
        # Ensemble prediction
        if self.method == 'voting':
            # Hard voting: mỗi model vote 1 phiếu
            votes = list(predictions.values())
            final_category = Counter(votes).most_common(1)[0][0]
            
        elif self.method == 'weighted_voting':
            # Weighted voting: mỗi model vote với trọng số
            vote_counts = {}
            for name, category in predictions.items():
                weight = self.weights.get(name, 1.0)
                vote_counts[category] = vote_counts.get(category, 0) + weight
            
            final_category = max(vote_counts.items(), key=lambda x: x[1])[0]
            
        elif self.method == 'average':
            # Average probability
            if not probabilities:
                # Nếu không có probability, fallback về voting
                votes = list(predictions.values())
                final_category = Counter(votes).most_common(1)[0][0]
            else:
                # Tính average probability
                all_categories = set()
                for proba_dict in probabilities.values():
                    all_categories.update(proba_dict.keys())
                
                avg_proba = {}
                for cat in all_categories:
                    probs = [proba_dict.get(cat, 0) for proba_dict in probabilities.values()]
                    avg_proba[cat] = np.mean(probs)
                
                final_category = max(avg_proba.items(), key=lambda x: x[1])[0]
                proba_dict = dict(sorted(avg_proba.items(), key=lambda x: x[1], reverse=True))
                
        elif self.method == 'weighted_average':
            # Weighted average probability
            if not probabilities:
                # Nếu không có probability, fallback về weighted voting
                vote_counts = {}
                for name, category in predictions.items():
                    weight = self.weights.get(name, 1.0)
                    vote_counts[category] = vote_counts.get(category, 0) + weight
                final_category = max(vote_counts.items(), key=lambda x: x[1])[0]
            else:
                # Tính weighted average probability
                all_categories = set()
                for proba_dict in probabilities.values():
                    all_categories.update(proba_dict.keys())
                
                weighted_proba = {}
                total_weight = sum(self.weights.values())
                
                for cat in all_categories:
                    weighted_sum = 0
                    for name, proba_dict in probabilities.items():
                        weight = self.weights.get(name, 1.0)
                        weighted_sum += proba_dict.get(cat, 0) * weight
                    weighted_proba[cat] = weighted_sum / total_weight
                
                final_category = max(weighted_proba.items(), key=lambda x: x[1])[0]
                proba_dict = dict(sorted(weighted_proba.items(), key=lambda x: x[1], reverse=True))
        
        else:
            raise ValueError(f"Phương pháp ensemble không hợp lệ: {self.method}")
        
        if return_proba:
            if 'proba_dict' not in locals():
                # Tạo proba_dict từ predictions nếu chưa có
                all_categories = set(predictions.values())
                proba_dict = {cat: 0.0 for cat in all_categories}
                for cat in predictions.values():
                    proba_dict[cat] += 1.0 / len(predictions)
                proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
            
            return final_category, proba_dict
        else:
            return final_category
    
    def predict_all_models(self, text):
        """
        Predict với tất cả models và trả về kết quả chi tiết
        
        Returns:
            Dict chứa kết quả từ từng model và ensemble
        """
        results = {
            'individual_predictions': {},
            'ensemble_prediction': None,
            'ensemble_probability': None
        }
        
        # Predict với từng model
        for name, model in self.models.items():
            try:
                category, proba_dict = model.predict(text, return_proba=True)
                results['individual_predictions'][name] = {
                    'category': category,
                    'probability': proba_dict
                }
            except Exception as e:
                results['individual_predictions'][name] = {
                    'error': str(e)
                }
        
        # Ensemble prediction
        try:
            category, proba_dict = self.predict(text, return_proba=True)
            results['ensemble_prediction'] = category
            results['ensemble_probability'] = proba_dict
        except Exception as e:
            results['ensemble_error'] = str(e)
        
        return results
    
    def save(self, model_dir='models/saved/ensemble'):
        """Lưu thông tin ensemble (chỉ lưu config, không lưu models)"""
        os.makedirs(model_dir, exist_ok=True)
        
        config = {
            'method': self.method,
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }
        
        config_path = os.path.join(model_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Đã lưu ensemble config vào {model_dir}")
        print("Lưu ý: Các mô hình riêng lẻ cần được load riêng")
    
    def load_config(self, model_dir='models/saved/ensemble'):
        """Load config (models cần được load riêng)"""
        config_path = os.path.join(model_dir, 'config.pkl')
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.method = config['method']
        self.weights = config['weights']
        
        print(f"Đã load ensemble config từ {model_dir}")
        print(f"Method: {self.method}")
        print(f"Models: {config['model_names']}")
        print("Lưu ý: Cần load các mô hình riêng lẻ trước khi sử dụng")

