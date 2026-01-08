"""
Logistic Regression Model cho phân loại tin tức tiếng Việt
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Thêm root project vào path để import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.utils import VietnameseTextPreprocessor


class NewsLogisticRegression:
    """Mô hình Logistic Regression cho phân loại tin tức"""
    
    def __init__(self, max_features=10000, random_state=42):
        """
        Args:
            max_features: Số lượng features tối đa cho TF-IDF
            random_state: Random seed
        """
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigram và bigram
            min_df=2,  # Từ phải xuất hiện ít nhất 2 lần
            max_df=0.95,  # Bỏ qua từ xuất hiện > 95% documents
            lowercase=True,
            token_pattern=r'\b\w+\b'  # Token pattern cho tiếng Việt
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',  # Solver phù hợp cho multi-class
            C=1.0  # Regularization strength
        )
        self.label_encoder = {}  # Map category -> label
        self.label_decoder = {}  # Map label -> category
        self.is_trained = False
    
    def prepare_data(self, df, text_column='combined_text', label_column='category'):
        """
        Chuẩn bị dữ liệu cho training
        
        Args:
            df: DataFrame chứa dữ liệu
            text_column: Tên cột chứa text (title + content)
            label_column: Tên cột chứa label (category)
        """
        # Tạo combined text nếu chưa có
        if text_column not in df.columns:
            if 'title' in df.columns and 'content' in df.columns:
                df[text_column] = df.apply(
                    lambda row: VietnameseTextPreprocessor.combine_title_content(
                        row['title'], row['content']
                    ),
                    axis=1
                )
            else:
                raise ValueError("Cần có cột 'title' và 'content' hoặc 'combined_text'")
        
        # Loại bỏ rows có text rỗng
        df = df[df[text_column].str.strip() != ''].copy()
        
        # Encode labels
        unique_labels = sorted(df[label_column].unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for idx, label in enumerate(unique_labels)}
        
        X = df[text_column].values
        y = df[label_column].map(self.label_encoder).values
        
        return X, y, df
    
    def train(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Train model
        
        Args:
            X: Features (text)
            y: Labels
            test_size: Tỷ lệ test set
            validation_size: Tỷ lệ validation set (từ train set)
        """
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=self.random_state, stratify=y_train
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Vectorize text
        print("Đang vectorize text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        # Train model
        print("Đang train model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        print("\n=== Kết quả trên Validation Set ===")
        y_val_pred = self.model.predict(X_val_vec)
        self._print_metrics(y_val, y_val_pred, "Validation")
        
        print("\n=== Kết quả trên Test Set ===")
        y_test_pred = self.model.predict(X_test_vec)
        self._print_metrics(y_test, y_test_pred, "Test")
        
        self.is_trained = True
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred
        }
    
    def _print_metrics(self, y_true, y_pred, set_name):
        """In các metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{set_name} Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        
        # Classification report
        print(f"\n{set_name} Classification Report:")
        labels = [self.label_decoder[i] for i in sorted(self.label_decoder.keys())]
        print(classification_report(
            y_true, y_pred,
            target_names=labels,
            zero_division=0
        ))
    
    def predict(self, text, return_proba=False):
        """
        Predict category cho text
        
        Args:
            text: Text cần predict (có thể là title hoặc title + content)
            return_proba: Có trả về probability không
        
        Returns:
            Category name (và probability nếu return_proba=True)
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train. Hãy gọi train() trước.")
        
        # Preprocess text
        if isinstance(text, tuple):
            # Nếu là (title, content)
            text = VietnameseTextPreprocessor.combine_title_content(text[0], text[1])
        else:
            text = VietnameseTextPreprocessor.normalize_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([text])
        
        # Predict
        if return_proba:
            proba = self.model.predict_proba(text_vec)[0]
            pred_label = self.model.predict(text_vec)[0]
            category = self.label_decoder[pred_label]
            
            # Tạo dict với probability cho mỗi category
            proba_dict = {
                self.label_decoder[i]: float(prob)
                for i, prob in enumerate(proba)
            }
            proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
            
            return category, proba_dict
        else:
            pred_label = self.model.predict(text_vec)[0]
            return self.label_decoder[pred_label]
    
    def save(self, model_dir='models/saved/logistic_regression'):
        """Lưu model"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'labels.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(label_encoder_path, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'label_decoder': self.label_decoder
            }, f)
        
        print(f"Đã lưu model vào {model_dir}")
    
    def load(self, model_dir='models/saved/logistic_regression'):
        """Load model"""
        model_path = os.path.join(model_dir, 'model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'labels.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            labels = pickle.load(f)
            self.label_encoder = labels['label_encoder']
            self.label_decoder = labels['label_decoder']
        
        self.is_trained = True
        print(f"Đã load model từ {model_dir}")

