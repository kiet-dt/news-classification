"""
PhoBERT Model cho phân loại tin tức tiếng Việt
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# Thêm root project vào path để import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.utils import VietnameseTextPreprocessor


class NewsDataset(Dataset):
    """Dataset class cho PhoBERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NewsPhoBERT:
    """Mô hình PhoBERT cho phân loại tin tức"""
    
    def __init__(self, model_name='vinai/phobert-base', max_length=256, random_state=42):
        """
        Args:
            model_name: Tên model PhoBERT (mặc định: 'vinai/phobert-base')
            max_length: Độ dài tối đa của sequence
            random_state: Random seed
        """
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        
        # Khởi tạo tokenizer và model
        print(f"Đang load tokenizer và model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model sẽ được load khi biết số lượng labels
        self.model = None
        self.label_encoder = {}  # Map category -> label
        self.label_decoder = {}  # Map label -> category
        self.is_trained = False
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {self.device}")
    
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
        
        # Load model với số lượng labels
        num_labels = len(self.label_encoder)
        print(f"Đang load model với {num_labels} labels...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        X = df[text_column].values
        y = df[label_column].map(self.label_encoder).values
        
        return X, y, df
    
    def train(self, X, y, test_size=0.2, validation_size=0.1, 
              batch_size=16, num_epochs=3, learning_rate=2e-5):
        """
        Train model
        
        Args:
            X: Features (text)
            y: Labels
            test_size: Tỷ lệ test set
            validation_size: Tỷ lệ validation set (từ train set)
            batch_size: Batch size
            num_epochs: Số epochs
            learning_rate: Learning rate
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
        
        # Tạo datasets
        train_dataset = NewsDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = NewsDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = NewsDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        # Training arguments
        output_dir = 'models/saved/phobert/training_output'
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            seed=self.random_state,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("\n=== Bắt đầu training ===")
        trainer.train()
        
        # Evaluate trên validation set
        print("\n=== Kết quả trên Validation Set ===")
        val_predictions = trainer.predict(val_dataset)
        y_val_pred = np.argmax(val_predictions.predictions, axis=1)
        self._print_metrics(y_val, y_val_pred, "Validation")
        
        # Evaluate trên test set
        print("\n=== Kết quả trên Test Set ===")
        test_predictions = trainer.predict(test_dataset)
        y_test_pred = np.argmax(test_predictions.predictions, axis=1)
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
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(proba)
        
        category = self.label_decoder[pred_label]
        
        if return_proba:
            # Tạo dict với probability cho mỗi category
            proba_dict = {
                self.label_decoder[i]: float(prob)
                for i, prob in enumerate(proba)
            }
            proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
            return category, proba_dict
        else:
            return category
    
    def save(self, model_dir='models/saved/phobert'):
        """Lưu model"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Lưu model và tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Lưu label encoders
        label_encoder_path = os.path.join(model_dir, 'labels.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'label_decoder': self.label_decoder,
                'model_name': self.model_name,
                'max_length': self.max_length
            }, f)
        
        print(f"Đã lưu model vào {model_dir}")
    
    def load(self, model_dir='models/saved/phobert'):
        """Load model"""
        # Load model và tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load label encoders
        label_encoder_path = os.path.join(model_dir, 'labels.pkl')
        with open(label_encoder_path, 'rb') as f:
            labels = pickle.load(f)
            self.label_encoder = labels['label_encoder']
            self.label_decoder = labels['label_decoder']
            self.model_name = labels.get('model_name', 'vinai/phobert-base')
            self.max_length = labels.get('max_length', 256)
        
        self.is_trained = True
        print(f"Đã load model từ {model_dir}")

