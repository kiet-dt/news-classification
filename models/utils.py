"""
Utilities chung cho tất cả các mô hình
"""
import pandas as pd
import re


class VietnameseTextPreprocessor:
    """Xử lý text tiếng Việt - dùng chung cho tất cả models"""
    
    @staticmethod
    def normalize_text(text):
        """Chuẩn hóa text tiếng Việt"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # Loại bỏ ký tự đặc biệt không cần thiết
        text = re.sub(r'[^\w\s]', ' ', text)
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        # Chuyển về lowercase
        text = text.lower().strip()
        return text
    
    @staticmethod
    def combine_title_content(title, content):
        """Kết hợp title và content"""
        title = VietnameseTextPreprocessor.normalize_text(title)
        content = VietnameseTextPreprocessor.normalize_text(content)
        
        # Kết hợp title và content (title có trọng số cao hơn)
        combined = f"{title} {title} {content}"  # Lặp title 2 lần để tăng trọng số
        return combined

