"""
Ensemble Model cho phân loại tin tức tiếng Việt
"""
import sys
import os
import importlib.util

# Thêm root project vào path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import trực tiếp từ file ensemble.py (không phải folder)
ensemble_file = os.path.join(project_root, 'models', 'ensemble.py')
spec = importlib.util.spec_from_file_location("ensemble_model", ensemble_file)
ensemble_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble_module)

NewsEnsemble = ensemble_module.NewsEnsemble

__all__ = ['NewsEnsemble']

