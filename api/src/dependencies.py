from .predictor import get_predictor
from typing import Dict, Any

def get_predictor():
    """Зависимость для получения предсказателя"""
    return get_predictor()

def get_model_info() -> Dict[str, Any]:
    """Зависимость для получения информации о модели"""
    predictor = get_predictor()
    return predictor.get_model_info()
