import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Класс для загрузки модели и выполнения предсказаний"""
    
    def __init__(self, model_path: str = "../models/final/churn_model.pkl"):
        self.model = None
        self.feature_names = None
        self.model_info = None
        self.model_version = "1.0.0"
        self.is_loaded = False
        
        # Определяем пути относительно текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))

        logger.info(f"✅ Модель загружена: {type(self.model).__name__}")
        
        # Полные пути к файлам модели
        full_model_path = os.path.join(base_dir, "models/final/churn_model.pkl")
        feature_names_path = os.path.join(base_dir, "models/final/feature_names.pkl")
        model_card_path = os.path.join(base_dir, "models/final/model_card.json")
        
        try:
            # Загрузка модели
            self.model = joblib.load(full_model_path)
            logger.info(f"✅ Модель загружена: {type(self.model).__name__}")
            
            # Загрузка названий признаков
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"✅ Загружено признаков: {len(self.feature_names)}")
            
            self.model_info = {
                "model_name": "Предиктор оттоков клиентов",
                "model_type": type(self.model).__name__,
                "performance": {},
                "training_date": "unknown",
                "feature_count": len(self.feature_names)
            }
            
            self.is_loaded = True
            logger.info("✅ Модель успешно инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.is_loaded = False
    
    def _prepare_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        # Создаем копию словаря для модификации
        processed_features = features.copy()
        
        # Вычисляем дополнительные признаки если они не предоставлены
        if processed_features.get('AvgOrdersPerMonth') is None:
            if processed_features.get('Tenure', 0) > 0:
                processed_features['AvgOrdersPerMonth'] = (
                    processed_features.get('OrderCount', 0) / processed_features['Tenure']
                )
            else:
                processed_features['AvgOrdersPerMonth'] = 0
        
        if processed_features.get('AvgCashbackPerOrder') is None:
            if processed_features.get('OrderCount', 0) > 0:
                processed_features['AvgCashbackPerOrder'] = (
                    processed_features.get('CashbackAmount', 0) / processed_features['OrderCount']
                )
            else:
                processed_features['AvgCashbackPerOrder'] = 0
        
        if processed_features.get('EngagementScore') is None:
            processed_features['EngagementScore'] = (
                processed_features.get('HourSpendOnApp', 0) * 
                processed_features.get('OrderCount', 0)
            )
        
        if processed_features.get('SatisfactionComplainRatio') is None:
            complain = processed_features.get('Complain', 0)
            processed_features['SatisfactionComplainRatio'] = (
                processed_features.get('SatisfactionScore', 0) / (complain + 1)
            )
        
        # Создаем DataFrame с правильным порядком признаков
        input_data = {}
        for feature in self.feature_names:
            if feature in processed_features:
                input_data[feature] = [processed_features[feature]]
            else:
                logger.warning(f"Признак '{feature}' отсутствует в входных данных")
                input_data[feature] = [0.0]  # Заполняем нулем
        
        return pd.DataFrame(input_data)
    
    def predict_single(self, features: Dict[str, Any], customer_id: Optional[int] = None, 
                       threshold: float = 0.5) -> Dict[str, Any]:
        """Предсказание для одного клиента"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        try:
            # Подготавливаем признаки
            input_df = self._prepare_features(features)
            
            # Выполняем предсказание
            probabilities = self.model.predict_proba(input_df)
            churn_probability = float(probabilities[0, 1])  # Вероятность класса 1 (отток)
            prediction = churn_probability > threshold
            
            return {
                "customer_id": customer_id,
                "churn_probability": churn_probability,
                "prediction": bool(prediction),
                "threshold": threshold,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise
    
    def predict_batch(self, customers: List[Dict[str, Any]], 
                     customer_ids: Optional[List[int]] = None,
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Пакетное предсказание для нескольких клиентов"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        predictions = []
        
        for i, customer_features in enumerate(customers):
            customer_id = customer_ids[i] if customer_ids and i < len(customer_ids) else None
            
            try:
                prediction = self.predict_single(customer_features, customer_id, threshold)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Ошибка предсказания для клиента {i}: {e}")
                # Добавляем ошибку вместо предсказания
                predictions.append({
                    "customer_id": customer_id,
                    "error": str(e),
                    "churn_probability": None,
                    "prediction": None,
                    "threshold": threshold,
                    "model_version": self.model_version,
                    "timestamp": datetime.now().isoformat()
                })
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        return {
            "model_name": self.model_info.get("model_name", "Customer Churn Predictor"),
            "model_version": self.model_version,
            "model_type": self.model_info.get("model_type", type(self.model).__name__),
            "feature_count": len(self.feature_names),
            "training_date": self.model_info.get("training_date", "unknown"),
            "performance": self.model_info.get("performance", {}),
            "feature_names": self.feature_names.tolist() if hasattr(self.feature_names, 'tolist') else list(self.feature_names)
        }

# Глобальный экземпляр предсказателя
_predictor_instance = None

def get_predictor() -> ChurnPredictor:
    """Функция для получения экземпляра предсказателя (синглтон)"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ChurnPredictor()
    return _predictor_instance