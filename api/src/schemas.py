from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict

class CustomerFeatures(BaseModel):
    """Модель для признаков клиента"""
    Tenure: float = Field(..., description="Время пользования услугами (месяцы)")
    CityTier: float = Field(..., description="Уровень города (1-3)")
    WarehouseToHome: float = Field(..., description="Расстояние до склада")
    HourSpendOnApp: float = Field(..., description="Часов в приложении")
    NumberOfDeviceRegistered: float = Field(..., description="Количество устройств")
    SatisfactionScore: float = Field(..., description="Удовлетворенность (1-5)")
    NumberOfAddress: float = Field(..., description="Количество адресов")
    Complain: float = Field(..., description="Жалобы (0-1)")
    OrderAmountHikeFromlastYear: float = Field(..., description="Рост заказов (%)")
    CouponUsed: float = Field(..., description="Использовано купонов")
    OrderCount: float = Field(..., description="Количество заказов")
    DaySinceLastOrder: float = Field(..., description="Дней с последнего заказа")
    CashbackAmount: float = Field(..., description="Сумма кэшбэка")
    
    # Опциональные признаки, которые могут быть вычислены
    AvgOrdersPerMonth: Optional[float] = Field(None, description="Средние заказы в месяц")
    AvgCashbackPerOrder: Optional[float] = Field(None, description="Средний кэшбэк за заказ")
    EngagementScore: Optional[float] = Field(None, description="Оценка вовлеченности")
    SatisfactionComplainRatio: Optional[float] = Field(None, description="Отношение удовлетворенности к жалобам")
    
    @field_validator('SatisfactionScore')
    @classmethod
    def validate_satisfaction(cls, v: float) -> float:
        if v < 1 or v > 5:
            raise ValueError('SatisfactionScore должен быть между 1 и 5')
        return v
    
    @field_validator('CityTier')
    @classmethod
    def validate_city_tier(cls, v: float) -> float:
        if v < 1 or v > 3:
            raise ValueError('CityTier должен быть между 1 и 3')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "Tenure": 12.0,
                "CityTier": 2.0,
                "WarehouseToHome": 8.0,
                "HourSpendOnApp": 3.5,
                "NumberOfDeviceRegistered": 3.0,
                "SatisfactionScore": 4.0,
                "NumberOfAddress": 5.0,
                "Complain": 0.0,
                "OrderAmountHikeFromlastYear": 15.0,
                "CouponUsed": 4.0,
                "OrderCount": 25.0,
                "DaySinceLastOrder": 3.0,
                "CashbackAmount": 250.0,
                "AvgOrdersPerMonth": 2.08,
                "AvgCashbackPerOrder": 10.0,
                "EngagementScore": 87.5,
                "SatisfactionComplainRatio": 4.0
            }
        }

class ChurnPrediction(BaseModel):
    """Модель для результата предсказания"""
    customer_id: Optional[int] = Field(None, description="ID клиента")
    churn_probability: float = Field(..., ge=0, le=1, description="Вероятность оттока")
    prediction: bool = Field(..., description="Предсказание оттока")
    threshold: float = Field(..., description="Порог классификации")
    model_version: str = Field(..., description="Версия модели")
    timestamp: str = Field(..., description="Время предсказания")

class BatchPredictionRequest(BaseModel):
    """Модель для batch предсказаний"""
    customers: List[CustomerFeatures] = Field(..., description="Список клиентов")
    customer_ids: Optional[List[int]] = Field(None, description="ID клиентов")

class BatchPredictionResponse(BaseModel):
    """Модель для batch ответа"""
    predictions: List[ChurnPrediction] = Field(..., description="Список предсказаний")
    total_customers: int = Field(..., description="Общее количество клиентов")
    churn_rate: float = Field(..., description="Процент оттока в батче")
    avg_probability: float = Field(..., description="Средняя вероятность оттока")

class ModelInfo(BaseModel):
    """Информация о модели"""
    model_name: str = Field(..., description="Название модели")
    model_version: str = Field(..., description="Версия модели")
    model_type: str = Field(..., description="Тип модели")
    feature_count: int = Field(..., description="Количество признаков")
    training_date: str = Field(..., description="Дата обучения")
    performance: Dict[str, float] = Field(..., description="Метрики производительности")
    feature_names: List[str] = Field(..., description="Список признаков")

class HealthCheck(BaseModel):
    """Модель для проверки здоровья"""
    status: str = Field(..., description="Статус API")
    model_loaded: bool = Field(..., description="Модель загружена")
    timestamp: str = Field(..., description="Временная метка")
    version: str = Field(..., description="Версия API")