import logging

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional

from datetime import datetime

from src.schemas import (
    CustomerFeatures, 
    ChurnPrediction, 
    BatchPredictionRequest,
    BatchPredictionResponse, 
    ModelInfo, 
    HealthCheck
)
from src.predictor import ChurnPredictor, get_predictor
from src.dependencies import get_model_info

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Предсказание оттока клиентов API",
    description="REST API для предсказания оттока клиентов с использованием машинного обучения",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Предсказание оттока клиентов API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check(predictor: ChurnPredictor = Depends(get_predictor)):
    """Проверка здоровья API"""
    return HealthCheck(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfo)
async def model_info(predictor: ChurnPredictor = Depends(get_predictor)):
    """Получение информации о модели"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    
    try:
        info = predictor.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения информации о модели: {str(e)}"
        )

@app.post("/predict", response_model=ChurnPrediction)
async def predict(
    features: CustomerFeatures,
    customer_id: Optional[int] = None,
    threshold: float = 0.5,
    predictor: ChurnPredictor = Depends(get_predictor)
):
    """Предсказание оттока для одного клиента"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    
    # Валидация threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Порог (threshold) должен быть между 0 и 1"
        )
    
    try:
        # Конвертируем Pydantic модель в словарь
        features_dict = features.model_dump()
        
        # Выполняем предсказание
        prediction = predictor.predict_single(features_dict, customer_id, threshold)
        
        logger.info(f"Предсказание для клиента {customer_id}: вероятность {prediction['churn_probability']:.2%}")
        return ChurnPrediction(**prediction)
        
    except ValueError as e:
        logger.error(f"Ошибка валидации данных: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка валидации данных: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка предсказания: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    threshold: float = 0.5,
    predictor: ChurnPredictor = Depends(get_predictor)
):
    """Пакетное предсказание оттока для нескольких клиентов"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    
    # Валидация threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Порог (threshold) должен быть между 0 и 1"
        )
    
    # Валидация customer_ids если они предоставлены
    if request.customer_ids and len(request.customer_ids) != len(request.customers):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Количество customer_ids должно совпадать с количеством customers"
        )
    
    try:
        # Конвертируем Pydantic модели в словари
        customers_dict = [customer.model_dump() for customer in request.customers]
        
        # Выполняем пакетное предсказание
        predictions = predictor.predict_batch(customers_dict, request.customer_ids, threshold)
        
        # Фильтруем успешные предсказания
        successful_predictions = [p for p in predictions if "error" not in p]
        errors = [p for p in predictions if "error" in p]
        
        if errors:
            logger.warning(f"Ошибки в пакетном предсказании: {len(errors)} ошибок")
        
        # Вычисляем метрики
        total_customers = len(successful_predictions)
        if total_customers > 0:
            churn_rate = sum(1 for p in successful_predictions if p["prediction"]) / total_customers
            avg_probability = sum(p["churn_probability"] for p in successful_predictions) / total_customers
        else:
            churn_rate = 0.0
            avg_probability = 0.0
        
        # Конвертируем в Pydantic модели
        prediction_models = []
        for pred in successful_predictions:
            # Создаем ChurnPrediction только если нет ошибки
            if "error" not in pred:
                prediction_models.append(ChurnPrediction(**pred))
        
        logger.info(f"Пакетное предсказание: {total_customers} клиентов, отток: {churn_rate:.2%}")
        
        return BatchPredictionResponse(
            predictions=prediction_models,
            total_customers=total_customers,
            churn_rate=churn_rate,
            avg_probability=avg_probability
        )
        
    except Exception as e:
        logger.error(f"Ошибка пакетного предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка пакетного предсказания: {str(e)}"
        )

@app.get("/features", summary="Получить список признаков модели")
async def get_features(predictor: ChurnPredictor = Depends(get_predictor)):
    """Получение списка признаков, которые ожидает модель"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )
    
    try:
        info = predictor.get_model_info()
        return {
            "features": info["feature_names"],
            "total_features": len(info["feature_names"]),
            "required_features": [
                "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
                "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
                "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
                "OrderCount", "DaySinceLastOrder", "CashbackAmount"
            ],
            "optional_features": [
                "AvgOrdersPerMonth", "AvgCashbackPerOrder",
                "EngagementScore", "SatisfactionComplainRatio"
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка получения списка признаков: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения списка признаков: {str(e)}"
        )

@app.get("/version", summary="Версия API")
async def get_version():
    """Получение версии API"""
    return {
        "api_version": "1.0.0",
        "model_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик общих исключений"""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Внутренняя ошибка сервера"}
    )

def get_app():
    return app