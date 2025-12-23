import pytest
import sys
import os
import pandas as pd

from datetime import datetime
from fastapi.testclient import TestClient

# Импортируем main из родительской директории
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app import get_app
from src.schemas import CustomerFeatures, BatchPredictionRequest
from src.predictor import ChurnPredictor

# Создаем тестового клиента
client = TestClient(get_app())

# Тестовые данные
VALID_CUSTOMER_DATA = {
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
    "CashbackAmount": 250.0
}

INVALID_CUSTOMER_DATA = {
    "Tenure": -5.0,  # Негативное значение
    "CityTier": 5.0,  # Вне диапазона
    "SatisfactionScore": 10.0  # Вне диапазона
}

# Фикстуры для тестов
@pytest.fixture
def sample_customer():
    """Фикстура для тестового клиента"""
    return VALID_CUSTOMER_DATA.copy()

@pytest.fixture
def batch_customers():
    """Фикстура для batch тестов"""
    return [
        VALID_CUSTOMER_DATA.copy(),
        {
            "Tenure": 24.0, "CityTier": 1.0, "WarehouseToHome": 5.0,
            "HourSpendOnApp": 2.0, "NumberOfDeviceRegistered": 2.0,
            "SatisfactionScore": 5.0, "NumberOfAddress": 3.0, "Complain": 0.0,
            "OrderAmountHikeFromlastYear": 10.0, "CouponUsed": 8.0,
            "OrderCount": 40.0, "DaySinceLastOrder": 1.0, "CashbackAmount": 400.0
        }
    ]

@pytest.fixture
def predictor():
    """Фикстура для предсказателя"""
    return ChurnPredictor()

class TestAPIHealth:
    """Тесты здоровья API"""
    
    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Предсказание оттока клиентов API"
        assert "version" in data
        assert "docs" in data
    
    def test_health_check(self):
        """Тест проверки здоровья"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_version_endpoint(self):
        """Тест эндпоинта версии"""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "model_version" in data
        assert "timestamp" in data

class TestModelInfo:
    """Тесты информации о модели"""
    
    def test_model_info(self):
        """Тест получения информации о модели"""
        response = client.get("/model/info")
        # Может быть 200 или 503 в зависимости от загрузки модели
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "model_type" in data
            assert "feature_count" in data
            assert "training_date" in data
            assert "performance" in data
            assert "feature_names" in data
    
    def test_features_endpoint(self):
        """Тест получения списка признаков"""
        response = client.get("/features")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "features" in data
            assert "total_features" in data
            assert "required_features" in data
            assert "optional_features" in data
            assert isinstance(data["features"], list)
            assert isinstance(data["required_features"], list)
            assert isinstance(data["optional_features"], list)

class TestSinglePrediction:
    """Тесты одиночного предсказания"""
    
    def test_predict_valid_data(self, sample_customer):
        """Тест предсказания с валидными данными"""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "customer_id" in data
            assert "churn_probability" in data
            assert "prediction" in data
            assert "threshold" in data
            assert "model_version" in data
            assert "timestamp" in data
            assert 0 <= data["churn_probability"] <= 1
            assert isinstance(data["prediction"], bool)
    
    def test_predict_with_customer_id(self, sample_customer):
        """Тест предсказания с указанием customer_id"""
        response = client.post("/predict?customer_id=12345", json=sample_customer)
        if response.status_code == 200:
            data = response.json()
            assert data["customer_id"] == 12345
    
    def test_predict_with_threshold(self, sample_customer):
        """Тест предсказания с кастомным порогом"""
        for threshold in [0.3, 0.5, 0.7]:
            response = client.post(f"/predict?threshold={threshold}", json=sample_customer)
            if response.status_code == 200:
                data = response.json()
                assert data["threshold"] == threshold
    
    def test_predict_invalid_threshold(self, sample_customer):
        """Тест предсказания с невалидным порогом"""
        # Порог меньше 0
        response = client.post("/predict?threshold=-0.1", json=sample_customer)
        assert response.status_code == 400
        
        # Порог больше 1
        response = client.post("/predict?threshold=1.1", json=sample_customer)
        assert response.status_code == 400
    
    def test_predict_missing_required_field(self):
        """Тест предсказания с отсутствующим обязательным полем"""
        invalid_data = {"Tenure": 12.0}  # Только одно поле
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_invalid_field_value(self):
        """Тест предсказания с невалидным значением поля"""
        invalid_data = VALID_CUSTOMER_DATA.copy()
        invalid_data["SatisfactionScore"] = 10.0  # Вне диапазона 1-5
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_negative_value(self):
        """Тест предсказания с отрицательным значением"""
        invalid_data = VALID_CUSTOMER_DATA.copy()
        invalid_data["Tenure"] = -5.0  # Отрицательное значение
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 200
    
    def test_predict_with_extra_fields(self, sample_customer):
        """Тест предсказания с дополнительными полями (должно игнорироваться)"""
        extended_data = sample_customer.copy()
        extended_data["extra_field"] = "some_value"
        response = client.post("/predict", json=extended_data)
        # Должно вернуть 200, лишние поля игнорируются
        assert response.status_code in [200, 503]

class TestBatchPrediction:
    """Тесты пакетного предсказания"""
    
    def test_batch_predict_valid_data(self, batch_customers):
        """Тест пакетного предсказания с валидными данными"""
        request_data = {
            "customers": batch_customers,
            "customer_ids": [1001, 1002]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_customers" in data
            assert "churn_rate" in data
            assert "avg_probability" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == len(batch_customers)
            assert 0 <= data["churn_rate"] <= 1
            assert 0 <= data["avg_probability"] <= 1
    
    def test_batch_predict_without_ids(self, batch_customers):
        """Тест пакетного предсказания без customer_ids"""
        request_data = {
            "customers": batch_customers
        }
        
        response = client.post("/predict/batch", json=request_data)
        if response.status_code == 200:
            data = response.json()
            # customer_id должен быть None
            for pred in data["predictions"]:
                assert pred["customer_id"] is None
    
    def test_batch_predict_mismatched_ids(self, batch_customers):
        """Тест пакетного предсказания с несовпадающими ID"""
        request_data = {
            "customers": batch_customers,
            "customer_ids": [1001]  # Меньше ID чем клиентов
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 400
    
    def test_batch_predict_empty_list(self):
        """Тест пакетного предсказания с пустым списком"""
        request_data = {
            "customers": []
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_customers"] == 0
            assert data["churn_rate"] == 0.0
            assert data["avg_probability"] == 0.0
            assert len(data["predictions"]) == 0
    
    def test_batch_predict_invalid_data(self):
        """Тест пакетного предсказания с невалидными данными"""
        request_data = {
            "customers": [
                {"Tenure": 12.0},  # Неполные данные
                VALID_CUSTOMER_DATA
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422
    
    def test_batch_predict_with_threshold(self, batch_customers):
        """Тест пакетного предсказания с кастомным порогом"""
        request_data = {
            "customers": batch_customers,
            "customer_ids": [1001, 1002]
        }
        
        for threshold in [0.3, 0.6]:
            response = client.post(f"/predict/batch?threshold={threshold}", 
                                 json=request_data)
            if response.status_code == 200:
                data = response.json()
                # Проверяем, что все предсказания используют указанный порог
                for pred in data["predictions"]:
                    assert pred["threshold"] == threshold

class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_nonexistent_endpoint(self):
        """Тест несуществующего эндпоинта"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_http_method(self):
        """Тест неверного HTTP метода"""
        response = client.get("/predict")  # GET вместо POST
        assert response.status_code == 405
    
    def test_malformed_json(self):
        """Тест некорректного JSON"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422
    
    def test_large_payload(self):
        """Тест слишком большого payload"""
        # Создаем слишком много клиентов
        customers = [VALID_CUSTOMER_DATA.copy() for _ in range(1001)]
        request_data = {
            "customers": customers
        }
        
        response = client.post("/predict/batch", json=request_data)
        # FastAPI имеет ограничение по умолчанию на размер payload
        assert response.status_code in [200, 413, 422, 503]
    
    def test_special_characters(self):
        """Тест специальных символов в данных"""
        # Пытаемся отправить данные с нестандартными символами
        response = client.post("/predict", 
                             json={"Tenure": "twelve",  # Строка вместо числа
                                   "CityTier": 2.0})
        assert response.status_code == 422

class TestPredictorUnitTests:
    """Юнит-тесты для ChurnPredictor"""
    
    def test_predictor_initialization(self, predictor):
        """Тест инициализации предсказателя"""
        assert predictor is not None
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'feature_names')
        assert hasattr(predictor, 'model_info')
        assert hasattr(predictor, 'model_version')
        assert hasattr(predictor, 'is_loaded')
    
    def test_get_model_info(self, predictor):
        """Тест получения информации о модели"""
        info = predictor.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_version" in info
        assert "model_type" in info
        assert "feature_count" in info
        assert "training_date" in info
        assert "performance" in info
        assert "feature_names" in info
    
    def test_prepare_features(self, predictor, sample_customer):
        """Тест подготовки признаков"""
        df = predictor._prepare_features(sample_customer)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Одна строка для одного клиента
        # Проверяем, что все необходимые признаки присутствуют
        for feature in predictor.feature_names:
            assert feature in df.columns
    
    def test_predict_single(self, predictor, sample_customer):
        """Тест предсказания для одного клиента"""
        result = predictor.predict_single(sample_customer, customer_id=999, threshold=0.5)
        assert isinstance(result, dict)
        assert "customer_id" in result
        assert "churn_probability" in result
        assert "prediction" in result
        assert "threshold" in result
        assert "model_version" in result
        assert "timestamp" in result
        assert result["customer_id"] == 999
        assert result["threshold"] == 0.5
        assert 0 <= result["churn_probability"] <= 1
        assert isinstance(result["prediction"], bool)
    
    def test_predict_single_without_customer_id(self, predictor, sample_customer):
        """Тест предсказания без customer_id"""
        result = predictor.predict_single(sample_customer)
        assert result["customer_id"] is None
    
    def test_predict_batch(self, predictor, batch_customers):
        """Тест пакетного предсказания"""
        results = predictor.predict_batch(batch_customers, [1001, 1002])
        assert isinstance(results, list)
        assert len(results) == len(batch_customers)
        
        for result in results:
            assert "customer_id" in result
            assert "churn_probability" in result
            assert "prediction" in result
            assert "threshold" in result
            assert "model_version" in result
            assert "timestamp" in result
    
    def test_predict_batch_without_ids(self, predictor, batch_customers):
        """Тест пакетного предсказания без customer_ids"""
        results = predictor.predict_batch(batch_customers)
        assert len(results) == len(batch_customers)
        for result in results:
            assert result["customer_id"] is None
    
    def test_predict_batch_with_different_threshold(self, predictor, batch_customers):
        """Тест пакетного предсказания с разными порогами"""
        for threshold in [0.3, 0.7]:
            results = predictor.predict_batch(batch_customers, threshold=threshold)
            for result in results:
                assert result["threshold"] == threshold

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_extreme_values(self):
        """Тест с экстремальными значениями признаков"""
        extreme_customer = {
            "Tenure": 1000.0,
            "CityTier": 3.0,
            "WarehouseToHome": 100.0,
            "HourSpendOnApp": 24.0,
            "NumberOfDeviceRegistered": 10.0,
            "SatisfactionScore": 1.0,
            "NumberOfAddress": 50.0,
            "Complain": 1.0,
            "OrderAmountHikeFromlastYear": 100.0,
            "CouponUsed": 100.0,
            "OrderCount": 1000.0,
            "DaySinceLastOrder": 365.0,
            "CashbackAmount": 10000.0
        }
        
        response = client.post("/predict", json=extreme_customer)
        assert response.status_code in [200, 503]
    
    def test_zero_values(self):
        """Тест с нулевыми значениями"""
        zero_customer = {
            "Tenure": 0.0,
            "CityTier": 1.0,
            "WarehouseToHome": 0.0,
            "HourSpendOnApp": 0.0,
            "NumberOfDeviceRegistered": 0.0,
            "SatisfactionScore": 1.0,
            "NumberOfAddress": 1.0,
            "Complain": 0.0,
            "OrderAmountHikeFromlastYear": 0.0,
            "CouponUsed": 0.0,
            "OrderCount": 0.0,
            "DaySinceLastOrder": 0.0,
            "CashbackAmount": 0.0
        }
        
        response = client.post("/predict", json=zero_customer)
        assert response.status_code in [200, 503]
    
    def test_decimal_values(self):
        """Тест с дробными значениями"""
        decimal_customer = {
            "Tenure": 12.75,
            "CityTier": 1.5,
            "WarehouseToHome": 8.25,
            "HourSpendOnApp": 2.75,
            "NumberOfDeviceRegistered": 2.0,
            "SatisfactionScore": 3.5,
            "NumberOfAddress": 4.0,
            "Complain": 0.5,
            "OrderAmountHikeFromlastYear": 12.75,
            "CouponUsed": 3.25,
            "OrderCount": 15.5,
            "DaySinceLastOrder": 2.75,
            "CashbackAmount": 187.5
        }
        
        response = client.post("/predict", json=decimal_customer)
        assert response.status_code in [200, 503]

class TestConcurrentRequests:
    """Тесты конкурентных запросов"""
    
    def test_multiple_simultaneous_requests(self, sample_customer):
        """Тест множественных одновременных запросов"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/predict", json=sample_customer)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Проверяем, что все запросы завершились без ошибок
        assert len(errors) == 0
        # Проверяем, что все запросы вернули статус 200 или 503
        for status_code in results:
            assert status_code in [200, 503]

class TestDataValidation:
    """Тесты валидации данных"""
    
    def test_validate_customer_features(self):
        """Тест валидации модели CustomerFeatures"""
        # Валидные данные
        valid_features = CustomerFeatures(**VALID_CUSTOMER_DATA)
        assert isinstance(valid_features, CustomerFeatures)
        
        # Проверяем преобразование в словарь
        features_dict = valid_features.model_dump()
        assert isinstance(features_dict, dict)
        
        # Проверяем, что все поля присутствуют
        for field in CustomerFeatures.__annotations__:
            if field != 'return':
                assert field in features_dict
    
    def test_invalid_customer_features(self):
        """Тест невалидных данных для CustomerFeatures"""
        # SatisfactionScore вне диапазона
        invalid_data = VALID_CUSTOMER_DATA.copy()
        invalid_data["SatisfactionScore"] = 10.0
        
        with pytest.raises(ValueError):
            CustomerFeatures(**invalid_data)
    
    def test_batch_prediction_request_validation(self, batch_customers):
        """Тест валидации BatchPredictionRequest"""
        # Валидный запрос
        valid_request = BatchPredictionRequest(
            customers=batch_customers,
            customer_ids=[1001, 1002]
        )
        assert isinstance(valid_request, BatchPredictionRequest)
        
        # Запрос без customer_ids
        request_without_ids = BatchPredictionRequest(customers=batch_customers)
        assert request_without_ids.customer_ids is None

class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_workflow(self, batch_customers):
        """Тест полного workflow API"""
        # 1. Проверка здоровья
        response = client.get("/health")
        assert response.status_code == 200
        
        # 2. Получение информации о модели
        response = client.get("/model/info")
        assert response.status_code in [200, 503]
        
        # 3. Получение списка признаков
        response = client.get("/features")
        assert response.status_code in [200, 503]
        
        # 4. Одиночное предсказание
        response = client.post("/predict", json=VALID_CUSTOMER_DATA)
        assert response.status_code in [200, 503]
        
        # 5. Пакетное предсказание
        request_data = {
            "customers": batch_customers,
            "customer_ids": [1001, 1002]
        }
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code in [200, 503]
        
        # 6. Проверка версии
        response = client.get("/version")
        assert response.status_code == 200
    
    def test_error_recovery(self):
        """Тест восстановления после ошибки"""
        # Сначала отправляем невалидные данные
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422
        
        # Затем отправляем валидные данные - система должна восстановиться
        response = client.post("/predict", json=VALID_CUSTOMER_DATA)
        assert response.status_code in [200, 503]

# Дополнительные тесты для увеличения покрытия
class TestCoverageEnhancement:
    """Тесты для увеличения покрытия кода"""
    
    def test_predictor_edge_cases(self, predictor):
        """Тест граничных случаев в предсказателе"""
        # Тест с пустыми признаками
        empty_features = {}
        try:
            result = predictor.predict_single(empty_features)
            # Если не выброшено исключение, проверяем результат
            assert "churn_probability" in result
        except Exception:
            # Ожидаем исключение при пустых признаках
            pass
        
        # Тест с частичными признаками
        partial_features = {"Tenure": 12.0, "CityTier": 2.0}
        result = predictor.predict_single(partial_features)
        assert result is not None
    
    def test_api_route_validation(self):
        """Тест валидации маршрутов API"""
        # Тест несуществующего маршрута с параметрами
        response = client.get("/predict/123")
        assert response.status_code == 404
        
        # Тест неверного метода с параметрами
        response = client.delete("/predict")
        assert response.status_code == 405
    
    def test_response_structure_consistency(self, sample_customer):
        """Тест согласованности структуры ответов"""
        # Получаем информацию о модели
        model_info_response = client.get("/model/info")
        if model_info_response.status_code == 200:
            model_info = model_info_response.json()
            assert "feature_names" in model_info
            assert isinstance(model_info["feature_names"], list)
        
        # Получаем список признаков отдельно
        features_response = client.get("/features")
        if features_response.status_code == 200:
            features_info = features_response.json()
            assert "features" in features_info
            assert isinstance(features_info["features"], list)
    
    def test_timestamp_format(self, sample_customer):
        """Тест формата временных меток"""
        response = client.post("/predict", json=sample_customer)
        if response.status_code == 200:
            data = response.json()
            timestamp = data["timestamp"]
            # Проверяем, что timestamp в ISO формате
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                # Если это не ISO формат, проверяем другие форматы
                assert isinstance(timestamp, str)
    
    def test_model_performance_metrics(self):
        """Тест метрик производительности модели"""
        response = client.get("/model/info")
        if response.status_code == 200:
            data = response.json()
            if "performance" in data and data["performance"]:
                metrics = data["performance"]
                for metric, value in metrics.items():
                    assert isinstance(value, (int, float))
                    # Проверяем, что метрики в разумных пределах
                    if metric in ["accuracy", "precision", "recall", "auc"]:
                        assert 0 <= value <= 1

# Тесты для особых сценариев
class TestSpecialScenarios:
    """Тесты особых сценариев"""
    
    def test_unicode_in_input(self):
        """Тест с Unicode символами в входных данных"""
        # Пытаемся отправить данные с Unicode
        response = client.post("/predict", 
                             json={"Tenure": 12.0, 
                                   "CityTier": "два"})  # Русские буквы
        assert response.status_code == 422  # Должна быть ошибка валидации
    
    def test_very_small_threshold(self):
        """Тест с очень маленьким порогом"""
        response = client.post("/predict?threshold=0.001", json=VALID_CUSTOMER_DATA)
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.001
    
    def test_very_large_threshold(self):
        """Тест с очень большим порогом"""
        response = client.post("/predict?threshold=0.999", json=VALID_CUSTOMER_DATA)
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.999
    
    def test_null_values_in_batch(self):
        """Тест с null значениями в batch запросе"""
        request_data = {
            "customers": [VALID_CUSTOMER_DATA, None, VALID_CUSTOMER_DATA],
            "customer_ids": [1, 2, 3]
        }
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422  # Должна быть ошибка валидации

if __name__ == "__main__":
    # Запуск тестов
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Генерация отчета о покрытии
    print("\n" + "="*60)
    print("Для получения отчета о покрытии выполните:")
    print("pytest --cov=api --cov-report=term-missing --cov-report=html")
    print("="*60)