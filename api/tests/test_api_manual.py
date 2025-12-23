import requests
import json
import sys
import os

# Добавляем текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://localhost:8000"

def test_api():
    """Ручное тестирование API"""
    
    print("🚀 Тестирование Customer Churn Prediction API\n")
    
    # 1. Проверка здоровья
    print("1. Проверка здоровья API:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Статус: {health['status']}")
            print(f"   ✅ Модель загружена: {health['model_loaded']}")
        else:
            print(f"   ❌ Ошибка: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Ошибка подключения: {e}")
        print(f"   ⚠ Запустите API: uvicorn api.main:app --reload")
        return
    
    # 2. Получение информации о модели
    print("\n2. Информация о модели:")
    response = requests.get(f"{BASE_URL}/model/info")
    if response.status_code == 200:
        model_info = response.json()
        print(f"   ✅ Название: {model_info['model_name']}")
        print(f"   ✅ Тип: {model_info['model_type']}")
        print(f"   ✅ Признаков: {model_info['feature_count']}")
    else:
        print(f"   ❌ Ошибка: {response.status_code}")
    
    # 3. Пример предсказания
    print("\n3. Пример предсказания:")
    
    # Данные клиента
    customer_data = {
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
    
    response = requests.post(f"{BASE_URL}/predict?customer_id=50001", json=customer_data)
    if response.status_code == 200:
        prediction = response.json()
        print(f"   ✅ Клиент ID: {prediction['customer_id']}")
        print(f"   ✅ Вероятность оттока: {prediction['churn_probability']:.2%}")
        print(f"   ✅ Предсказание: {'ОТТОК' if prediction['prediction'] else 'НЕТ ОТТОКА'}")
        print(f"   ✅ Порог: {prediction['threshold']}")
    else:
        print(f"   ❌ Ошибка: {response.status_code}")
        print(f"   Детали: {response.text}")
    
    # 4. Пакетное предсказание
    print("\n4. Пакетное предсказание:")
    
    batch_data = {
        "customers": [customer_data, customer_data],
        "customer_ids": [50002, 50003]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    if response.status_code == 200:
        batch_result = response.json()
        print(f"   ✅ Клиентов обработано: {batch_result['total_customers']}")
        print(f"   ✅ Процент оттока: {batch_result['churn_rate']:.2%}")
        print(f"   ✅ Средняя вероятность: {batch_result['avg_probability']:.2%}")
    else:
        print(f"   ❌ Ошибка: {response.status_code}")
    
    # 5. Получение списка признаков
    print("\n5. Список признаков:")
    response = requests.get(f"{BASE_URL}/features")
    if response.status_code == 200:
        features = response.json()
        print(f"   ✅ Всего признаков: {features['total_features']}")
        print(f"   ✅ Обязательные признаки: {len(features['required_features'])}")
        print(f"   ✅ Опциональные признаки: {len(features['optional_features'])}")
    else:
        print(f"   ❌ Ошибка: {response.status_code}")
    
    print("\n🎉 Тестирование завершено!")
    print(f"\n📚 Документация API доступна по адресу: {BASE_URL}/docs")

if __name__ == "__main__":
    test_api()
