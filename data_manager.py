import pandas as pd
import numpy as np
from datetime import datetime
import json


def load_environmental_data(file_path, required_columns=None):
    """
    Загрузка и комплексная валидация данных экологического мониторинга
    С ПРИНУДИТЕЛЬНОЙ КОДИРОВКОЙ
    """
    if required_columns is None:
        required_columns = ['so2', 'no2', 'date']

    validation_report = {
        'status': 'success',
        'errors': [],
        'warnings': [],
        'records_loaded': 0,
        'data_period': None
    }

    try:
        print(f"🔄 Попытка загрузки: {file_path}")

        # Пробуем разные кодировки ПОСЛЕДОВАТЕЛЬНО
        encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8', 'windows-1252']

        data = None
        used_encoding = None

        for encoding in encodings:
            try:
                print(f"   Пробуем кодировку: {encoding}")
                data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                used_encoding = encoding
                print(f"   ✅ Успешно с кодировкой: {encoding}")
                break
            except UnicodeDecodeError as e:
                print(f"   ❌ Ошибка Unicode с {encoding}: {e}")
                continue
            except Exception as e:
                print(f"   ❌ Другая ошибка с {encoding}: {e}")
                continue

        if data is None:
            validation_report['status'] = 'error'
            validation_report['errors'].append("Не удалось загрузить файл ни с одной кодировкой")
            return pd.DataFrame(), validation_report

        validation_report['records_loaded'] = len(data)
        validation_report['encoding_used'] = used_encoding

        print(f"✅ Успешно загружено {len(data)} записей с кодировкой {used_encoding}")

        # Проверка обязательных колонок
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_report['warnings'].append(f"Отсутствуют некоторые колонки: {missing_columns}")
            print(f"⚠ Внимание: отсутствуют колонки: {missing_columns}")

        # Показываем доступные колонки
        print(f"📋 Доступные колонки: {data.columns.tolist()}")

        # Преобразование даты
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            invalid_dates = data['date'].isna().sum()
            if invalid_dates > 0:
                validation_report['warnings'].append(f"Обнаружено {invalid_dates} некорректных дат")
                data = data.dropna(subset=['date'])
        else:
            # Если нет колонки date, ищем альтернативы
            date_alternatives = ['sampling_date', 'timestamp', 'time']
            for alt in date_alternatives:
                if alt in data.columns:
                    print(f"🕐 Используем альтернативную колонку даты: {alt}")
                    data['date'] = pd.to_datetime(data[alt], errors='coerce')
                    invalid_dates = data['date'].isna().sum()
                    if invalid_dates > 0:
                        validation_report['warnings'].append(f"Обнаружено {invalid_dates} некорректных дат в {alt}")
                        data = data.dropna(subset=['date'])
                    break

        # Преобразование числовых колонок
        numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                invalid_values = data[col].isna().sum()
                if invalid_values > 0:
                    validation_report['warnings'].append(f"Обнаружено {invalid_values} некорректных значений в {col}")

        # Определение периода данных
        if len(data) > 0 and 'date' in data.columns:
            validation_report['data_period'] = {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            }
            print(f"📅 Период данных: {validation_report['data_period']}")

        return data, validation_report

    except Exception as e:
        validation_report['status'] = 'error'
        validation_report['errors'].append(f"Критическая ошибка загрузки: {str(e)}")
        print(f"❌ Критическая ошибка: {e}")
        return pd.DataFrame(), validation_report


def detect_anomalies_mad(data, pollutant_column, sensitivity='auto'):
    """
    Обнаружение аномалий методом MAD для вашего формата данных
    """

    print(f"🔍 Поиск аномалий в колонке: {pollutant_column}")

    if pollutant_column not in data.columns:
        print(f"❌ Колонка {pollutant_column} не найдена в данных")
        print(f"📋 Доступные колонки: {data.columns.tolist()}")
        return data, {'anomalies_detected': 0, 'error': f'Колонка {pollutant_column} не найдена'}

    # Создаем копию данных для работы
    working_data = data.copy()

    # Удаляем пропуски в целевом показателе
    initial_count = len(working_data)
    working_data = working_data.dropna(subset=[pollutant_column])
    removed_nulls = initial_count - len(working_data)

    if removed_nulls > 0:
        print(f"⚠ Удалено {removed_nulls} записей с пропусками в {pollutant_column}")

    if len(working_data) == 0:
        print(f"❌ Нет данных для анализа в колонке {pollutant_column}")
        return data, {'anomalies_detected': 0, 'error': 'Нет данных для анализа'}

    # Настройка порога
    sensitivity_map = {'low': 3.5, 'medium': 3.0, 'high': 2.5}
    threshold = sensitivity_map.get(sensitivity, 3.0)

    if sensitivity == 'auto':
        cv = working_data[pollutant_column].std() / working_data[pollutant_column].mean()
        threshold = 3.5 if cv > 1.0 else 2.8 if cv > 0.5 else 2.5

    # Расчет MAD
    median = working_data[pollutant_column].median()
    mad = (working_data[pollutant_column] - median).abs().median()

    if mad == 0:
        mad = working_data[pollutant_column].std() / 1.4826

    # Определение границ
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad

    # Идентификация аномалий
    anomalies_mask = (working_data[pollutant_column] < lower_bound) | (working_data[pollutant_column] > upper_bound)
    clean_data = working_data[~anomalies_mask]
    anomalies_data = working_data[anomalies_mask]

    # Статистика
    stats = {
        'anomalies_detected': len(anomalies_data),
        'anomaly_percentage': (len(anomalies_data) / len(working_data)) * 100,
        'threshold_used': threshold,
        'median': median,
        'bounds': {'lower': lower_bound, 'upper': upper_bound}
    }

    print(f"✅ Обнаружено {stats['anomalies_detected']} аномалий ({stats['anomaly_percentage']:.1f}%)")

    return clean_data, stats


def normalize_measurements(data):
    """
    Нормализация измерений
    """
    print("📊 Нормализация данных...")

    normalized_data = data.copy()

    numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']

    for col in numeric_columns:
        if col in normalized_data.columns:
            # Заполняем пропуски медианой
            if normalized_data[col].isna().sum() > 0:
                median_val = normalized_data[col].median()
                normalized_data[col] = normalized_data[col].fillna(median_val)

            # Минимальная нормализация
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[f'{col}_normalized'] = (normalized_data[col] - min_val) / (max_val - min_val)

    print("✅ Нормализация завершена")
    return normalized_data


def prepare_analysis_dataset(data):
    """
    Подготовка финального датасета для анализа
    """
    print("🛠 Подготовка датасета для анализа...")

    analysis_data = data.copy()

    # Базовые колонки для анализа
    base_columns = ['date', 'so2', 'no2', 'rspm', 'spm']
    available_columns = [col for col in base_columns if col in analysis_data.columns]

    analysis_data = analysis_data[available_columns]

    # Сортируем по дате если есть
    if 'date' in analysis_data.columns:
        analysis_data = analysis_data.sort_values('date')

    # Заполняем пропуски
    for col in available_columns:
        if col != 'date' and col in analysis_data.columns:
            if analysis_data[col].isna().sum() > 0:
                analysis_data[col] = analysis_data[col].fillna(analysis_data[col].median())

    print(f"✅ Подготовлен датасет: {len(analysis_data)} записей, {len(available_columns)} колонок")
    return analysis_data

def detect_anomalies_iqr(data, pollutant_column):
    """Обнаружение аномалий методом IQR"""
    if pollutant_column not in data.columns:
        return data, {'error': f'Колонка {pollutant_column} не найдена'}

    working_data = data.copy()
    working_data = working_data.dropna(subset=[pollutant_column])

    if len(working_data) == 0:
        return data, {'error': 'Нет данных для анализа'}

    Q1 = working_data[pollutant_column].quantile(0.25)
    Q3 = working_data[pollutant_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies_mask = (working_data[pollutant_column] < lower_bound) | (working_data[pollutant_column] > upper_bound)
    clean_data = working_data[~anomalies_mask]
    anomalies_data = working_data[anomalies_mask]

    stats = {
        'anomalies_detected': len(anomalies_data),
        'anomaly_percentage': (len(anomalies_data) / len(working_data)) * 100,
        'bounds': {'lower': lower_bound, 'upper': upper_bound}
    }

    return clean_data, stats

def detect_anomalies_zscore(data, pollutant_column, threshold=3):
    """Обнаружение аномалий методом Z-score"""
    if pollutant_column not in data.columns:
        return data, {'error': f'Колонка {pollutant_column} не найдена'}

    working_data = data.copy()
    working_data = working_data.dropna(subset=[pollutant_column])

    if len(working_data) == 0:
        return data, {'error': 'Нет данных для анализа'}

    mean = working_data[pollutant_column].mean()
    std = working_data[pollutant_column].std()

    z_scores = (working_data[pollutant_column] - mean) / std
    anomalies_mask = abs(z_scores) > threshold

    clean_data = working_data[~anomalies_mask]
    anomalies_data = working_data[anomalies_mask]

    stats = {
        'anomalies_detected': len(anomalies_data),
        'anomaly_percentage': (len(anomalies_data) / len(working_data)) * 100,
        'threshold_used': threshold
    }

    return clean_data, stats