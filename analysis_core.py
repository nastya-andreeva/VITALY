import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


def calculate_pollution_trend(data, pollutant, method='composite'):
    """
    Комплексный расчет тенденций изменения концентраций

    Parameters:
    data (pd.DataFrame): временной ряд данных
    pollutant (str): целевой загрязнитель
    method (str): метод расчета ('linear', 'moving_avg', 'decomposition', 'composite')

    Returns:
    dict: результаты анализа трендов
    """

    if pollutant not in data.columns:
        return {'error': f'Загрязнитель {pollutant} не найден в данных'}

    # Подготовка временного ряда
    time_series = data[['timestamp', pollutant]].dropna().sort_values('timestamp')

    if len(time_series) < 10:
        return {'error': 'Недостаточно данных для анализа тренда'}

    trend_results = {
        'pollutant': pollutant,
        'period_days': (time_series['timestamp'].max() - time_series['timestamp'].min()).days,
        'data_points': len(time_series)
    }

    # Линейный тренд
    if method in ['linear', 'composite']:
        time_numeric = (time_series['timestamp'] - time_series['timestamp'].min()).dt.total_seconds().values.reshape(-1,
                                                                                                                     1)
        concentration = time_series[pollutant].values

        lin_model = LinearRegression()
        lin_model.fit(time_numeric, concentration)
        linear_trend = lin_model.predict(time_numeric)

        trend_results['linear_trend'] = {
            'slope': lin_model.coef_[0],
            'intercept': lin_model.intercept_,
            'r_squared': lin_model.score(time_numeric, concentration),
            'values': linear_trend
        }

    # Скользящее среднее
    if method in ['moving_avg', 'composite']:
        window_size = min(24, len(time_series) // 10)  # Адаптивный размер окна
        moving_avg = time_series[pollutant].rolling(window=window_size, center=True).mean()

        trend_results['moving_avg'] = {
            'window_size': window_size,
            'values': moving_avg.values
        }

    # Сезонная декомпозиция
    if method in ['decomposition', 'composite'] and len(time_series) >= 50:
        try:
            # Установка временного индекса
            ts_indexed = time_series.set_index('timestamp')[pollutant]

            # Сезонная декомпозиция
            decomposition = seasonal_decompose(ts_indexed, model='additive', period=24)

            trend_results['decomposition'] = {
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.values
            }
        except Exception as e:
            trend_results['decomposition_error'] = str(e)

    # Композитный тренд (усреднение методов)
    if method == 'composite':
        composite_trend = np.zeros(len(time_series))
        method_count = 0

        if 'linear_trend' in trend_results:
            composite_trend += trend_results['linear_trend']['values']
            method_count += 1

        if 'moving_avg' in trend_results:
            valid_ma = ~np.isnan(trend_results['moving_avg']['values'])
            composite_trend[valid_ma] += trend_results['moving_avg']['values'][valid_ma]
            method_count += 1

        if method_count > 0:
            composite_trend /= method_count
            trend_results['composite_trend'] = composite_trend

    # Расчет общего направления тренда
    if 'composite_trend' in trend_results:
        first_val = trend_results['composite_trend'][0]
        last_val = trend_results['composite_trend'][-1]
        trend_direction = 'рост' if last_val > first_val else 'снижение' if last_val < first_val else 'стабильный'
        trend_results['overall_direction'] = trend_direction
        trend_results['change_percentage'] = ((last_val - first_val) / first_val) * 100

    return trend_results


def predict_future_levels(data, pollutant, forecast_horizon=24, method='hybrid'):
    """
    Прогнозирование уровней загрязнения на заданный горизонт

    Parameters:
    data (pd.DataFrame): исторические данные
    pollutant (str): целевой загрязнитель
    forecast_horizon (int): горизонт прогнозирования (часы)
    method (str): метод прогнозирования ('arima', 'ensemble', 'hybrid')

    Returns:
    dict: результаты прогнозирования
    """

    if pollutant not in data.columns:
        return {'error': f'Загрязнитель {pollutant} не найден'}

    # Подготовка временного ряда
    ts_data = data[['timestamp', pollutant]].dropna().sort_values('timestamp')

    if len(ts_data) < 48:  # Минимум 2 дня данных
        return {'error': 'Недостаточно данных для прогнозирования'}

    forecast_results = {
        'pollutant': pollutant,
        'forecast_horizon': forecast_horizon,
        'last_historical_date': ts_data['timestamp'].max()
    }

    # Установка временного индекса
    ts_indexed = ts_data.set_index('timestamp')[pollutant]

    predictions = {}

    # Метод ARIMA
    if method in ['arima', 'hybrid']:
        try:
            # Автоматический подбор параметров ARIMA
            arima_model = ARIMA(ts_indexed, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=forecast_horizon)

            predictions['arima'] = arima_forecast.values
            forecast_results['arima_confidence'] = 0.8  # Упрощенная оценка доверия
        except Exception as e:
            forecast_results['arima_error'] = str(e)

    # Ансамблевый метод на основе Random Forest
    if method in ['ensemble', 'hybrid']:
        try:
            # Создание признаков для временного ряда
            features = pd.DataFrame(index=ts_indexed.index)
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            features['value_lag_1'] = ts_indexed.shift(1)
            features['value_lag_24'] = ts_indexed.shift(24)
            features['rolling_mean_7'] = ts_indexed.rolling(7).mean()

            features = features.dropna()
            target = ts_indexed.loc[features.index]

            if len(features) > 100:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(features, target)

                # Прогноз на будущие периоды
                last_date = features.index.max()
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='H')[1:]

                # Создаем список для хранения прогнозов пошагово
                future_predictions = []
                current_features = features.iloc[-1:].copy()

                for i, date in enumerate(future_dates):
                    # Создаем фичи для будущей даты
                    future_feature_row = {
                        'hour': date.hour,
                        'day_of_week': date.dayofweek,
                        'month': date.month,
                        'value_lag_1': ts_indexed.iloc[-1] if i == 0 else future_predictions[-1],
                        'value_lag_24': ts_indexed.iloc[-24] if len(ts_indexed) >= 24 else ts_indexed.iloc[-1],
                        'rolling_mean_7': ts_indexed.tail(7).mean()
                    }

                    future_feature_df = pd.DataFrame([future_feature_row])
                    prediction = rf_model.predict(future_feature_df)[0]
                    future_predictions.append(prediction)

                predictions['ensemble'] = np.array(future_predictions)
                forecast_results['ensemble_features_used'] = list(features.columns)

        except Exception as e:
            forecast_results['ensemble_error'] = str(e)

    # Гибридный подход (усреднение методов)
    if method == 'hybrid' and len(predictions) >= 2:
        hybrid_forecast = np.mean(list(predictions.values()), axis=0)
        predictions['hybrid'] = hybrid_forecast
        forecast_results['final_forecast'] = hybrid_forecast.tolist()
        forecast_results['method_used'] = 'hybrid'
    elif predictions:
        # Использование лучшего доступного метода
        best_method = list(predictions.keys())[0]
        forecast_results['final_forecast'] = predictions[best_method].tolist()
        forecast_results['method_used'] = best_method
    else:
        forecast_results['error'] = 'Не удалось построить прогноз'
        return forecast_results

    # Генерация дат прогноза
    forecast_dates = pd.date_range(
        start=forecast_results['last_historical_date'],
        periods=forecast_horizon + 1,
        freq='H'
    )[1:]

    forecast_results['forecast_dates'] = forecast_dates.tolist()
    forecast_results['all_predictions'] = {k: v.tolist() for k, v in predictions.items()}

    # Добавляем статистику прогноза
    if 'final_forecast' in forecast_results:
        forecast_values = forecast_results['final_forecast']
        forecast_results['forecast_stats'] = {
            'min': float(np.min(forecast_values)),
            'max': float(np.max(forecast_values)),
            'mean': float(np.mean(forecast_values)),
            'std': float(np.std(forecast_values))
        }

    print(f"Построен прогноз для {pollutant} на {forecast_horizon} часов вперед")

    return forecast_results


def compute_air_quality_index(data, pollutants_config=None):
    """
    Расчет комплексных индексов качества воздуха

    Parameters:
    data (pd.DataFrame): данные измерений
    pollutants_config (dict): конфигурация загрязнителей и порогов

    Returns:
    dict: расчетные индексы и классификации
    """

    if pollutants_config is None:
        pollutants_config = {
            'PM2.5': {'unit': 'μg/m³', 'breakpoints': [12, 35.4, 55.4, 150.4, 250.4]},
            'PM10': {'unit': 'μg/m³', 'breakpoints': [54, 154, 254, 354, 424]},
            'NO2': {'unit': 'ppb', 'breakpoints': [53, 100, 360, 649, 1249]},
            'SO2': {'unit': 'ppb', 'breakpoints': [35, 75, 185, 304, 604]},
            'CO': {'unit': 'ppm', 'breakpoints': [4.4, 9.4, 12.4, 15.4, 30.4]}
        }

    aqi_results = {}

    for pollutant, config in pollutants_config.items():
        if pollutant in data.columns:
            concentrations = data[pollutant].dropna()

            if len(concentrations) > 0:
                current_level = concentrations.iloc[-1]
                breakpoints = config['breakpoints']

                # Расчет AQI для текущего загрязнителя
                aqi_value = calculate_single_aqi(current_level, breakpoints)

                # Классификация качества воздуха
                if aqi_value <= 50:
                    category = "Хорошо"
                    color = "green"
                elif aqi_value <= 100:
                    category = "Удовлетворительно"
                    color = "yellow"
                elif aqi_value <= 150:
                    category = "Вредно для чувствительных групп"
                    color = "orange"
                elif aqi_value <= 200:
                    category = "Вредно"
                    color = "red"
                elif aqi_value <= 300:
                    category = "Очень вредно"
                    color = "purple"
                else:
                    category = "Опасно"
                    color = "maroon"

                aqi_results[pollutant] = {
                    'concentration': current_level,
                    'unit': config['unit'],
                    'aqi': aqi_value,
                    'category': category,
                    'color': color,
                    'health_advice': generate_health_advice(category)
                }

    # Расчет общего AQI (максимальный из индивидуальных)
    if aqi_results:
        overall_aqi = max([result['aqi'] for result in aqi_results.values()])
        dominant_pollutant = max(aqi_results.items(), key=lambda x: x[1]['aqi'])[0]

        aqi_results['overall'] = {
            'aqi': overall_aqi,
            'dominant_pollutant': dominant_pollutant,
            'category': aqi_results[dominant_pollutant]['category'],
            'color': aqi_results[dominant_pollutant]['color']
        }

    return aqi_results


def calculate_single_aqi(concentration, breakpoints):
    """Расчет AQI для одного загрязнителя"""
    aqi_breakpoints = [0, 50, 100, 150, 200, 300, 500]

    for i in range(len(breakpoints)):
        if concentration <= breakpoints[i]:
            c_low = breakpoints[i - 1] if i > 0 else 0
            c_high = breakpoints[i]
            i_low = aqi_breakpoints[i]
            i_high = aqi_breakpoints[i + 1]

            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)

    return 500  # Максимальное значение


def generate_health_advice(category):
    """Генерация рекомендаций по здоровью"""
    advice_map = {
        "Хорошо": "Качество воздуха удовлетворительное, опасности для здоровья нет",
        "Удовлетворительно": "Качество воздуха приемлемое, однако у некоторых людей может быть легкое раздражение",
        "Вредно для чувствительных групп": "Члены чувствительных групп могут испытывать последствия для здоровья",
        "Вредно": "Каждый может начать испытывать последствия для здоровья",
        "Очень вредно": "Предупреждение о вреде для здоровья: риск воздействия повышен",
        "Опасно": "Чрезвычайная ситуация для здоровья: все население может пострадать"
    }
    return advice_map.get(category, "Рекомендации недоступны")


def analyze_seasonal_patterns(data, pollutant, period='daily'):
    """
    Анализ сезонных паттернов загрязнения

    Parameters:
    data (pd.DataFrame): данные измерений
    pollutant (str): целевой загрязнитель
    period (str): тип анализа ('daily', 'weekly', 'monthly')

    Returns:
    dict: результаты сезонного анализа
    """

    if pollutant not in data.columns:
        return {'error': f'Загрязнитель {pollutant} не найден'}

    ts_data = data[['timestamp', pollutant]].dropna().sort_values('timestamp')

    if len(ts_data) < 168:  # Минимум 1 неделя данных
        return {'error': 'Недостаточно данных для сезонного анализа'}

    # Добавление временных признаков
    ts_data['hour'] = ts_data['timestamp'].dt.hour
    ts_data['day_of_week'] = ts_data['timestamp'].dt.dayofweek
    ts_data['month'] = ts_data['timestamp'].dt.month

    seasonal_results = {
        'pollutant': pollutant,
        'analysis_period': period
    }

    if period == 'daily':
        # Суточные паттерны
        hourly_patterns = ts_data.groupby('hour')[pollutant].agg(['mean', 'std', 'count']).reset_index()
        seasonal_results['hourly_patterns'] = hourly_patterns.to_dict('records')

        # Пиковые часы
        max_hour = hourly_patterns.loc[hourly_patterns['mean'].idxmax()]
        seasonal_results['peak_hour'] = {
            'hour': int(max_hour['hour']),
            'concentration': float(max_hour['mean'])
        }

    elif period == 'weekly':
        # Недельные паттерны
        daily_patterns = ts_data.groupby('day_of_week')[pollutant].agg(['mean', 'std']).reset_index()
        seasonal_results['daily_patterns'] = daily_patterns.to_dict('records')

    elif period == 'monthly':
        # Месячные паттерны
        monthly_patterns = ts_data.groupby('month')[pollutant].agg(['mean', 'std']).reset_index()
        seasonal_results['monthly_patterns'] = monthly_patterns.to_dict('records')

    # Статистическая значимость паттернов
    try:
        from scipy import stats

        if period == 'daily':
            # Проверка значимости суточных колебаний
            hourly_groups = [group[pollutant].values for _, group in ts_data.groupby('hour')]
            if len(hourly_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*hourly_groups)
                seasonal_results['statistical_significance'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    except ImportError:
        seasonal_results['statistical_note'] = 'Для статистического анализа требуется scipy'

    return seasonal_results