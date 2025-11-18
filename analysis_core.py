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
        window_size = min(24, len(time_series) // 10)
        moving_avg = time_series[pollutant].rolling(window=window_size, center=True).mean()

        trend_results['moving_avg'] = {
            'window_size': window_size,
            'values': moving_avg.values
        }

    # Сезонная декомпозиция
    if method in ['decomposition', 'composite'] and len(time_series) >= 50:
        try:
            ts_indexed = time_series.set_index('timestamp')[pollutant]
            decomposition = seasonal_decompose(ts_indexed, model='additive', period=min(24, len(ts_indexed) // 2))

            trend_results['decomposition'] = {
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.values
            }
        except Exception as e:
            trend_results['decomposition_error'] = str(e)

    # Композитный тренд
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
        trend_results['change_percentage'] = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0

    return trend_results


def predict_future_levels(data, pollutant, forecast_horizon=24, method='hybrid'):
    """
    Прогнозирование уровней загрязнения
    """
    if pollutant not in data.columns:
        return {'error': f'Загрязнитель {pollutant} не найден'}

    # Подготовка временного ряда
    ts_data = data[['timestamp', pollutant]].dropna().sort_values('timestamp')

    if len(ts_data) < 48:
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
    if method == 'arima':
        try:
            arima_model = ARIMA(ts_indexed, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=forecast_horizon)

            predictions['arima'] = arima_forecast.values
            forecast_results['arima_confidence'] = 0.8
            forecast_results['final_forecast'] = arima_forecast.values.tolist()
            forecast_results['method_used'] = 'arima'

        except Exception as e:
            forecast_results['arima_error'] = str(e)
            forecast_results['error'] = f'Ошибка ARIMA: {str(e)}'

    # Генерация дат прогноза
    forecast_dates = pd.date_range(
        start=forecast_results['last_historical_date'],
        periods=forecast_horizon + 1,
        freq='H'
    )[1:]

    forecast_results['forecast_dates'] = forecast_dates.tolist()
    forecast_results['all_predictions'] = {k: v.tolist() for k, v in predictions.items()}

    # Статистика прогноза
    if 'final_forecast' in forecast_results:
        forecast_values = forecast_results['final_forecast']
        forecast_results['forecast_stats'] = {
            'min': float(np.min(forecast_values)),
            'max': float(np.max(forecast_values)),
            'mean': float(np.mean(forecast_values)),
            'std': float(np.std(forecast_values))
        }

    return forecast_results


def compute_air_quality_index(data, pollutants_config=None):
    """
    Расчет комплексных индексов качества воздуха
    """
    if pollutants_config is None:
        pollutants_config = {
            'PM2.5': {'unit': 'μg/m³', 'breakpoints': [12.0, 35.4, 55.4, 150.4, 250.4, 500.4]},
            'PM10': {'unit': 'μg/m³', 'breakpoints': [54.0, 154.0, 254.0, 354.0, 424.0, 604.0]},
            'NO2': {'unit': 'ppb', 'breakpoints': [53.0, 100.0, 360.0, 649.0, 1249.0, 2049.0]},
            'SO2': {'unit': 'ppb', 'breakpoints': [35.0, 75.0, 185.0, 304.0, 604.0, 1004.0]},
            'CO': {'unit': 'ppm', 'breakpoints': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]},
            'O3': {'unit': 'ppb', 'breakpoints': [54.0, 70.0, 85.0, 105.0, 200.0, 400.0]}
        }

    aqi_results = {}

    # Сопоставление названий колонок
    column_mapping = {
        'pm2_5': 'PM2.5',
        'rspm': 'PM10',
        'spm': 'PM10',
        'no2': 'NO2',
        'so2': 'SO2'
    }

    for data_col, standard_name in column_mapping.items():
        if data_col in data.columns and standard_name in pollutants_config:
            concentrations = data[data_col].dropna()

            if len(concentrations) > 0:
                # Используем среднее значение за последние сутки или доступные данные
                current_level = concentrations.mean()
                config = pollutants_config[standard_name]
                breakpoints = config['breakpoints']

                # Расчет AQI
                aqi_value = calculate_single_aqi(current_level, breakpoints)

                # Классификация
                category, color = get_aqi_category(aqi_value)

                aqi_results[standard_name] = {
                    'concentration': float(current_level),
                    'unit': config['unit'],
                    'aqi': int(aqi_value),
                    'category': category,
                    'color': color,
                    'health_advice': generate_health_advice(category)
                }

    # Расчет общего AQI (максимальный из индивидуальных)
    if aqi_results:
        overall_aqi = max([result['aqi'] for result in aqi_results.values()])
        dominant_pollutant = max(aqi_results.items(), key=lambda x: x[1]['aqi'])[0]

        overall_category, overall_color = get_aqi_category(overall_aqi)

        aqi_results['overall'] = {
            'aqi': int(overall_aqi),
            'dominant_pollutant': dominant_pollutant,
            'category': overall_category,
            'color': overall_color
        }

    return aqi_results


def calculate_single_aqi(concentration, breakpoints):
    """Расчет AQI для одного загрязнителя"""
    aqi_breakpoints = [0, 50, 100, 150, 200, 300, 500]

    # Находим соответствующий диапазон
    for i in range(len(breakpoints) - 1):
        if concentration <= breakpoints[i]:
            c_low = breakpoints[i - 1] if i > 0 else 0
            c_high = breakpoints[i]
            i_low = aqi_breakpoints[i]
            i_high = aqi_breakpoints[i + 1]

            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)

    return 500  # Максимальное значение


def get_aqi_category(aqi_value):
    """Определение категории AQI"""
    if aqi_value <= 50:
        return "Хорошо", "green"
    elif aqi_value <= 100:
        return "Удовлетворительно", "yellow"
    elif aqi_value <= 150:
        return "Вредно для чувствительных групп", "orange"
    elif aqi_value <= 200:
        return "Вредно", "red"
    elif aqi_value <= 300:
        return "Очень вредно", "purple"
    else:
        return "Опасно", "maroon"


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
    """
    if pollutant not in data.columns:
        return {'error': f'Загрязнитель {pollutant} не найден'}

    # Проверяем наличие колонки с датой
    if 'date' not in data.columns and 'timestamp' not in data.columns:
        return {'error': 'Не найдена колонка с датой'}

    # Используем date или timestamp
    date_col = 'date' if 'date' in data.columns else 'timestamp'

    # Подготавливаем данные
    ts_data = data[[date_col, pollutant]].dropna().sort_values(date_col)

    # Переименовываем для единообразия
    ts_data = ts_data.rename(columns={date_col: 'timestamp'})

    if len(ts_data) < 24:  # Минимум 1 день данных
        return {'error': 'Недостаточно данных для сезонного анализа'}

    seasonal_results = {
        'pollutant': pollutant,
        'analysis_period': period
    }

    # Добавление временных признаков
    ts_data['hour'] = ts_data['timestamp'].dt.hour
    ts_data['day_of_week'] = ts_data['timestamp'].dt.dayofweek
    ts_data['month'] = ts_data['timestamp'].dt.month

    if period == 'daily':
        # Суточные паттерны
        hourly_stats = ts_data.groupby('hour')[pollutant].agg(['mean', 'std', 'count']).reset_index()
        seasonal_results['hourly_patterns'] = hourly_stats.to_dict('records')

        # Пиковый час
        if not hourly_stats.empty:
            max_idx = hourly_stats['mean'].idxmax()
            peak_hour_data = hourly_stats.loc[max_idx]
            seasonal_results['peak_hour'] = {
                'hour': int(peak_hour_data['hour']),
                'concentration': float(peak_hour_data['mean'])
            }

    elif period == 'weekly':
        # Недельные паттерны
        daily_stats = ts_data.groupby('day_of_week')[pollutant].agg(['mean', 'std']).reset_index()
        seasonal_results['daily_patterns'] = daily_stats.to_dict('records')

    elif period == 'monthly':
        # Месячные паттерны
        monthly_stats = ts_data.groupby('month')[pollutant].agg(['mean', 'std']).reset_index()
        seasonal_results['monthly_patterns'] = monthly_stats.to_dict('records')

    # Базовая статистика
    seasonal_results['basic_stats'] = {
        'mean': float(ts_data[pollutant].mean()),
        'std': float(ts_data[pollutant].std()),
        'min': float(ts_data[pollutant].min()),
        'max': float(ts_data[pollutant].max()),
        'total_records': len(ts_data)
    }

    return seasonal_results


def analyze_correlation_matrix(data, pollutants=None):
    """
    Анализ корреляционной матрицы между показателями
    """
    if pollutants is None:
        pollutants = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']

    available_pollutants = [p for p in pollutants if p in data.columns]

    if len(available_pollutants) < 2:
        return {'error': 'Недостаточно показателей для анализа корреляции'}

    # Вычисляем корреляционную матрицу
    corr_matrix = data[available_pollutants].corr()

    # Находим наиболее коррелирующие пары
    correlations = []
    for i in range(len(available_pollutants)):
        for j in range(i + 1, len(available_pollutants)):
            corr_value = corr_matrix.iloc[i, j]
            correlations.append({
                'pollutant1': available_pollutants[i],
                'pollutant2': available_pollutants[j],
                'correlation': float(corr_value),
                'strength': get_correlation_strength(abs(corr_value))
            })

    # Сортируем по абсолютному значению корреляции
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'top_correlations': correlations[:5],  # Топ-5 корреляций
        'pollutants_analyzed': available_pollutants
    }


def get_correlation_strength(corr_value):
    """Определение силы корреляции"""
    if corr_value >= 0.8:
        return "очень сильная"
    elif corr_value >= 0.6:
        return "сильная"
    elif corr_value >= 0.4:
        return "умеренная"
    elif corr_value >= 0.2:
        return "слабая"
    else:
        return "очень слабая"