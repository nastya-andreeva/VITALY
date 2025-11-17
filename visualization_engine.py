import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def setup_visualization_style():
    """Настройка единого стиля визуализаций"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Кастомные настройки
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def create_comprehensive_timeseries_plot(data, trends_data, forecasts_data, pollutant, save_path=None):
    """
    Создание комплексного временного графика с трендами и прогнозами

    Parameters:
    data (pd.DataFrame): исходные данные
    trends_data (dict): результаты анализа трендов
    forecasts_data (dict): результаты прогнозирования
    pollutant (str): целевой загрязнитель
    save_path (str): путь для сохранения графика

    Returns:
    matplotlib.figure.Figure: созданный график
    """

    setup_visualization_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Фильтрация данных для графика
    if 'timestamp' in data.columns:
        time_col = 'timestamp'
    elif 'date' in data.columns:
        time_col = 'date'
    else:
        print(f"❌ Не найдена колонка времени. Доступные колонки: {data.columns.tolist()}")
        return None

    plot_data = data[[time_col, pollutant]].dropna().sort_values(time_col)

    if len(plot_data) == 0:
        print("Нет данных для визуализации")
        return None

    # Основной график временного ряда
    ax1.plot(plot_data['timestamp'], plot_data[pollutant],
             label='Измерения', color='steelblue', alpha=0.7, linewidth=1)

    # Добавление трендов
    if trends_data and 'composite_trend' in trends_data:
        ax1.plot(plot_data['timestamp'], trends_data['composite_trend'],
                 label='Основной тренд', color='red', linewidth=2, linestyle='--')

    if trends_data and 'moving_avg' in trends_data:
        moving_avg_values = trends_data['moving_avg']['values']
        valid_indices = ~np.isnan(moving_avg_values)
        ax1.plot(plot_data['timestamp'][valid_indices], moving_avg_values[valid_indices],
                 label='Скользящее среднее', color='orange', linewidth=1.5, alpha=0.8)

    # Добавление прогнозов
    if forecasts_data and 'final_forecast' in forecasts_data:
        forecast_dates = [datetime.fromisoformat(str(date)) for date in forecasts_data['forecast_dates']]
        forecast_values = forecasts_data['final_forecast']

        ax1.plot(forecast_dates, forecast_values,
                 label='Прогноз', color='green', linewidth=2, marker='o', markersize=4)

        # Область uncertainty (если доступно)
        if 'all_predictions' in forecasts_data and len(forecasts_data['all_predictions']) > 1:
            all_forecasts = list(forecasts_data['all_predictions'].values())
            forecast_std = np.std(all_forecasts, axis=0)

            ax1.fill_between(forecast_dates,
                             np.array(forecast_values) - forecast_std,
                             np.array(forecast_values) + forecast_std,
                             alpha=0.2, color='green', label='Доверительный интервал')

    ax1.set_title(f'Динамика концентрации {pollutant}', fontsize=16, pad=20)
    ax1.set_xlabel('Дата и время')
    ax1.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Форматирование оси времени
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # График остатков (если доступно)
    if trends_data and 'decomposition' in trends_data:
        residuals = trends_data['decomposition']['residual']
        valid_residuals = ~np.isnan(residuals)

        if np.any(valid_residuals):
            ax2.plot(plot_data['timestamp'][valid_residuals], residuals[valid_residuals],
                     color='gray', alpha=0.7, linewidth=1)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Остатки сезонной декомпозиции')
            ax2.set_xlabel('Дата и время')
            ax2.set_ylabel('Остатки')
            ax2.grid(True, alpha=0.3)

    else:
        # Альтернативно: график суточных паттернов
        try:
            plot_data['hour'] = plot_data['timestamp'].dt.hour
            hourly_avg = plot_data.groupby('hour')[pollutant].mean()

            ax2.bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='lightblue')
            ax2.set_title('Средние суточные паттерны')
            ax2.set_xlabel('Час дня')
            ax2.set_ylabel(f'Средняя концентрация ({get_pollutant_unit(pollutant)})')
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, 'Дополнительные данные недоступны',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Дополнительный анализ')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")

    return fig


def create_aqi_dashboard(aqi_results, save_path=None):
    """
    Создание дашборда индексов качества воздуха

    Parameters:
    aqi_results (dict): результаты расчета AQI
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный дашборд
    """

    setup_visualization_style()

    if not aqi_results or 'overall' not in aqi_results:
        print("Нет данных AQI для визуализации")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Общий AQI gauge
    overall_aqi = aqi_results['overall']['aqi']
    aqi_category = aqi_results['overall']['category']
    aqi_color = aqi_results['overall']['color']

    # Простой gauge chart
    categories = ['Хорошо', 'Удовл.', 'Вредно\nдля групп', 'Вредно', 'Очень\nвредно', 'Опасно']
    ranges = [0, 50, 100, 150, 200, 300, 500]
    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

    ax1.barh(0, overall_aqi, color=aqi_color, height=0.5)
    ax1.set_xlim(0, 500)
    ax1.set_title('Общий индекс качества воздуха (AQI)', fontsize=14, pad=20)
    ax1.set_xlabel('Значение AQI')
    ax1.set_yticks([])

    # Добавление меток диапазонов
    for i in range(len(ranges) - 1):
        ax1.axvline(x=ranges[i], color='gray', linestyle='--', alpha=0.5)
        ax1.text(ranges[i] + (ranges[i + 1] - ranges[i]) / 2, 0.3, categories[i],
                 ha='center', va='center', fontsize=8)

    ax1.text(overall_aqi, 0, f'{overall_aqi}', ha='center', va='bottom',
             fontweight='bold', fontsize=16)
    ax1.text(overall_aqi, -0.3, aqi_category, ha='center', va='top',
             fontweight='bold', color=aqi_color)

    # 2. Сравнение загрязнителей
    pollutants_data = []
    for poll, data in aqi_results.items():
        if poll != 'overall':
            pollutants_data.append({
                'pollutant': poll,
                'aqi': data['aqi'],
                'concentration': data['concentration'],
                'category': data['category'],
                'color': data['color']
            })

    if pollutants_data:
        pollutants_df = pd.DataFrame(pollutants_data)

        bars = ax2.bar(pollutants_df['pollutant'], pollutants_df['aqi'],
                       color=pollutants_df['color'], alpha=0.7)
        ax2.set_title('Индексы AQI по загрязнителям', fontsize=14)
        ax2.set_ylabel('Значение AQI')
        ax2.set_ylim(0, max(pollutants_df['aqi']) * 1.1)

        # Добавление значений на столбцы
        for bar, aqi_val in zip(bars, pollutants_df['aqi']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{aqi_val:.0f}', ha='center', va='bottom', fontweight='bold')

    # 3. Рекомендации по здоровью
    health_advice = aqi_results['overall']['category']
    dominant_poll = aqi_results['overall']['dominant_pollutant']
    specific_advice = aqi_results[dominant_poll]['health_advice']

    advice_text = f"Основной загрязнитель: {dominant_poll}\n\n"
    advice_text += f"Общая оценка: {health_advice}\n\n"
    advice_text += f"Рекомендации:\n{specific_advice}"

    ax3.text(0.05, 0.95, advice_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax3.set_title('Рекомендации для здоровья', fontsize=14)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # 4. Легенда категорий AQI
    legend_elements = [
        Patch(facecolor='green', label='Хорошо (0-50)'),
        Patch(facecolor='yellow', label='Удовлетворительно (51-100)'),
        Patch(facecolor='orange', label='Вредно для групп (101-150)'),
        Patch(facecolor='red', label='Вредно (151-200)'),
        Patch(facecolor='purple', label='Очень вредно (201-300)'),
        Patch(facecolor='maroon', label='Опасно (301-500)')
    ]

    ax4.legend(handles=legend_elements, loc='center', fontsize=10)
    ax4.set_title('Категории качества воздуха', fontsize=14)
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Дашборд AQI сохранен: {save_path}")

    return fig


def create_seasonal_analysis_plot(seasonal_results, save_path=None):
    """
    Визуализация сезонных паттернов загрязнения

    Parameters:
    seasonal_results (dict): результаты сезонного анализа
    save_path (str): путь для сохранения
    """

    setup_visualization_style()

    if not seasonal_results or 'error' in seasonal_results:
        print("Нет данных для сезонного анализа")
        return None

    pollutant = seasonal_results['pollutant']
    period = seasonal_results['analysis_period']

    fig, ax = plt.subplots(figsize=(12, 6))

    if period == 'daily' and 'hourly_patterns' in seasonal_results:
        patterns = seasonal_results['hourly_patterns']
        hours = [p['hour'] for p in patterns]
        means = [p['mean'] for p in patterns]
        stds = [p['std'] for p in patterns]

        ax.plot(hours, means, marker='o', linewidth=2, label='Средняя концентрация')
        ax.fill_between(hours,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, label='Стандартное отклонение')

        ax.set_xlabel('Час дня')
        ax.set_ylabel(f'Концентрация {pollutant} ({get_pollutant_unit(pollutant)})')
        ax.set_title(f'Суточные паттерны концентрации {pollutant}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Выделение пикового часа
        peak_hour = seasonal_results.get('peak_hour', {})
        if peak_hour:
            ax.axvline(x=peak_hour['hour'], color='red', linestyle='--', alpha=0.7)
            ax.text(peak_hour['hour'], peak_hour['concentration'],
                    f'Пик: {peak_hour["hour"]}:00',
                    ha='center', va='bottom', color='red', fontweight='bold')

    elif period == 'weekly' and 'daily_patterns' in seasonal_results:
        patterns = seasonal_results['daily_patterns']
        days = [p['day_of_week'] for p in patterns]
        means = [p['mean'] for p in patterns]

        day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        day_labels = [day_names[d] for d in days]

        ax.bar(day_labels, means, alpha=0.7, color='skyblue')
        ax.set_xlabel('День недели')
        ax.set_ylabel(f'Концентрация {pollutant} ({get_pollutant_unit(pollutant)})')
        ax.set_title(f'Недельные паттерны концентрации {pollutant}', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сезонности сохранен: {save_path}")

    return fig


def get_pollutant_unit(pollutant):
    """Получение единиц измерения для загрязнителя"""
    units = {
        'PM2.5': 'μg/m³',
        'PM10': 'μg/m³',
        'NO2': 'ppb',
        'SO2': 'ppb',
        'CO': 'ppm',
        'O3': 'ppb'
    }
    return units.get(pollutant, 'units')


def save_visualization(fig, filename, dpi=300):
    """Универсальная функция сохранения визуализации"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Визуализация сохранена: {filename}")