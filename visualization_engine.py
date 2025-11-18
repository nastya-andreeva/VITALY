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

    # Кастомные настройки для лучшего отображения
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14


def create_comprehensive_timeseries_plot(data, trends_data, forecasts_data, pollutant, save_path=None):
    """
    Создание комплексного временного графика с трендами и прогнозами
    """

    setup_visualization_style()

    # Создаем фигуру только с одним графиком
    fig, ax = plt.subplots(figsize=(14, 6))

    # Фильтрация данных для графика
    if 'timestamp' in data.columns:
        time_col = 'timestamp'
    elif 'date' in data.columns:
        time_col = 'date'
    else:
        print(f"❌ Не найдена колонка времени. Доступные колонки: {data.columns.tolist()}")
        return None

    # Очистка и подготовка данных
    plot_data = data[[time_col, pollutant]].dropna().copy()

    if len(plot_data) == 0:
        print("Нет данных для визуализации")
        return None

    # Преобразуем в datetime если нужно
    if not pd.api.types.is_datetime64_any_dtype(plot_data[time_col]):
        plot_data[time_col] = pd.to_datetime(plot_data[time_col], errors='coerce')
        plot_data = plot_data.dropna(subset=[time_col])

    plot_data = plot_data.sort_values(time_col)

    # АГРЕГАЦИЯ ДАННЫХ - ключевое изменение!
    # Группируем по дате и берем среднее значение
    daily_data = plot_data.groupby(plot_data[time_col].dt.date)[pollutant].mean().reset_index()
    daily_data[time_col] = pd.to_datetime(daily_data[time_col])

    if len(daily_data) == 0:
        print("Нет данных после агрегации")
        return None

    # Определяем интервал агрегации в зависимости от диапазона дат
    date_range = daily_data[time_col].max() - daily_data[time_col].min()

    if date_range.days > 365 * 10:  # Более 10 лет
        aggregated_data = daily_data.set_index(time_col).resample('Y').mean().dropna()
        interval_label = "Годовые средние"
    elif date_range.days > 365 * 5:  # Более 5 лет
        aggregated_data = daily_data.set_index(time_col).resample('6M').mean().dropna()
        interval_label = "Полугодовые средние"
    elif date_range.days > 365 * 2:  # Более 2 лет
        aggregated_data = daily_data.set_index(time_col).resample('Q').mean().dropna()
        interval_label = "Квартальные средние"
    elif date_range.days > 365:  # Более 1 года
        aggregated_data = daily_data.set_index(time_col).resample('M').mean().dropna()
        interval_label = "Месячные средние"
    elif date_range.days > 180:  # Более 6 месяцев
        aggregated_data = daily_data.set_index(time_col).resample('2W').mean().dropna()
        interval_label = "Двухнедельные средние"
    else:  # Меньше 6 месяцев
        aggregated_data = daily_data.set_index(time_col).resample('W').mean().dropna()
        interval_label = "Недельные средние"

    # Основной график
    ax.plot(aggregated_data.index, aggregated_data[pollutant],
            label=f'Измерения ({interval_label})',
            color='steelblue', alpha=0.8, linewidth=2, marker='o', markersize=3)

    # Добавление трендов (если доступны)
    if trends_data and 'composite_trend' in trends_data:
        # Адаптируем тренд к нашим агрегированным данным
        trend_length = min(len(aggregated_data), len(trends_data['composite_trend']))
        if trend_length > 0:
            ax.plot(aggregated_data.index[:trend_length],
                    trends_data['composite_trend'][:trend_length],
                    label='Основной тренд', color='red', linewidth=2.5, linestyle='--')

    # Добавление прогнозов
    if forecasts_data and 'final_forecast' in forecasts_data:
        forecast_dates = [datetime.fromisoformat(str(date)) for date in forecasts_data['forecast_dates']]
        forecast_values = forecasts_data['final_forecast']

        ax.plot(forecast_dates, forecast_values,
                label='Прогноз', color='green', linewidth=2.5, marker='s', markersize=4)

    ax.set_title(f'Динамика концентрации {pollutant.upper()}', fontsize=14, pad=15)
    ax.set_xlabel('Дата', fontsize=11)
    ax.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})', fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Форматирование оси времени
    if date_range.days > 365 * 10:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif date_range.days > 365 * 2:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif date_range.days > 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сохранен: {save_path}")

    return fig


def create_aqi_dashboard(aqi_results, save_path=None):
    """
    Создание дашборда индексов качества воздуха - УПРОЩЕННАЯ ВЕРСИЯ
    """

    setup_visualization_style()

    if not aqi_results or 'overall' not in aqi_results:
        print("Нет данных AQI для визуализации")
        return None

    # Создаем ОДИН график вместо дашборда
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Индекс качества воздуха (AQI)', fontsize=14, y=0.95)

    # ЛЕВАЯ ЧАСТЬ: AQI по загрязнителям
    pollutants_data = []
    for poll, data in aqi_results.items():
        if poll != 'overall':
            pollutants_data.append({
                'pollutant': poll.upper(),
                'aqi': data['aqi'],
                'concentration': data['concentration'],
                'category': data['category'],
                'color': data['color']
            })

    if pollutants_data:
        pollutants_df = pd.DataFrame(pollutants_data)

        bars = ax1.bar(range(len(pollutants_df)), pollutants_df['aqi'],
                       color=pollutants_df['color'], alpha=0.8,
                       edgecolor='black', linewidth=1)

        ax1.set_title('AQI по загрязнителям', fontsize=12, pad=10)
        ax1.set_ylabel('Значение AQI', fontsize=10)
        ax1.set_ylim(0, max(pollutants_df['aqi']) * 1.1)
        ax1.set_xticks(range(len(pollutants_df)))
        ax1.set_xticklabels(pollutants_df['pollutant'], fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # Значения НАД столбцами
        for bar, aqi_val in zip(bars, pollutants_df['aqi']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f'{aqi_val:.0f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=9)

    # ПРАВАЯ ЧАСТЬ: Информация и рекомендации
    overall_aqi = aqi_results['overall']['aqi']
    health_advice = aqi_results['overall']['category']
    dominant_poll = aqi_results['overall']['dominant_pollutant']
    specific_advice = aqi_results[dominant_poll]['health_advice']

    # Обрезаем длинный текст
    if len(specific_advice) > 120:
        specific_advice = specific_advice[:117] + "..."

    info_text = f"Общий AQI: {overall_aqi}\n"
    info_text += f"Категория: {health_advice}\n"
    info_text += f"Основной загрязнитель: {dominant_poll.upper()}\n\n"
    info_text += f"Рекомендации:\n{specific_advice}"

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', linespacing=1.3)
    ax2.set_title('Информация и рекомендации', fontsize=12, pad=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Рамка вокруг текста
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    # Легенда внизу
    legend_elements = [
        Patch(facecolor='#00E400', label='Хорошо (0-50)'),
        Patch(facecolor='#FFFF00', label='Удовл. (51-100)'),
        Patch(facecolor='#FF7E00', label='Вредно для групп (101-150)'),
        Patch(facecolor='#FF0000', label='Вредно (151-200)'),
        Patch(facecolor='#8F3F97', label='Очень вредно (201-300)'),
        Patch(facecolor='#7E0023', label='Опасно (301-500)')
    ]

    ax2.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Место для легенды

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Дашборд AQI сохранен: {save_path}")

    return fig


def create_seasonal_analysis_plot(seasonal_results, save_path=None):
    """
    Визуализация сезонных паттернов загрязнения
    """

    setup_visualization_style()

    if not seasonal_results or 'error' in seasonal_results:
        print("Нет данных для сезонного анализа")
        return None

    pollutant = seasonal_results['pollutant']
    period = seasonal_results['analysis_period']

    fig, ax = plt.subplots(figsize=(10, 5))

    if period == 'daily' and 'hourly_patterns' in seasonal_results:
        patterns = seasonal_results['hourly_patterns']
        hours = [p['hour'] for p in patterns]
        means = [p['mean'] for p in patterns]

        ax.plot(hours, means, marker='o', linewidth=2, color='steelblue', markersize=4)
        ax.set_xlabel('Час дня', fontsize=10)
        ax.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})', fontsize=10)
        ax.set_title(f'Суточные паттерны {pollutant.upper()}', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))

    elif period == 'weekly' and 'daily_patterns' in seasonal_results:
        patterns = seasonal_results['daily_patterns']
        days = [p['day_of_week'] for p in patterns]
        means = [p['mean'] for p in patterns]

        day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        day_labels = [day_names[d] for d in days]

        ax.bar(day_labels, means, alpha=0.8, color='skyblue', edgecolor='navy')
        ax.set_xlabel('День недели', fontsize=10)
        ax.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})', fontsize=10)
        ax.set_title(f'Недельные паттерны {pollutant.upper()}', fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
        'O3': 'ppb',
        'so2': 'μg/m³',
        'no2': 'μg/m³',
        'rspm': 'μg/m³',
        'spm': 'μg/m³',
        'pm2_5': 'μg/m³'
    }
    return units.get(pollutant.lower(), 'units')


def save_visualization(fig, filename, dpi=300):
    """Универсальная функция сохранения визуализации"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Визуализация сохранена: {filename}")