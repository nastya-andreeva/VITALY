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
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 3


def create_simple_timeseries_plot(data, pollutant, region="Все регионы", period_text="", save_path=None):
    """
    Упрощенный временной график с агрегацией по дням/неделям

    Parameters:
    data (pd.DataFrame): данные с колонками 'date' и pollutant
    pollutant (str): целевой загрязнитель
    region (str): название региона
    period_text (str): текст периода для заголовка
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный график
    """
    setup_visualization_style()

    if data is None or data.empty:
        print("Нет данных для построения графика")
        return None

    # Проверяем наличие необходимых колонок
    if 'date' not in data.columns or pollutant not in data.columns:
        print(f"Отсутствуют необходимые колонки: date или {pollutant}")
        return None

    # Создаем копию и очищаем данные
    plot_data = data[['date', pollutant]].copy()
    plot_data = plot_data.dropna()

    if plot_data.empty:
        print("Нет данных после очистки")
        return None

    # Сортируем по дате
    plot_data = plot_data.sort_values('date')

    # Агрегируем по дням для уменьшения шума
    daily_data = plot_data.set_index('date').resample('D').mean()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Основной график
    ax.plot(daily_data.index, daily_data[pollutant],
            linewidth=1.5, color='steelblue', alpha=0.8, label='Среднесуточные значения')

    # Добавляем скользящее среднее для тренда
    if len(daily_data) > 7:
        rolling_avg = daily_data[pollutant].rolling(window=7, min_periods=1).mean()
        ax.plot(daily_data.index, rolling_avg,
                linewidth=2, color='red', alpha=0.9, label='Скользящее среднее (7 дней)')

    # Настройка оформления
    region_info = f" - {region}" if region != "Все регионы" else ""
    period_info = f" {period_text}" if period_text else ""

    ax.set_title(f'Динамика {pollutant.upper()}{region_info}{period_info}', fontsize=16, pad=15)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Форматирование дат
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Добавляем статистику в углу
    stats_text = f"Статистика:\nМин: {daily_data[pollutant].min():.1f}\nМакс: {daily_data[pollutant].max():.1f}\nСреднее: {daily_data[pollutant].mean():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сохранен: {save_path}")

    return fig


def create_pollutant_comparison_plot(data, pollutants, region="Все регионы", period_text="", save_path=None):
    """
    Сравнение нескольких загрязнителей на одном графике

    Parameters:
    data (pd.DataFrame): исходные данные
    pollutants (list): список загрязнителей для сравнения
    region (str): регион
    period_text (str): период
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный график
    """
    setup_visualization_style()

    if data is None or data.empty:
        print("Нет данных для построения графика")
        return None

    # Оставляем только существующие загрязнители
    available_pollutants = [p for p in pollutants if p in data.columns]
    if not available_pollutants:
        print("Нет доступных загрязнителей для сравнения")
        return None

    # Агрегируем по месяцам для сравнения
    monthly_data = data.set_index('date')[available_pollutants].resample('M').mean()

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['steelblue', 'red', 'green', 'orange', 'purple']

    for i, pollutant in enumerate(available_pollutants):
        color = colors[i % len(colors)]
        ax.plot(monthly_data.index, monthly_data[pollutant],
                label=pollutant.upper(), color=color, linewidth=2, marker='o', markersize=3)

    region_info = f" - {region}" if region != "Все регионы" else ""
    period_info = f" {period_text}" if period_text else ""

    ax.set_title(f'Сравнение загрязнителей{region_info}{period_info}', fontsize=16, pad=15)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Концентрация', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Форматирование дат
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сравнения сохранен: {save_path}")

    return fig


def create_regional_comparison_plot(data, pollutant, region_col='state', top_n=10, save_path=None):
    """
    Сравнение средних концентраций по регионам

    Parameters:
    data (pd.DataFrame): данные
    pollutant (str): загрязнитель
    region_col (str): колонка с регионами
    top_n (int): количество топ регионов для показа
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный график
    """
    setup_visualization_style()

    if data is None or pollutant not in data.columns or region_col not in data.columns:
        print("Нет данных для регионального сравнения")
        return None

    # Группируем по регионам и вычисляем средние
    regional_means = data.groupby(region_col)[pollutant].mean().sort_values(ascending=False)

    # Берем топ-N регионов
    top_regions = regional_means.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(range(len(top_regions)), top_regions.values,
                   color='lightcoral', alpha=0.8, edgecolor='darkred')

    ax.set_yticks(range(len(top_regions)))
    ax.set_yticklabels([str(region)[:20] + '...' if len(str(region)) > 20 else str(region)
                        for region in top_regions.index])
    ax.set_xlabel(f'Средняя концентрация {pollutant} ({get_pollutant_unit(pollutant)})', fontsize=12)
    ax.set_title(f'Топ-{top_n} регионов по загрязнению {pollutant.upper()}', fontsize=16, pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{width:.1f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График регионов сохранен: {save_path}")

    return fig


def create_monthly_trend_plot(data, pollutant, region="Все регионы", save_path=None):
    """
    График средних месячных значений за все годы

    Parameters:
    data (pd.DataFrame): данные
    pollutant (str): загрязнитель
    region (str): регион
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный график
    """
    setup_visualization_style()

    if data is None or 'date' not in data.columns or pollutant not in data.columns:
        print("Нет данных для месячного тренда")
        return None

    # Извлекаем год и месяц
    plot_data = data[['date', pollutant]].copy()
    plot_data = plot_data.dropna()
    plot_data['year'] = plot_data['date'].dt.year
    plot_data['month'] = plot_data['date'].dt.month

    # Средние по месяцам
    monthly_avg = plot_data.groupby('month')[pollutant].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
              'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

    ax.plot(range(1, 13), monthly_avg.values, marker='o', linewidth=2.5,
            color='teal', markersize=8, markerfacecolor='orange')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_xlabel('Месяц', fontsize=12)
    ax.set_ylabel(f'Концентрация ({get_pollutant_unit(pollutant)})', fontsize=12)

    region_info = f" - {region}" if region != "Все регионы" else ""
    ax.set_title(f'Сезонные паттерны {pollutant.upper()}{region_info}', fontsize=16, pad=15)
    ax.grid(True, alpha=0.3)

    # Подсветка максимального и минимального месяцев
    max_month = monthly_avg.idxmax()
    min_month = monthly_avg.idxmin()

    ax.axvline(x=max_month, color='red', alpha=0.3, linestyle='--', label=f'Макс: {months[max_month - 1]}')
    ax.axvline(x=min_month, color='green', alpha=0.3, linestyle='--', label=f'Мин: {months[min_month - 1]}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График месячных трендов сохранен: {save_path}")

    return fig


def create_yearly_summary_plot(data, pollutant, region="Все регионы", save_path=None):
    """
    График изменения по годам

    Parameters:
    data (pd.DataFrame): данные
    pollutant (str): загрязнитель
    region (str): регион
    save_path (str): путь для сохранения

    Returns:
    matplotlib.figure.Figure: созданный график
    """
    setup_visualization_style()

    if data is None or 'date' not in data.columns or pollutant not in data.columns:
        print("Нет данных для годового обзора")
        return None

    # Извлекаем год
    plot_data = data[['date', pollutant]].copy()
    plot_data = plot_data.dropna()
    plot_data['year'] = plot_data['date'].dt.year

    # Средние по годам
    yearly_avg = plot_data.groupby('year')[pollutant].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(yearly_avg.index, yearly_avg.values,
                  color='skyblue', alpha=0.8, edgecolor='navy')

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel(f'Средняя концентрация ({get_pollutant_unit(pollutant)})', fontsize=12)

    region_info = f" - {region}" if region != "Все регионы" else ""
    ax.set_title(f'Динамика по годам: {pollutant.upper()}{region_info}', fontsize=16, pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    # Линия тренда
    if len(yearly_avg) > 1:
        z = np.polyfit(yearly_avg.index, yearly_avg.values, 1)
        p = np.poly1d(z)
        ax.plot(yearly_avg.index, p(yearly_avg.index), "r--", alpha=0.8, linewidth=2, label='Тренд')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График годовой динамики сохранен: {save_path}")

    return fig


# Сохраняем оригинальный дашборд AQI без изменений
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
    if len(specific_advice) > 100:
        words = specific_advice.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= 40:
                current_line += " " + word
            else:
                lines.append(current_line.strip())
                current_line = word
        lines.append(current_line.strip())
        specific_advice = "\n".join(lines)

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