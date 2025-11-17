import data_manager as dm
import analysis_core as ac
import pandas as pd
import json
import os
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Кастомный энкодер для numpy типов"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def run_complete_analysis(file_path):
    """
    Запуск полного анализа для использования в GUI

    Parameters:
    file_path (str): путь к файлу данных

    Returns:
    dict: результаты анализа
    """

    print(f"🔄 Запуск анализа для файла: {file_path}")

    # Загрузка данных
    raw_data, validation_report = dm.load_environmental_data(file_path)

    if raw_data.empty:
        return {'error': 'Не удалось загрузить данные'}

    results = {
        'validation_report': validation_report,
        'data_info': {
            'records_loaded': len(raw_data),
            'columns': raw_data.columns.tolist(),
            'period': validation_report.get('data_period')
        }
    }

    # Определение целевого показателя
    numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
    target_pollutant = None

    for col in numeric_columns:
        if col in raw_data.columns and raw_data[col].notna().sum() > 1000:
            target_pollutant = col
            break

    if not target_pollutant:
        return {'error': 'Нет показателей с достаточным количеством данных'}

    results['target_pollutant'] = target_pollutant

    # Очистка данных
    cleaned_data, anomalies_stats = dm.detect_anomalies_mad(
        raw_data, target_pollutant, sensitivity='auto'
    )

    results['anomalies_stats'] = anomalies_stats

    # Подготовка данных для анализа
    analysis_data = dm.prepare_analysis_dataset(cleaned_data)

    # Базовый анализ трендов
    if 'date' in analysis_data.columns:
        analysis_data['year'] = analysis_data['date'].dt.year
        yearly_avg = analysis_data.groupby('year')[target_pollutant].mean().reset_index()

        if len(yearly_avg) > 1:
            first_val = float(yearly_avg.iloc[0][target_pollutant])
            last_val = float(yearly_avg.iloc[-1][target_pollutant])
            change_percent = float(((last_val - first_val) / first_val) * 100)

            results['trend_analysis'] = {
                'overall_direction': 'рост' if change_percent > 0 else 'снижение',
                'change_percentage': abs(change_percent),
                'first_year_avg': first_val,
                'last_year_avg': last_val,
                'years_analyzed': len(yearly_avg),
                'period': f"{yearly_avg['year'].min()}-{yearly_avg['year'].max()}"
            }

    # Базовая статистика
    if target_pollutant in analysis_data.columns:
        stats = analysis_data[target_pollutant].describe()
        results['basic_statistics'] = {
            'mean': float(stats['mean']),
            'median': float(stats['50%']),
            'max': float(stats['max']),
            'min': float(stats['min']),
            'std': float(stats['std']),
            'count': int(stats['count'])
        }

    print("✅ Анализ завершен успешно")
    return results


if __name__ == "__main__":
    # Тестирование функции
    results = run_complete_analysis('data/air_quality_data.csv')
    print(json.dumps(results, indent=2, ensure_ascii=False, cls=NumpyEncoder))