import pandas as pd
import numpy as np
from analysis_core import calculate_pollution_trend, predict_future_levels


class RegionalAnalyzer:
    """Класс для регионального анализа данных"""

    def __init__(self, data):
        self.data = data
        self.regions = self._extract_regions()

    def _extract_regions(self):
        """Извлечение информации о регионах из данных"""
        regions = {}
        region_columns = ['state', 'city', 'location', 'region', 'area']

        for col in region_columns:
            if col in self.data.columns:
                unique_regions = self.data[col].dropna().unique()
                for region in unique_regions:
                    regions[region] = col
                break  # Используем первую найденную колонку

        return regions

    def get_region_data(self, region_name):
        """Получить данные для конкретного региона"""
        if region_name not in self.regions:
            return None

        region_col = self.regions[region_name]
        return self.data[self.data[region_col] == region_name].copy()

    def compare_regions(self, regions, pollutant, metric='mean'):
        """Сравнение регионов по указанному показателю"""
        comparison = {}

        for region in regions:
            region_data = self.get_region_data(region)
            if region_data is not None and pollutant in region_data.columns:
                values = region_data[pollutant].dropna()

                if metric == 'mean':
                    comparison[region] = values.mean()
                elif metric == 'median':
                    comparison[region] = values.median()
                elif metric == 'max':
                    comparison[region] = values.max()
                elif metric == 'min':
                    comparison[region] = values.min()
                elif metric == 'std':
                    comparison[region] = values.std()

        return comparison

    def regional_trend_analysis(self, regions, pollutant):
        """Анализ трендов по регионам"""
        trends = {}

        for region in regions:
            region_data = self.get_region_data(region)
            if region_data is not None and 'date' in region_data.columns:
                region_data = region_data.rename(columns={'date': 'timestamp'})
                trend = calculate_pollution_trend(region_data, pollutant, method='composite')
                trends[region] = trend

        return trends

    def regional_forecast(self, regions, pollutant, horizon=24):
        """Прогнозирование по регионам"""
        forecasts = {}

        for region in regions:
            region_data = self.get_region_data(region)
            if region_data is not None and 'date' in region_data.columns:
                region_data = region_data.rename(columns={'date': 'timestamp'})
                forecast = predict_future_levels(region_data, pollutant,
                                                 forecast_horizon=horizon, method='hybrid')
                forecasts[region] = forecast

        return forecasts