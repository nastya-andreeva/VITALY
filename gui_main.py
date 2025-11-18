import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
import numpy as np

# Добавляем пути к нашим модулям
sys.path.append('.')
import data_manager as dm
import analysis_core as ac
import visualization_engine as ve


class AnalysisThread(threading.Thread):
    """Поток для выполнения анализа без блокировки GUI"""

    def __init__(self, target, args=(), kwargs={}, callback=None):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.result = None
        self.exception = None

    def run(self):
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            if self.callback:
                self.callback(self)


class AirQualityAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Система анализа качества воздуха")
        self.root.geometry("1200x800")

        self.data = None
        self.analysis_results = {}
        self.current_plots = []
        self.regions = {}
        self.analysis_thread = None
        self.is_analyzing = False

        # Переменные для прогресс-бара
        self.progress_var = tk.DoubleVar()
        self.progress_label_var = tk.StringVar(value="Готов")

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка загрузки данных
        self.setup_data_tab()

        # Вкладка анализа
        self.setup_analysis_tab()

        # Вкладка визуализации
        self.setup_visualization_tab()

        # Вкладка результатов
        self.setup_results_tab()

    def setup_data_tab(self):
        """Вкладка загрузки и просмотра данных"""
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Данные")

        # Фрейм загрузки
        load_frame = ttk.LabelFrame(self.data_tab, text="Загрузка данных", padding=10)
        load_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(load_frame, text="Выбрать файл CSV",
                   command=self.load_data).pack(side='left', padx=5)

        self.file_label = ttk.Label(load_frame, text="Файл не выбран")
        self.file_label.pack(side='left', padx=10)

        # Фрейм фильтров данных
        filter_frame = ttk.LabelFrame(self.data_tab, text="Фильтры данных", padding=10)
        filter_frame.pack(fill='x', padx=5, pady=5)

        # Первая строка фильтров
        filter_row1 = ttk.Frame(filter_frame)
        filter_row1.pack(fill='x', pady=2)

        ttk.Label(filter_row1, text="Регион:").pack(side='left', padx=5)
        self.data_region_var = tk.StringVar(value="Все регионы")
        self.data_region_combo = ttk.Combobox(filter_row1, textvariable=self.data_region_var, width=20)
        self.data_region_combo.pack(side='left', padx=5)

        ttk.Label(filter_row1, text="Показатель:").pack(side='left', padx=5)
        self.data_pollutant_var = tk.StringVar(value="Все показатели")
        self.data_pollutant_combo = ttk.Combobox(filter_row1, textvariable=self.data_pollutant_var,
                                                 values=["Все показатели", "so2", "no2", "rspm", "spm", "pm2_5"])
        self.data_pollutant_combo.pack(side='left', padx=5)

        # Вторая строка фильтров - даты
        filter_row2 = ttk.Frame(filter_frame)
        filter_row2.pack(fill='x', pady=2)

        ttk.Label(filter_row2, text="Период:").pack(side='left', padx=5)

        # Фрейм для полей ввода дат
        date_frame = ttk.Frame(filter_row2)
        date_frame.pack(side='left', padx=5)

        ttk.Label(date_frame, text="Нач. дата:").pack(side='left')
        self.data_start_date_var = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.data_start_date_var, width=12).pack(side='left', padx=2)

        ttk.Label(date_frame, text="Кон. дата:").pack(side='left', padx=(10, 0))
        self.data_end_date_var = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.data_end_date_var, width=12).pack(side='left', padx=2)

        # Кнопки управления фильтрами
        button_frame = ttk.Frame(filter_row2)
        button_frame.pack(side='left', padx=10)

        ttk.Button(button_frame, text="Применить фильтры",
                   command=self.apply_data_filters).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Сбросить фильтры",
                   command=self.reset_data_filters).pack(side='left', padx=2)

        # Подсказка по формату дат
        ttk.Label(filter_row2, text="Формат: ГГГГ-ММ-ДД", foreground="gray").pack(side='left', padx=5)

        # Фрейм информации о данных
        info_frame = ttk.LabelFrame(self.data_tab, text="Информация о данных", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=6, width=100)
        self.info_text.pack(fill='both', expand=True)

        # Фрейм просмотра данных
        view_frame = ttk.LabelFrame(self.data_tab, text="Просмотр данных", padding=10)
        view_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Treeview для отображения данных
        columns = ("Дата", "Регион", "SO2", "NO2", "RSPM", "SPM", "PM2.5")
        self.data_tree = ttk.Treeview(view_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(view_frame, orient='vertical', command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)

        self.data_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_analysis_tab(self):
        """Вкладка анализа данных"""
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Анализ")

        # Фрейм выбора параметров
        params_frame = ttk.LabelFrame(self.analysis_tab, text="Параметры анализа", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)

        # Первая строка параметров
        row1 = ttk.Frame(params_frame)
        row1.pack(fill='x', pady=2)

        ttk.Label(row1, text="Целевой показатель:").pack(side='left', padx=5)
        self.pollutant_var = tk.StringVar(value="so2")
        self.pollutant_combo = ttk.Combobox(row1, textvariable=self.pollutant_var,
                                            values=["so2", "no2", "rspm", "spm", "pm2_5"])
        self.pollutant_combo.pack(side='left', padx=5)

        ttk.Label(row1, text="Регион:").pack(side='left', padx=5)
        self.region_var = tk.StringVar(value="Все регионы")
        self.region_combo = ttk.Combobox(row1, textvariable=self.region_var)
        self.region_combo.pack(side='left', padx=5)

        # Вторая строка параметров
        row2 = ttk.Frame(params_frame)
        row2.pack(fill='x', pady=2)

        ttk.Label(row2, text="Метод анализа:").pack(side='left', padx=5)
        self.trend_method_var = tk.StringVar(value="composite")
        trend_combo = ttk.Combobox(row2, textvariable=self.trend_method_var,
                                   values=["linear", "moving_avg", "decomposition", "composite"])
        trend_combo.pack(side='left', padx=5)

        ttk.Label(row2, text="Горизонт прогноза (ч):").pack(side='left', padx=5)
        self.forecast_horizon_var = tk.StringVar(value="24")
        ttk.Spinbox(row2, from_=1, to=168, textvariable=self.forecast_horizon_var,
                    width=5).pack(side='left', padx=5)

        # Прогресс-бар
        progress_frame = ttk.Frame(params_frame)
        progress_frame.pack(fill='x', pady=5)

        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_label_var)
        self.progress_label.pack(anchor='w')

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', pady=2)

        # Кнопки анализа
        button_frame = ttk.Frame(params_frame)
        button_frame.pack(fill='x', pady=10)

        self.analyze_trends_btn = ttk.Button(button_frame, text="Анализ трендов",
                                             command=self.analyze_trends)
        self.analyze_trends_btn.pack(side='left', padx=5)

        self.analyze_forecast_btn = ttk.Button(button_frame, text="Прогнозирование",
                                               command=self.analyze_forecast)
        self.analyze_forecast_btn.pack(side='left', padx=5)

        self.calculate_aqi_btn = ttk.Button(button_frame, text="Расчет AQI",
                                            command=self.calculate_aqi)
        self.calculate_aqi_btn.pack(side='left', padx=5)

        self.analyze_seasonal_btn = ttk.Button(button_frame, text="Сезонный анализ",
                                               command=self.analyze_seasonal)
        self.analyze_seasonal_btn.pack(side='left', padx=5)

        self.cancel_analysis_btn = ttk.Button(button_frame, text="Отменить",
                                              command=self.cancel_analysis, state='disabled')
        self.cancel_analysis_btn.pack(side='left', padx=5)

        # Фрейм результатов анализа
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Результаты анализа", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.analysis_text = scrolledtext.ScrolledText(results_frame, height=15, width=100)
        self.analysis_text.pack(fill='both', expand=True)

    def setup_visualization_tab(self):
        """Вкладка визуализации"""
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Визуализация")

        # Фрейм управления визуализацией
        control_frame = ttk.LabelFrame(self.viz_tab, text="Параметры визуализации", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)

        # Строка выбора параметров
        params_row = ttk.Frame(control_frame)
        params_row.pack(fill='x', pady=5)

        ttk.Label(params_row, text="Показатель:").pack(side='left', padx=5)
        self.viz_pollutant_var = tk.StringVar(value="so2")
        viz_pollutant_combo = ttk.Combobox(params_row, textvariable=self.viz_pollutant_var,
                                           values=["so2", "no2", "rspm", "spm", "pm2_5"])
        viz_pollutant_combo.pack(side='left', padx=5)

        ttk.Label(params_row, text="Регион:").pack(side='left', padx=5)
        self.viz_region_var = tk.StringVar(value="Все регионы")
        self.viz_region_combo = ttk.Combobox(params_row, textvariable=self.viz_region_var)
        self.viz_region_combo.pack(side='left', padx=5)

        # Строка временного фильтра
        time_row = ttk.Frame(control_frame)
        time_row.pack(fill='x', pady=5)

        ttk.Label(time_row, text="Период:").pack(side='left', padx=5)

        # Фрейм для полей ввода дат
        date_frame = ttk.Frame(time_row)
        date_frame.pack(side='left', padx=5)

        ttk.Label(date_frame, text="Нач. дата:").pack(side='left')
        self.start_date_var = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.start_date_var, width=12).pack(side='left', padx=2)

        ttk.Label(date_frame, text="Кон. дата:").pack(side='left', padx=(10, 0))
        self.end_date_var = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.end_date_var, width=12).pack(side='left', padx=2)

        ttk.Button(time_row, text="Применить фильтр",
                   command=self.apply_viz_filters).pack(side='left', padx=10)

        # Подсказка по формату дат
        ttk.Label(time_row, text="Формат: ГГГГ-ММ-ДД", foreground="gray").pack(side='left', padx=5)

        # Строка кнопок графиков
        buttons_row = ttk.Frame(control_frame)
        buttons_row.pack(fill='x', pady=5)

        ttk.Button(buttons_row, text="Временной ряд",
                   command=self.plot_timeseries).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Сравнение загрязнителей",
                   command=self.plot_comparison).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Региональное сравнение",
                   command=self.plot_regional).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Сезонные паттерны",
                   command=self.plot_seasonal).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Годовая динамика",
                   command=self.plot_yearly).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Дашборд AQI",
                   command=self.plot_aqi).pack(side='left', padx=2)
        ttk.Button(buttons_row, text="Очистить графики",
                   command=self.clear_plots).pack(side='left', padx=2)

        # Фрейм для графиков
        self.plot_frame = ttk.Frame(self.viz_tab)
        self.plot_frame.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_results_tab(self):
        """Вкладка итоговых результатов"""
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Результаты")

        # Фрейм экспорта
        export_frame = ttk.LabelFrame(self.results_tab, text="Экспорт результатов", padding=10)
        export_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(export_frame, text="Сохранить отчет",
                   command=self.save_report).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Экспорт графиков",
                   command=self.export_plots).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Сводный отчет",
                   command=self.show_summary).pack(side='left', padx=5)
        # НОВАЯ КНОПКА для экспорта полного прогноза
        ttk.Button(export_frame, text="Экспорт полного прогноза",
                   command=self.export_full_forecast).pack(side='left', padx=5)

        # Фрейм сводного отчета - ИСПРАВЛЕНО: создаем self.summary_text
        summary_frame = ttk.LabelFrame(self.results_tab, text="Сводный отчет", padding=10)
        summary_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # СОЗДАЕМ self.summary_text
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=20, width=100)
        self.summary_text.pack(fill='both', expand=True)

    def export_full_forecast(self):
        """Экспорт полного прогноза в CSV"""
        if 'forecast' not in self.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет данных прогноза для экспорта")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить полный прогноз"
            )

            if file_path:
                forecast_data = self.analysis_results['forecast']

                if 'final_forecast' in forecast_data and 'forecast_dates' in forecast_data:
                    # Создаем DataFrame с полным прогнозом
                    df = pd.DataFrame({
                        'datetime': forecast_data['forecast_dates'],
                        'forecast': forecast_data['final_forecast']
                    })

                    # Добавляем дополнительные методы прогнозирования если есть
                    if 'all_predictions' in forecast_data:
                        for method, values in forecast_data['all_predictions'].items():
                            if len(values) == len(df):
                                df[f'forecast_{method}'] = values

                    df.to_csv(file_path, index=False, encoding='utf-8')
                    messagebox.showinfo("Успех", f"Полный прогноз экспортирован: {file_path}")
                else:
                    messagebox.showwarning("Предупреждение", "Нет данных для экспорта")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def show_summary(self):
        """Показать сводный отчет"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        try:
            summary_text = "📊 СВОДНЫЙ ОТЧЕТ ПО АНАЛИЗУ КАЧЕСТВА ВОЗДУХА\n\n"
            summary_text += "=" * 50 + "\n\n"

            # Информация о данных
            summary_text += "📁 ДАННЫЕ:\n"
            summary_text += f"• Записей: {len(self.data)}\n"
            if 'date' in self.data.columns:
                min_date = self.data['date'].min()
                max_date = self.data['date'].max()
                summary_text += f"• Период: {min_date} - {max_date}\n"
                summary_text += f"• Дней данных: {(max_date - min_date).days}\n"
            summary_text += f"• Целевой показатель: {self.pollutant_var.get()}\n"
            summary_text += f"• Регион: {self.region_var.get()}\n\n"

            # Статистика по данным
            summary_text += "📈 СТАТИСТИКА ДАННЫХ:\n"
            pollutant = self.pollutant_var.get()
            if pollutant in self.data.columns:
                data_stats = self.data[pollutant].describe()
                summary_text += f"• Среднее: {data_stats.get('mean', 0):.2f}\n"
                summary_text += f"• Медиана: {data_stats.get('50%', 0):.2f}\n"
                summary_text += f"• Стандартное отклонение: {data_stats.get('std', 0):.2f}\n"
                summary_text += f"• Минимум: {data_stats.get('min', 0):.2f}\n"
                summary_text += f"• Максимум: {data_stats.get('max', 0):.2f}\n"
                summary_text += f"• Непустых значений: {self.data[pollutant].notna().sum()}\n\n"

            # Результаты анализа
            summary_text += "🔍 РЕЗУЛЬТАТЫ АНАЛИЗА:\n"

            if 'trends' in self.analysis_results:
                trends = self.analysis_results['trends']
                direction = trends.get('overall_direction', 'Не определен')
                change_pct = trends.get('change_percentage', 0)
                summary_text += f"• Тренд: {direction} ({change_pct:+.2f}%)\n"

            if 'aqi' in self.analysis_results and 'overall' in self.analysis_results['aqi']:
                aqi = self.analysis_results['aqi']['overall']
                summary_text += f"• Общий AQI: {aqi.get('aqi', 'N/A')} ({aqi.get('category', 'N/A')})\n"
                summary_text += f"• Основной загрязнитель: {aqi.get('dominant_pollutant', 'N/A')}\n"

            if 'forecast' in self.analysis_results:
                forecast = self.analysis_results['forecast']
                horizon = forecast.get('forecast_horizon', 'N/A')
                if 'forecast_stats' in forecast:
                    stats = forecast['forecast_stats']
                    summary_text += f"• Прогноз ({horizon} ч): среднее {stats.get('mean', 0):.2f} "
                    summary_text += f"(диапазон: {stats.get('min', 0):.2f}-{stats.get('max', 0):.2f})\n"

            if 'seasonal' in self.analysis_results:
                seasonal = self.analysis_results['seasonal']
                if 'peak_hour' in seasonal:
                    peak = seasonal['peak_hour']
                    summary_text += f"• Пиковый час: {peak.get('hour', 'N/A')}:00 "
                    summary_text += f"({peak.get('concentration', 0):.2f})\n"

            summary_text += "\n" + "=" * 50 + "\n"
            summary_text += f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

            # Очищаем и вставляем текст
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания отчета: {str(e)}")

    def export_full_forecast(self):
        """Экспорт полного прогноза в CSV"""
        if 'forecast' not in self.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет данных прогноза для экспорта")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить полный прогноз"
            )

            if file_path:
                forecast_data = self.analysis_results['forecast']

                if 'final_forecast' in forecast_data and 'forecast_dates' in forecast_data:
                    # Создаем DataFrame с полным прогнозом
                    df = pd.DataFrame({
                        'datetime': forecast_data['forecast_dates'],
                        'forecast': forecast_data['final_forecast']
                    })

                    # Добавляем дополнительные методы прогнозирования если есть
                    if 'all_predictions' in forecast_data:
                        for method, values in forecast_data['all_predictions'].items():
                            if len(values) == len(df):
                                df[f'forecast_{method}'] = values

                    df.to_csv(file_path, index=False, encoding='utf-8')
                    messagebox.showinfo("Успех", f"Полный прогноз экспортирован: {file_path}")
                else:
                    messagebox.showwarning("Предупреждение", "Нет данных для экспорта")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def update_progress(self, value, label=""):
        """Обновление прогресс-бара"""
        self.progress_var.set(value)
        if label:
            self.progress_label_var.set(label)
        self.root.update_idletasks()

    def set_analysis_buttons_state(self, enabled):
        """Включение/выключение кнопок анализа"""
        state = 'normal' if enabled else 'disabled'
        self.analyze_trends_btn.config(state=state)
        self.analyze_forecast_btn.config(state=state)
        self.calculate_aqi_btn.config(state=state)
        self.analyze_seasonal_btn.config(state=state)
        self.cancel_analysis_btn.config(state='normal' if not enabled else 'disabled')
        self.is_analyzing = not enabled

    def cancel_analysis(self):
        """Отмена текущего анализа"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # В Python нет прямого способа остановить поток, но мы можем пометить его для отмены
            self.update_progress(0, "Анализ отменен...")
            self.set_analysis_buttons_state(True)

    def analyze_in_thread(self, analysis_func, func_name, *args, **kwargs):
        """Запуск анализа в отдельном потоке"""
        if self.is_analyzing:
            messagebox.showwarning("Предупреждение", "Уже выполняется другой анализ")
            return

        self.set_analysis_buttons_state(False)
        self.update_progress(10, f"Запуск {func_name}...")

        # Запуск в отдельном потоке
        self.analysis_thread = AnalysisThread(
            target=analysis_func,
            args=args,
            kwargs=kwargs,
            callback=self.on_analysis_complete
        )
        self.analysis_thread.start()

        # Запуск мониторинга прогресса
        self.monitor_analysis_progress(func_name)

    def monitor_analysis_progress(self, func_name):
        """Мониторинг прогресса анализа"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # Имитация прогресса для длительных операций
            current_progress = self.progress_var.get()
            if current_progress < 80:
                new_progress = current_progress + 5
                self.update_progress(new_progress, f"Выполняется {func_name}...")
                self.root.after(1000, lambda: self.monitor_analysis_progress(func_name))
            else:
                self.root.after(500, lambda: self.monitor_analysis_progress(func_name))

    def on_analysis_complete(self, thread):
        """Обработчик завершения анализа"""
        self.set_analysis_buttons_state(True)

        if thread.exception:
            self.update_progress(0, "Ошибка")
            messagebox.showerror("Ошибка", f"Ошибка при анализе: {thread.exception}")
        else:
            self.update_progress(100, "Завершено")
            self.root.after(1000, lambda: self.update_progress(0, "Готов"))

            # Обновляем UI с результатами
            if hasattr(self, 'pending_update') and self.pending_update:
                self.pending_update(thread.result)

    def analyze_trends(self):
        """Анализ трендов в отдельном потоке"""
        data = self.get_filtered_data()
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        def trends_analysis():
            self.update_progress(20, "Анализ трендов...")
            analysis_data = data.copy()
            if 'date' in analysis_data.columns:
                analysis_data = analysis_data.rename(columns={'date': 'timestamp'})

            result = ac.calculate_pollution_trend(
                analysis_data,
                self.pollutant_var.get(),
                self.trend_method_var.get()
            )
            self.update_progress(80, "Формирование результатов...")
            return result

        self.analyze_in_thread(trends_analysis, "анализ трендов")

        # Сохраняем функцию для обновления UI
        self.pending_update = self.display_trends_results

    def analyze_forecast(self):
        """Прогнозирование в отдельном потоке"""
        data = self.get_filtered_data()
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        def forecast_analysis():
            self.update_progress(20, "Подготовка данных...")
            horizon = int(self.forecast_horizon_var.get())
            analysis_data = data.copy()

            if 'date' in analysis_data.columns:
                analysis_data = analysis_data.rename(columns={'date': 'timestamp'})

            self.update_progress(40, "Построение прогноза...")
            result = ac.predict_future_levels(
                analysis_data,
                self.pollutant_var.get(),
                forecast_horizon=horizon,
                method='hybrid'
            )
            self.update_progress(80, "Формирование результатов...")
            return result

        self.analyze_in_thread(forecast_analysis, "прогнозирование")
        self.pending_update = self.display_forecast_results

    def calculate_aqi(self):
        """Расчет AQI в отдельном потоке"""
        data = self.get_filtered_data()
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        def aqi_analysis():
            self.update_progress(30, "Расчет индексов AQI...")
            result = ac.compute_air_quality_index(data)
            self.update_progress(80, "Формирование отчета...")
            return result

        self.analyze_in_thread(aqi_analysis, "расчет AQI")
        self.pending_update = self.display_aqi_results

    def analyze_seasonal(self):
        """Сезонный анализ в отдельном потоке"""
        data = self.get_filtered_data()
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        def seasonal_analysis():
            self.update_progress(25, "Анализ сезонных паттернов...")
            result = ac.analyze_seasonal_patterns(data, self.pollutant_var.get(), period='daily')
            self.update_progress(75, "Обработка результатов...")
            return result

        self.analyze_in_thread(seasonal_analysis, "сезонный анализ")
        self.pending_update = self.display_seasonal_results

    def display_trends_results(self, trends):
        """Отображение результатов анализа трендов"""
        if not trends or 'error' in trends:
            result_text = "❌ Не удалось выполнить анализ трендов\n"
            if trends and 'error' in trends:
                result_text += f"Ошибка: {trends['error']}\n"
        else:
            region_info = f" ({self.region_var.get()})" if self.region_var.get() != "Все регионы" else ""
            result_text = f"📈 АНАЛИЗ ТРЕНДОВ: {self.pollutant_var.get().upper()}{region_info}\n"
            result_text += f"Метод: {self.trend_method_var.get()}\n"
            result_text += f"Период анализа: {trends.get('period_days', 'N/A')} дней\n"
            result_text += f"Точек данных: {trends.get('data_points', 'N/A')}\n\n"

            if 'overall_direction' in trends:
                result_text += f"Общее направление: {trends['overall_direction']}\n"
                result_text += f"Изменение: {trends.get('change_percentage', 0):.2f}%\n"

            if 'linear_trend' in trends:
                lin_trend = trends['linear_trend']
                result_text += f"\nЛинейный тренд:\n"
                result_text += f"  Наклон: {lin_trend.get('slope', 0):.6f}\n"
                result_text += f"  R²: {lin_trend.get('r_squared', 0):.3f}\n"

            self.analysis_results['trends'] = trends

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, result_text)

    def display_forecast_results(self, forecast):
        """Отображение результатов прогнозирования"""
        if not forecast or 'error' in forecast:
            result_text = "❌ Не удалось выполнить прогнозирование\n"
            if forecast and 'error' in forecast:
                result_text += f"Ошибка: {forecast['error']}\n"
        else:
            region_info = f" ({self.region_var.get()})" if self.region_var.get() != "Все регионы" else ""
            result_text = f"🔮 ПРОГНОЗ: {self.pollutant_var.get().upper()}{region_info}\n"
            result_text += f"Горизонт: {forecast.get('forecast_horizon', 'N/A')} часов\n"
            result_text += f"Метод: {forecast.get('method_used', 'N/A')}\n\n"

            if 'forecast_stats' in forecast:
                stats = forecast['forecast_stats']
                result_text += "Статистика прогноза:\n"
                result_text += f"  Среднее: {stats.get('mean', 0):.2f}\n"
                result_text += f"  Мин: {stats.get('min', 0):.2f}\n"
                result_text += f"  Макс: {stats.get('max', 0):.2f}\n"
                result_text += f"  Станд. откл.: {stats.get('std', 0):.2f}\n"

            # Детальный прогноз по часам - ВЫВОДИМ ВСЕ ЧАСЫ ИЛИ РАЗУМНОЕ КОЛИЧЕСТВО
            if 'final_forecast' in forecast and 'forecast_dates' in forecast:
                forecast_horizon = forecast.get('forecast_horizon', 24)

                # Если горизонт большой, показываем первые 24 часа и последние 6 часов
                if forecast_horizon > 30:
                    result_text += f"\nДетальный прогноз (первые 24 часа и последние 6 часов):\n"

                    # Первые 24 часа
                    first_forecasts = forecast['final_forecast'][:24]
                    first_dates = forecast['forecast_dates'][:24]

                    for i, (date, value) in enumerate(zip(first_dates, first_forecasts)):
                        time_str = pd.to_datetime(date).strftime('%m-%d %H:%M')
                        result_text += f"  {time_str}: {value:.2f}\n"

                    result_text += f"  ... (пропущено {forecast_horizon - 30} часов) ...\n"

                    # Последние 6 часов
                    last_forecasts = forecast['final_forecast'][-6:]
                    last_dates = forecast['forecast_dates'][-6:]

                    for i, (date, value) in enumerate(zip(last_dates, last_forecasts)):
                        time_str = pd.to_datetime(date).strftime('%m-%d %H:%M')
                        result_text += f"  {time_str}: {value:.2f}\n"

                else:
                    # Для небольших горизонтов показываем все часы
                    result_text += f"\nДетальный прогноз (все {forecast_horizon} часов):\n"
                    forecasts = forecast['final_forecast']
                    dates = forecast['forecast_dates']

                    for i, (date, value) in enumerate(zip(dates, forecasts)):
                        time_str = pd.to_datetime(date).strftime('%m-%d %H:%M')
                        result_text += f"  {time_str}: {value:.2f}\n"

            self.analysis_results['forecast'] = forecast

        current_text = self.analysis_text.get(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

    def display_aqi_results(self, aqi_results):
        """Отображение результатов AQI"""
        if not aqi_results:
            result_text = "❌ Не удалось рассчитать AQI\n"
            result_text += "Проверьте наличие данных по SO2, NO2, PM2.5, PM10.\n"
        else:
            region_info = f" ({self.region_var.get()})" if self.region_var.get() != "Все регионы" else ""
            result_text = f"🌍 ИНДЕКС КАЧЕСТВА ВОЗДУХА (AQI){region_info}\n\n"

            if 'overall' in aqi_results:
                overall = aqi_results['overall']
                result_text += f"ОБЩИЙ AQI: {overall['aqi']} - {overall['category']}\n"
                result_text += f"Основной загрязнитель: {overall['dominant_pollutant']}\n\n"

            # Показываем все доступные загрязнители
            for poll, poll_data in aqi_results.items():
                if poll != 'overall':
                    result_text += f"{poll}:\n"
                    result_text += f"  Концентрация: {poll_data.get('concentration', 0):.2f} {poll_data.get('unit', '')}\n"
                    result_text += f"  AQI: {poll_data.get('aqi', 0)}\n"
                    result_text += f"  Категория: {poll_data.get('category', 'N/A')}\n"
                    result_text += f"  Рекомендации: {poll_data.get('health_advice', 'N/A')}\n\n"

            self.analysis_results['aqi'] = aqi_results

        current_text = self.analysis_text.get(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

    def display_seasonal_results(self, seasonal):
        """Отображение результатов сезонного анализа"""
        if not seasonal or 'error' in seasonal:
            result_text = "❌ Не удалось выполнить сезонный анализ\n"
            if seasonal and 'error' in seasonal:
                result_text += f"Ошибка: {seasonal['error']}\n"
        else:
            region_info = f" ({self.region_var.get()})" if self.region_var.get() != "Все регионы" else ""
            result_text = f"📅 СЕЗОННЫЙ АНАЛИЗ: {self.pollutant_var.get().upper()}{region_info}\n\n"

            if 'basic_stats' in seasonal:
                stats = seasonal['basic_stats']
                result_text += f"Общая статистика:\n"
                result_text += f"  Среднее: {stats.get('mean', 0):.2f}\n"
                result_text += f"  Стандартное отклонение: {stats.get('std', 0):.2f}\n"
                result_text += f"  Минимум: {stats.get('min', 0):.2f}\n"
                result_text += f"  Максимум: {stats.get('max', 0):.2f}\n"
                result_text += f"  Записей: {stats.get('total_records', 0)}\n\n"

            if 'hourly_patterns' in seasonal:
                result_text += "Суточные паттерны (первые 6 часов):\n"
                patterns = seasonal['hourly_patterns']
                for pattern in patterns[:6]:
                    result_text += f"  {int(pattern['hour'])}:00 - {pattern['mean']:.2f} (σ={pattern.get('std', 0):.2f})\n"

            if 'peak_hour' in seasonal:
                peak = seasonal['peak_hour']
                result_text += f"\n🏆 Пиковый час: {peak['hour']}:00\n"
                result_text += f"Концентрация: {peak['concentration']:.2f}\n"

            self.analysis_results['seasonal'] = seasonal

        current_text = self.analysis_text.get(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

    def load_data(self):
        """Загрузка данных из CSV файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.file_label.config(text=f"Загружается: {os.path.basename(file_path)}")
            self.root.update()

            # Загрузка данных
            self.data, validation_report = dm.load_environmental_data(file_path)

            if self.data.empty:
                messagebox.showerror("Ошибка", "Не удалось загрузить данные")
                return

            # Обновление информации
            self.update_data_info(validation_report)

            # Обновление treeview
            self.update_data_treeview()

            # Обновление регионов
            self.update_regions()

            # Обновление выбора показателей
            self.update_pollutant_choices()

            messagebox.showinfo("Успех", "Данные успешно загружены!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def update_data_info(self, validation_report):
        """Обновление информации о данных"""
        info_text = f"✅ Успешно загружено: {validation_report['records_loaded']} записей\n"

        if 'data_period' in validation_report and validation_report['data_period']:
            period = validation_report['data_period']
            info_text += f"📅 Период: {period.get('start', 'N/A')} - {period.get('end', 'N/A')}\n"

        info_text += f"📊 Колонки: {', '.join(self.data.columns)}\n\n"

        # Статистика по показателям
        numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
        for col in numeric_columns:
            if col in self.data.columns:
                non_null = self.data[col].notna().sum()
                percentage = (non_null / len(self.data)) * 100
                if non_null > 0:
                    avg = self.data[col].mean()
                    info_text += f"{col}: {non_null} записей ({percentage:.1f}%), среднее: {avg:.2f}\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    def update_regions(self):
        """Обновление списка регионов"""
        if self.data is None:
            return

        # Поиск колонки с регионами
        region_columns = ['state', 'city', 'location', 'region', 'area']
        region_col = None

        for col in region_columns:
            if col in self.data.columns:
                region_col = col
                break

        if region_col:
            regions = self.data[region_col].dropna().unique()
            self.regions = {region: region for region in regions}

            # Обновление комбобоксов
            region_values = ["Все регионы"] + list(regions)
            self.region_combo['values'] = region_values
            self.viz_region_combo['values'] = region_values
            self.data_region_combo['values'] = region_values
        else:
            self.regions = {"Все данные": "Все данные"}
            self.region_combo['values'] = ["Все регионы"]
            self.viz_region_combo['values'] = ["Все регионы"]
            self.data_region_combo['values'] = ["Все регионы"]

    def get_filtered_data(self, use_viz_filters=False, use_data_filters=False):
        """Получить отфильтрованные данные"""
        if self.data is None:
            return None

        filtered_data = self.data.copy()

        # Выбор источника фильтров
        if use_viz_filters:
            region_var = self.viz_region_var
            start_date_var = self.start_date_var
            end_date_var = self.end_date_var
        elif use_data_filters:
            region_var = self.data_region_var
            start_date_var = self.data_start_date_var
            end_date_var = self.data_end_date_var
        else:
            region_var = self.region_var
            start_date_var = tk.StringVar()  # Пустые для анализа
            end_date_var = tk.StringVar()

        # Фильтрация по региону
        current_region = region_var.get()
        if current_region != "Все регионы":
            region_columns = ['state', 'city', 'location', 'region', 'area']
            for col in region_columns:
                if col in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[col] == current_region]
                    break

        # Фильтрация по дате
        if (use_viz_filters or use_data_filters) and 'date' in filtered_data.columns:
            try:
                start_date = start_date_var.get()
                end_date = end_date_var.get()

                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    filtered_data = filtered_data[filtered_data['date'] >= start_dt]

                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    filtered_data = filtered_data[filtered_data['date'] <= end_dt]

            except Exception as e:
                print(f"Ошибка фильтрации дат: {e}")

        return filtered_data

    def apply_data_filters(self):
        """Применить фильтры для данных"""
        filtered_data = self.get_filtered_data(use_data_filters=True)
        if filtered_data is not None:
            # Обновляем информацию о данных
            self.update_filtered_data_info(filtered_data)

            # Обновляем treeview
            self.update_data_treeview(filtered_data)

            messagebox.showinfo("Успех", f"Фильтры применены. Отобрано записей: {len(filtered_data)}")

    def reset_data_filters(self):
        """Сбросить фильтры данных"""
        self.data_region_var.set("Все регионы")
        self.data_pollutant_var.set("Все показатели")
        self.data_start_date_var.set("")
        self.data_end_date_var.set("")

        # Обновляем информацию и treeview
        self.update_data_info({'records_loaded': len(self.data) if self.data else 0})
        self.update_data_treeview(self.data)

        messagebox.showinfo("Успех", "Фильтры сброшены")

    def update_filtered_data_info(self, filtered_data):
        """Обновление информации о отфильтрованных данных"""
        info_text = f"✅ Отфильтровано записей: {len(filtered_data)}\n"

        if 'date' in filtered_data.columns:
            info_text += f"📅 Период: {filtered_data['date'].min()} - {filtered_data['date'].max()}\n"

        region_info = f" ({self.data_region_var.get()})" if self.data_region_var.get() != "Все регионы" else ""
        info_text += f"📍 Регион: {self.data_region_var.get()}{region_info}\n\n"

        # Статистика по показателям
        numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
        for col in numeric_columns:
            if col in filtered_data.columns:
                non_null = filtered_data[col].notna().sum()
                percentage = (non_null / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
                if non_null > 0:
                    avg = filtered_data[col].mean()
                    info_text += f"{col}: {non_null} записей ({percentage:.1f}%), среднее: {avg:.2f}\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    def apply_viz_filters(self):
        """Применить фильтры для визуализации"""
        filtered_data = self.get_filtered_data(use_viz_filters=True)
        if filtered_data is not None:
            messagebox.showinfo("Успех", f"Фильтры применены. Отобрано записей: {len(filtered_data)}")

    def update_data_treeview(self, data=None):
        """Обновление отображения данных в treeview"""
        if data is None:
            data = self.data

        if data is None:
            return

        # Очистка существующих данных
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # Фильтрация по показателю если выбран
        pollutant = self.data_pollutant_var.get()
        if pollutant != "Все показатели" and pollutant in data.columns:
            # Показываем только записи с данными по выбранному показателю
            data = data[data[pollutant].notna()]

        # Добавление записей для предпросмотра (максимум 100)
        preview_data = data.head(100)

        for _, row in preview_data.iterrows():
            values = []
            for col_name in ["Дата", "Регион", "SO2", "NO2", "RSPM", "SPM", "PM2.5"]:
                if col_name == "Дата" and 'date' in data.columns:
                    values.append(str(row['date'])[:19] if pd.notna(row.get('date')) else "")
                elif col_name == "Регион":
                    # Поиск региона в данных
                    region = "N/A"
                    region_columns = ['state', 'city', 'location', 'region', 'area']
                    for r_col in region_columns:
                        if r_col in data.columns and pd.notna(row.get(r_col)):
                            region = str(row[r_col])
                            break
                    values.append(region)
                elif col_name.lower() in data.columns:
                    val = row[col_name.lower()]
                    values.append(f"{val:.2f}" if pd.notna(val) else "")
                else:
                    values.append("")

            self.data_tree.insert("", "end", values=values)

    def update_pollutant_choices(self):
        """Обновление доступных показателей для анализа"""
        available_pollutants = []
        for col in ['so2', 'no2', 'rspm', 'spm', 'pm2_5']:
            if col in self.data.columns and self.data[col].notna().sum() > 100:
                available_pollutants.append(col)

        if available_pollutants:
            self.pollutant_var.set(available_pollutants[0])
            self.pollutant_combo['values'] = available_pollutants
            self.viz_pollutant_var.set(available_pollutants[0])

    def plot_timeseries(self):
        """Построение упрощенного графика временного ряда"""
        data = self.get_filtered_data(use_viz_filters=True)
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.viz_pollutant_var.get()
        region = self.viz_region_var.get()

        # Формируем текст периода
        period_text = ""
        if self.start_date_var.get() or self.end_date_var.get():
            start = self.start_date_var.get() or "начало"
            end = self.end_date_var.get() or "конец"
            period_text = f"[{start} - {end}]"

        try:
            fig = ve.create_simple_timeseries_plot(data, pollutant, region, period_text)
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить график. Проверьте данные.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_comparison(self):
        """Построение графика сравнения загрязнителей"""
        data = self.get_filtered_data(use_viz_filters=True)
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        region = self.viz_region_var.get()
        period_text = ""
        if self.start_date_var.get() or self.end_date_var.get():
            start = self.start_date_var.get() or "начало"
            end = self.end_date_var.get() or "конец"
            period_text = f"[{start} - {end}]"

        # Все доступные загрязнители
        pollutants = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
        available_pollutants = [p for p in pollutants if p in data.columns]

        if len(available_pollutants) < 2:
            messagebox.showwarning("Предупреждение", "Нужно хотя бы 2 загрязнителя для сравнения")
            return

        try:
            fig = ve.create_pollutant_comparison_plot(data, available_pollutants, region, period_text)
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить график сравнения")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_regional(self):
        """Построение графика регионального сравнения"""
        data = self.get_filtered_data(use_viz_filters=True)
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.viz_pollutant_var.get()

        # Находим колонку с регионами
        region_col = None
        region_columns = ['state', 'city', 'location', 'region', 'area']
        for col in region_columns:
            if col in data.columns:
                region_col = col
                break

        if not region_col:
            messagebox.showwarning("Предупреждение", "Не найдена колонка с регионами")
            return

        try:
            fig = ve.create_regional_comparison_plot(data, pollutant, region_col, top_n=10)
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить график регионов")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_seasonal(self):
        """Построение графика сезонных паттернов"""
        data = self.get_filtered_data(use_viz_filters=True)
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.viz_pollutant_var.get()
        region = self.viz_region_var.get()

        try:
            fig = ve.create_monthly_trend_plot(data, pollutant, region)
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить график сезонности")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_yearly(self):
        """Построение графика годовой динамики"""
        data = self.get_filtered_data(use_viz_filters=True)
        if data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.viz_pollutant_var.get()
        region = self.viz_region_var.get()

        try:
            fig = ve.create_yearly_summary_plot(data, pollutant, region)
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить график годовой динамики")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_aqi(self):
        """Построение дашборда AQI"""
        if 'aqi' not in self.analysis_results:
            messagebox.showwarning("Предупреждение", "Сначала выполните расчет AQI")
            return

        try:
            aqi_data = self.analysis_results['aqi']
            fig = ve.create_aqi_dashboard(aqi_data)

            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось построить дашборд AQI")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения дашборда AQI: {str(e)}")

    def display_plot(self, fig):
        """Отображение графика в GUI"""
        # Очистка предыдущих графиков
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Создание canvas для matplotlib
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Сохранение ссылки на текущий график
        self.current_plots.append((fig, canvas))

    def clear_plots(self):
        """Очистка всех графиков"""
        for fig, canvas in self.current_plots:
            plt.close(fig)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.current_plots = []

    def save_report(self):
        """Сохранение отчета"""
        if not self.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'data_info': {
                        'records': len(self.data) if self.data else 0,
                        'pollutant': self.pollutant_var.get(),
                        'region': self.region_var.get()
                    },
                    'analysis_results': self.make_serializable(self.analysis_results)
                }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

                messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")

    def export_plots(self):
        """Экспорт графиков"""
        if not self.current_plots:
            messagebox.showwarning("Предупреждение", "Нет графиков для экспорта")
            return

        try:
            folder_path = filedialog.askdirectory(title="Выберите папку для сохранения графиков")

            if folder_path:
                for i, (fig, _) in enumerate(self.current_plots):
                    fig.savefig(f"{folder_path}/plot_{i + 1}.png", dpi=300, bbox_inches='tight')

                messagebox.showinfo("Успех", f"Графики экспортированы в: {folder_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def show_summary(self):
        """Показать сводный отчет"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        try:
            summary_text = "📊 СВОДНЫЙ ОТЧЕТ ПО АНАЛИЗУ КАЧЕСТВА ВОЗДУХА\n\n"

            # Информация о данных
            summary_text += "ДАННЫЕ:\n"
            summary_text += f"Записей: {len(self.data)}\n"
            if 'date' in self.data.columns:
                summary_text += f"Период: {self.data['date'].min()} - {self.data['date'].max()}\n"
            summary_text += f"Целевой показатель: {self.pollutant_var.get()}\n"
            summary_text += f"Регион: {self.region_var.get()}\n\n"

            # Результаты анализа
            summary_text += "РЕЗУЛЬТАТЫ АНАЛИЗА:\n"

            if 'trends' in self.analysis_results:
                trends = self.analysis_results['trends']
                summary_text += f"Тренды: {trends.get('overall_direction', 'Не определен')}\n"

            if 'aqi' in self.analysis_results and 'overall' in self.analysis_results['aqi']:
                aqi = self.analysis_results['aqi']['overall']
                summary_text += f"AQI: {aqi.get('aqi', 'N/A')} ({aqi.get('category', 'N/A')})\n"

            if 'forecast' in self.analysis_results:
                forecast = self.analysis_results['forecast']
                if 'forecast_stats' in forecast:
                    stats = forecast['forecast_stats']
                    summary_text += f"Прогноз (среднее): {stats.get('mean', 0):.2f}\n"

            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания отчета: {str(e)}")

    def make_serializable(self, obj):
        """Рекурсивное преобразование объекта в сериализуемый формат"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {key: self.make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        elif hasattr(obj, 'dtype'):  # numpy types
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        else:
            return str(obj)


def main():
    """Запуск GUI приложения"""
    root = tk.Tk()
    app = AirQualityAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()