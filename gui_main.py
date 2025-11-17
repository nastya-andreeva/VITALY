import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import sys

# Добавляем пути к нашим модулям
sys.path.append('.')
import data_manager as dm
import analysis_core as ac
import visualization_engine as ve


class AirQualityAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Система анализа качества воздуха")
        self.root.geometry("1200x800")

        self.data = None
        self.analysis_results = {}
        self.current_plots = []

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

        # Фрейм информации о данных
        info_frame = ttk.LabelFrame(self.data_tab, text="Информация о данных", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=100)
        self.info_text.pack(fill='both', expand=True)

        # Фрейм просмотра данных
        view_frame = ttk.LabelFrame(self.data_tab, text="Просмотр данных", padding=10)
        view_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Treeview для отображения данных
        columns = ("Дата", "SO2", "NO2", "RSPM", "SPM", "PM2.5")
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

        ttk.Label(params_frame, text="Целевой показатель:").grid(row=0, column=0, sticky='w', padx=5)
        self.pollutant_var = tk.StringVar(value="so2")
        pollutant_combo = ttk.Combobox(params_frame, textvariable=self.pollutant_var,
                                       values=["so2", "no2", "rspm", "spm", "pm2_5"])
        pollutant_combo.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(params_frame, text="Метод анализа трендов:").grid(row=1, column=0, sticky='w', padx=5)
        self.trend_method_var = tk.StringVar(value="composite")
        trend_combo = ttk.Combobox(params_frame, textvariable=self.trend_method_var,
                                   values=["linear", "moving_avg", "decomposition", "composite"])
        trend_combo.grid(row=1, column=1, sticky='w', padx=5)

        # Кнопки анализа
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Анализ трендов",
                   command=self.analyze_trends).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Прогнозирование",
                   command=self.analyze_forecast).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Расчет AQI",
                   command=self.calculate_aqi).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Сезонный анализ",
                   command=self.analyze_seasonal).pack(side='left', padx=5)

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
        control_frame = ttk.LabelFrame(self.viz_tab, text="Управление визуализацией", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Временной ряд",
                   command=self.plot_timeseries).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Распределение",
                   command=self.plot_distribution).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Сезонность",
                   command=self.plot_seasonal).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Дашборд AQI",
                   command=self.plot_aqi).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Очистить графики",
                   command=self.clear_plots).pack(side='left', padx=5)

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

        # Фрейм сводного отчета
        summary_frame = ttk.LabelFrame(self.results_tab, text="Сводный отчет", padding=10)
        summary_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=20, width=100)
        self.summary_text.pack(fill='both', expand=True)

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
            info_text = f"✅ Успешно загружено: {validation_report['records_loaded']} записей\n"
            info_text += f"📅 Период: {validation_report.get('data_period', 'Не определен')}\n"
            info_text += f"📊 Колонки: {', '.join(self.data.columns)}\n\n"

            # Статистика по показателям
            numeric_columns = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
            for col in numeric_columns:
                if col in self.data.columns:
                    non_null = self.data[col].notna().sum()
                    percentage = (non_null / len(self.data)) * 100
                    info_text += f"{col}: {non_null} записей ({percentage:.1f}%)\n"

            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)

            # Обновление treeview
            self.update_data_treeview()

            # Обновление выбора показателей
            self.update_pollutant_choices()

            messagebox.showinfo("Успех", "Данные успешно загружены!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def update_data_treeview(self):
        """Обновление отображения данных в treeview"""
        # Очистка существующих данных
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # Добавление первых 100 записей для предпросмотра
        preview_data = self.data.head(100)

        for _, row in preview_data.iterrows():
            values = []
            for col in ["Дата", "SO2", "NO2", "RSPM", "SPM", "PM2.5"]:
                if col == "Дата" and 'date' in self.data.columns:
                    values.append(str(row['date'])[:19] if pd.notna(row.get('date')) else "")
                elif col.lower() in self.data.columns:
                    val = row[col.lower()]
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

    def analyze_trends(self):
        """Анализ трендов"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.pollutant_var.get()
        method = self.trend_method_var.get()

        try:
            # Подготовка данных для анализа
            analysis_data = self.data.copy()
            if 'date' in analysis_data.columns:
                analysis_data = analysis_data.rename(columns={'date': 'timestamp'})

            # Анализ трендов
            trends = ac.calculate_pollution_trend(analysis_data, pollutant, method)

            # Отображение результатов
            result_text = f"📈 АНАЛИЗ ТРЕНДОВ: {pollutant.upper()}\n"
            result_text += f"Метод: {method}\n"
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

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа трендов: {str(e)}")

    def analyze_forecast(self):
        """Прогнозирование уровней загрязнения"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.pollutant_var.get()

        try:
            # Подготовка данных
            analysis_data = self.data.copy()
            if 'date' in analysis_data.columns:
                analysis_data = analysis_data.rename(columns={'date': 'timestamp'})

            # Прогнозирование
            forecast = ac.predict_future_levels(analysis_data, pollutant, forecast_horizon=24, method='hybrid')

            # Отображение результатов
            result_text = f"🔮 ПРОГНОЗ: {pollutant.upper()}\n"
            result_text += f"Горизонт: {forecast.get('forecast_horizon', 'N/A')} часов\n"
            result_text += f"Метод: {forecast.get('method_used', 'N/A')}\n\n"

            if 'forecast_stats' in forecast:
                stats = forecast['forecast_stats']
                result_text += "Статистика прогноза:\n"
                result_text += f"  Среднее: {stats.get('mean', 0):.2f}\n"
                result_text += f"  Мин: {stats.get('min', 0):.2f}\n"
                result_text += f"  Макс: {stats.get('max', 0):.2f}\n"
                result_text += f"  Станд. откл.: {stats.get('std', 0):.2f}\n"

            self.analysis_results['forecast'] = forecast
            current_text = self.analysis_text.get(1.0, tk.END)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка прогнозирования: {str(e)}")

    def calculate_aqi(self):
        """Расчет индекса качества воздуха"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        try:
            # Расчет AQI
            aqi_results = ac.compute_air_quality_index(self.data)

            # Отображение результатов
            result_text = "🌍 ИНДЕКС КАЧЕСТВА ВОЗДУХА (AQI)\n\n"

            if 'overall' in aqi_results:
                overall = aqi_results['overall']
                result_text += f"ОБЩИЙ AQI: {overall['aqi']} - {overall['category']}\n"
                result_text += f"Основной загрязнитель: {overall['dominant_pollutant']}\n\n"

            for poll, data in aqi_results.items():
                if poll != 'overall':
                    result_text += f"{poll}:\n"
                    result_text += f"  Концентрация: {data.get('concentration', 0):.2f} {data.get('unit', '')}\n"
                    result_text += f"  AQI: {data.get('aqi', 0)}\n"
                    result_text += f"  Категория: {data.get('category', 'N/A')}\n"
                    result_text += f"  Рекомендации: {data.get('health_advice', 'N/A')}\n\n"

            self.analysis_results['aqi'] = aqi_results
            current_text = self.analysis_text.get(1.0, tk.END)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка расчета AQI: {str(e)}")

    def analyze_seasonal(self):
        """Сезонный анализ"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.pollutant_var.get()

        try:
            # Сезонный анализ
            seasonal = ac.analyze_seasonal_patterns(self.data, pollutant, period='daily')

            # Отображение результатов
            result_text = f"📅 СЕЗОННЫЙ АНАЛИЗ: {pollutant.upper()}\n\n"

            if 'hourly_patterns' in seasonal:
                result_text += "Суточные паттерны:\n"
                for pattern in seasonal['hourly_patterns'][:6]:  # Первые 6 часов
                    result_text += f"  {int(pattern['hour'])}:00 - {pattern['mean']:.2f}\n"

            if 'peak_hour' in seasonal:
                peak = seasonal['peak_hour']
                result_text += f"\nПиковый час: {peak['hour']}:00\n"
                result_text += f"Концентрация: {peak['concentration']:.2f}\n"

            self.analysis_results['seasonal'] = seasonal
            current_text = self.analysis_text.get(1.0, tk.END)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, current_text + "\n\n" + result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сезонного анализа: {str(e)}")

    def plot_timeseries(self):
        """Построение графика временного ряда"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.pollutant_var.get()

        try:
            # Подготовка данных
            plot_data = self.data.copy()
            if 'date' in plot_data.columns:
                plot_data = plot_data.rename(columns={'date': 'timestamp'})

            # Создание графика
            fig, ax = plt.subplots(figsize=(10, 5))

            if 'timestamp' in plot_data.columns:
                valid_data = plot_data[['timestamp', pollutant]].dropna()
                ax.plot(valid_data['timestamp'], valid_data[pollutant],
                        alpha=0.7, linewidth=1, label=pollutant)

                ax.set_title(f'Временной ряд: {pollutant}')
                ax.set_xlabel('Дата')
                ax.set_ylabel('Концентрация')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Форматирование дат
                fig.autofmt_xdate()

            # Отображение в GUI
            self.display_plot(fig)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_distribution(self):
        """Построение гистограммы распределения"""
        if self.data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return

        pollutant = self.pollutant_var.get()

        try:
            fig, ax = plt.subplots(figsize=(10, 5))

            valid_data = self.data[pollutant].dropna()
            ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')

            ax.set_title(f'Распределение: {pollutant}')
            ax.set_xlabel('Концентрация')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)

            # Добавление статистики
            mean = valid_data.mean()
            median = valid_data.median()
            ax.axvline(mean, color='red', linestyle='--', label=f'Среднее: {mean:.2f}')
            ax.axvline(median, color='green', linestyle='--', label=f'Медиана: {median:.2f}')
            ax.legend()

            self.display_plot(fig)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения гистограммы: {str(e)}")

    def plot_seasonal(self):
        """Построение графика сезонности"""
        if self.data is None or 'seasonal' not in self.analysis_results:
            messagebox.showwarning("Предупреждение", "Сначала выполните сезонный анализ")
            return

        try:
            seasonal_data = self.analysis_results['seasonal']
            fig = ve.create_seasonal_analysis_plot(seasonal_data)

            if fig:
                self.display_plot(fig)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика сезонности: {str(e)}")

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
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Преобразование в сериализуемый формат
                    serializable_results = self.make_serializable(self.analysis_results)
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)

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
            summary_text += f"Целевой показатель: {self.pollutant_var.get()}\n\n"

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