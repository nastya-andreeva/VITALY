import pandas as pd
import os


def diagnose_data_issue():
    """Диагностика проблемы с данными"""
    file_path = "data/air_quality_data.csv"

    print("🔍 ДИАГНОСТИКА ФАЙЛА ДАННЫХ")
    print("-" * 40)

    # 1. Проверка существования файла
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return

    print(f"✅ Файл найден: {file_path}")

    # 2. Проверка размера файла
    file_size = os.path.getsize(file_path)
    print(f"📏 Размер файла: {file_size} байт")

    # 3. Попытка загрузки напрямую pandas
    try:
        # Пробуем разные кодировки
        encodings = ['utf-8', 'latin-1', 'cp1251', 'iso-8859-1']

        for encoding in encodings:
            try:
                test_df = pd.read_csv(file_path, encoding=encoding, nrows=5)
                print(f"✅ Успешная загрузка с кодировкой: {encoding}")
                print(f"   Колонки: {test_df.columns.tolist()}")
                print(f"   Первые строки:")
                print(test_df.head(2))
                break
            except UnicodeDecodeError:
                print(f"❌ Ошибка кодировки: {encoding}")
                continue
            except Exception as e:
                print(f"❌ Другая ошибка с {encoding}: {e}")
                continue
        else:
            print("❌ Не удалось загрузить ни с одной кодировкой")
            return

        # 4. Полная загрузка
        full_df = pd.read_csv(file_path, encoding=encoding)
        print(f"✅ Полная загрузка успешна")
        print(f"   Размер данных: {full_df.shape}")
        print(f"   Колонки: {full_df.columns.tolist()}")
        print(f"   Типы данных:")
        print(full_df.dtypes)

        return full_df, encoding

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None, None


# Запустите диагностику
data, encoding = diagnose_data_issue()