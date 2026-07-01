import os

import gspread
import pandas as pd
from utils import get_creds


def clean_text_field(text):
    """Очищает ячейки от случайных артефактов экранирования строк."""
    if not isinstance(text, str):
        return str(text) if pd.notna(text) else ""
    cleaned = text.strip()

    # Убираем внешние кавычки-обертки, если они проросли при чтении ячеек
    if cleaned.startswith('"""') and cleaned.endswith('"""'):
        cleaned = cleaned[3:-3]
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    return cleaned.strip()


def main():
    # Извлекаем ID таблицы из переменных окружения воркспейса
    GSHEET_ID = os.getenv("SOURCE_GSHEET")
    if not GSHEET_ID:
        raise ValueError(
            "Критическая ошибка: Переменная окружения 'SOURCE_GSHEET' не задана."
        )

    print("==================================================")
    print(" 🧠 МИИБА: Сборщик системного промпта из Google Sheets")
    print("==================================================")

    print("1. Подключение к Google Sheets API и авторизация...")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    # Используем сквозной корпоративный токен авторизации курса
    creds = get_creds(scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GSHEET_ID)

    print("2. Скачивание облачных вкладок 'main_prompt' и 'task_decomposition'...")

    # Парсинг глобальных настроек роли и тона ИИ
    try:
        ws_main = sh.worksheet("main_prompt")
        df_main = pd.DataFrame(ws_main.get_all_records())
    except Exception as e:
        print(f"❌ Ошибка доступа к вкладке 'main_prompt': {e}")
        return

    # Парсинг пошаговой декомпозиции семинаров
    try:
        ws_decomp = sh.worksheet("task_decomposition")
        df_decomp = pd.DataFrame(ws_decomp.get_all_records())
    except Exception as e:
        print(f"❌ Ошибка доступа к вкладке 'task_decomposition': {e}")
        return

    # Убираем возможные невидимые пробелы в заголовках колонок
    df_main.columns = df_main.columns.str.strip()
    df_decomp.columns = df_decomp.columns.str.strip()

    # Инициализация итогового документа
    md_lines = ["# SYSTEM PROMPT & CONTEXT: Personal AI Tutor\n"]

    # 3. ПОСТРОЕНИЕ БАЗОВОЙ АРХИТЕКТУРЫ ПОВЕДЕНИЯ (main_prompt)
    if not df_main.empty:
        print("3. Маппинг фундаментальных блоков педагогической стратегии...")
        row_context = df_main.iloc[0]

        # Строгое соответствие заголовков секций колонкам в GSheets
        sections = [
            ("1. Твоя роль и Тон общения", "## 1. Твоя роль и Тон общения"),
            (
                "2. Педагогическая стратегия (ПРАВИЛА ИГРЫ)",
                "## 2. Педагогическая стратегия (ПРАВИЛА ИГРЫ)",
            ),
            (
                "3. Контекст курса (Границы твоих знаний)",
                "## 3. Контекст курса (Границы твоих знаний)",
            ),
            (
                "4. Как отвечать на технические вопросы",
                "## 4. Как отвечать на технические вопросы",
            ),
            ("5. СИСТЕМА НАВИГАЦИИ (ХЭШТЕГИ)", "## 5. СИСТЕМА НАВИГАЦИИ (ХЭШТЕГИ)"),
        ]

        for col_name, heading in sections:
            if col_name in df_main.columns:
                text_content = clean_text_field(row_context[col_name])
                md_lines.append(f"{heading}")
                md_lines.append(f"{text_content}\n")

    # 4. ИНТЕГРАЦИЯ СЦЕНАРИЕВ СЕМИНАРОВ И ТАСКОВ (task_decomposition)
    print("4. Компиляция пошаговых сценариев ведения дебаггинга...")
    md_lines.append("### 🔎 ПОДРОБНЫЕ СЦЕНАРИИ СЕМИНАРОВ (Шпаргалка тьютора)\n")

    # Безопасное определение колонок декомпозиции
    col_key = "Ключ" if "Ключ" in df_decomp.columns else df_decomp.columns[0]
    col_prompt = "Промпт" if "Промпт" in df_decomp.columns else df_decomp.columns[1]
    col_comm = (
        "Комментарий"
        if "Комментарий" in df_decomp.columns
        else (df_decomp.columns[2] if len(df_decomp.columns) > 2 else None)
    )

    for _, row in df_decomp.iterrows():
        key_title = clean_text_field(row[col_key])
        prompt_body = clean_text_field(row[col_prompt])

        if not key_title or not prompt_body:
            continue

        md_lines.append(f"#### {key_title}")
        md_lines.append(f"{prompt_body}\n")

        # Обогащаем промпт методическими подсказками, если они заполнены
        if col_comm and col_comm in df_decomp.columns and pd.notna(row[col_comm]):
            commentary = clean_text_field(row[col_comm])
            if commentary:
                md_lines.append(f"> 💡 **Методический комментарий:** {commentary}\n")

    # 5. ЭКСПОРТ РЕЗУЛЬТАТА В СТУДЕНЧЕСКИЙ КОНТУР
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_md_path = os.path.join(repo_root, "seminars", "help_me.md")

    print("5. Экспорт монолитного Markdown-файла в рабочую область...")
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(
        f"✅ Успех! Системный контекст ИИ-тьютора обновлен из облака: {output_md_path}"
    )
    print("==================================================")


if __name__ == "__main__":
    main()
