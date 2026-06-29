import glob
import json
import os
import re

import gspread
import nbformat
import requests
from utils import get_creds

# ==========================================
# CONFIGURATION
# ==========================================
GSHEET_ID = os.getenv("SOURCE_GSHEET")
SHEET_NAME = "task_decomposition"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = "gemma3:4b"
SOURCE_DIR = os.path.join(".", "examples", "seminars")


# ==========================================
# OLLAMA API CALL (Instructions Only)
# ==========================================
def generate_tutor_instructions(solution_code, seminar_prefix, task_number):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    prompt = f"""You are an educational AI content developer for the university course "MIIBA".
I will provide you with a raw Python code block from a completed seminar solution.

Your job is to:
1. Come up with a short, professional, descriptive title for this specific programming task in Russian.
2. Write a short explanation of what the student needs to accomplish.
3. Write tutor guidance rules for our AI system instructing it on how to guide the student conceptually without spoiling the answer.

Code to process:
{solution_code}

Return EXACTLY a valid JSON object matching this schema. Do not output markdown codeblocks:
{{
    "task_title": "A short descriptive title in Russian",
    "student_task": "A short explanation of what the student needs to accomplish",
    "tags": "#{seminar_prefix}_TASK{task_number}_START, #{seminar_prefix}_TASK{task_number}_BUG",
    "tutor_instruction": "Strict instructions for the AI tutor explaining how to guide the student conceptually"
}}
"""
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return json.loads(response.json()["response"])
    except Exception as e:
        print(f"   [!] Network/JSON exception with Ollama: {e}")
        return {}


# ==========================================
# MAIN INSTRUCTION ROUTINE
# ==========================================
def main():
    if not GSHEET_ID:
        raise ValueError("Environment variable 'SOURCE_GSHEET' is not set.")

    print("1. Establishing Google Sheets Connection...")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = get_creds(scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GSHEET_ID)
    worksheet = sh.worksheet(SHEET_NAME)

    print(f" -> Clearing old data in tab '{SHEET_NAME}' and rewriting headers...")
    worksheet.clear()
    worksheet.append_row(["Ключ", "Промпт", "Комментарий"])

    search_path = os.path.join(SOURCE_DIR, "*.ipynb")
    notebook_files = glob.glob(search_path)

    if not notebook_files:
        print(f"No notebooks found in source directory: {SOURCE_DIR}")
        return

    print(f"2. Found {len(notebook_files)} notebooks for INSTRUCTION generation.")
    all_new_rows = []

    for file_path in notebook_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing File: {filename}")

        digit_match = re.search(r"\d+", filename)
        seminar_num = int(digit_match.group()) if digit_match else 1
        seminar_prefix = f"SEM{seminar_num}"

        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        task_counter = 1

        for cell in nb.cells:
            if cell.cell_type == "code":
                solution_code = cell.source.strip()

                if not solution_code:
                    continue

                # Rule check 1: Skip if it contains an import statement
                if any(
                    line.strip().startswith(("import ", "from "))
                    for line in solution_code.split("\n")
                ):
                    print("    -> Skipping block (contains imports).")
                    continue

                # Rule check 2: Skip autocheck validation blocks
                if "run_autocheck" in solution_code:
                    print("    -> Skipping block (contains run_autocheck validation).")
                    continue

                print(f"    -> Generating instructions for block #{task_counter}...")

                ai_data = generate_tutor_instructions(
                    solution_code, seminar_prefix, task_counter
                )

                generated_title = ai_data.get(
                    "task_title", "Анализ данных и вычисления"
                ).strip()
                task_key = f"ЗАДАНИЕ {task_counter}: {generated_title}"

                student_task = ai_data.get(
                    "student_task", "Выполнить расчеты по шагу семинара."
                )
                tags = ai_data.get(
                    "tags",
                    f"#{seminar_prefix}_TASK{task_counter}_START, #{seminar_prefix}_TASK{task_counter}_BUG",
                )
                tutor_instruction = ai_data.get(
                    "tutor_instruction",
                    "Помоги студенту разобраться с логикой и синтаксисом этой ячейки.",
                )

                combined_prompt = (
                    f"Задача студента: {student_task}\n"
                    f"Теги: {tags}\n"
                    f"Инструкция Тьютору: {tutor_instruction}"
                )

                all_new_rows.append([task_key, combined_prompt, ""])
                task_counter += 1

    if all_new_rows:
        print(
            f"\n3. Uploading {len(all_new_rows)} metadata rows to Google Sheet '{SHEET_NAME}'..."
        )
        worksheet.append_rows(all_new_rows)
        print("GSheet pipeline processing complete!")
    else:
        print("\nNo code targets were parsed.")


if __name__ == "__main__":
    main()
