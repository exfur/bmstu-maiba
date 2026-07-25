import glob
import os

import nbformat

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(__file__)

# Список пар (исходная папка -> целевая папка)
TASKS = [
    {
        "source": os.path.join(".", "examples", "seminars"),
        "target": os.path.abspath(os.path.join(BASE_DIR, "..", "seminars")),
    },
    {
        "source": os.path.join(".", "examples", "course_project", "notebooks"),
        "target": os.path.abspath(os.path.join(BASE_DIR, "..", "course_project", "notebooks")),
    },
]


# ==========================================
# HELPER ROUTINE
# ==========================================
def process_directory(source_dir: str, target_dir: str) -> None:
    """Очищает ноутбуки из source_dir и сохраняет студенческие версии в target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    search_path = os.path.join(source_dir, "*.ipynb")
    notebook_files = glob.glob(search_path)

    if not notebook_files:
        print(f"No notebooks found in source directory: {source_dir}")
        return

    print("\n==========================================")
    print(f"Processing: {source_dir} -> {target_dir}")
    print(f"Found {len(notebook_files)} notebooks for student version generation.")
    print("==========================================")

    for file_path in notebook_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing File: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        new_cells = []
        removed_masters_count = 0
        processed_templates_count = 0

        # Пошагово фильтруем структуру ячеек ноутбука
        for cell in nb.cells:
            if cell.cell_type == "code":
                source_code = cell.source

                # Шаг 1. Если это блок решения преподавателя — полностью удаляем его из студенческой версии
                if "[MASTER SOLUTION]" in source_code:
                    removed_masters_count += 1
                    continue  # Пропускаем ячейку, она не попадет в финальный файл

                # Шаг 2. Если это шаблон для студента — сохраняем его и убираем служебный маркер для чистоты
                if "[STUDENT TEMPLATE]" in source_code:
                    # Вырезаем служебную строку-маркер, чтобы оставить код pristine чистым
                    cleaned_code = source_code.replace("# [STUDENT TEMPLATE]\n", "").replace("# [STUDENT TEMPLATE]", "")
                    cell.source = cleaned_code
                    processed_templates_count += 1

            # Все остальные ячейки (markdown-описания, базовые импорты, настройки отображения и автотесты)
            # автоматически сохраняются без каких-либо изменений
            new_cells.append(cell)

        # Перезаписываем список ячеек отфильтрованным результатом
        nb.cells = new_cells

        # Сохраняем готовую студенческую версию в целевую директорию
        target_file_path = os.path.join(target_dir, filename)
        with open(target_file_path, "w", encoding="utf-8") as f:
            if hasattr(nbformat, "normalize"):
                nbformat.normalize(nb)
            nbformat.write(nb, f)

        print(f" -> Successfully cleaned: Removed {removed_masters_count} master solutions.")
        print(f" -> Successfully prepared: Processed {processed_templates_count} student templates.")
        print(f" -> Saved production notebook to: {target_file_path}")


# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    for task in TASKS:
        process_directory(task["source"], task["target"])


if __name__ == "__main__":
    main()
