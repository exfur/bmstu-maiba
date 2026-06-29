import glob
import os

import nbformat

# ==========================================
# CONFIGURATION
# ==========================================
# Папка, где лежат исходные мастер-ноутбуки (с решениями и шаблонами)
SOURCE_DIR = os.path.join(".", "examples", "seminars")
# Папка, куда сохраняются чистые студенческие версии
TARGET_DIR = r"C:\Users\User\Projects\bmstu\maiba\seminars"


# ==========================================
# MAIN TEMPLATE ROUTINE
# ==========================================
def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    search_path = os.path.join(SOURCE_DIR, "*.ipynb")
    notebook_files = glob.glob(search_path)

    if not notebook_files:
        print(f"No notebooks found in source directory: {SOURCE_DIR}")
        return

    print(f"Found {len(notebook_files)} notebooks for student version generation.")

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
                    cleaned_code = source_code.replace(
                        "# [STUDENT TEMPLATE]\n", ""
                    ).replace("# [STUDENT TEMPLATE]", "")
                    cell.source = cleaned_code
                    processed_templates_count += 1

            # Все остальные ячейки (markdown-описания, базовые импорты, настройки отображения и автотесты)
            # автоматически сохраняются без каких-либо изменений
            new_cells.append(cell)

        # Перезаписываем список ячеек отфильтрованным результатом
        nb.cells = new_cells

        # Сохраняем готовую студенческую версию в целевую директорию
        target_file_path = os.path.join(TARGET_DIR, filename)
        with open(target_file_path, "w", encoding="utf-8") as f:
            if hasattr(nbformat, "normalize"):
                nbformat.normalize(nb)
            nbformat.write(nb, f)

        print(
            f" -> Successfully cleaned: Removed {removed_masters_count} master solutions."
        )
        print(
            f" -> Successfully prepared: Processed {processed_templates_count} student templates."
        )
        print(f" -> Saved production notebook to: {target_file_path}")


if __name__ == "__main__":
    main()
