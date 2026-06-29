import glob
import json
import os

import nbformat
import requests

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = "gemma3:4b"

SOURCE_DIR = os.path.join(".", "examples", "seminars")
TARGET_DIR = r"C:\Users\User\Projects\bmstu\maiba\seminars"


# ==========================================
# OLLAMA API CALL (Templates Only)
# ==========================================
def generate_student_template(solution_code):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    prompt = f"""You are an educational Python developer. I will provide you with a raw Python code block from a completed solution.
Your ONLY job is to convert this code into a template for students. 
Replace core metrics, parameters, functions, or algorithmic lines with an explicit line: raise NotImplementedError("Ваш код здесь"). 
Keep the baseline scaffolding (variable setups, standard print structures) visible so they know contextually where to add their code.

Code to process:
{solution_code}

Return EXACTLY a valid JSON object matching this schema. Do not output markdown codeblocks:
{{
    "student_template_code": "The code string with raise NotImplementedError(\\\"Ваш код здесь\\\") replacements"
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
# MAIN TEMPLATE ROUTINE
# ==========================================
def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    search_path = os.path.join(SOURCE_DIR, "*.ipynb")
    notebook_files = glob.glob(search_path)

    if not notebook_files:
        print(f"No notebooks found in source directory: {SOURCE_DIR}")
        return

    print(f"Found {len(notebook_files)} notebooks for TEMPLATE generation.")

    for file_path in notebook_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing File: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        file_has_tasks = False
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

                print(f"    -> Templating code block #{task_counter}...")

                ai_data = generate_student_template(solution_code)

                student_code = ai_data.get(
                    "student_template_code",
                    '# TODO: Восстановите код решения\nraise NotImplementedError("Ваш код здесь")',
                )
                cell.source = student_code

                task_counter += 1
                file_has_tasks = True

        if file_has_tasks:
            target_file_path = os.path.join(TARGET_DIR, filename)
            with open(target_file_path, "w", encoding="utf-8") as f:
                if hasattr(nbformat, "normalize"):
                    nbformat.normalize(nb)
                nbformat.write(nb, f)
            print(f" -> Successfully saved templates to: {target_file_path}")


if __name__ == "__main__":
    main()
