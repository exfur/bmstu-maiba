import json
import os


def convert_txt_to_ipynb(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Process all text files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    if not files:
        print(f"No .txt files found in '{input_dir}'.")
        return

    for filename in files:
        input_path = os.path.join(input_dir, filename)

        # Determine output filename (.txt -> .ipynb)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.ipynb"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        cells = []
        current_type = None
        current_source = []

        def append_current_cell():
            if not current_type or not current_source:
                return

            # Auto truncate leading empty rows
            start = 0
            while start < len(current_source) and not current_source[start].strip():
                start += 1

            # Auto truncate trailing empty rows
            end = len(current_source)
            while end > start and not current_source[end - 1].strip():
                end -= 1

            cleaned_source = current_source[start:end]

            # Skip appending if the cell is completely empty after truncation
            if not cleaned_source:
                return

            if current_type == "markdown":
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": cleaned_source,
                    }
                )
            elif current_type == "code":
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": cleaned_source,
                    }
                )

        for line in lines:
            stripped = line.strip()

            if stripped == "-- markdown":
                append_current_cell()
                current_type = "markdown"
                current_source = []
            elif stripped == "-- python":
                append_current_cell()
                current_type = "code"
                current_source = []
            else:
                # Only accumulate if a cell type has been initialized
                if current_type is not None:
                    current_source.append(line)

        # Append the final cell remaining in the buffer
        append_current_cell()

        # Build valid Jupyter Notebook JSON structure
        notebook_data = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 2,
        }

        # Write the JSON payload to the target .ipynb file
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(notebook_data, out_f, indent=1, ensure_ascii=False)

    print("\nConversion layout complete structure initialized.")


if __name__ == "__main__":
    # Configure production relative paths
    INPUT_DIRECTORY = os.path.join("data", "land_data")
    OUTPUT_DIRECTORY = os.path.join("examples", "seminars")

    convert_txt_to_ipynb(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
