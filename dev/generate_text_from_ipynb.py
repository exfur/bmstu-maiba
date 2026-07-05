import json
import os


def convert_ipynb_to_txt(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Process all Jupyter Notebook files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(".ipynb")]

    if not files:
        print(f"No .ipynb files found in '{input_dir}'.")
        return

    for filename in files:
        input_path = os.path.join(input_dir, filename)

        # Determine output filename (.ipynb -> .txt)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            try:
                notebook_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Failed to parse valid JSON from {filename}")
                continue

        cells = notebook_data.get("cells", [])
        txt_lines = []

        for cell in cells:
            cell_type = cell.get("cell_type")
            source = cell.get("source", [])

            # Jupyter cell source can be either a string or a list of strings
            if isinstance(source, str):
                source_lines = source.splitlines(keepends=True)
            else:
                source_lines = source

            # Append the appropriate type marker
            if cell_type == "markdown":
                txt_lines.append("-- markdown\n")
            elif cell_type == "code":
                txt_lines.append("-- python\n")
            else:
                # Skip unknown cell types (like 'raw')
                continue

            # Add the block's content lines
            txt_lines.extend(source_lines)

            # Ensure the last line of the cell content ends with a newline character
            if txt_lines and not txt_lines[-1].endswith("\n"):
                txt_lines[-1] += "\n"

            # Add an extra newline between blocks for clean separation
            txt_lines.append("\n")

        # Strip out the trailing spacer newline at the end of the file
        if txt_lines and txt_lines[-1] == "\n":
            txt_lines.pop()

        # Write the reconstructed plain-text content
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.writelines(txt_lines)

    print(f"Reverse structure reconstruction complete for: {input_dir} -> {output_dir}")


if __name__ == "__main__":
    # 1. Seminars Reverse Processing
    SEMINARS_INPUT = os.path.join("examples", "seminars")
    SEMINARS_OUTPUT = os.path.join("data/land_data", "seminars")

    print("--- Reverse Processing Seminars Batch ---")
    convert_ipynb_to_txt(SEMINARS_INPUT, SEMINARS_OUTPUT)

    # 2. Course Project Notebooks Reverse Processing
    # Using a dynamic path to handle different environments
    NOTEBOOKS_INPUT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "examples", "course_project", "notebooks")
    )
    NOTEBOOKS_OUTPUT = os.path.join("data/land_data", "notebooks")

    print("\n--- Reverse Processing Course Project Notebooks Batch ---")
    convert_ipynb_to_txt(NOTEBOOKS_INPUT, NOTEBOOKS_OUTPUT)
