import csv
import os


def generate_table_index():
    # Targets relative to execution path: C:\Users\User\Projects\bmstu\maiba\dev
    processed_dir = os.path.join("data", "processed")
    output_file = os.path.join(processed_dir, "index.txt")

    if not os.path.exists(processed_dir):
        print(
            f"Error: Target path '{processed_dir}' does not exist. Check your working directory."
        )
        return

    output_lines = [
        "==================================================",
        "          DATASET TABLE STRUCTURE INDEX           ",
        "==================================================",
        "",
    ]

    # Get sorted list of all subdirectories (01_telco_customer_churn, etc.)
    datasets = sorted(
        [
            d
            for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ]
    )

    for dataset in datasets:
        output_lines.append(f"## Dataset: {dataset}")
        dataset_path = os.path.join(processed_dir, dataset)

        # Find all CSV files in the directory
        csv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".csv")])

        if not csv_files:
            output_lines.append("   └── (No CSV tables found)")
            output_lines.append("")
            continue

        for csv_file in csv_files:
            file_path = os.path.join(dataset_path, csv_file)
            output_lines.append(f"   └── Table: {csv_file}")

            try:
                with open(file_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Grab only the first row
                    columns_str = ", ".join(headers)
                    output_lines.append(f"       Columns: {columns_str}")
            except Exception as e:
                output_lines.append(
                    f"       Columns: [Error reading headers - {str(e)}]"
                )

        output_lines.append("")  # Spacer between datasets

    # Write out the final index file
    with open(output_file, mode="w", encoding="utf-8") as out_f:
        out_f.write("\n".join(output_lines))

    print(f"Success! Structure index generated at: {output_file}")


if __name__ == "__main__":
    generate_table_index()
