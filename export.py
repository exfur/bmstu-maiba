import os
import shutil
import tempfile
import zipfile
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
DEV_PROCESSED_DIR = PROJECT_ROOT / "dev" / "data" / "processed"

# Files/Directories to ignore globally when packaging
IGNORE_PATTERNS = {
    "__pycache__",
    ".pyc",
    ".venv",
    "venv",
    ".git",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
}


def should_ignore(path: Path) -> bool:
    """Check if a file or directory should be excluded globally."""
    return any(part in IGNORE_PATTERNS or part.endswith(".pyc") for part in path.parts)


def should_ignore_master(path: Path) -> bool:
    """Exclusion rules specifically for master.zip (*.txt and auto-generated 'data' folders)."""
    if should_ignore(path):
        return True
    # Exclude all .txt files
    if path.suffix == ".txt":
        return True
    # Exclude any folder named 'data'
    if "data" in path.parts:
        return True
    return False


def zip_folder(source_dir: Path, output_zip: Path, archive_prefix: str = "", ignore_fn=should_ignore):
    """Recursively zip a directory using a custom filter function."""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)

            # Filter out ignored directories in-place to avoid traversing into them
            dirs[:] = [d for d in dirs if not ignore_fn(root_path / d)]

            for file in files:
                file_path = root_path / file
                if ignore_fn(file_path):
                    continue

                rel_path = file_path.relative_to(source_dir)
                arc_name = Path(archive_prefix) / rel_path if archive_prefix else rel_path
                zipf.write(file_path, arc_name)


def create_student_workspace_zip(temp_dir: Path) -> Path:
    """Pack student workspace into student_workspace.zip."""
    print("📦 Creating student_workspace.zip...")
    student_zip_path = temp_dir / "student_workspace.zip"

    items_to_pack = [
        ("pyproject.toml", PROJECT_ROOT / "pyproject.toml", False),
        ("README.md", PROJECT_ROOT / "README.md", False),
        (".vscode", PROJECT_ROOT / ".vscode", True),
        ("course_project", PROJECT_ROOT / "course_project", True),
    ]

    # Handle 'seminars' directory (root 'seminars' or fallback to 'dev/examples/seminars')
    seminars_dir = PROJECT_ROOT / "seminars"
    if not seminars_dir.exists():
        seminars_dir = PROJECT_ROOT / "dev" / "examples" / "seminars"

    if seminars_dir.exists():
        items_to_pack.append(("seminars", seminars_dir, True))

    with zipfile.ZipFile(student_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for arc_name, source_path, is_dir in items_to_pack:
            if not source_path.exists():
                print(f"  ⚠️ Warning: {source_path.name} missing, skipping...")
                continue

            if is_dir:
                for root, dirs, files in os.walk(source_path):
                    dirs[:] = [d for d in dirs if d not in IGNORE_PATTERNS]
                    for file in files:
                        file_path = Path(root) / file
                        if should_ignore(file_path):
                            continue
                        rel_path = file_path.relative_to(source_path)
                        zipf.write(file_path, Path(arc_name) / rel_path)
            else:
                zipf.write(source_path, arc_name)

    return student_zip_path


def create_master_zip(temp_dir: Path) -> Path:
    """Pack master solutions and completed notebooks into master.zip (excluding *.txt and data/)."""
    print("📦 Creating master.zip (excluding *.txt and data/)...")
    master_zip_path = temp_dir / "master.zip"
    dev_examples = PROJECT_ROOT / "dev" / "examples"

    source_dir = dev_examples if dev_examples.exists() else PROJECT_ROOT / "dev"
    zip_folder(source_dir, master_zip_path, ignore_fn=should_ignore_master)

    return master_zip_path


def prepare_variant_zips(temp_dir: Path) -> list[Path]:
    """Find or pack ready variants from dev/data/processed into individual ZIP files."""
    print("📦 Processing variant data files from dev/data/processed...")
    variant_zips = []
    variants_out_dir = temp_dir / "variants"
    variants_out_dir.mkdir(parents=True, exist_ok=True)

    if not DEV_PROCESSED_DIR.exists():
        print(f"  ⚠️ Directory not found: {DEV_PROCESSED_DIR}")
        return variant_zips

    for item in DEV_PROCESSED_DIR.iterdir():
        if should_ignore(item):
            continue

        if item.is_dir():
            out_zip = variants_out_dir / f"{item.name}.zip"
            zip_folder(item, out_zip)
            variant_zips.append(out_zip)
            print(f"  + Packed variant directory: {item.name}.zip")
        elif item.suffix == ".zip":
            out_zip = variants_out_dir / item.name
            shutil.copy2(item, out_zip)
            variant_zips.append(out_zip)
            print(f"  + Included existing variant archive: {item.name}")
        elif item.is_file():
            var_name = item.stem
            out_zip = variants_out_dir / f"variant_{var_name}.zip"
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(item, item.name)
            variant_zips.append(out_zip)
            print(f"  + Packed variant file: {out_zip.name}")

    return variant_zips


def build_final_distribution():
    """Main orchestration to assemble all components into a single distribution ZIP."""
    print("🚀 Starting Distribution Export Pipeline...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    final_dist_zip = DIST_DIR / "distribution.zip"

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        temp_dir = Path(tmp_dir_str)

        # 1. Generate master zip
        master_zip = create_master_zip(temp_dir)

        # 2. Generate student workspace zip
        student_zip = create_student_workspace_zip(temp_dir)

        # 3. Generate or collect variant zips
        variant_zips = prepare_variant_zips(temp_dir)

        # 4. Bundle everything into the single final distribution ZIP
        print(f"\n📦 Bundling into single distribution ZIP: {final_dist_zip.resolve()}")
        with zipfile.ZipFile(final_dist_zip, "w", zipfile.ZIP_DEFLATED) as dist_zip:
            dist_zip.write(master_zip, "master.zip")
            dist_zip.write(student_zip, "student_workspace.zip")

            for v_zip in variant_zips:
                dist_zip.write(v_zip, Path("variants") / v_zip.name)

    print("\n" + "=" * 60)
    print("✅ DISTRIBUTION PACKAGE SUCCESSFULLY CREATED!")
    print(f"📍 Location: {final_dist_zip.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    build_final_distribution()
