"""Build otoium as a standalone executable with PyInstaller."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    for name in ("build", "dist"):
        path = ROOT / name
        if path.exists():
            shutil.rmtree(path)

    command = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--name", "otoium",
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--collect-all", "numpy",
        "--collect-all", "scipy",
        "--collect-all", "soundfile",
        str(ROOT / "run_otoium.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    output = ROOT / "dist" / f"otoium{suffix}"
    if not output.is_file():
        raise FileNotFoundError(f"Build output was not created: {output}")
    print(f"Build complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
