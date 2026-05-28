#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "package"
BUILD_SCRIPT = ROOT / "scripts" / "build_reference_spritesheet.py"


def load_pet_id() -> str:
    pet_json = json.loads((PACKAGE_DIR / "pet.json").read_text(encoding="utf-8"))
    pet_id = pet_json.get("id")
    if not pet_id:
        raise SystemExit("package/pet.json is missing the pet id.")
    return pet_id


def build_package() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True, cwd=str(ROOT))


def install_package(install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pet.json", "avatar.json", "spritesheet.webp"):
        source = PACKAGE_DIR / name
        if not source.exists():
            raise SystemExit(f"Missing build artifact: {source}")
        shutil.copy2(source, install_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and install the Shian Codex pet.")
    parser.add_argument("--build", action="store_true", help="Rebuild the pet package before installing.")
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=None,
        help="Override the target install directory. Defaults to ~/.codex/pets/<pet-id>.",
    )
    args = parser.parse_args()

    if args.build:
        build_package()

    pet_id = load_pet_id()
    install_dir = args.install_dir or (Path.home() / ".codex" / "pets" / pet_id)
    install_package(install_dir)
    print(f"Installed {pet_id} to {install_dir}")


if __name__ == "__main__":
    main()
