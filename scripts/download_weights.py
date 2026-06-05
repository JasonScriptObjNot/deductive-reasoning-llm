"""
Download the trained LoRA adapter weights from the GitHub release.

Usage:
    python scripts/download_weights.py

Downloads reasoner_adapter.zip from the v1.0 release and extracts it
into outputs/reasoner_adapter/, which is where all demo and eval scripts
expect to find the adapter.
"""

import os
import sys
import urllib.request
import zipfile

RELEASE_URL = (
    "https://github.com/JasonScriptObjNot/deductive-reasoning-llm"
    "/releases/download/v1.0/reasoner_adapter.zip"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reasoner_adapter")
ZIP_PATH = os.path.join(os.path.dirname(__file__), "..", "reasoner_adapter.zip")


def _progress(count, block_size, total):
    pct = min(count * block_size / total * 100, 100)
    mb_done = count * block_size / 1_048_576
    mb_total = total / 1_048_576
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)


def main() -> None:
    adapter_path = os.path.join(OUT_DIR, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        print("Adapter weights already present at outputs/reasoner_adapter/ — nothing to do.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading adapter weights (~145 MB) …")
    urllib.request.urlretrieve(RELEASE_URL, ZIP_PATH, reporthook=_progress)
    print()

    print("Extracting …")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(OUT_DIR)

    os.remove(ZIP_PATH)
    print(f"Done. Adapter extracted to outputs/reasoner_adapter/")
    print("You can now run:  python scripts/demo.py --benchmark B6")


if __name__ == "__main__":
    main()
