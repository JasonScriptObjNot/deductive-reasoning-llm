"""python scripts/run_preprocess.py [--force]"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dras.config import Config
from dras.preprocess import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Rebuild even if files exist")
args = parser.parse_args()

build_dataset(Config(), force=args.force)
