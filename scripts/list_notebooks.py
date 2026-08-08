#!/usr/bin/env python3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
    print(p.relative_to(ROOT))
