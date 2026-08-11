import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = (ROOT / ".." / "market-data-hub" / "market_data.duckdb").resolve()
os.environ.setdefault("MARKET_DATA_DB", str(DB_PATH))
sys.argv = [sys.argv[0]] + (sys.argv[1:] or ["8766"])

from project import tree_studio

tree_studio.main()
