import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    database = (root / ".." / "market-data-hub" / "market_data.duckdb").resolve()
    os.environ.setdefault("MARKET_DATA_DB", str(database))
    sys.argv = [sys.argv[0], *(sys.argv[1:] or ["8766"])]

    from project import tree_studio

    return tree_studio.main()


if __name__ == "__main__":
    raise SystemExit(main())
