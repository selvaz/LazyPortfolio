"""Guided post-install setup, callable as ``lazyportfolio-setup``.

Ships inside the package (unlike a root-level script) so it is available
after *any* ``pip install`` of lazyportfolio - editable, a local checkout, or
straight from GitHub - with no separate clone or download step. Always
installs and configures market-data-hub (the one supported way every
function here reads price/return data - packaged as the ``[datacore]``
extra only because :mod:`lazyportfolio.backend` imports it lazily, never
because it's meant to be skipped), asks only about the genuinely optional
pieces, and locates or asks for the Market Data Hub ``.duckdb`` database.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGE_NAME = "lazyportfolio"
MARKET_DATA_HUB_GIT_URL = "https://github.com/selvaz/market-data-hub.git"


def _find_repo_root() -> Path | None:
    """If running from a repo checkout (editable install or plain clone), return its root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / PACKAGE_NAME).is_dir():
            return parent
    return None


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def _extra_requirements(extra: str, seen: set[str] | None = None) -> list[str]:
    """Flatten one of this package's own extras into concrete pip specs.

    Deliberately never returns a spec naming this package itself (e.g. a
    nested ``lazyportfolio[test]`` inside the ``dev`` extra is expanded
    recursively instead) - reinstalling the package this script ships in,
    from a process running as that package's own installed console-script
    .exe, fails on Windows (the file is locked while executing). Anything
    this script needs from its own extras must resolve to *other* packages.
    """
    seen = seen if seen is not None else set()
    if extra in seen:
        return []
    seen.add(extra)
    reqs: list[str] = []
    for raw in metadata.metadata(PACKAGE_NAME).get_all("Requires-Dist") or []:
        spec, _, marker = raw.partition(";")
        if f"extra == '{extra}'" not in marker and f'extra == "{extra}"' not in marker:
            continue
        spec = spec.strip()
        name = spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        if name.lower() == PACKAGE_NAME.lower():
            nested = spec[spec.index("[") + 1 : spec.index("]")] if "[" in spec else ""
            for nested_extra in nested.split(","):
                if nested_extra.strip():
                    reqs.extend(_extra_requirements(nested_extra.strip(), seen))
            continue
        reqs.append(spec)
    return reqs


def _install_extras(extras: list[str]) -> None:
    specs: list[str] = []
    for extra in extras:
        specs.extend(_extra_requirements(extra))
    seen_specs: set[str] = set()
    unique_specs: list[str] = []
    for spec in specs:
        if spec not in seen_specs:
            seen_specs.add(spec)
            unique_specs.append(spec)
    if unique_specs:
        _pip("install", *unique_specs)


def _ask_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default_yes
    return answer in {"y", "yes", "s", "si", "sì"}


def _set_persistent_env_var(name: str, value: str) -> None:
    os.environ[name] = value
    if os.name == "nt":
        subprocess.run(["setx", name, value], check=True, stdout=subprocess.DEVNULL)
    else:
        print(
            f"  (non-Windows: add 'export {name}=\"{value}\"' to your shell profile "
            "to persist this across sessions)"
        )


def _select_market_data_db(requested: str | None, sibling_hub: Path | None) -> str | None:
    if requested:
        return str(Path(requested).expanduser().resolve())

    existing = os.environ.get("MARKET_DATA_DB")
    if existing and Path(existing).is_file():
        return existing

    candidates: list[Path] = []
    if sibling_hub is not None:
        candidates = sorted(sibling_hub.glob("*.duckdb"))

    print("\nMarket Data Hub database")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  {index}) {candidate}")
    enter_choice = len(candidates) + 1
    skip_choice = len(candidates) + 2
    print(f"  {enter_choice}) Enter a .duckdb path")
    print(f"  {skip_choice}) Skip for now (set MARKET_DATA_DB later)")

    default = "1" if candidates else str(enter_choice)
    selection = input(f"Select [{default}]: ").strip() or default
    if not selection.isdigit():
        return None
    choice = int(selection)
    if 1 <= choice <= len(candidates):
        return str(candidates[choice - 1])
    if choice == enter_choice:
        entered = input("Full path to the Market Data Hub .duckdb file: ").strip()
        return str(Path(entered).expanduser().resolve()) if entered else None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lazyportfolio-setup", description=__doc__)
    parser.add_argument(
        "--market-data-hub-path", help="Path to a local market-data-hub checkout"
    )
    parser.add_argument(
        "--db-path", help="Skip the interactive prompt and use this .duckdb file"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Don't run the test suite even if dev tooling is installed",
    )
    args = parser.parse_args(argv)

    repo_root = _find_repo_root()
    mode = f"repo checkout ({repo_root})" if repo_root else "installed package"
    print(f"Python: {sys.executable}")
    print(f"Mode: {mode}")
    print(
        f"LazyPortfolio {metadata.version(PACKAGE_NAME)} already installed "
        "(that's how this command exists)."
    )

    # --- market-data-hub: ALWAYS installed, never asked. It's the one
    # supported way every function here reads real data - treat it as core.
    sibling_hub = None
    if args.market_data_hub_path:
        sibling_hub = Path(args.market_data_hub_path).expanduser().resolve()
    elif repo_root is not None:
        candidate = repo_root.parent / "market-data-hub"
        if candidate.is_dir():
            sibling_hub = candidate

    print("\n==> Installing market-data-hub")
    if sibling_hub is not None:
        _pip("install", "-e", str(sibling_hub))
    else:
        _pip("install", f"market-data-hub @ git+{MARKET_DATA_HUB_GIT_URL}")
        print(
            "\nNo local market-data-hub checkout found. That's fine for using the "
            "package, but you'll need a populated .duckdb database (see below) and "
            "won't get market-data-hub's own ingestion/scheduling scripts. Clone "
            f"{MARKET_DATA_HUB_GIT_URL} and run its own setup for the full data pipeline."
        )

    # --- Genuinely optional pieces.
    extras: list[str] = []
    have_dev = _ask_yes_no("\nInstall dev/test tooling (pytest, ruff, mypy)?", True)
    if have_dev:
        extras.append("dev")
    if extras:
        print(f"\n==> Installing extras: {', '.join(extras)}")
        _install_extras(extras)

    have_node = False
    if repo_root is not None and _which("npm"):
        if _ask_yes_no("Install Tree Studio's JS test harness (npm install)?", False):
            print("\n==> Running npm install")
            subprocess.run(["npm", "install"], cwd=repo_root, check=True)
            have_node = True

    # --- Locate / configure MARKET_DATA_DB.
    resolved_db = _select_market_data_db(args.db_path, sibling_hub)
    if resolved_db:
        _set_persistent_env_var("MARKET_DATA_DB", resolved_db)
        print(f"\nMARKET_DATA_DB set (persisted for your user account): {resolved_db}")
        if not Path(resolved_db).is_file():
            print("  (file does not exist yet - market-data-hub will create it on first run)")
    else:
        print(
            '\nNo database configured. Set it later, e.g.: '
            'setx MARKET_DATA_DB "<path-to.duckdb>"'
        )

    print("\n==> Verifying imports")
    verify_code = "import lazyportfolio, market_data_hub; print('LazyPortfolio environment OK')"
    subprocess.run([sys.executable, "-c", verify_code], check=True)

    if have_dev and not args.skip_tests and repo_root is not None:
        print("\n==> Running the test suite (includes SLSQP solves, can take ~20 min)")
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo_root, check=True)

    print("\n==> Setup complete.")
    if repo_root is not None:
        tree_studio = repo_root / "project" / "tree_studio.py"
        print(f"Run Tree Studio with:\n  {sys.executable} {tree_studio} 8766")
    if have_node:
        print("Run Tree Studio's JS contract tests with:\n  npm test")
    return 0


def _which(command: str) -> str | None:
    from shutil import which

    return which(command)


if __name__ == "__main__":
    raise SystemExit(main())
