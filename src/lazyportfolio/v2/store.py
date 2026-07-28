"""Shared, file-based persistence for named V2 tree configurations.

Both Tree Studio (the local visual editor, ``project/tree_studio.py``) and any
external caller (LazyTools' MCP ``portfolio_tree_*`` tools) read and write
through this module, never through their own copy of the logic -- so a tree
saved by one is immediately visible to the other: same directory, same
filename sanitization, same validate-before-write gate. Stdlib-only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from lazyportfolio.v2.model import V2Model

#: Same character policy Tree Studio has always used for a model's on-disk
#: name: collapse anything else to a hyphen, then trim stray separators.
_MODEL_NAME = re.compile(r"[^A-Za-z0-9._ -]+")

#: Environment variable both processes read to agree on one shared directory.
_ENV_VAR = "LAZYPORTFOLIO_TREE_MODELS_DIR"


class ModelStoreError(ValueError):
    """A model name or configuration cannot be persisted or found."""


def _as_json(value: Any) -> Any:
    """Best-effort JSON coercion, same fallback Tree Studio's own responses use."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return _as_json(asdict(value))
    if hasattr(value, "to_dict"):
        return {str(k): _as_json(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


def resolve_models_dir(store_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the one shared directory saved tree configurations live in.

    Precedence: an explicit ``store_dir`` argument, then the ``LAZYPORTFOLIO_TREE_MODELS_DIR``
    env var (the interop mechanism between Tree Studio and any other caller),
    then the historical Tree Studio default -- ``<repo>/reports/tree_studio/models``,
    computed from this installed module rather than a script's ``__file__`` so
    it resolves correctly however the package is imported.
    """
    if store_dir:
        return Path(store_dir).resolve()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).resolve()
    # .../src/lazyportfolio/v2/store.py -> v2 -> lazyportfolio -> src -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "reports" / "tree_studio" / "models"


def sanitize_model_name(name: Any) -> str:
    """Reduce a model name to a safe, stable on-disk stem (no extension)."""
    cleaned = _MODEL_NAME.sub("-", str(name).strip()).strip(" .-")
    if not cleaned:
        raise ModelStoreError("model name cannot be blank")
    return cleaned[:120]


def model_path(name: Any, *, store_dir: str | os.PathLike[str] | None = None) -> Path:
    """The on-disk path a given model name resolves to (whether or not it exists)."""
    return resolve_models_dir(store_dir) / f"{sanitize_model_name(name)}.json"


def list_saved_models(*, store_dir: str | os.PathLike[str] | None = None) -> list[dict[str, str]]:
    """List saved models as ``{"name", "file"}`` pairs, newest first."""
    directory = resolve_models_dir(store_dir)
    if not directory.exists():
        return []
    return [
        {"name": path.stem, "file": path.name}
        for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def read_model(name: Any, *, store_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read a saved model's raw configuration by name (no re-validation)."""
    path = model_path(name, store_dir=store_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no saved model named {sanitize_model_name(name)!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_model(
    name: Any,
    config: dict[str, Any],
    *,
    store_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Validate ``config`` and persist it; never writes on a validation failure.

    Validation is the same gate Tree Studio's own save endpoint has always
    used: constructing ``V2Model.from_config(config)`` and discarding the
    result (this call is for the side-effecting validation, not the model).
    """
    if not isinstance(config, dict):
        raise ModelStoreError("model config must be an object")
    V2Model.from_config(config)
    path = model_path(name, store_dir=store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=_as_json) + "\n", encoding="utf-8")
    return path


def delete_model(name: Any, *, store_dir: str | os.PathLike[str] | None = None) -> Path:
    """Delete a saved model by name, returning the path that was removed."""
    path = model_path(name, store_dir=store_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no saved model named {sanitize_model_name(name)!r}")
    path.unlink()
    return path


__all__ = [
    "ModelStoreError",
    "delete_model",
    "list_saved_models",
    "model_path",
    "read_model",
    "resolve_models_dir",
    "sanitize_model_name",
    "write_model",
]
