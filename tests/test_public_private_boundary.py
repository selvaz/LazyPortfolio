"""The public side must not know the Investment Committee exists.

The node and tree engine is the universal part: it belongs to whoever runs it,
and the committee is only one caller. Producer identity, policy and rationale
arrive from that caller — they are never named here.

Today that is true. Nothing enforced it, which is the reason for this file: a
property that holds only because everyone remembered is one commit away from
not holding, and the failure is silent — the engine keeps working, for the one
caller whose name leaked in.

The sibling check lives in LazyStats (``test_etf_stats_plans.py``); this is the
same guarantee for the node advisor and the optimisation trees.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

#: Every root that ships as the public engine. ``project/`` holds the advisor
#: and the tree assembly, ``src/lazyportfolio/`` the library they build on.
PUBLIC_ROOTS = ("src/lazyportfolio", "project")

#: The private repository, in the spellings a real leak would take: an import,
#: a producer id, a database key, a config path.
PRIVATE_NAMES = ("investmentcommittee", "investment_committee")


def public_modules() -> list[Path]:
    found: list[Path] = []
    for root in PUBLIC_ROOTS:
        directory = REPOSITORY / root
        assert directory.is_dir(), f"public root missing: {root}"
        found.extend(sorted(directory.rglob("*.py")))
    assert found, "no public modules found; the roots above are wrong"
    return found


def parsed(module: Path) -> ast.Module:
    return ast.parse(module.read_text(encoding="utf-8", errors="ignore"))


def relative(module: Path) -> str:
    return module.relative_to(REPOSITORY).as_posix()


@pytest.mark.parametrize("module", public_modules(), ids=relative)
def test_no_public_module_imports_the_private_repository(module: Path) -> None:
    """An import is the hard dependency: it would make the engine unusable
    for anyone who does not have the committee's repository."""
    offenders = []
    for node in ast.walk(parsed(module)):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in PRIVATE_NAMES:
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PRIVATE_NAMES:
                    offenders.append(f"import {alias.name} (line {node.lineno})")
    assert not offenders, f"{relative(module)} imports the private repository: {offenders}"


@pytest.mark.parametrize("module", public_modules(), ids=relative)
def test_no_public_module_names_the_private_repository_in_a_value(module: Path) -> None:
    """A hardcoded identity is the soft dependency, and the likelier one.

    A default ``producer_id`` of ``"investmentcommittee:advisor"`` needs no
    import and breaks no test: the engine simply stops being neutral, and every
    proposal it writes carries one caller's name. Only string *values* are
    checked — a comment or docstring explaining this boundary is allowed to say
    the name, and should.
    """
    offenders = []
    for node in ast.walk(parsed(module)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(name in lowered for name in PRIVATE_NAMES):
                offenders.append(f"{node.value!r} (line {node.lineno})")
    assert not offenders, (
        f"{relative(module)} carries the private repository's name in a value: "
        f"{offenders} — identity belongs to the caller, not to the engine"
    )
