"""The DataSpace adapter for this repository's Tree Studio store.

Skipped entirely when ``lazydataspace`` is not installed: it is an optional
extra, and the repo must keep working standalone without it.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

lazydataspace = pytest.importorskip("lazydataspace", reason="optional [lazydataspace] extra")

from lazydataspace import DataSpace, Health, Source, SourceInfo  # noqa: E402

from lazyportfolio.dataspace_source import PortfolioSource  # noqa: E402
from lazyportfolio.v2.db import connect  # noqa: E402


@pytest.fixture
def real_store(tmp_path):
    """A store created by the repo's own schema, not a hand-rolled one."""
    path = tmp_path / "tree_studio.sqlite3"
    con = connect(path)
    con.close()
    return str(path)


class TestProtocolConformance:
    def test_satisfies_the_source_protocol(self, real_store):
        assert isinstance(PortfolioSource(real_store), Source)

    def test_identity(self, real_store):
        source = PortfolioSource(real_store)
        assert source.name == "portfolio"
        assert source.owner == "lazyportfolio"

    def test_registrable_in_a_dataspace(self, real_store):
        space = DataSpace(PortfolioSource(real_store))
        assert space.list() == ["portfolio"]


class TestDescribe:
    def test_returns_source_info(self, real_store):
        info = PortfolioSource(real_store).describe()
        assert isinstance(info, SourceInfo)
        assert "portfolio.trees" in info.capabilities

    def test_description_does_not_leak_the_path(self, real_store, tmp_path):
        info = PortfolioSource(real_store).describe()
        assert real_store not in info.description
        assert str(tmp_path) not in info.description
        assert not re.search(r"[A-Za-z]:[\\/]", info.description), "no absolute path"


class TestHealth:
    def test_ready_against_a_real_store(self, real_store):
        health = PortfolioSource(real_store).health()
        assert isinstance(health, Health)
        assert health.ready is True

    def test_unready_when_the_file_is_absent(self, tmp_path):
        health = PortfolioSource(str(tmp_path / "missing.sqlite3")).health()
        assert health.ready is False
        assert "does not exist" in health.detail

    def test_absent_store_is_not_created_by_the_check(self, tmp_path):
        """connect() would CREATE the file — the reason health() opens with
        mode=ro instead of going through it."""
        missing = tmp_path / "missing.sqlite3"
        PortfolioSource(str(missing)).health()
        assert not missing.exists()

    def test_unready_when_the_file_is_not_a_database(self, tmp_path):
        junk = tmp_path / "not-a-db.sqlite3"
        junk.write_text("this is not sqlite", encoding="utf-8")
        health = PortfolioSource(str(junk)).health()
        assert health.ready is False
        assert "cannot open" in health.detail

    def test_unready_when_pointed_at_the_wrong_database(self, tmp_path):
        other = tmp_path / "other.sqlite3"
        con = sqlite3.connect(str(other))
        con.execute("CREATE TABLE something_else (x INTEGER)")
        con.commit()
        con.close()
        health = PortfolioSource(str(other)).health()
        assert health.ready is False
        assert "trees" in health.detail

    def test_failure_detail_never_contains_the_path(self, tmp_path):
        junk = tmp_path / "secret-location.sqlite3"
        junk.write_text("junk", encoding="utf-8")
        detail = PortfolioSource(str(junk)).health().detail
        assert str(junk) not in detail
        assert "secret-location" not in detail

    def test_unset_env_reports_the_default_store_as_missing(self, tmp_path, monkeypatch):
        """This repo's resolver always yields a path, so an unset env var
        shows up as 'the default store does not exist yet' rather than as a
        'nothing configured' state."""
        monkeypatch.setenv("LAZYPORTFOLIO_TREE_DB", str(tmp_path / "nonexistent.sqlite3"))
        health = PortfolioSource().health()
        assert health.ready is False
        assert "does not exist" in health.detail


class TestReadinessGate:
    def test_gate_passes_with_a_real_store(self, real_store):
        DataSpace(PortfolioSource(real_store)).require_ready()

    def test_gate_fails_before_a_workflow_writes(self, tmp_path):
        space = DataSpace(PortfolioSource(str(tmp_path / "missing.sqlite3")))
        with pytest.raises(lazydataspace.SourceNotReadyError) as exc:
            space.require_ready()
        assert "portfolio" in str(exc.value)

    def test_registering_a_source_opens_nothing(self, tmp_path):
        """A Source is a description, not a connection."""
        missing = tmp_path / "untouched.sqlite3"
        DataSpace(PortfolioSource(str(missing)))
        assert not missing.exists()


class TestStandaloneIndependence:
    def test_the_package_does_not_import_the_adapter(self):
        """Importing lazyportfolio must not require lazydataspace."""
        import ast
        import pathlib

        import lazyportfolio

        package_dir = pathlib.Path(lazyportfolio.__file__).parent
        importers = []
        for module in package_dir.rglob("*.py"):
            if module.name == "dataspace_source.py":
                continue
            tree = ast.parse(module.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    if root == "lazydataspace" or node.module.endswith("dataspace_source"):
                        importers.append(module.name)
                elif isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] == "lazydataspace" for a in node.names):
                        importers.append(module.name)
        assert not importers, f"these modules would make lazydataspace mandatory: {importers}"
