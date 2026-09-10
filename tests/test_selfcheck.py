"""评测自检的自检：文本类破坏场景秒级可跑，每个都必须贴出它声明的标签；场景表覆盖全部归因标签。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import selfcheck  # noqa: E402
from attribution import LABELS  # noqa: E402

TEXT_ONLY = [sc for sc in selfcheck.SCENARIOS if sc.get("text_only") and sc["reachable"]]


@pytest.fixture(scope="module")
def ctx():
    return selfcheck.Ctx(fast=True)


@pytest.mark.parametrize("sc", TEXT_ONLY, ids=[sc["name"] for sc in TEXT_ONLY])
def test_text_only_scenario_produces_its_label(sc, ctx):
    res = selfcheck.run_scenario(sc, ctx)
    assert res["passed"], f"{sc['name']}: {res.get('evidence')} {res.get('error', '')}"
    if sc["label"] in LABELS:
        assert res["observed"].get(sc["label"], 0) > 0, res["observed"]


def test_registry_covers_every_attribution_label():
    covered = {sc["label"] for sc in selfcheck.SCENARIOS}
    assert set(LABELS) <= covered, set(LABELS) - covered


def test_registry_entries_are_well_formed():
    for sc in selfcheck.SCENARIOS:
        for k in ("name", "label", "reachable", "run", "setup", "subset", "expect", "note"):
            assert k in sc, (sc.get("name"), k)
        assert callable(sc["run"])


def test_report_has_five_sections(tmp_path):
    rep = selfcheck.run_selfcheck(fast=True, only=["skip", "db_missing"], out_dir=tmp_path, quiet=True)
    assert set(selfcheck.SECTIONS) <= set(rep["sections"])
    assert (tmp_path / "latest.md").exists() and (tmp_path / "latest.json").exists()
    assert not rep["failed"], rep["failed"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
