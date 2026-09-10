"""参数文件与 rank 模块属性必须一致：参数是数据不是代码，改了 JSON 就改了行为，反之亦然。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialect_addr import rank as rank_mod  # noqa: E402

PARAMS = ROOT / "data" / "params" / "rank_params.json"


def test_json_params_equal_module_attributes():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))
    for k, v in data["params"].items():
        assert getattr(rank_mod, k) == pytest.approx(v), k
    assert set(data["params"]) == set(rank_mod.TUNABLE)


def test_metadata_fields_present():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))
    for k in ("version", "source", "created", "params", "replayable", "retrieval_only", "bounds", "history"):
        assert k in data, k
    assert data["source"] in ("hand-set", "tuned", "tuned-below-threshold")
    for k in data["replayable"] + data["retrieval_only"]:
        assert k in data["params"], k
    info = rank_mod.params_info()
    assert info["version"] == data["version"] and info["source"] == data["source"]


def test_weights_form_a_simplex():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))
    keys = data["constraints"]["simplex"]
    assert sum(data["params"][k] for k in keys) == pytest.approx(1.0, abs=1e-9)


def test_apply_params_round_trip_and_rejects_unknown_names():
    before = rank_mod.current_params()
    try:
        rank_mod.apply_params({"MARGIN_MIN": 0.11, "W_CONFLICT": 0.3})
        assert rank_mod.MARGIN_MIN == pytest.approx(0.11) and rank_mod.W_CONFLICT == pytest.approx(0.3)
        with pytest.raises(KeyError):
            rank_mod.apply_params({"NOT_A_PARAM": 1.0})
    finally:
        rank_mod.apply_params(before)
    assert rank_mod.current_params() == before


def test_calibrated_prob_is_none_without_table_and_steps_with_one():
    assert rank_mod.calibrated_prob(0.9) is None
    saved = rank_mod._PARAMS
    try:
        rank_mod._PARAMS = {**saved, "calibration": {"breakpoints": [[0.0, 0.1], [0.6, 0.5], [0.8, 0.9]]}}
        assert rank_mod.calibrated_prob(0.5) == pytest.approx(0.1)
        assert rank_mod.calibrated_prob(0.7) == pytest.approx(0.5)
        assert rank_mod.calibrated_prob(0.95) == pytest.approx(0.9)
        assert rank_mod.params_info()["calibrated"] is True
    finally:
        rank_mod._PARAMS = saved


def test_missing_params_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        rank_mod.load_params(tmp_path / "nope.json")


def test_max_dist_comes_from_params():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))
    from dialect_addr.romanize import resolve_space

    eff, n = rank_mod.effective_max_dist("解放碑街道", resolve_space(None))
    assert n == 5 and eff == pytest.approx(data["params"]["MAX_DIST"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
