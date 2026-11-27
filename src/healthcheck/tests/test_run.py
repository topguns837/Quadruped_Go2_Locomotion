# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the health-check runner (src/healthcheck/run.py) and its generic assertions."""

import os
import sys
from pathlib import Path

import pytest
import yaml

HC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HC))
import asserts  # noqa: E402
import run  # noqa: E402


def case(**kw):
    kw.setdefault("id", "c")
    kw.setdefault("group", "g")
    kw.setdefault("cmd", "true")
    return run.Case(**kw)


@pytest.fixture
def env(tmp_path):
    return dict(os.environ), tmp_path


def run_one(c, tmp_path):
    return run.run_case(c, tmp_path, dict(os.environ), "python3")


class TestExpand:
    def test_matrix_cartesian_product_and_substitution(self):
        raw = [{"id": "x-{a}-{b}", "cmd": "echo {a} {b} {run} ${HOME}", "matrix": {"a": [1, 2], "b": ["u", "v"]}}]
        cases = run.expand_cases(raw)
        assert [c.id for c in cases] == ["x-1-u", "x-1-v", "x-2-u", "x-2-v"]
        assert cases[0].cmd == "echo 1 u {run} ${HOME}"  # only matrix keys are substituted

    def test_expect_block_and_defaults(self):
        c = run.expand_cases([{"id": "a", "cmd": "x", "expect": {"rc": "nonzero", "contains": ["z"]}}])[0]
        assert c.rc == "nonzero" and c.contains == ["z"] and c.timeout == 600 and not c.gpu

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            run.expand_cases([{"id": "a", "cmd": "x"}, {"id": "a", "cmd": "y"}])

    def test_shipped_cases_file_is_valid(self):
        raw = yaml.safe_load((HC / "cases.yaml").read_text())
        cases = run.expand_cases(raw["cases"])
        ids = {c.id for c in cases}
        assert len(cases) > 50
        for c in cases:
            assert set(c.needs) <= ids, f"{c.id} needs an unknown case"
            for chk in c.checks:
                name = chk if isinstance(chk, str) else chk["fn"]
                assert callable(getattr(asserts, name)), f"{c.id}: unknown check {name}"


class TestSelect:
    CASES = [case(id="a-1", group="x"), case(id="a-2", group="x", needs=["a-1"]), case(id="b", group="y", needs=["a-2"])]

    def test_no_filter_returns_all(self):
        assert run.select(self.CASES, None) == self.CASES

    def test_group_filter(self):
        assert [c.id for c in run.select(self.CASES, "x")] == ["a-1", "a-2"]

    def test_prefix_filter_pulls_dependencies(self):
        assert [c.id for c in run.select(self.CASES, "b")] == ["a-1", "a-2", "b"]

    def test_no_match_is_empty(self):
        assert run.select(self.CASES, "zzz") == []


class TestRunCase:
    def test_pass(self, tmp_path):
        r = run_one(case(cmd="echo hello"), tmp_path)
        assert r.status == "PASS" and not r.reasons

    def test_unexpected_exit_code_fails(self, tmp_path):
        r = run_one(case(cmd="exit 3"), tmp_path)
        assert r.status == "FAIL" and "exit code 3" in r.reasons[0]

    def test_expected_nonzero(self, tmp_path):
        assert run_one(case(cmd="exit 2", rc="nonzero"), tmp_path).status == "PASS"
        assert run_one(case(cmd="exit 0", rc="nonzero"), tmp_path).status == "FAIL"

    def test_specific_exit_code(self, tmp_path):
        assert run_one(case(cmd="exit 7", rc=7), tmp_path).status == "PASS"

    def test_contains_and_not_contains(self, tmp_path):
        assert run_one(case(cmd="echo abc", contains=["a.c"]), tmp_path).status == "PASS"
        assert run_one(case(cmd="echo abc", contains=["zzz"]), tmp_path).status == "FAIL"
        assert run_one(case(cmd="echo abc", not_contains=["abc"]), tmp_path).status == "FAIL"

    def test_traceback_fails_a_zero_exit_unless_allowed(self, tmp_path):
        cmd = "echo 'Traceback (most recent call last):'"
        assert run_one(case(cmd=cmd), tmp_path).status == "FAIL"
        assert run_one(case(cmd=cmd, allow_traceback=True), tmp_path).status == "PASS"

    def test_timeout_kills_the_process_group(self, tmp_path):
        r = run_one(case(cmd="sleep 30 & sleep 30", timeout=1), tmp_path)
        assert r.status == "FAIL" and "timed out" in r.reasons[0] and r.seconds < 25

    def test_known_issue_is_reported_not_failed(self, tmp_path):
        r = run_one(case(cmd="exit 1", known="documented"), tmp_path)
        assert r.status == "KNOWN"

    def test_artifacts_plain_and_glob(self, tmp_path):
        (tmp_path / "out.txt").write_text("x")
        c = case(cmd="true", artifacts=["{run}/out.txt", "{run}/o*.txt"])
        assert run_one(c, tmp_path).status == "PASS"
        assert run_one(case(cmd="true", artifacts=["{run}/missing.txt"]), tmp_path).status == "FAIL"

    def test_run_dir_and_py_substitution(self, tmp_path):
        r = run_one(case(cmd="echo {run} {py}", contains=[str(tmp_path), "python3"]), tmp_path)
        assert r.status == "PASS"

    def test_check_function_errors_and_crashes_are_failures(self, tmp_path):
        assert run_one(case(cmd="echo hi", checks=[{"fn": "log_count", "args": {"pattern": "(?m)^hi$", "exactly": 1}}]),
                       tmp_path).status == "PASS"
        assert run_one(case(cmd="echo hi", checks=[{"fn": "log_count", "args": {"pattern": "(?m)^hi$", "exactly": 2}}]),
                       tmp_path).status == "FAIL"
        r = run_one(case(cmd="echo hi", checks=["does_not_exist"]), tmp_path)
        assert r.status == "FAIL" and "crashed" in r.reasons[0]


class TestReports:
    def test_report_and_junit(self, tmp_path):
        ok = run.Result(case(id="a"), status="PASS", seconds=1.0)
        bad = run.Result(case(id="b"), status="FAIL", seconds=2.0, reasons=["boom"], log="b.log")
        skip = run.Result(case(id="c"), status="SKIP", reasons=["dependency"])
        cov = {"total": 91.2, "files": {"src/x.py": (50.0, 5, 10)}}
        meta = {"sha": "abc", "run_id": "r", "seconds": 60.0}
        assert run.write_reports([ok, bad, skip], cov, 90.0, tmp_path, meta) is False
        text = (tmp_path / "report.md").read_text()
        assert "FAIL" in text.splitlines()[0] and "boom" in text and "91.2%" in text and "`src/x.py`" in text
        assert (tmp_path / "junit.xml").read_text().count("<failure") == 1

    def test_coverage_below_threshold_fails_the_run(self, tmp_path):
        ok = run.Result(case(id="a"), status="PASS")
        meta = {"sha": "abc", "run_id": "r", "seconds": 1.0}
        assert run.write_reports([ok], {"total": 80.0, "files": {}}, 90.0, tmp_path, meta) is False
        assert run.write_reports([ok], {"total": 95.0, "files": {}}, 90.0, tmp_path, meta) is True
        assert run.write_reports([ok], {"total": 10.0, "files": {}}, 0.0, tmp_path, meta) is True  # threshold off


class TestAsserts:
    def ctx(self, tmp_path, log=""):
        return asserts.Context(repo=tmp_path, run_dir=tmp_path, log=log, py="python3")

    def test_no_nan_in_log(self, tmp_path):
        assert asserts.no_nan_in_log(self.ctx(tmp_path, "values 1.0 2.0 inference informative")) == []
        assert asserts.no_nan_in_log(self.ctx(tmp_path, "loss nan at step 3"))
        assert asserts.no_nan_in_log(self.ctx(tmp_path, "x = -inf"))

    def test_file_contains(self, tmp_path):
        (tmp_path / "f.txt").write_text("hello world")
        assert asserts.file_contains(self.ctx(tmp_path), "f.txt", "wor") == []
        assert asserts.file_contains(self.ctx(tmp_path), "f.txt", "xyz")
        assert asserts.file_contains(self.ctx(tmp_path), "nope.txt", "x")

    def test_files_exist_with_glob(self, tmp_path):
        (tmp_path / "a1.pt").write_text("")
        assert asserts.files_exist(self.ctx(tmp_path), ["a*.pt"]) == []
        assert asserts.files_exist(self.ctx(tmp_path), ["b*.pt"])

    def test_finite_numbers_in_file(self, tmp_path):
        (tmp_path / "r.md").write_text("a 1.5 b 2.5 c 3.5")
        assert asserts.finite_numbers_in_file(self.ctx(tmp_path), "r.md", min_numbers=3) == []
        (tmp_path / "bad.md").write_text("a 1.5 nan")
        assert asserts.finite_numbers_in_file(self.ctx(tmp_path), "bad.md")
        assert asserts.finite_numbers_in_file(self.ctx(tmp_path), "r.md", min_numbers=9)

    def test_npz_fields(self, tmp_path):
        import numpy as np

        np.savez(tmp_path / "t.npz", obs=np.zeros((20, 3)), bad=np.array([[np.nan]] * 20))
        assert asserts.npz_fields(self.ctx(tmp_path), "t.npz", ["obs"], min_rows=10) == []
        assert asserts.npz_fields(self.ctx(tmp_path), "t.npz", ["obs"], min_rows=50)
        assert asserts.npz_fields(self.ctx(tmp_path), "t.npz", ["nope"])
        assert asserts.npz_fields(self.ctx(tmp_path), "t.npz", ["bad"])

    def test_fake_robot_summary(self, tmp_path):
        import json

        (tmp_path / "fake").mkdir()
        good = {"lowcmd_total": 500, "checks_total": {"crc_bad": 0, "header_bad": 0, "level_bad": 0, "mode_bad": 0,
                                                      "crc_checked": 50}}
        (tmp_path / "fake" / "summary.json").write_text(json.dumps(good))
        assert asserts.fake_robot_summary(self.ctx(tmp_path), str(tmp_path), min_lowcmds=100) == []
        assert asserts.fake_robot_summary(self.ctx(tmp_path), str(tmp_path), min_lowcmds=900)
        good["checks_total"]["crc_bad"] = 2
        (tmp_path / "fake" / "summary.json").write_text(json.dumps(good))
        assert any("crc_bad" in e for e in asserts.fake_robot_summary(self.ctx(tmp_path), str(tmp_path)))
        assert asserts.fake_robot_summary(self.ctx(tmp_path), str(tmp_path / "other"))
