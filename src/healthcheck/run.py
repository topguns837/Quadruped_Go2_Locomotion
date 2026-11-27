# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo health-check runner. Meant to run INSIDE the testing container, started by ``./healthcheck.sh``.

Reads ``src/healthcheck/cases.yaml``, runs every case as a subprocess (own process group, hard timeout), applies the
case's expectations (exit code, log patterns, artifacts, Python checks from ``asserts.py``), collects coverage from
every Python process, and writes ``report.md`` + ``junit.xml`` + ``htmlcov/`` into ``logs/healthcheck/<run id>/``.

Exit code: 0 if every case passed (or is a documented known issue / skipped), 1 if anything failed or coverage is
below the threshold, 2 on a usage error.

Standard library + PyYAML only, so it also runs on a bare Python for ``--list``.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import asserts  # noqa: E402

DEFAULT_PY = "/workspace/isaaclab/_isaac_sim/python.sh"
TRACEBACK_RE = r"Traceback \(most recent call last\)"
# Training outputs this suite creates; removed before and (unless --keep) after the run.
OUTPUT_GLOBS = ["logs/rsl_rl/healthcheck_*", "logs/hydra/healthcheck_*", "logs/batch_status_healthcheck_*",
                "logs/relay_runs/hc_*"]


@dataclass
class Case:
    id: str
    group: str
    cmd: str
    timeout: int = 600
    gpu: bool = False
    needs: list = field(default_factory=list)
    rc: object = 0  # 0, an int, or "nonzero"
    contains: list = field(default_factory=list)
    not_contains: list = field(default_factory=list)
    allow_traceback: bool = False
    artifacts: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    known: str = ""
    description: str = ""


@dataclass
class Result:
    case: Case
    status: str = "PENDING"  # PASS FAIL KNOWN SKIP
    seconds: float = 0.0
    reasons: list = field(default_factory=list)
    log: str = ""


def expand_cases(raw: list[dict]) -> list[Case]:
    """Turn the YAML list into Cases, expanding ``matrix:`` entries (cartesian product, ``{key}`` substitution)."""
    cases: list[Case] = []
    for entry in raw:
        entry = dict(entry)
        matrix = entry.pop("matrix", None)
        combos = [{}]
        if matrix:
            keys = list(matrix)
            combos = [dict(zip(keys, vals)) for vals in itertools.product(*(matrix[k] for k in keys))]
        for combo in combos:
            spec = {k: _subst(v, combo) for k, v in entry.items()}
            expect = spec.pop("expect", {}) or {}
            cases.append(
                Case(
                    id=spec["id"],
                    group=spec.get("group", "misc"),
                    cmd=spec["cmd"],
                    timeout=int(spec.get("timeout", 600)),
                    gpu=bool(spec.get("gpu", False)),
                    needs=list(spec.get("needs", [])),
                    rc=expect.get("rc", 0),
                    contains=list(expect.get("contains", [])),
                    not_contains=list(expect.get("not_contains", [])),
                    allow_traceback=bool(spec.get("allow_traceback", False)),
                    artifacts=list(spec.get("artifacts", [])),
                    checks=list(spec.get("checks", [])),
                    known=spec.get("known", ""),
                    description=spec.get("description", ""),
                )
            )
    ids = [c.id for c in cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate case ids: {sorted(dup)}")
    return cases


def _subst(v, combo: dict):
    """Substitute ``{key}`` for matrix keys only, leaving every other brace (shell, {run}, {py}) untouched."""
    if isinstance(v, str):
        for k, val in combo.items():
            v = v.replace("{" + k + "}", str(val))
        return v
    if isinstance(v, list):
        return [_subst(x, combo) for x in v]
    if isinstance(v, dict):
        return {k: _subst(x, combo) for k, x in v.items()}
    return v


def select(cases: list[Case], only: str | None) -> list[Case]:
    """Filter by comma-separated group names or case ids (or id prefixes); pulls in every ``needs`` dependency."""
    if not only:
        return cases
    wanted = {w.strip() for w in only.split(",") if w.strip()}
    by_id = {c.id: c for c in cases}
    keep: set[str] = set()

    def add(cid: str) -> None:
        if cid in keep:
            return
        keep.add(cid)
        for n in by_id[cid].needs:
            add(n)

    for c in cases:
        if c.group in wanted or c.id in wanted or any(c.id.startswith(w) for w in wanted):
            add(c.id)
    return [c for c in cases if c.id in keep]


# ---------------------------------------------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------------------------------------------
def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return (total - used) / 1024.0
    except Exception:
        return None


def wait_for_gpu(min_free_gb: float, wait_min: float) -> bool:
    """True once at least ``min_free_gb`` of VRAM is free (another job, e.g. real training, may be using the card)."""
    deadline = time.monotonic() + wait_min * 60
    while True:
        free = gpu_free_gb()
        if free is None:
            print("[healthcheck] nvidia-smi unavailable: cannot check the GPU, assuming it is free", flush=True)
            return True
        if free >= min_free_gb:
            return True
        if time.monotonic() >= deadline:
            print(f"[healthcheck] GPU still has only {free:.1f} GB free after {wait_min:.0f} min", flush=True)
            return False
        print(f"[healthcheck] GPU busy ({free:.1f} GB free, need {min_free_gb:.0f}); waiting...", flush=True)
        time.sleep(30)


# ---------------------------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------------------------
COVERAGERC = """\
[run]
parallel = True
sigterm = True
branch = False
source =
    {repo}/src
omit =
{omit}
data_file = {data}

[report]
exclude_lines =
    pragma: no cover
    if __name__ == .__main__.:
    raise NotImplementedError
"""


HOOK_SOURCE = """\
# Installed by the health check into the throwaway container's site-packages (see src/healthcheck/run.py).
# Isaac Sim processes leave through a C-level exit with no Python teardown, so atexit never runs there: besides the
# usual atexit/os._exit save, a daemon thread saves the measurements every 0.5 s (at most 0.5 s of data can be lost).
import atexit, os, threading, time
import coverage
_c = coverage.process_startup()
if _c is not None:
    def _save():
        try:
            _c.save()
        except Exception:
            pass
    def _loop():
        while True:
            time.sleep(0.5)
            _save()
    threading.Thread(target=_loop, name="hc-cov-save", daemon=True).start()
    atexit.register(_save)
    _orig_exit = os._exit
    def _exit(code):
        _save()
        _orig_exit(code)
    os._exit = _exit
"""


def setup_coverage(py: str, run_dir: Path, omit: list[str]) -> dict:
    """Write a coveragerc and a ``.pth`` hook so EVERY python process (train.py under Kit, bash-launched chains,
    pytest) measures itself. The hook lives in this container's site-packages, which is thrown away with it."""
    data = run_dir / ".coverage"
    rc = run_dir / "coveragerc"
    omit_lines = "\n".join(f"    {REPO}/{o}" for o in omit)
    rc.write_text(COVERAGERC.format(repo=REPO, omit=omit_lines, data=data))
    site = subprocess.check_output([py, "-c", "import site; print(site.getsitepackages()[0])"], text=True)
    site_dir = Path(site.strip().splitlines()[-1])
    (site_dir / "hc_cov.py").write_text(HOOK_SOURCE)
    (site_dir / "hc_cov.pth").write_text("import hc_cov\n")
    return {"COVERAGE_PROCESS_START": str(rc), "COVERAGE_FILE": str(data), "HEALTHCHECK_COVERAGE": "1"}


def report_coverage(py: str, run_dir: Path, rcfile: Path, omit: list[str] | None = None) -> dict:
    """Combine the per-process data files; return {'total': pct, 'files': {path: (pct, missing)}} and write htmlcov."""
    env = dict(os.environ, COVERAGE_FILE=str(run_dir / ".coverage"))
    base = [py, "-m", "coverage"]
    subprocess.run(base + ["combine", "--rcfile", str(rcfile), "--append"], cwd=run_dir, env=env,
                   capture_output=True, text=True)
    js = run_dir / "coverage.json"
    r = subprocess.run(base + ["json", "--rcfile", str(rcfile), "-o", str(js), "--ignore-errors"], cwd=run_dir,
                       env=env, capture_output=True, text=True)
    subprocess.run(base + ["html", "--rcfile", str(rcfile), "-d", str(run_dir / "htmlcov"), "--ignore-errors"],
                   cwd=run_dir, env=env, capture_output=True, text=True)
    subprocess.run(base + ["xml", "--rcfile", str(rcfile), "-o", str(run_dir / "coverage.xml"), "--ignore-errors"],
                   cwd=run_dir, env=env, capture_output=True, text=True)
    txt = subprocess.run(base + ["report", "--rcfile", str(rcfile), "--show-missing", "--ignore-errors"], cwd=run_dir,
                         env=env, capture_output=True, text=True)
    (run_dir / "coverage.txt").write_text(txt.stdout)
    if r.returncode != 0 or not js.exists():
        return {"total": None, "files": {}, "error": (r.stderr or r.stdout)[-400:]}
    import json

    data = json.loads(js.read_text())
    files = {}
    for path, info in data["files"].items():
        s = info["summary"]
        files[os.path.relpath(path, REPO)] = (s["percent_covered"], s["missing_lines"], s["num_statements"])
    n_stmt = sum(v[2] for v in files.values())
    n_miss = sum(v[1] for v in files.values())
    # Files no process ever imported are absent from coverage's data: count their statements as missed, so the total
    # is not flattered by code that nothing ran.
    for rel, n in unmeasured_files(files, omit or []).items():
        files[rel] = (0.0, n, n)
        n_stmt += n
        n_miss += n
    total = 100.0 * (n_stmt - n_miss) / n_stmt if n_stmt else 0.0
    return {"total": total, "files": files}


def unmeasured_files(measured: dict, omit: list[str]) -> dict[str, int]:
    """{repo-relative path: statement count} for every non-omitted ``src/**/*.py`` that coverage never saw."""
    import fnmatch

    from coverage.python import PythonParser

    out = {}
    for f in sorted((REPO / "src").rglob("*.py")):
        rel = str(f.relative_to(REPO))
        if rel in measured or any(fnmatch.fnmatch(rel, o) for o in omit):
            continue
        try:
            parser = PythonParser(filename=str(f))
            parser.parse_source()
            n = len(parser.statements)
        except Exception:
            continue
        if n:
            out[rel] = n
    return out


# ---------------------------------------------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------------------------------------------
def run_case(case: Case, run_dir: Path, env: dict, py: str) -> Result:
    res = Result(case)
    log_path = run_dir / f"{case.id}.log"
    res.log = str(log_path)
    cmd = case.cmd.replace("{py}", py).replace("{run}", str(run_dir)).replace("{repo}", str(REPO))
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, "w") as lf:
        lf.write(f"$ {cmd}\n\n")
        lf.flush()
        proc = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=case.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # The testing container has its own PID namespace subtree: killing this group cannot reach anything else.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    continue
            rc = -9
    res.seconds = time.monotonic() - t0
    text = log_path.read_text(errors="replace")

    errors: list[str] = []
    if timed_out:
        errors.append(f"timed out after {case.timeout}s")
    elif case.rc == "nonzero":
        if rc == 0:
            errors.append("expected a non-zero exit code, got 0")
    elif rc != int(case.rc):
        errors.append(f"exit code {rc} (expected {case.rc})")
    for pat in case.contains:
        if not re.search(pat, text):
            errors.append(f"log lacks /{pat}/")
    for pat in case.not_contains:
        if re.search(pat, text):
            errors.append(f"log has unexpected /{pat}/")
    if case.rc == 0 and not case.allow_traceback and re.search(TRACEBACK_RE, text):
        errors.append("a Traceback is in the log")
    for art in case.artifacts:
        a = Path(art.replace("{run}", str(run_dir)))
        if not a.is_absolute():
            a = REPO / a
        if not any(Path(p).exists() for p in ([a] if not any(ch in str(a) for ch in "*?[") else
                                              [*a.parent.glob(a.name)])):
            errors.append(f"missing artifact {art}")
    ctx = asserts.Context(repo=REPO, run_dir=run_dir, log=text, py=py)
    for chk in case.checks:
        name, args = (chk, {}) if isinstance(chk, str) else (chk["fn"], chk.get("args", {}))
        try:
            errors += getattr(asserts, name)(ctx, **args)
        except Exception as e:  # a broken check is a failure of the check, reported as such
            errors.append(f"check {name} crashed: {type(e).__name__}: {e}")

    res.reasons = errors
    if not errors:
        res.status = "PASS"
    elif case.known:
        res.status = "KNOWN"
    else:
        res.status = "FAIL"
    return res


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------
def write_reports(results: list[Result], cov: dict | None, threshold: float, run_dir: Path, meta: dict) -> bool:
    n = {s: sum(r.status == s for r in results) for s in ("PASS", "FAIL", "KNOWN", "SKIP")}
    cov_ok = True
    if cov is not None and threshold:
        cov_ok = cov.get("total") is not None and cov["total"] >= threshold
    ok = n["FAIL"] == 0 and cov_ok
    L = [f"# Repo health check: {'PASS' if ok else 'FAIL'}", ""]
    L.append(f"- commit `{meta['sha']}`, run `{meta['run_id']}`, {meta['seconds'] / 60:.1f} min")
    L.append(f"- cases: **{n['PASS']} pass**, **{n['FAIL']} fail**, {n['KNOWN']} known issue, {n['SKIP']} skipped")
    if cov is not None:
        t = cov.get("total")
        L.append(f"- coverage of `src/`: **{t:.1f}%** (threshold {threshold:.0f}%)" if t is not None
                 else f"- coverage: unavailable ({cov.get('error', '')})")
    L += ["", "| Group | Case | Result | Time | Notes |", "|---|---|---|---|---|"]
    for r in results:
        note = "; ".join(r.reasons)[:200] if r.reasons else ""
        if r.case.known and r.reasons:
            note = f"KNOWN: {r.case.known} ({note})"
        L.append(f"| {r.case.group} | `{r.case.id}` | {r.status} | {r.seconds:.0f}s | {note} |")
    if cov and cov.get("files"):
        L += ["", "## Coverage by file (lowest first)", "", "| File | Covered | Statements |", "|---|---|---|"]
        for f, (pct, _miss, nst) in sorted(cov["files"].items(), key=lambda kv: kv[1][0]):
            L.append(f"| `{f}` | {pct:.0f}% | {nst} |")
    failed = [r for r in results if r.status == "FAIL"]
    if failed:
        L += ["", "## Failures", ""]
        for r in failed:
            L.append(f"### `{r.case.id}`\n\n- " + "\n- ".join(r.reasons) + f"\n- log: `{r.log}`\n")
    (run_dir / "report.md").write_text("\n".join(L) + "\n")

    suite = ET.Element("testsuite", name="healthcheck", tests=str(len(results)), failures=str(n["FAIL"]),
                       skipped=str(n["SKIP"] + n["KNOWN"]), time=f"{meta['seconds']:.1f}")
    for r in results:
        tc = ET.SubElement(suite, "testcase", classname=r.case.group, name=r.case.id, time=f"{r.seconds:.1f}")
        if r.status == "FAIL":
            ET.SubElement(tc, "failure", message="; ".join(r.reasons)[:500])
        elif r.status in ("SKIP", "KNOWN"):
            ET.SubElement(tc, "skipped", message="; ".join(r.reasons) or r.case.known)
    ET.ElementTree(suite).write(run_dir / "junit.xml", encoding="utf-8", xml_declaration=True)
    return ok


def print_coverage_summary(cov: dict, threshold: float, run_dir: Path, worst: int = 12) -> None:
    """Console summary of the coverage run: total, the weakest files, and where the full reports are."""
    print("\n" + "=" * 78)
    print("CODE COVERAGE (src/, every Python process the cases started)")
    print("=" * 78)
    if cov.get("total") is None:
        print(f"coverage unavailable: {cov.get('error', 'no data collected')}")
        return
    files = cov["files"]
    stmts = sum(v[2] for v in files.values())
    miss = sum(v[1] if isinstance(v[1], int) else len(v[1]) for v in files.values())
    verdict = "" if not threshold else ("  [meets threshold]" if cov["total"] >= threshold else f"  [BELOW {threshold:.0f}%]")
    print(f"total: {cov['total']:.1f}%  ({stmts - miss}/{stmts} statements in {len(files)} files){verdict}")
    print("\nlowest-covered files:")
    for f, (pct, _m, n) in sorted(files.items(), key=lambda kv: kv[1][0])[:worst]:
        print(f"  {pct:5.1f}%  {n:5d} stmts  {f}")
    print(f"\nfull reports in {run_dir}:")
    print("  htmlcov/index.html   browsable, per-line (open in a browser)")
    print("  coverage.txt         per-file table with the missing line numbers")
    print("  coverage.xml         Cobertura XML for CI tools;  coverage.json  raw data")
    print("=" * 78)


def give_back_ownership(run_dir: Path) -> None:
    """The container runs as root: hand the run folder (and ``logs/healthcheck`` itself) to the user who started
    ./healthcheck.sh, so they can read, move and delete the reports without sudo."""
    uid, gid = os.environ.get("HEALTHCHECK_HOST_UID"), os.environ.get("HEALTHCHECK_HOST_GID")
    if not (uid and gid):
        return
    for target, recursive in ((run_dir, True), (run_dir.parent, False)):
        cmd = ["chown"] + (["-R"] if recursive else []) + [f"{uid}:{gid}", str(target)]
        subprocess.run(cmd, capture_output=True)


def clean_outputs() -> None:
    for pattern in OUTPUT_GLOBS:
        for p in REPO.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=str(HERE / "cases.yaml"))
    ap.add_argument("--only", default=None, help="Comma-separated groups, case ids or id prefixes to run.")
    ap.add_argument("--list", action="store_true", help="List the selected cases and exit.")
    ap.add_argument("--keep", action="store_true", help="Keep training outputs afterwards.")
    ap.add_argument("--no_coverage", action="store_true", help="Do not collect coverage.")
    ap.add_argument("--fail_under", type=float, default=90.0, help="Coverage threshold in percent (0 = never fail).")
    ap.add_argument("--gpu_min_free_gb", type=float, default=8.0)
    ap.add_argument("--gpu_wait_min", type=float, default=20.0, help="Wait this long for a busy GPU, then skip.")
    ap.add_argument("--stop_on_fail", action="store_true")
    args = ap.parse_args(argv)

    raw = yaml.safe_load(Path(args.cases).read_text())
    cfg = raw if isinstance(raw, dict) else {"cases": raw}
    cases = select(expand_cases(cfg["cases"]), args.only)
    if args.list:
        for c in cases:
            print(f"{c.group:12s} {c.id:44s} {'gpu ' if c.gpu else '    '}{c.timeout:>5d}s  {c.description}")
        print(f"\n{len(cases)} cases")
        return 0

    py = os.environ.get("HEALTHCHECK_PY", DEFAULT_PY)
    sha = os.environ.get("HEALTHCHECK_SHA") or subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    run_id = os.environ.get("HEALTHCHECK_RUN_ID") or f"{time.strftime('%Y%m%d_%H%M%S')}_{sha or 'nogit'}"
    run_dir = REPO / "logs" / "healthcheck" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs()

    env = dict(os.environ, PYTHONUNBUFFERED="1", HEALTHCHECK_RUN_DIR=str(run_dir))
    rcfile = None
    if not args.no_coverage:
        env.update(setup_coverage(py, run_dir, cfg.get("coverage_omit", [])))
        rcfile = run_dir / "coveragerc"

    print(f"[healthcheck] {len(cases)} cases, output {run_dir}", flush=True)
    t_start = time.monotonic()
    results: list[Result] = []
    status: dict[str, str] = {}
    gpu_ok: bool | None = None
    for i, case in enumerate(cases, 1):
        res = Result(case)
        blocked = [n for n in case.needs if status.get(n) not in ("PASS", "KNOWN")]
        if blocked:
            res.status, res.reasons = "SKIP", [f"dependency not passing: {', '.join(blocked)}"]
        elif case.gpu:
            if gpu_ok is None:
                gpu_ok = wait_for_gpu(args.gpu_min_free_gb, args.gpu_wait_min)
            if not gpu_ok:
                res.status, res.reasons = "SKIP", ["GPU busy (another job is using it)"]
        if res.status == "PENDING":
            print(f"[{i}/{len(cases)}] {case.id} ...", end=" ", flush=True)
            res = run_case(case, run_dir, env, py)
            print(f"{res.status} ({res.seconds:.0f}s)" + (f"  <- {'; '.join(res.reasons)[:160]}" if res.reasons else ""),
                  flush=True)
        else:
            print(f"[{i}/{len(cases)}] {case.id} SKIP ({res.reasons[0]})", flush=True)
        status[case.id] = res.status
        results.append(res)
        if args.stop_on_fail and res.status == "FAIL":
            break

    cov = None
    if rcfile is not None:
        cov = report_coverage(py, run_dir, rcfile, cfg.get("coverage_omit", []))
    if not args.keep:
        clean_outputs()
    meta = {"sha": sha, "run_id": run_id, "seconds": time.monotonic() - t_start}
    if cov is not None:
        print_coverage_summary(cov, 0.0 if args.only else args.fail_under, run_dir)
    ok = write_reports(results, cov, 0.0 if args.only else args.fail_under, run_dir, meta)
    give_back_ownership(run_dir)
    print(f"\n[healthcheck] {'PASS' if ok else 'FAIL'}  report: {run_dir / 'report.md'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
