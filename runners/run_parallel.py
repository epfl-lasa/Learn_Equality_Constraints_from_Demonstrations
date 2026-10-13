#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.unified_experiment import VALID_METHODS


DEFAULT_WORKERS = int(os.environ.get("LEARNEQ_PARALLEL_WORKERS", "6"))


@dataclass(frozen=True)
class Job:
    index: int
    method: str
    dataset: str
    seed: int | None
    overrides: tuple[str, ...]
    name: str


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _resolve_path(path: str, default: str) -> str:
    p = str(path or default).strip()
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _clear_directory(path: str) -> None:
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _prepare_outdir(path: str, *, clear: bool) -> None:
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"outdir exists but is not a directory: {path}")
    if os.path.isdir(path) and clear:
        print(f"[clear] removing existing outdir contents: {path}", flush=True)
        _clear_directory(path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "runs"), exist_ok=True)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)


def _parse_json_value(text: str) -> Any:
    s = str(text).strip()
    if not s:
        return ""
    try:
        return json.loads(s)
    except Exception:
        return s


def _json_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=False)}"


def _parse_sweep(expr: str) -> tuple[str, list[Any]]:
    if "=" not in str(expr):
        raise ValueError(f"invalid --sweep {expr!r}; expected key=v1,v2")
    key, raw_values = str(expr).split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"invalid --sweep {expr!r}; empty key")
    values = next(csv.reader([raw_values], skipinitialspace=True))
    parsed = [_parse_json_value(v) for v in values if str(v).strip()]
    if not parsed:
        raise ValueError(f"invalid --sweep {expr!r}; no values")
    return key, parsed


def _normalize_job_overrides(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [_json_override(str(k), v) for k, v in value.items()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(_normalize_job_overrides(item))
            else:
                raise ValueError(f"job override list items must be strings or objects, got {item!r}")
        return out
    if isinstance(value, str):
        return [value]
    raise ValueError(f"unsupported override value: {value!r}")


def _load_job_file(path: str) -> list[dict[str, Any]]:
    p = Path(_resolve_path(path, path))
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"job json must be a list: {p}")
        if not all(isinstance(x, dict) for x in data):
            raise ValueError(f"job json entries must be objects: {p}")
        return list(data)

    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"job jsonl line {lineno} must be an object: {p}")
        rows.append(obj)
    return rows


def _safe_slug(text: str, *, max_len: int = 64) -> str:
    chars: list[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "job")[:max_len]


def _job_name(index: int, method: str, dataset: str, seed: int | None, overrides: list[str], explicit: str | None) -> str:
    if explicit:
        return _safe_slug(str(explicit), max_len=96)
    payload = json.dumps(
        {"method": method, "dataset": dataset, "seed": seed, "overrides": overrides},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    seed_part = "none" if seed is None else str(seed)
    return _safe_slug(f"{index:04d}_{method}_{dataset}_seed{seed_part}_{digest}", max_len=120)


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    common_overrides = list(args.override or [])
    jobs: list[Job] = []

    if args.job_file:
        rows = _load_job_file(str(args.job_file))
        for i, row in enumerate(rows, start=1):
            method = str(row.get("method", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            if not method or not dataset:
                raise ValueError(f"job #{i} must define method and dataset")
            seed_raw = row.get("seed", None)
            seed = None if seed_raw in (None, "") else int(seed_raw)
            overrides = [*common_overrides, *_normalize_job_overrides(row.get("override", row.get("overrides", [])))]
            name = _job_name(i, method, dataset, seed, overrides, row.get("name"))
            jobs.append(Job(i, method, dataset, seed, tuple(overrides), name))
        return jobs

    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    seeds_text = _split_csv(args.seeds)
    seeds: list[int | None] = [int(s) for s in seeds_text] if seeds_text else [None]
    if not methods:
        raise ValueError("no methods provided")
    if not datasets:
        raise ValueError("no datasets provided")

    sweep_specs = [_parse_sweep(s) for s in list(args.sweep or [])]
    if sweep_specs:
        keys = [k for k, _ in sweep_specs]
        value_sets = [vals for _, vals in sweep_specs]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_sets)]
    else:
        combos = [{}]

    index = 1
    for method, dataset, seed, combo in itertools.product(methods, datasets, seeds, combos):
        dynamic_overrides = [_json_override(k, v) for k, v in combo.items()]
        overrides = [*common_overrides, *dynamic_overrides]
        name = _job_name(index, method, dataset, seed, overrides, None)
        jobs.append(Job(index, method, dataset, seed, tuple(overrides), name))
        index += 1
    return jobs


def _validate_jobs(jobs: list[Job]) -> None:
    if not jobs:
        raise ValueError("no jobs to run")
    bad = sorted({j.method for j in jobs if j.method not in VALID_METHODS})
    if bad:
        raise ValueError(f"unsupported methods: {bad}; valid={sorted(VALID_METHODS)}")


def _summary_path(run_dir: str, job: Job) -> str:
    return os.path.join(run_dir, f"summary_{job.method}_{job.dataset}.json")


def _job_result_path(outdir: str, job: Job) -> str:
    return os.path.join(outdir, "runs", job.name, "job_result.json")


def _run_job(job: Job, args: argparse.Namespace, outdir: str, config_root: str) -> dict[str, Any]:
    run_dir = os.path.join(outdir, "runs", job.name)
    log_path = os.path.join(outdir, "logs", f"{job.name}.log")
    result_path = _job_result_path(outdir, job)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [
        str(args.python),
        "-m",
        "runners.run_one",
        "--method",
        job.method,
        "--dataset",
        job.dataset,
        "--outdir",
        run_dir,
        "--retrain",
        "--config-root",
        config_root,
    ]
    if job.seed is not None:
        cmd.extend(["--seed", str(job.seed)])
    for override in job.overrides:
        cmd.extend(["--override", str(override)])

    started = time.time()
    row: dict[str, Any] = {
        "job": job.name,
        "index": job.index,
        "method": job.method,
        "dataset": job.dataset,
        "seed": job.seed,
        "overrides": list(job.overrides),
        "run_dir": run_dir,
        "log_path": log_path,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
    }
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    ended = time.time()
    row["returncode"] = int(proc.returncode)
    row["seconds"] = float(ended - started)
    row["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended))
    row["status"] = "success" if proc.returncode == 0 else "failed"

    summary_path = _summary_path(run_dir, job)
    if proc.returncode == 0 and os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        row["metrics"] = result.get("metrics", {}) if isinstance(result, dict) else {}
        row["config"] = result.get("config", {}) if isinstance(result, dict) else {}
        row["summary_path"] = summary_path
    elif proc.returncode == 0:
        row["status"] = "failed"
        row["returncode"] = 2
        row["error"] = f"missing summary file: {summary_path}"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    return row


def _skip_row(outdir: str, job: Job) -> dict[str, Any] | None:
    path = _job_result_path(outdir, job)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
    except Exception:
        return None
    if isinstance(row, dict) and row.get("status") == "success":
        row = dict(row)
        row["skipped"] = True
        return row
    return None


def _write_outputs(outdir: str, rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r.get("index", 0)))
    jsonl = os.path.join(outdir, "per_run_metrics.jsonl")
    summary = os.path.join(outdir, "parallel_summary.json")
    csv_path = os.path.join(outdir, "per_case_metrics.csv")

    with open(jsonl, "w", encoding="utf-8") as f:
        for row in rows_sorted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2, ensure_ascii=False)

    metric_keys: set[str] = set()
    for row in rows_sorted:
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            metric_keys.update(str(k) for k in metrics.keys())
    metric_cols = sorted(metric_keys)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "status", "method", "dataset", "seed", "seconds", "overrides", *metric_cols])
        for row in rows_sorted:
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
            w.writerow([
                row.get("job", ""),
                row.get("status", ""),
                row.get("method", ""),
                row.get("dataset", ""),
                row.get("seed", ""),
                row.get("seconds", ""),
                json.dumps(row.get("overrides", []), ensure_ascii=False),
                *[metrics.get(k, "") for k in metric_cols],
            ])
    print(f"[saved] {jsonl}", flush=True)
    print(f"[saved] {summary}", flush=True)
    print(f"[saved] {csv_path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run method/dataset/seed/parameter sweeps in parallel subprocesses."
    )
    p.add_argument("--methods", default="oncl", help="comma-separated methods")
    p.add_argument("--datasets", default="3d_sphere_surface", help="comma-separated datasets")
    p.add_argument("--seeds", default="10", help="comma-separated seeds; empty means no seed override")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2", help="Cartesian parameter sweep. Can be repeated.")
    p.add_argument("--job-file", default="", help="JSON list or JSONL jobs with method,dataset,seed,override/name fields")
    p.add_argument("--override", action="append", default=[], help="Common dotted key=value override for every job")
    p.add_argument("--outdir", default="outputs/parallel")
    p.add_argument("--config-root", default="configs")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel subprocesses; defaults to 6")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true", help="skip jobs with successful job_result.json")
    p.add_argument("--rewrite", action="store_true", help="rerun jobs even if they already succeeded")
    p.add_argument("--clear", action="store_true", help="clear outdir before running")
    p.add_argument("--dry-run", action="store_true", help="print planned jobs without running them")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.resume and args.rewrite:
        raise ValueError("--resume and --rewrite are mutually exclusive")
    outdir = _resolve_path(str(args.outdir), "outputs/parallel")
    config_root = _resolve_path(str(args.config_root), "configs")
    jobs = _build_jobs(args)
    _validate_jobs(jobs)
    workers = max(1, min(int(args.workers), len(jobs)))

    print(f"[plan] jobs={len(jobs)} workers={workers} outdir={outdir}", flush=True)
    for job in jobs:
        print(
            f"[job] {job.name} method={job.method} dataset={job.dataset} "
            f"seed={job.seed} overrides={list(job.overrides)}",
            flush=True,
        )
    if args.dry_run:
        return

    _prepare_outdir(outdir, clear=bool(args.clear))

    rows: list[dict[str, Any]] = []
    pending: list[Job] = []
    for job in jobs:
        skipped = None if args.rewrite else _skip_row(outdir, job)
        if skipped is not None and args.resume:
            print(f"[skip] {job.name} already succeeded", flush=True)
            rows.append(skipped)
        else:
            pending.append(job)

    started = time.time()
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending) or 1))) as pool:
        fut_to_job = {
            pool.submit(_run_job, job, args, outdir, config_root): job
            for job in pending
        }
        for fut in concurrent.futures.as_completed(fut_to_job):
            job = fut_to_job[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "job": job.name,
                    "index": job.index,
                    "method": job.method,
                    "dataset": job.dataset,
                    "seed": job.seed,
                    "overrides": list(job.overrides),
                    "status": "failed",
                    "error": repr(e),
                }
            if row.get("status") != "success":
                failures += 1
            rows.append(row)
            print(
                f"[done] {row.get('job')} status={row.get('status')} "
                f"seconds={float(row.get('seconds', 0.0)):.2f}",
                flush=True,
            )
            _write_outputs(outdir, rows)

    total_seconds = float(time.time() - started)
    _write_outputs(outdir, rows)
    successes = sum(1 for r in rows if r.get("status") == "success")
    print(
        f"[summary] success={successes}/{len(jobs)} failed={failures} "
        f"total_seconds={total_seconds:.2f} workers={workers}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
