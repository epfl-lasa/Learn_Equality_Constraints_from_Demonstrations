from __future__ import annotations

import json
import os
from typing import Any


def _load_file(path: str) -> dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"yaml config requested but PyYAML not available: {path}") from e
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"unsupported config extension: {path}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be object/dict: {path}")
    return data


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_first_existing(base_no_ext: str) -> str | None:
    for ext in (".json", ".yaml", ".yml"):
        p = base_no_ext + ext
        if os.path.exists(p):
            return p
    return None


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError(f"invalid override key: '{dotted_key}'")
    cur = cfg
    for k in keys[:-1]:
        nxt = cur.get(k)
        if nxt is None:
            nxt = {}
            cur[k] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"override path collides with scalar at '{k}' in '{dotted_key}'")
        cur = nxt
    cur[keys[-1]] = value


def _parse_scalar(text: str) -> Any:
    t = text.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(t)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        pass
    return t


def parse_override(expr: str) -> tuple[str, Any]:
    if "=" not in expr:
        raise ValueError(f"invalid override '{expr}', expected key=value")
    k, v = expr.split("=", 1)
    key = k.strip()
    if not key:
        raise ValueError(f"invalid override '{expr}', empty key")
    try:
        val = json.loads(v)
    except Exception:
        val = _parse_scalar(v)
    return key, val


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = dict(cfg)
    for expr in overrides:
        key, val = parse_override(expr)
        _set_dotted(out, key, val)
    return out


def load_layered_config(config_root: str, method: str, dataset: str) -> tuple[dict[str, Any], list[str]]:
    root = os.path.abspath(config_root)
    dataset_for_config = str(dataset)
    try:
        from datasets.real_iiwa.registry import canonical_real_iiwa_dataset_name

        canonical = canonical_real_iiwa_dataset_name(str(dataset))
        if canonical is not None:
            dataset_for_config = str(canonical)
    except Exception:
        dataset_for_config = str(dataset)
    layers = [
        _resolve_first_existing(os.path.join(root, "base")),
        _resolve_first_existing(os.path.join(root, "methods", method)),
        _resolve_first_existing(os.path.join(root, "datasets", dataset_for_config)),
        _resolve_first_existing(os.path.join(root, "method_dataset", f"{method}_{dataset_for_config}")),
    ]
    loaded_paths: list[str] = []
    merged: dict[str, Any] = {}
    for p in layers:
        if p is None:
            continue
        merged = _deep_merge(merged, _load_file(p))
        loaded_paths.append(p)
    return merged, loaded_paths
