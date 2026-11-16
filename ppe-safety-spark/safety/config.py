"""Configuration, SH17 class taxonomy and small path/report helpers shared by every stage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# SH17 class ids as used in the label files (NOT the order listed on the dataset page).
CLASS_NAMES = [
    "person", "ear", "earmuffs", "face", "face-guard", "face-mask-medical", "foot", "tools",
    "glasses", "gloves", "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
BODY_PARTS = ["head", "face", "ear", "hands", "foot"]
PPE = ["helmet", "glasses", "face-mask-medical", "face-guard", "earmuffs", "gloves", "shoes",
       "safety-vest", "safety-suit", "medical-suit"]
VIOLATION_STATUSES = ["missing", "not_worn", "partial"]


def parse_args(extra: callable | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "config.yaml"))
    if extra:
        extra(p)
    return p.parse_args()


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else (REPO_ROOT / p).resolve())
    for k in ("weights",):
        p = Path(cfg["inference"][k])
        cfg["inference"][k] = str(p if p.is_absolute() else (REPO_ROOT / p).resolve())
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    s = cfg["data"]["split"]
    assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-6, "split fractions must sum to 1"
    for name, r in cfg["compliance"]["requirements"].items():
        bad = [c for c in r["ppe"] if c not in CLASS_ID]
        assert not bad, f"{name}: unknown PPE classes {bad}"
        assert r.get("part") in (None, *BODY_PARTS), f"{name}: unknown body part {r.get('part')}"
    reqs = set(cfg["compliance"]["requirements"])
    for z, zc in cfg["risk"]["zones"].items():
        assert set(zc["required"]) <= reqs, f"zone {z} references unknown requirements"
    assert cfg["data"]["phash_max_hamming"] <= 7, "8-band LSH is only exact for hamming <= 7"


def lake(cfg: dict, layer: str, name: str) -> str:
    return os.path.join(cfg["paths"]["lake"], layer, name)


def report_path(cfg: dict, *parts: str) -> str:
    p = Path(cfg["paths"]["reports"], *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def write_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
