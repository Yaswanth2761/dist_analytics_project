"""YOLO inference inside Spark workers (batch job and streaming job share this)."""
from __future__ import annotations

import sys
from typing import Iterator

import pandas as pd

DET_SCHEMA = ("image_id string, det_id string, cls_id int, cls_name string, conf double, "
              "x1 double, y1 double, x2 double, y2 double")
_DET_COLS = [c.split()[0] for c in DET_SCHEMA.split(", ")]

_MODEL = None  # one model per Python worker process, reused across batches (spark.python.worker.reuse)


def _get_model(weights: str):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
    return _MODEL


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        _DET_COLS, ["object", "object", "int32", "object", "float64", "float64", "float64", "float64", "float64"])})


def make_detect_fn(opts: dict):
    """opts: weights, imgsz, conf, iou, max_det, batch_size, device, half.
    Input columns: image_id, path. Output: one row per detection (normalized xyxy)."""

    def _predict(model, paths):
        return model.predict(paths, imgsz=opts["imgsz"], conf=opts["conf"], iou=opts["iou"],
                             max_det=opts["max_det"], device=opts["device"],
                             half=opts["half"] and str(opts["device"]) != "cpu", verbose=False)

    def _fn(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        model = _get_model(opts["weights"])
        names = model.names
        for pdf in batches:
            ids, paths = pdf["image_id"].tolist(), pdf["path"].tolist()
            rows = []
            bs = opts["batch_size"]
            for s in range(0, len(paths), bs):
                chunk_ids, chunk_paths = ids[s:s + bs], paths[s:s + bs]
                try:
                    results = list(zip(chunk_ids, _predict(model, chunk_paths)))
                except Exception:
                    # one unreadable file must not kill the whole partition: retry image by image
                    results = []
                    for i, p in zip(chunk_ids, chunk_paths):
                        try:
                            results.append((i, _predict(model, [p])[0]))
                        except Exception as e:
                            print(f"[detector] skipping {p}: {e}", file=sys.stderr)
                for image_id, r in results:
                    b = r.boxes
                    if b is None or len(b) == 0:
                        continue
                    xyxy = b.xyxyn.cpu().numpy()
                    cls = b.cls.cpu().numpy().astype(int)
                    conf = b.conf.cpu().numpy()
                    for k in range(len(cls)):
                        rows.append((image_id, f"{image_id}_{k}", int(cls[k]), names[int(cls[k])],
                                     float(conf[k]), *map(float, xyxy[k])))
            yield pd.DataFrame(rows, columns=_DET_COLS).astype({"cls_id": "int32"}) if rows else _empty()

    return _fn
