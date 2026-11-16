"""Simulated plant cameras: drops JSON event files (one frame reference per event) into stream/in.

Files are written to a temp name and atomically renamed — Spark's file source must never see a
half-written file. Event time runs on a simulated clock so windows close quickly in a demo.
"""
import _bootstrap  # noqa: F401
import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from safety.config import load_config, parse_args
from safety.risk import cameras


def _args(p):
    p.add_argument("--files", type=int, default=200, help="number of files to emit (0 = forever)")
    p.add_argument("--source-split", default="test")


def main():
    args = parse_args(_args)
    cfg = load_config(args.config)
    s = cfg["streaming"]
    src = Path(cfg["paths"]["yolo_root"], args.source_split, "images")
    frames = sorted(str(p) for p in src.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {src}; run p03_export_yolo.py first")
    out = Path(cfg["paths"]["stream"], "in")
    out.mkdir(parents=True, exist_ok=True)
    cams = cameras(cfg)
    rng = random.Random(cfg["data"]["seed"])
    clock = datetime.now(timezone.utc).replace(microsecond=0)
    i = 0
    while args.files == 0 or i < args.files:
        events = []
        for _ in range(s["events_per_file"]):
            zone, cam = rng.choice(cams)
            frame = rng.choice(frames)
            clock += timedelta(seconds=s["sim_seconds_per_event"])
            events.append({"event_id": uuid.uuid4().hex, "camera_id": cam, "zone": zone,
                           "ts": clock.strftime("%Y-%m-%dT%H:%M:%SZ"), "image_id": Path(frame).stem, "path": frame})
        tmp = out / f".tmp_{i:06d}.json"
        with open(tmp, "w") as f:
            f.write("\n".join(json.dumps(e) for e in events))
        os.replace(tmp, out / f"events_{i:06d}.json")
        i += 1
        print(f"emitted file {i} (sim time {clock.isoformat()})", end="\r")
        time.sleep(s["file_interval_sec"])


if __name__ == "__main__":
    main()
