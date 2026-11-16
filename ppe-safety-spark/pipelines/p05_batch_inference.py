"""Stage 5 — distributed batch inference with Spark -> gold/detections.

Each Spark task holds ONE model replica (loaded once per Python worker, reused across batches).
`inference.concurrency` sets the number of parallel tasks: 1 per GPU, or N on CPU-only machines.
Every run appends a throughput row to reports/05_inference_benchmark.csv so you can plot scaling
(e.g. run with --concurrency 1, 2, 4, 8 on CPU).
"""
import _bootstrap  # noqa: F401
import csv
import time
from pathlib import Path

from pyspark.sql import functions as F

from safety.config import lake, load_config, parse_args, report_path
from safety.detector import DET_SCHEMA, make_detect_fn
from safety.spark import get_spark, write


def _args(p):
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--device", default=None)


def main():
    args = parse_args(_args)
    cfg = load_config(args.config)
    inf = dict(cfg["inference"])
    if args.concurrency:
        inf["concurrency"] = args.concurrency
    if args.device:
        inf["device"] = args.device
    if not Path(inf["weights"]).exists():
        raise FileNotFoundError(f"weights not found: {inf['weights']} (run p04_train.py first)")

    spark = get_spark(cfg, master=f"local[{inf['concurrency']}]")
    root = cfg["paths"]["yolo_root"]
    images = (spark.read.parquet(lake(cfg, "silver", "images"))
              .filter(F.col("split").isin(inf["splits"]))
              .withColumn("path", F.concat_ws("/", F.lit(root), "split", F.lit("images"),
                                              F.concat("image_id", F.lit(".jpg"))))
              .select("image_id", "path", "split"))
    n = images.count()
    opts = {k: inf[k] for k in ("weights", "imgsz", "conf", "iou", "max_det", "batch_size", "device", "half")}

    t0 = time.perf_counter()
    dets = (images.repartition(inf["concurrency"] * 4, "image_id")   # several tasks per replica: load balancing
            .mapInPandas(make_detect_fn(opts), DET_SCHEMA))
    write(dets, lake(cfg, "gold", "detections"))
    dt = time.perf_counter() - t0

    det = spark.read.parquet(lake(cfg, "gold", "detections"))
    n_det = det.count()
    row = {"concurrency": inf["concurrency"], "device": inf["device"], "imgsz": inf["imgsz"],
           "images": n, "detections": n_det, "seconds": round(dt, 2), "images_per_sec": round(n / dt, 3)}
    p = report_path(cfg, "05_inference_benchmark.csv")
    new = not Path(p).exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)
    det.groupBy("cls_name").agg(F.count("*").alias("n"), F.round(F.avg("conf"), 3).alias("mean_conf")) \
       .orderBy(F.desc("n")).show(20, truncate=False)


if __name__ == "__main__":
    main()
