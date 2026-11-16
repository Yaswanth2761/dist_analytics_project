"""Stage 8 — real-time safety monitoring with Spark Structured Streaming.

Query 1 (foreachBatch): events -> YOLO -> compliance -> risk -> gold/stream_person_risk (+ gold/stream_alerts)
Query 2 (event-time):   sliding windows per zone with a watermark -> gold/stream_zone_windows
Run alongside:  python pipelines/p08_stream_simulator.py
"""
import _bootstrap  # noqa: F401
import os
from pathlib import Path

from pyspark.sql import functions as F

from safety import compliance as C
from safety import risk as R
from safety.config import lake, load_config, parse_args
from safety.detector import DET_SCHEMA, make_detect_fn
from safety.spark import get_spark

EVENT_SCHEMA = "event_id string, camera_id string, zone string, ts timestamp, image_id string, path string"


def main():
    cfg = load_config(parse_args().config)
    inf, st = cfg["inference"], cfg["streaming"]
    spark = get_spark(cfg, master=f"local[{max(2, inf['concurrency'] + 1)}]")
    spark.conf.set("spark.sql.shuffle.partitions", 4)  # micro-batches are small
    base = Path(cfg["paths"]["stream"])
    in_dir, ckpt = base / "in", base / "checkpoints"
    pr_path, alert_path, win_path = (lake(cfg, "gold", n) for n in ("stream_person_risk", "stream_alerts", "stream_zone_windows"))
    for p in (in_dir, ckpt, pr_path):
        os.makedirs(p, exist_ok=True)

    zone_req = R.zone_requirements(spark, cfg).cache()
    detect = make_detect_fn({k: inf[k] for k in ("weights", "imgsz", "conf", "iou", "max_det", "batch_size", "device", "half")})
    alert_thr = cfg["risk"]["levels"]["high"]

    def process(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        frames = batch_df.select(F.col("event_id").alias("image_id"), "path")   # each event is its own frame
        dets = frames.repartition(inf["concurrency"]).mapInPandas(detect, DET_SCHEMA)
        persons, assigned = C.assign_to_persons(dets, cfg)
        comp = C.compute_compliance(persons, assigned, cfg)
        ctx = batch_df.select(F.col("event_id").alias("image_id"), "zone", "camera_id", "ts").withColumn("shift", R.shift_of(F.col("ts")))
        pr = R.score_persons(comp, ctx, zone_req, cfg).cache()
        pr.write.mode("append").parquet(pr_path)
        alerts = pr.filter(F.col("risk") >= alert_thr)
        if not alerts.isEmpty():
            alerts.withColumn("batch_id", F.lit(batch_id)).write.mode("append").parquet(alert_path)
            for a in alerts.select("ts", "camera_id", "violations", "risk").collect():
                print(f"[ALERT] {a.ts} {a.camera_id} missing={a.violations} risk={a.risk:.2f}")
        pr.unpersist()

    events = spark.readStream.schema(EVENT_SCHEMA).option("maxFilesPerTrigger", 2).json(str(in_dir))
    q1 = (events.writeStream.foreachBatch(process)
          .option("checkpointLocation", str(ckpt / "scoring"))
          .trigger(processingTime=st["trigger"]).start())

    pr_stream = spark.readStream.schema(R.PERSON_RISK_SCHEMA).parquet(pr_path)
    win = (pr_stream.withWatermark("ts", st["watermark"])
           .groupBy(F.window("ts", st["window"], st["slide"]).alias("w"), "zone")
           .agg(F.count("*").alias("persons"),
                F.sum((F.col("n_violations") > 0).cast("int")).alias("violators"),
                F.avg("risk").alias("mean_risk"))
           .withColumn("violation_rate", F.col("violators") / F.col("persons"))
           .withColumn("alert", F.col("violation_rate") >= st["window_violation_rate_alert"])
           .select(F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"), "zone",
                   "persons", "violators", "mean_risk", "violation_rate", "alert"))
    q2 = (win.writeStream.outputMode("append").format("parquet")
          .option("path", win_path).option("checkpointLocation", str(ckpt / "windows"))
          .trigger(processingTime=st["trigger"]).start())

    print("streaming… Ctrl+C to stop")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        q1.stop()
        q2.stop()


if __name__ == "__main__":
    main()
