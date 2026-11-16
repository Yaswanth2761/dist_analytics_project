"""Stage 1 — ingest raw SH17 into BRONZE tables (nothing is dropped here, only flagged).

bronze/images : one row per image file — size, EXIF orientation, integrity, duplicate-id flag
bronze/labels : one row per label line — parsed bbox (or polygon -> bbox), validation `issue`
"""
import _bootstrap  # noqa: F401
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

from safety.config import CLASS_NAMES, lake, load_config, parse_args, report_path, write_json
from safety.io_images import META_SCHEMA, image_meta
from safety.spark import get_spark, write


def ingest_images(spark, cfg):
    img_dir = Path(cfg["paths"]["raw_root"]) / "images"
    files = (spark.read.format("binaryFile")
             .option("pathGlobFilter", cfg["data"]["image_glob"])
             .option("recursiveFileLookup", "true")
             .load(str(img_dir))
             .select("path", F.col("length").alias("bytes")))  # `content` is pruned -> bytes never loaded
    files = files.withColumn("image_id", F.regexp_extract("path", r"([^/]+)\.[^./]+$", 1))
    n = files.count()
    meta = (files.select("image_id", "path", "bytes")
            .repartition(max(8, n // 250))
            .mapInPandas(image_meta, META_SCHEMA))
    dup = F.count("*").over(Window.partitionBy("image_id"))
    return meta.withColumn("dup_image_id", dup > 1)


def _tok(i: int) -> F.Column:
    # try_* functions keep this safe under ANSI mode (default in Spark 4): bad tokens -> NULL, not a crash
    return F.expr(f"try_cast(try_element_at(tok, {i}) as double)")


def ingest_labels(spark, cfg):
    tol = cfg["data"]["coord_tolerance"]
    lbl_dir = Path(cfg["paths"]["raw_root"]) / "labels"
    raw = (spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.txt")
           .text(str(lbl_dir))
           .select(F.col("value").alias("raw_line"), F.col("_metadata.file_path").alias("label_file")))
    df = (raw.withColumn("image_id", F.regexp_extract("label_file", r"([^/]+)\.txt$", 1))
          .filter(F.length(F.trim("raw_line")) > 0)
          .withColumn("tok", F.split(F.trim("raw_line"), r"\s+"))
          .withColumn("ntok", F.size("tok"))
          .withColumn("fmt", F.when(F.col("ntok") == 5, "bbox")
                      .when((F.col("ntok") >= 7) & (F.col("ntok") % 2 == 1), "polygon")
                      .otherwise("malformed"))
          .withColumn("cls_raw", _tok(1)))
    # bbox: cx cy w h  |  polygon (segmentation export): x1 y1 x2 y2 ... -> enclosing box
    xs = F.expr("transform(sequence(1, int((ntok - 1) / 2)), k -> try_cast(try_element_at(tok, 2 * k) as double))")
    ys = F.expr("transform(sequence(1, int((ntok - 1) / 2)), k -> try_cast(try_element_at(tok, 2 * k + 1) as double))")
    is_poly = F.col("fmt") == "polygon"
    cx, cy, w, h = _tok(2), _tok(3), _tok(4), _tok(5)
    df = (df.withColumn("rx1", F.when(is_poly, F.array_min(xs)).otherwise(cx - w / 2))
          .withColumn("ry1", F.when(is_poly, F.array_min(ys)).otherwise(cy - h / 2))
          .withColumn("rx2", F.when(is_poly, F.array_max(xs)).otherwise(cx + w / 2))
          .withColumn("ry2", F.when(is_poly, F.array_max(ys)).otherwise(cy + h / 2)))

    coords = [F.col(c) for c in ("rx1", "ry1", "rx2", "ry2")]
    any_null = F.lit(False)
    for c in coords:
        any_null = any_null | c.isNull() | F.isnan(c)
    valid_cls = (F.col("cls_raw").isNotNull() & (F.col("cls_raw") == F.floor("cls_raw"))
                 & (F.col("cls_raw") >= 0) & (F.col("cls_raw") < len(CLASS_NAMES)))
    oob = ((F.col("rx1") < -tol) | (F.col("ry1") < -tol) | (F.col("rx2") > 1 + tol) | (F.col("ry2") > 1 + tol))
    df = df.withColumn("issue",
                       F.when(F.col("fmt") == "malformed", "malformed")
                       .when(~valid_cls, "bad_class")
                       .when(any_null, "bad_number")
                       .when(oob, "out_of_bounds")
                       .when((F.col("rx2") <= F.col("rx1")) | (F.col("ry2") <= F.col("ry1")), "degenerate")
                       .otherwise("ok"))
    clip = lambda c: F.least(F.lit(1.0), F.greatest(F.lit(0.0), F.col(c)))  # noqa: E731
    return df.select("image_id", "label_file", "raw_line", "fmt",
                     F.col("cls_raw").cast("int").alias("cls_id"),
                     clip("rx1").alias("x1"), clip("ry1").alias("y1"),
                     clip("rx2").alias("x2"), clip("ry2").alias("y2"), "issue")


def main():
    cfg = load_config(parse_args().config)
    spark = get_spark(cfg)
    images = ingest_images(spark, cfg)
    write(images, lake(cfg, "bronze", "images"))
    labels = ingest_labels(spark, cfg)
    write(labels, lake(cfg, "bronze", "labels"))

    images = spark.read.parquet(lake(cfg, "bronze", "images"))
    labels = spark.read.parquet(lake(cfg, "bronze", "labels"))
    summary = {
        "images": images.count(),
        "unreadable_images": images.filter(~F.col("read_ok")).count(),
        "duplicate_image_ids": images.filter("dup_image_id").select("image_id").distinct().count(),
        "exif_rotated_images": images.filter(F.col("exif_orientation") > 1).count(),
        "label_lines": labels.count(),
        "label_issues": {r["issue"]: r["count"] for r in labels.groupBy("issue").count().collect()},
        "label_formats": {r["fmt"]: r["count"] for r in labels.groupBy("fmt").count().collect()},
    }
    write_json(summary, report_path(cfg, "01_ingest_summary.json"))
    print(summary)
    if summary["exif_rotated_images"]:
        print("NOTE: some images carry EXIF rotation. Export bakes it in (Ultralytics convention). "
              "Spot-check a few with their boxes drawn to confirm labels match the rotated view.")


if __name__ == "__main__":
    main()
