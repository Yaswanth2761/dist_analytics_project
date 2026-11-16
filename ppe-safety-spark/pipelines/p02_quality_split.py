"""Stage 2 — BRONZE -> SILVER: validation, EDA, near-duplicate grouping, leakage-safe stratified split.

silver/labels : clean boxes (ok rows, too-small and exact-duplicate boxes removed) with cx,cy,w,h + pixel sizes
silver/images : readable images with phash, dup_group, stratum and split
reports/eda/* : class stats, small-object analysis, co-occurrence, resolution, split balance
"""
import _bootstrap  # noqa: F401
from pathlib import Path

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from safety.config import CLASS_NAMES, lake, load_config, parse_args, report_path, write_json
from safety.io_images import PHASH_SCHEMA, image_phash
from safety.spark import get_spark, write

def cls_name_col() -> F.Column:
    return F.element_at(F.array(*[F.lit(n) for n in CLASS_NAMES]), F.col("cls_id") + 1)


# --------------------------------------------------------------------------- cleaning
def clean_labels(images, labels, cfg):
    min_px = cfg["data"]["min_box_px"]
    lab = (labels.filter(F.col("issue") == "ok")
           .join(images.select("image_id", "eff_width", "eff_height"), "image_id")
           .withColumn("w", F.col("x2") - F.col("x1")).withColumn("h", F.col("y2") - F.col("y1"))
           .withColumn("cx", (F.col("x1") + F.col("x2")) / 2).withColumn("cy", (F.col("y1") + F.col("y2")) / 2)
           .withColumn("w_px", F.col("w") * F.col("eff_width")).withColumn("h_px", F.col("h") * F.col("eff_height"))
           .withColumn("area", F.col("w") * F.col("h")))
    too_small = lab.filter((F.col("w_px") < min_px) | (F.col("h_px") < min_px))
    lab = lab.filter((F.col("w_px") >= min_px) & (F.col("h_px") >= min_px))
    # exact duplicate annotations (same class, same box up to 1e-4) inflate counts and confuse NMS training
    keys = ["image_id", "cls_id"] + [F.round(c, 4).alias(f"r_{c}") for c in ("x1", "y1", "x2", "y2")]
    before = lab.count()
    lab = lab.select("*", *keys[2:]).dropDuplicates(["image_id", "cls_id", "r_x1", "r_y1", "r_x2", "r_y2"]) \
             .drop("r_x1", "r_y1", "r_x2", "r_y2")
    stats = {"too_small_boxes": too_small.count(), "exact_duplicate_boxes": before - lab.count()}
    return lab.withColumn("cls_name", cls_name_col()), stats


# --------------------------------------------------------------------------- EDA
def eda(spark, images, lab, cfg):
    out = Path(report_path(cfg, "eda", "x")).parent
    imgsz, small_px = cfg["train"]["imgsz"], cfg["export"]["small_box_px"]
    n_img = images.count()
    # size of the box's short side once the image is letterboxed to `imgsz` for training
    scale = F.lit(float(imgsz)) / F.greatest("eff_width", "eff_height")
    lab_s = lab.withColumn("short_px_train", F.least("w_px", "h_px") * scale)

    cls_stats = (lab_s.groupBy("cls_id", "cls_name").agg(
        F.count("*").alias("instances"),
        F.countDistinct("image_id").alias("images"),
        F.percentile_approx("area", 0.5).alias("median_area"),
        F.avg((F.col("area") < 0.001).cast("int")).alias("frac_area_lt_0.1pct"),
        F.avg((F.col("area") < 0.01).cast("int")).alias("frac_area_lt_1pct"),
        F.avg((F.col("short_px_train") < small_px).cast("int")).alias(f"frac_lt_{small_px}px_at_{imgsz}"),
    ).withColumn("image_share", F.col("images") / n_img).orderBy("cls_id").toPandas())
    cls_stats.to_csv(out / "class_stats.csv", index=False)

    per_img = lab.groupBy("image_id").count()
    q = per_img.approxQuantile("count", [0.5, 0.9, 0.99, 1.0], 0.001)
    res = images.select("eff_width", "eff_height").toPandas().describe()
    res.to_csv(out / "resolution_describe.csv")

    # image-level class co-occurrence: P(row class & col class) counts
    ic = lab.select("image_id", "cls_name").distinct()
    co = (ic.alias("a").join(ic.alias("b"), "image_id")
          .groupBy(F.col("a.cls_name").alias("a"), F.col("b.cls_name").alias("b")).count()
          .toPandas().pivot(index="a", columns="b", values="count").fillna(0).astype(int))
    co.to_csv(out / "cooccurrence.csv")

    summary = {"images": n_img, "boxes": lab.count(),
               "boxes_per_image_p50_p90_p99_max": q,
               "images_without_labels": images.join(lab.select("image_id").distinct(), "image_id", "left_anti").count()}
    return cls_stats, summary


# --------------------------------------------------------------------------- near-duplicates
def near_duplicate_groups(spark, images, cfg):
    """8 bands x 8 bits LSH on the 64-bit pHash. Two hashes differing in <= 7 bits must agree on at least
    one band (pigeonhole), so the candidate set is exact for the configured threshold."""
    thr = cfg["data"]["phash_max_hamming"]
    ph = (images.select("image_id", "path").repartition(max(8, images.count() // 250))
          .mapInPandas(image_phash, PHASH_SCHEMA).filter(F.col("phash").isNotNull()).cache())
    bands = ph.select("image_id", "phash", F.posexplode(F.array(*[F.substring("phash", 2 * i + 1, 2) for i in range(8)]))
                      .alias("band", "bval"))
    hi = lambda c: F.conv(F.substring(c, 1, 8), 16, 10).cast("long")   # noqa: E731
    lo = lambda c: F.conv(F.substring(c, 9, 8), 16, 10).cast("long")   # noqa: E731
    pairs = (bands.alias("a").join(bands.alias("b"), ["band", "bval"])
             .filter(F.col("a.image_id") < F.col("b.image_id"))
             .select(F.col("a.image_id").alias("a"), F.col("b.image_id").alias("b"),
                     F.col("a.phash").alias("ha"), F.col("b.phash").alias("hb")).distinct()
             .withColumn("hamming", F.bit_count(hi("ha").bitwiseXOR(hi("hb"))) + F.bit_count(lo("ha").bitwiseXOR(lo("hb"))))
             .filter(F.col("hamming") <= thr))
    edges = pairs.select("a", "b", "hamming").toPandas()  # near-dups are rare -> tiny; use GraphFrames at scale

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges["a"], edges["b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = pd.DataFrame([(k, find(k)) for k in list(parent)], columns=["image_id", "dup_group"])
    gdf = spark.createDataFrame(groups if len(groups) else pd.DataFrame({"image_id": pd.Series(dtype=str),
                                                                         "dup_group": pd.Series(dtype=str)}))
    out = (ph.join(gdf, "image_id", "left")
           .withColumn("dup_group", F.coalesce("dup_group", "image_id")))
    edges.to_csv(report_path(cfg, "eda", "near_duplicate_pairs.csv"), index=False)
    return out, {"near_duplicate_pairs": int(len(edges)),
                 "images_in_dup_groups": int(groups["image_id"].nunique()) if len(groups) else 0}


# --------------------------------------------------------------------------- split
def stratified_split(spark, images, lab, cfg):
    """Split by duplicate GROUP (no near-duplicate leakage), stratified on each group's rarest class so
    rare PPE (face-guard, medical-suit, ...) appears in every split."""
    s, seed = cfg["data"]["split"], cfg["data"]["seed"]
    ic = lab.select("image_id", "cls_id").distinct()
    freq = ic.groupBy("cls_id").agg(F.count("*").alias("cls_images"))
    rare = ic.join(freq, "cls_id").groupBy("image_id").agg(F.min_by("cls_id", "cls_images").alias("img_stratum"),
                                                         F.min("cls_images").alias("img_rarity"))
    g = (images.join(rare, "image_id", "left").fillna({"img_stratum": -1, "img_rarity": 10 ** 9})
         .groupBy("dup_group").agg(F.min_by("img_stratum", "img_rarity").alias("stratum")))
    r = F.percent_rank().over(Window.partitionBy("stratum").orderBy(F.xxhash64("dup_group", F.lit(seed))))
    g = g.withColumn("_r", r).withColumn(
        "split", F.when(F.col("_r") < s["train"], "train").when(F.col("_r") < s["train"] + s["val"], "val").otherwise("test"))
    out = images.join(g.select("dup_group", "stratum", "split"), "dup_group")

    if cfg["data"]["use_official_split"]:
        out = apply_official_split(spark, out, cfg)
    return out


def apply_official_split(spark, images, cfg):
    d = Path(cfg["paths"]["official_split_dir"])
    rows = []
    for name, split in (("train_files.txt", "train"), ("val_files.txt", "val")):
        p = d / name
        if not p.exists():
            raise FileNotFoundError(f"use_official_split=true but {p} is missing")
        rows += [(Path(line.strip()).stem, split) for line in p.read_text().splitlines() if line.strip()]
    off = spark.createDataFrame(rows, "image_id string, official string")
    s = cfg["data"]["split"]
    test_frac = s["test"] / (s["train"] + s["test"])
    # keep official val; carve a stratified test set out of official train
    r = F.percent_rank().over(Window.partitionBy("stratum").orderBy(F.xxhash64("dup_group", F.lit(cfg["data"]["seed"] + 1))))
    return (images.drop("split").join(off, "image_id", "left")
            .withColumn("_r", r)
            .withColumn("split", F.when(F.col("official") == "val", "val")
                        .when(F.col("_r") < test_frac, "test").otherwise("train"))
            .drop("_r", "official"))


def split_report(images, lab, cfg):
    sp = images.select("image_id", "split")
    df = (lab.join(sp, "image_id").groupBy("cls_name", "split").count().toPandas()
          .pivot(index="cls_name", columns="split", values="count").fillna(0).astype(int))
    df = df.join(df.div(df.sum(axis=1), axis=0).add_suffix("_share").round(3))
    df.to_csv(report_path(cfg, "eda", "split_balance.csv"))
    leak = (images.groupBy("dup_group").agg(F.countDistinct("split").alias("n")).filter("n > 1").count())
    return df, leak


def main():
    cfg = load_config(parse_args().config)
    spark = get_spark(cfg)
    images = (spark.read.parquet(lake(cfg, "bronze", "images"))
              .filter(F.col("read_ok") & ~F.col("dup_image_id")).cache())
    labels = spark.read.parquet(lake(cfg, "bronze", "labels"))

    orphans = labels.select("image_id").distinct().join(images, "image_id", "left_anti").count()
    lab, clean_stats = clean_labels(images, labels, cfg)
    write(lab, lake(cfg, "silver", "labels"))
    lab = spark.read.parquet(lake(cfg, "silver", "labels")).cache()

    cls_stats, eda_summary = eda(spark, images, lab, cfg)
    ph, dup_stats = near_duplicate_groups(spark, images, cfg)
    images = images.join(ph, "image_id", "left").withColumn("dup_group", F.coalesce("dup_group", "image_id"))
    images = stratified_split(spark, images, lab, cfg)
    n_boxes = lab.groupBy("image_id").agg(F.count("*").alias("n_boxes"))
    images = images.join(n_boxes, "image_id", "left").fillna({"n_boxes": 0})
    write(images.select("image_id", "path", "eff_width", "eff_height", "exif_orientation", "phash",
                        "dup_group", "stratum", "split", "n_boxes"), lake(cfg, "silver", "images"))

    images = spark.read.parquet(lake(cfg, "silver", "images"))
    bal, leak = split_report(images, lab, cfg)
    summary = {**eda_summary, **clean_stats, **dup_stats, "orphan_label_files": orphans,
               "split_sizes": {r["split"]: r["count"] for r in images.groupBy("split").count().collect()},
               "dup_groups_spanning_splits": leak}
    write_json(summary, report_path(cfg, "02_quality_summary.json"))
    print(cls_stats.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
