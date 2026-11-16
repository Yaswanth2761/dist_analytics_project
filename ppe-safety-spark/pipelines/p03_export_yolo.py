"""Stage 3 — SILVER -> YOLO training set (parallel resize/export with Spark workers).

data/yolo_sh17/{train,val,test,test_cctv}/{images,labels}
data/yolo_sh17/train_rfs/   repeat-factor-sampled view of train (symlinks) to counter class imbalance
data/yolo_sh17/data.yaml, data_cctv.yaml
"""
import _bootstrap  # noqa: F401
import os
import shutil
from pathlib import Path

import yaml
from pyspark.sql import functions as F

from safety.config import CLASS_NAMES, lake, load_config, parse_args, report_path, write_json
from safety.io_images import EXPORT_SCHEMA, make_export_fn
from safety.spark import get_spark, write


def export_images(spark, cfg):
    e, root = cfg["export"], cfg["paths"]["yolo_root"]
    for split in ("train", "val", "test", "test_cctv"):
        for sub in ("images", "labels"):
            Path(root, split, sub).mkdir(parents=True, exist_ok=True)

    images = spark.read.parquet(lake(cfg, "silver", "images"))
    labels = spark.read.parquet(lake(cfg, "silver", "labels"))
    lines = labels.select("image_id", F.format_string("%d %.6f %.6f %.6f %.6f", "cls_id", "cx", "cy", "w", "h").alias("line"))
    txt = lines.groupBy("image_id").agg(F.concat_ws("\n", F.sort_array(F.collect_list("line"))).alias("label_txt"))
    todo = (images.join(txt, "image_id", "left").fillna({"label_txt": ""})
            .select("image_id", "path", "split", "label_txt"))
    n = todo.count()
    fn = make_export_fn(root, e["long_side"], e["jpeg_quality"], e["skip_existing"], e["build_cctv_test"], cfg["data"]["seed"])
    manifest = todo.repartition(max(8, n // 100)).mapInPandas(fn, EXPORT_SCHEMA)
    write(manifest, lake(cfg, "silver", "export_manifest"))
    manifest = spark.read.parquet(lake(cfg, "silver", "export_manifest"))
    errors = manifest.filter(F.col("status") != "ok")
    n_err = errors.count()
    if n_err:
        errors.toPandas().to_csv(report_path(cfg, "03_export_errors.csv"), index=False)
        print(f"WARNING: {n_err} images failed to export (see reports/03_export_errors.csv)")
    return n, n_err


def repeat_factor_sampling(spark, cfg):
    """LVIS repeat-factor sampling: r_c = max(1, sqrt(t / f_c)), f_c = share of train images containing c;
    each image is repeated r_i = max_c r_c times (stochastic rounding, deterministic hash)."""
    e, root, seed = cfg["export"], Path(cfg["paths"]["yolo_root"]), cfg["data"]["seed"]
    images = spark.read.parquet(lake(cfg, "silver", "images")).filter("split = 'train'").select("image_id")
    n_train = images.count()
    ic = spark.read.parquet(lake(cfg, "silver", "labels")).select("image_id", "cls_id", "cls_name").distinct() \
        .join(images, "image_id")
    fc = (ic.groupBy("cls_id", "cls_name").agg((F.count("*") / n_train).alias("f_c"))
          .withColumn("r_c", F.greatest(F.lit(1.0), F.sqrt(F.lit(e["rfs_threshold"]) / F.col("f_c")))))
    ri = ic.join(fc.select("cls_id", "r_c"), "cls_id").groupBy("image_id").agg(F.max("r_c").alias("r"))
    u = (F.abs(F.xxhash64("image_id", F.lit(seed))) % 10000) / 10000.0
    reps = (images.join(ri, "image_id", "left").fillna({"r": 1.0})
            .withColumn("r", F.least(F.col("r"), F.lit(float(e["rfs_max_repeat"]))))
            .withColumn("repeats", (F.floor("r") + (u < (F.col("r") - F.floor("r"))).cast("int")).cast("int"))
            .toPandas())

    # effective class frequency after resampling (shows what RFS actually did)
    eff = (ic.join(spark.createDataFrame(reps[["image_id", "repeats"]]), "image_id")
           .groupBy("cls_name").agg(F.sum("repeats").alias("eff_images")).toPandas())
    fcp = fc.toPandas().merge(eff, on="cls_name").sort_values("cls_id")
    fcp["eff_share"] = fcp.eff_images / reps.repeats.sum()
    fcp.to_csv(report_path(cfg, "03_rfs_class_factors.csv"), index=False)

    dst = root / "train_rfs"
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("images", "labels"):
        (dst / sub).mkdir(parents=True)
    for iid, k in zip(reps.image_id, reps.repeats):
        for j in range(int(k)):
            for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                src, link = root / "train" / sub / f"{iid}{ext}", dst / sub / f"{iid}__r{j}{ext}"
                try:
                    os.symlink(src, link)
                except OSError:  # e.g. Windows without symlink rights
                    shutil.copy2(src, link)
    return {"train_images": n_train, "train_rfs_images": int(reps.repeats.sum())}


def write_yaml(cfg):
    root = cfg["paths"]["yolo_root"]
    base = {"path": root, "train": "train_rfs/images", "val": "val/images", "test": "test/images",
            "names": dict(enumerate(CLASS_NAMES))}
    with open(Path(root, "data.yaml"), "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    with open(Path(root, "data_cctv.yaml"), "w") as f:
        yaml.safe_dump({**base, "test": "test_cctv/images"}, f, sort_keys=False)


def main():
    cfg = load_config(parse_args().config)
    spark = get_spark(cfg)
    n, n_err = export_images(spark, cfg)
    rfs = repeat_factor_sampling(spark, cfg)
    write_yaml(cfg)
    summary = {"exported": n, "export_errors": n_err, **rfs}
    write_json(summary, report_path(cfg, "03_export_summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
