"""Stage 6 — detections -> person-level PPE compliance, evaluated against ground truth.

gold/person_compliance   long: (person, requirement, status)
gold/person_summary      wide: one row per person, one column per requirement
reports/06_compliance_metrics.csv     violation precision / recall / F1 per requirement (test split)
reports/06_compliance_confusion.csv   GT status x predicted status
reports/06_gt_violation_profile.csv   how many real violations SH17 contains (from GT boxes)
"""
import _bootstrap  # noqa: F401
from pyspark.sql import functions as F

from safety import compliance as C
from safety.config import VIOLATION_STATUSES, lake, load_config, parse_args, report_path
from safety.spark import get_spark, write


def gt_detections(spark, cfg, split):
    imgs = spark.read.parquet(lake(cfg, "silver", "images")).filter(F.col("split") == split).select("image_id")
    lab = spark.read.parquet(lake(cfg, "silver", "labels")).join(imgs, "image_id")
    return lab.select("image_id",
                      F.concat_ws("_", "image_id", F.lit("gt"), F.monotonically_increasing_id().cast("string")).alias("det_id"),
                      "cls_name", F.lit(1.0).alias("conf"), "x1", "y1", "x2", "y2")


def main():
    cfg = load_config(parse_args().config)
    spark = get_spark(cfg)
    dets = spark.read.parquet(lake(cfg, "gold", "detections"))

    persons, assigned = C.assign_to_persons(dets, cfg)
    comp = C.compute_compliance(persons, assigned, cfg)
    write(comp, lake(cfg, "gold", "person_compliance"))
    comp = spark.read.parquet(lake(cfg, "gold", "person_compliance"))
    write(C.wide_summary(comp), lake(cfg, "gold", "person_summary"))
    write(persons, lake(cfg, "gold", "persons"))

    # ---- evaluation on the held-out test split
    split = cfg["inference"]["eval_split"]
    test_ids = spark.read.parquet(lake(cfg, "silver", "images")).filter(F.col("split") == split).select("image_id")
    gt = gt_detections(spark, cfg, split)
    gt_p, gt_a = C.assign_to_persons(gt, cfg)
    gt_comp = C.compute_compliance(gt_p, gt_a, cfg).cache()
    pred_p = persons.join(test_ids, "image_id")
    pred_comp = comp.join(test_ids, "image_id")
    matches = C.match_persons(pred_p, gt_p, cfg["compliance"]["person_match_iou"])
    confusion, metrics = C.evaluate(pred_comp, gt_comp, matches)

    confusion.toPandas().sort_values(["requirement", "gt_status", "pred_status"]) \
        .to_csv(report_path(cfg, "06_compliance_confusion.csv"), index=False)
    metrics.to_csv(report_path(cfg, "06_compliance_metrics.csv"), index=False)
    prof = (gt_comp.groupBy("requirement").pivot("status").count().fillna(0).toPandas())
    prof.to_csv(report_path(cfg, "06_gt_violation_profile.csv"), index=False)

    print("GT status distribution (test):\n", prof.to_string(index=False))
    print("\nViolation detection metrics:\n", metrics.round(3).to_string(index=False))
    n_pred = pred_p.count()
    print(f"\npersons: pred={n_pred} gt={gt_p.count()} matched={matches.count()}")
    print("predicted violations (all scored images):")
    comp.filter(F.col("status").isin(VIOLATION_STATUSES)).groupBy("requirement", "status").count() \
        .orderBy("requirement").show(50, truncate=False)


if __name__ == "__main__":
    main()
