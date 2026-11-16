"""Person-level PPE compliance from detections, in pure Spark SQL.

Detections schema (predictions or ground truth): image_id, det_id, cls_name, conf, x1, y1, x2, y2.

Step 1  assign every body-part / PPE box to the person box that contains most of it.
Step 2  for each requirement, check whether each visible body part is covered by matching PPE.
Step 3  derive a status per (person, requirement):
          worn        PPE present and every visible matching body part is covered
          partial     some parts covered, some bare (e.g. one glove)        [multi-instance parts]
          not_worn    PPE assigned to the person but NOT on the body part (helmet carried in hand)
          missing     body part visible, no PPE at all
          not_visible body part not visible and no PPE -> no evidence either way (never a violation)
Body-protection has no anchor part: if the person is visible and has no vest/suit -> missing.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from safety import geometry
from safety.config import VIOLATION_STATUSES

_XY = ["x1", "y1", "x2", "y2"]


def assign_to_persons(dets: DataFrame, cfg: dict) -> tuple[DataFrame, DataFrame]:
    c = cfg["compliance"]
    reqs = c["requirements"].values()
    relevant = sorted({p for r in reqs for p in r["ppe"]} | {r["part"] for r in reqs if r.get("part")})

    persons = (dets.filter(F.col("cls_name") == "person")
               .withColumn("area", (F.col("x2") - F.col("x1")) * (F.col("y2") - F.col("y1")))
               .filter(F.col("area") >= c["min_person_area"])
               .select("image_id", *[F.col(k).alias(f"p_{k}") for k in ["det_id", "conf", *_XY, "area"]]))

    objs = (dets.filter(F.col("cls_name").isin(relevant))
            .select("image_id", *[F.col(k).alias(f"o_{k}") for k in ["det_id", "cls_name", "conf", *_XY]]))

    cand = (objs.join(persons, "image_id")
            .withColumn("ioa", geometry.ioa("o_", "p_"))
            .filter(F.col("ioa") >= c["person_ioa"]))
    # overlapping workers: give the object to the person containing most of it; ties -> the smaller person
    w = Window.partitionBy("o_det_id").orderBy(F.desc("ioa"), F.asc("p_area"))
    assigned = cand.withColumn("_rk", F.row_number().over(w)).filter("_rk = 1").drop("_rk")
    return persons, assigned


def _status(r: dict) -> F.Column:
    n_ppe, n_parts, n_unc = F.col("n_ppe"), F.col("n_parts"), F.col("n_uncovered")
    if not r.get("part"):
        return F.when(n_ppe > 0, "worn").otherwise("missing")
    covered = n_parts - n_unc
    return (F.when(n_parts == 0, F.when(n_ppe > 0, "worn").otherwise("not_visible"))
            .when(n_unc == 0, "worn")
            .when(covered > 0, F.lit("partial" if r.get("multi") else "worn"))
            .when(n_ppe > 0, "not_worn")
            .otherwise("missing"))


def compute_compliance(persons: DataFrame, assigned: DataFrame, cfg: dict) -> DataFrame:
    """Long format: one row per (person, requirement)."""
    thr = cfg["compliance"]["cover_overlap"]
    base_p = persons.select("image_id", "p_det_id", "p_conf")
    out = []
    for name, r in cfg["compliance"]["requirements"].items():
        ppe = assigned.filter(F.col("o_cls_name").isin(r["ppe"]))
        n_ppe = ppe.groupBy("p_det_id").agg(F.count("*").alias("n_ppe"))
        df = base_p.join(n_ppe, "p_det_id", "left")
        if r.get("part"):
            parts = assigned.filter(F.col("o_cls_name") == r["part"]).select(
                "p_det_id", F.col("o_det_id").alias("part_id"), *[F.col(f"o_{k}").alias(f"a_{k}") for k in _XY])
            ppe_b = ppe.select("p_det_id", *[F.col(f"o_{k}").alias(f"b_{k}") for k in _XY])
            cov = (parts.join(ppe_b, "p_det_id", "left")
                   .withColumn("ov", F.coalesce(geometry.overlap_coef("a_", "b_"), F.lit(0.0)))
                   .groupBy("p_det_id", "part_id").agg(F.max("ov").alias("best"))
                   .groupBy("p_det_id").agg(F.count("*").alias("n_parts"),
                                            F.sum((F.col("best") < thr).cast("int")).alias("n_uncovered")))
            df = df.join(cov, "p_det_id", "left")
        else:
            df = df.withColumn("n_parts", F.lit(0)).withColumn("n_uncovered", F.lit(0))
        df = (df.fillna(0, ["n_ppe", "n_parts", "n_uncovered"])
              .withColumn("requirement", F.lit(name))
              .withColumn("status", _status(r)))
        out.append(df.select("image_id", F.col("p_det_id").alias("person_id"),
                             F.col("p_conf").cast("double").alias("person_conf"), "requirement", "status",
                             F.col("n_parts").cast("long"), F.col("n_uncovered").cast("long"),
                             F.col("n_ppe").cast("long")))
    res = out[0]
    for d in out[1:]:
        res = res.unionByName(d)
    return res


def wide_summary(comp: DataFrame) -> DataFrame:
    reqs = sorted(r.requirement for r in comp.select("requirement").distinct().collect())
    return (comp.groupBy("image_id", "person_id", "person_conf").pivot("requirement", reqs)
            .agg(F.first("status"))
            .withColumn("n_violations", sum(F.col(r).isin(VIOLATION_STATUSES).cast("int") for r in reqs)))


# --------------------------------------------------------------------------- evaluation vs ground truth
def match_persons(pred_p: DataFrame, gt_p: DataFrame, iou_thr: float) -> DataFrame:
    """Mutual-best IoU matching between predicted and GT person boxes."""
    a = pred_p.select("image_id", F.col("p_det_id").alias("pred_id"), *[F.col(f"p_{k}").alias(f"a_{k}") for k in _XY])
    b = gt_p.select("image_id", F.col("p_det_id").alias("gt_id"), *[F.col(f"p_{k}").alias(f"b_{k}") for k in _XY])
    j = a.join(b, "image_id").withColumn("iou", geometry.iou("a_", "b_")).filter(F.col("iou") >= iou_thr)
    ra = F.row_number().over(Window.partitionBy("pred_id").orderBy(F.desc("iou")))
    rb = F.row_number().over(Window.partitionBy("gt_id").orderBy(F.desc("iou")))
    return (j.withColumn("ra", ra).withColumn("rb", rb).filter("ra = 1 AND rb = 1")
            .select("image_id", "pred_id", "gt_id", "iou"))


def evaluate(pred_comp: DataFrame, gt_comp: DataFrame, matches: DataFrame) -> tuple[DataFrame, pd.DataFrame]:
    """Violation-detection precision/recall per requirement.
    Missed GT persons count as false negatives, hallucinated persons as false positives."""
    g = gt_comp.select(F.col("person_id").alias("gt_id"), "requirement", F.col("status").alias("gt_status"))
    p = pred_comp.select(F.col("person_id").alias("pred_id"), "requirement", F.col("status").alias("pred_status"))
    m = matches.select("pred_id", "gt_id")
    matched = g.join(m, "gt_id").join(p, ["pred_id", "requirement"], "left")
    missed = g.join(m, "gt_id", "left_anti").withColumn("pred_status", F.lit("person_missed"))
    halluc = p.join(m, "pred_id", "left_anti").withColumn("gt_status", F.lit("no_person"))
    cols = ["requirement", "gt_status", "pred_status"]
    rows = matched.select(cols).unionByName(missed.select(cols)).unionByName(halluc.select(cols))
    confusion = rows.groupBy(cols).count()

    rows = (rows.withColumn("gt_v", F.col("gt_status").isin(VIOLATION_STATUSES))
            .withColumn("pr_v", F.col("pred_status").isin(VIOLATION_STATUSES)))
    agg = [F.sum((F.col("gt_v") & F.col("pr_v")).cast("int")).alias("tp"),
           F.sum((~F.col("gt_v") & F.col("pr_v")).cast("int")).alias("fp"),
           F.sum((F.col("gt_v") & ~F.col("pr_v")).cast("int")).alias("fn"),
           F.sum((F.col("gt_status") == F.col("pred_status")).cast("int")).alias("exact_status_match"),
           F.count("*").alias("n")]
    per = rows.groupBy("requirement").agg(*agg).toPandas()
    tot = rows.agg(*agg).toPandas().assign(requirement="ALL (micro)")
    met = pd.concat([per.sort_values("requirement"), tot], ignore_index=True)
    met["precision"] = met.tp / (met.tp + met.fp).where(lambda s: s > 0)
    met["recall"] = met.tp / (met.tp + met.fn).where(lambda s: s > 0)
    met["f1"] = 2 * met.precision * met.recall / (met.precision + met.recall)
    met["status_accuracy"] = met.exact_status_match / met.n
    return confusion, met
