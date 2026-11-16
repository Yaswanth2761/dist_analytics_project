"""Plant-context simulation and person-level risk scoring (shared by batch and streaming jobs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

SHIFTS = {"morning": 6, "evening": 14, "night": 22}  # start hour, 8 h each

PERSON_RISK_SCHEMA = (
    "person_id string, image_id string, zone string, camera_id string, shift string, ts timestamp, "
    "hazard double, person_conf double, raw_score double, n_checked long, n_violations long, "
    "violations array<string>, risk double, risk_level string")


def cameras(cfg: dict) -> list[tuple[str, str]]:
    return [(z, f"{z}_cam{k + 1}") for z in cfg["risk"]["zones"] for k in range(cfg["risk"]["cameras_per_zone"])]


def shift_of(ts: Column) -> Column:
    h = F.hour(ts)
    return F.when((h >= 6) & (h < 14), "morning").when((h >= 14) & (h < 22), "evening").otherwise("night")


def simulate_context(img: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """img: image_id, n_violations. Assigns zone/camera/shift/timestamp to each image.
    Images that contain violations are `strength`x more likely to land in the planted (zone, shift),
    giving the analytics a known pattern to recover."""
    sim = cfg["risk"]["simulation"]
    rng = np.random.default_rng(sim["seed"])
    zones, shifts = list(cfg["risk"]["zones"]), list(SHIFTS)
    cells = [(z, s) for z in zones for s in shifts]
    planted = (sim["planted"]["zone"], sim["planted"]["shift"])
    w_clean = np.ones(len(cells))
    w_viol = np.array([sim["planted"]["strength"] if c == planted else 1.0 for c in cells])
    cams = {z: [c for zz, c in cameras(cfg) if zz == z] for z in zones}
    start = pd.Timestamp(sim["start"])

    img = img.sort_values("image_id").reset_index(drop=True)  # deterministic
    rows = []
    for iid, nv in zip(img["image_id"], img["n_violations"]):
        w = w_viol if nv > 0 else w_clean
        z, s = cells[rng.choice(len(cells), p=w / w.sum())]
        day = int(rng.integers(sim["days"]))
        ts = start + pd.Timedelta(days=day, hours=SHIFTS[s]) + pd.Timedelta(seconds=int(rng.integers(8 * 3600)))
        rows.append((iid, z, rng.choice(cams[z]), s, ts.to_pydatetime()))
    return pd.DataFrame(rows, columns=["image_id", "zone", "camera_id", "shift", "ts"])


def zone_requirements(spark: SparkSession, cfg: dict) -> DataFrame:
    sev = cfg["risk"]["severity"]
    rows = [(z, float(zc["hazard"]), r, float(sev[r])) for z, zc in cfg["risk"]["zones"].items() for r in zc["required"]]
    return spark.createDataFrame(rows, "zone string, hazard double, requirement string, severity double")


def _lookup(d: dict) -> Column:
    return F.create_map(*[F.lit(x) for kv in d.items() for x in kv])


def score_persons(comp: DataFrame, ctx: DataFrame, zone_req: DataFrame, cfg: dict) -> DataFrame:
    """comp: long compliance table. ctx: image_id, zone, camera_id, shift, ts.
    risk = zone hazard x sum_{required r} severity_r x status_weight(status_r) x person detection confidence."""
    sw, lv = _lookup(cfg["risk"]["status_weight"]), cfg["risk"]["levels"]
    c = (comp.join(ctx, "image_id")
         .join(zone_req, ["zone", "requirement"])       # only requirements mandated in that zone count
         .withColumn("sw", F.coalesce(sw[F.col("status")], F.lit(0.0)))
         .withColumn("contrib", F.col("severity") * F.col("sw")))
    p = (c.groupBy("person_id", "image_id", "zone", "camera_id", "shift", "ts", "hazard", "person_conf")
         .agg(F.sum("contrib").alias("raw_score"),
              F.count(F.when(F.col("status") != "not_visible", 1)).alias("n_checked"),
              F.count(F.when(F.col("sw") > 0, 1)).alias("n_violations"),
              F.array_sort(F.collect_list(F.when(F.col("sw") > 0, F.col("requirement")))).alias("violations"))
         .withColumn("risk", F.col("hazard") * F.col("raw_score") * F.col("person_conf"))
         .withColumn("risk_level", F.when(F.col("risk") >= lv["high"], "high")
                     .when(F.col("risk") >= lv["medium"], "medium")
                     .when(F.col("risk") > 0, "low").otherwise("none")))
    cols = [c.split()[0] for c in PERSON_RISK_SCHEMA.split(", ")]
    return p.select(*cols)


def requirement_rates(comp: DataFrame, ctx: DataFrame, zone_req: DataFrame, keys: list[str]) -> DataFrame:
    """Violation rate per requirement among persons where the requirement was observable."""
    from safety.config import VIOLATION_STATUSES
    c = comp.join(ctx, "image_id").join(zone_req, ["zone", "requirement"]).filter(F.col("status") != "not_visible")
    return (c.groupBy(*keys, "requirement")
            .agg(F.count("*").alias("n_observed"),
                 F.sum(F.col("status").isin(VIOLATION_STATUSES).cast("int")).alias("n_violations"))
            .withColumn("violation_rate", F.col("n_violations") / F.col("n_observed")))
