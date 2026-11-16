"""Stage 7 — plant-level risk analytics (Spark SQL + MLlib).

gold/camera_events        simulated zone / camera / shift / timestamp per image (SH17 has none — say so!)
gold/person_risk          risk score per detected person
gold/zone_shift_rates     violation rate per zone x shift x requirement
gold/zone_windows         time-windowed zone KPIs + anomaly flags (rolling z-score)
gold/profile_clusters     KMeans clusters of camera-shift-day violation profiles
reports/07_planted_pattern_check.json   did the analytics recover the planted hotspot?
"""
import _bootstrap  # noqa: F401
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import Window
from pyspark.sql import functions as F

from safety import risk as R
from safety.config import VIOLATION_STATUSES, lake, load_config, parse_args, report_path, write_json
from safety.spark import get_spark, write


def build_context(spark, cfg, comp):
    scored = (spark.read.parquet(lake(cfg, "silver", "images"))
              .filter(F.col("split").isin(cfg["inference"]["splits"])).select("image_id"))
    nv = comp.filter(F.col("status").isin(VIOLATION_STATUSES)).groupBy("image_id").agg(F.count("*").alias("n_violations"))
    img = scored.join(nv, "image_id", "left").fillna({"n_violations": 0}).toPandas()
    ctx = spark.createDataFrame(R.simulate_context(img, cfg),
                                "image_id string, zone string, camera_id string, shift string, ts timestamp")
    return ctx


def windows_with_anomalies(person_risk, cfg):
    a = cfg["risk"]["anomaly"]
    win = (person_risk.groupBy("zone", F.window("ts", cfg["risk"]["window"]).alias("w"))
           .agg(F.count("*").alias("persons"),
                F.sum((F.col("n_violations") > 0).cast("int")).alias("violators"),
                F.avg("risk").alias("mean_risk"), F.max("risk").alias("max_risk"))
           .withColumn("window_start", F.col("w.start")).drop("w")
           .withColumn("violation_rate", F.col("violators") / F.col("persons")))
    hist = Window.partitionBy("zone").orderBy("window_start").rowsBetween(-a["lookback_windows"], -1)
    return (win.withColumn("base_mean", F.avg("violation_rate").over(hist))
            .withColumn("base_std", F.stddev("violation_rate").over(hist))
            .withColumn("z", (F.col("violation_rate") - F.col("base_mean")) / F.greatest(F.col("base_std"), F.lit(0.05)))
            .withColumn("anomaly", (F.col("z") >= a["z"]) & (F.col("persons") >= a["min_persons"])))


def cluster_profiles(spark, comp, ctx, zone_req, cfg):
    """Cluster camera x shift x day units by their per-requirement violation rates."""
    rates = R.requirement_rates(comp, ctx.withColumn("day", F.to_date("ts")), zone_req, ["camera_id", "zone", "shift", "day"])
    reqs = sorted(cfg["compliance"]["requirements"])
    prof = (rates.groupBy("camera_id", "zone", "shift", "day").pivot("requirement", reqs)
            .agg(F.first("violation_rate")).fillna(0.0))
    feats = VectorAssembler(inputCols=reqs, outputCol="raw").transform(prof)
    feats = StandardScaler(inputCol="raw", outputCol="features", withMean=True).fit(feats).transform(feats).cache()
    n = feats.count()
    best, scores = None, {}
    for k in cfg["risk"]["kmeans_k"]:
        if k >= n:
            continue
        model = KMeans(k=k, seed=cfg["data"]["seed"], featuresCol="features").fit(feats)
        s = ClusteringEvaluator(featuresCol="features").evaluate(model.transform(feats))
        scores[k] = s
        if best is None or s > best[1]:
            best = (model, s, k)
    if best is None:
        return None, scores
    out = best[0].transform(feats).drop("raw", "features")
    centers = out.groupBy("prediction").agg(*[F.round(F.avg(r), 3).alias(r) for r in reqs], F.count("*").alias("units"))
    centers.toPandas().to_csv(report_path(cfg, "07_cluster_centers.csv"), index=False)
    return out, scores


def main():
    cfg = load_config(parse_args().config)
    spark = get_spark(cfg)
    comp = spark.read.parquet(lake(cfg, "gold", "person_compliance")).cache()
    zone_req = R.zone_requirements(spark, cfg)

    ctx = build_context(spark, cfg, comp)
    write(ctx, lake(cfg, "gold", "camera_events"))
    ctx = spark.read.parquet(lake(cfg, "gold", "camera_events"))

    pr = R.score_persons(comp, ctx, zone_req, cfg)
    write(pr, lake(cfg, "gold", "person_risk"))
    pr = spark.read.parquet(lake(cfg, "gold", "person_risk")).cache()

    zs = R.requirement_rates(comp, ctx, zone_req, ["zone", "shift"])
    write(zs, lake(cfg, "gold", "zone_shift_rates"))

    zone_shift = (pr.groupBy("zone", "shift")
                  .agg(F.count("*").alias("persons"),
                       F.avg((F.col("n_violations") > 0).cast("int")).alias("violator_rate"),
                       F.avg("risk").alias("mean_risk"),
                       F.percentile_approx("risk", 0.95).alias("p95_risk"),
                       F.sum((F.col("risk_level") == "high").cast("int")).alias("high_risk_persons"))
                  .orderBy(F.desc("violator_rate")))
    zs_pdf = zone_shift.toPandas()
    zs_pdf.to_csv(report_path(cfg, "07_zone_shift_kpis.csv"), index=False)

    wins = windows_with_anomalies(pr, cfg)
    write(wins, lake(cfg, "gold", "zone_windows"))
    n_anom = spark.read.parquet(lake(cfg, "gold", "zone_windows")).filter("anomaly").count()

    clusters, sil = cluster_profiles(spark, comp, ctx, zone_req, cfg)
    if clusters is not None:
        write(clusters, lake(cfg, "gold", "profile_clusters"))

    # Planted-pattern recovery: lift = P(cell | image has violations) / P(cell | clean image).
    # Uniform assignment gives lift ~1; the planted cell should approach `strength`.
    planted = cfg["risk"]["simulation"]["planted"]
    viol_img = comp.filter(F.col("status").isin(VIOLATION_STATUSES)).select("image_id").distinct() \
        .withColumn("has_v", F.lit(1))
    cells = (ctx.join(viol_img, "image_id", "left").fillna({"has_v": 0})
             .groupBy("zone", "shift").agg(F.sum("has_v").alias("v"), F.sum(1 - F.col("has_v")).alias("c"))
             .toPandas())
    cells["lift"] = (cells.v / max(cells.v.sum(), 1)) / ((cells.c + 0.5) / (cells.c.sum() + 0.5 * len(cells)))
    cells = cells.sort_values("lift", ascending=False)
    cells.to_csv(report_path(cfg, "07_cell_lift.csv"), index=False)
    top = cells.iloc[0]
    check = {"planted": planted,
             "top_cell": {"zone": top["zone"], "shift": top["shift"], "lift": round(float(top["lift"]), 2)},
             "recovered": bool(top["zone"] == planted["zone"] and top["shift"] == planted["shift"]),
             "anomalous_windows": n_anom, "kmeans_silhouette": sil}
    write_json(check, report_path(cfg, "07_planted_pattern_check.json"))
    print(zs_pdf.round(3).to_string(index=False))
    print(check)


if __name__ == "__main__":
    main()
