"""SparkSession factory tuned for image workloads."""
from __future__ import annotations

import os
import sys

from pyspark.sql import DataFrame, SparkSession

from safety.config import REPO_ROOT

_JAVA_OPTS = ("-Dio.netty.tryReflectionSetAccessible=true "
              "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED")


def get_spark(cfg: dict, master: str | None = None, extra: dict | None = None) -> SparkSession:
    # Python workers inherit PYTHONPATH from the JVM's environment, so the `safety` package must be on it
    # BEFORE the JVM starts; otherwise mapInPandas UDFs defined in safety.* fail to unpickle.
    pp = os.environ.get("PYTHONPATH", "")
    if str(REPO_ROOT) not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([pp] if pp else []))

    s = cfg["spark"]
    b = (SparkSession.builder.appName(s["app_name"])
         .master(master or s["master"])
         .config("spark.driver.memory", s["driver_memory"])
         .config("spark.sql.shuffle.partitions", s["shuffle_partitions"])
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         # small Arrow batches: each row of an image UDF may decode a multi-megapixel image
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", s["arrow_batch_images"])
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.python.worker.reuse", "true")  # keeps the YOLO model cached across batches
         # Arrow (pandas UDFs) on Java 17/21 needs reflective access to direct buffers
         .config("spark.driver.extraJavaOptions", _JAVA_OPTS)
         .config("spark.executor.extraJavaOptions", _JAVA_OPTS))
    if sys.platform == "darwin":
        # macOS: forked Python workers cannot use Metal (MPS), so start workers fresh instead of forking
        b = b.config("spark.python.use.daemon", "false")
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def write(df: DataFrame, path: str, mode: str = "overwrite", partition_by: list[str] | None = None) -> None:
    w = df.write.mode(mode)
    if partition_by:
        w = w.partitionBy(*partition_by)
    w.parquet(path)
