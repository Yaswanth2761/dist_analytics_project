"""Box geometry as Spark Column expressions (normalized x1,y1,x2,y2 columns sharing a prefix)."""
from pyspark.sql import Column
from pyspark.sql import functions as F

_EPS = 1e-12  # plain float: Columns cannot be built before a SparkSession exists


def _c(prefix: str, k: str) -> Column:
    return F.col(f"{prefix}{k}")


def area(p: str) -> Column:
    return (_c(p, "x2") - _c(p, "x1")) * (_c(p, "y2") - _c(p, "y1"))


def inter_area(a: str, b: str) -> Column:
    """NULL if either box is NULL (e.g. after a left join). Needed because Spark's greatest/least
    silently SKIP nulls, which would make a missing box look like a perfect overlap."""
    present = F.lit(True)
    for k in ("x1", "y1", "x2", "y2"):
        present = present & _c(a, k).isNotNull() & _c(b, k).isNotNull()
    iw = F.greatest(F.lit(0.0), F.least(_c(a, "x2"), _c(b, "x2")) - F.greatest(_c(a, "x1"), _c(b, "x1")))
    ih = F.greatest(F.lit(0.0), F.least(_c(a, "y2"), _c(b, "y2")) - F.greatest(_c(a, "y1"), _c(b, "y1")))
    return F.when(present, iw * ih)


def ioa(a: str, b: str) -> Column:
    """Fraction of box `a` lying inside box `b` (used for part/PPE -> person assignment)."""
    return inter_area(a, b) / F.greatest(area(a), F.lit(_EPS))


def iou(a: str, b: str) -> Column:
    i = inter_area(a, b)
    return i / F.greatest(area(a) + area(b) - i, F.lit(_EPS))


def overlap_coef(a: str, b: str) -> Column:
    """Intersection over the SMALLER box. Glasses cover a small part of a face box and a helmet often
    extends past the head box; IoU would call both 'not covering', the overlap coefficient does not."""
    return inter_area(a, b) / F.greatest(F.least(area(a), area(b)), F.lit(_EPS))  # NULL-safe via inter_area
