"""Safety & risk dashboard over the gold tables.   Run:  streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety.config import REPO_ROOT, load_config  # noqa: E402

cfg = load_config(REPO_ROOT / "configs" / "config.yaml")
LAKE, REPORTS = Path(cfg["paths"]["lake"]), Path(cfg["paths"]["reports"])
st.set_page_config(page_title="Plant PPE Safety", layout="wide")


@st.cache_data(ttl=30)
def load(layer: str, name: str) -> pd.DataFrame:
    p = LAKE / layer / name
    return pd.read_parquet(p) if p.exists() and any(p.rglob("*.parquet")) else pd.DataFrame()


st.title("Manufacturing PPE safety & risk analytics")
st.caption("Detector + compliance results are real (SH17). Zones, cameras and timestamps are SIMULATED.")

pr = load("gold", "person_risk")
if pr.empty:
    st.warning("Run pipelines p01–p07 first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Persons analysed", f"{len(pr):,}")
c2.metric("With ≥1 violation", f"{(pr.n_violations > 0).mean():.1%}")
c3.metric("High-risk persons", int((pr.risk_level == "high").sum()))
c4.metric("Mean risk", f"{pr.risk.mean():.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Zones", "Trends & anomalies", "Patterns", "Model quality", "Live stream"])

with tab1:
    zs = load("gold", "zone_shift_rates")
    if not zs.empty:
        agg = zs.groupby(["zone", "requirement"], as_index=False)[["n_violations", "n_observed"]].sum()
        agg["violation_rate"] = agg.n_violations / agg.n_observed
        st.subheader("Violation rate by zone and requirement")
        st.altair_chart(alt.Chart(agg).mark_rect().encode(
            x="requirement:N", y="zone:N", color=alt.Color("violation_rate:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["zone", "requirement", alt.Tooltip("violation_rate:Q", format=".1%"), "n_observed"]),
            use_container_width=True)
    kpi = pr.assign(violator=pr.n_violations > 0).groupby(["zone", "shift"], as_index=False).agg(
        persons=("person_id", "count"), violator_rate=("violator", "mean"), mean_risk=("risk", "mean"))
    st.subheader("Zone × shift")
    st.dataframe(kpi.sort_values("violator_rate", ascending=False), use_container_width=True)

with tab2:
    w = load("gold", "zone_windows")
    if not w.empty:
        st.altair_chart(alt.Chart(w).mark_line(point=True).encode(
            x="window_start:T", y="violation_rate:Q", color="zone:N",
            tooltip=["zone", "window_start", "persons", "violation_rate"]), use_container_width=True)
        st.subheader("Anomalous windows (rolling z-score)")
        st.dataframe(w[w.anomaly].sort_values("z", ascending=False), use_container_width=True)
    top = pr.explode("violations").dropna(subset=["violations"]).groupby(["zone", "violations"]).size()
    st.subheader("Most frequent violations")
    st.dataframe(top.rename("count").reset_index().sort_values("count", ascending=False).head(20))

with tab3:
    p = REPORTS / "07_planted_pattern_check.json"
    if p.exists():
        st.json(json.loads(p.read_text()))
    cc = REPORTS / "07_cluster_centers.csv"
    if cc.exists():
        st.subheader("KMeans cluster centres (violation-rate profiles)")
        st.dataframe(pd.read_csv(cc), use_container_width=True)

with tab4:
    for f in sorted(REPORTS.glob("04_per_class_ap_*.csv")):
        st.subheader(f"Detector — {f.stem.replace('04_per_class_ap_', '')}")
        st.dataframe(pd.read_csv(f), use_container_width=True)
    m = REPORTS / "06_compliance_metrics.csv"
    if m.exists():
        st.subheader("Violation detection vs ground truth (test split)")
        st.dataframe(pd.read_csv(m).round(3), use_container_width=True)
    b = REPORTS / "05_inference_benchmark.csv"
    if b.exists():
        st.subheader("Spark inference throughput")
        st.dataframe(pd.read_csv(b), use_container_width=True)

with tab5:
    sw, sa = load("gold", "stream_zone_windows"), load("gold", "stream_alerts")
    if sw.empty and sa.empty:
        st.info("Start p08_streaming.py and p08_stream_simulator.py to see live data.")
    if not sw.empty:
        st.altair_chart(alt.Chart(sw).mark_line(point=True).encode(
            x="window_start:T", y="violation_rate:Q", color="zone:N"), use_container_width=True)
    if not sa.empty:
        st.subheader("Latest alerts")
        st.dataframe(sa.sort_values("ts", ascending=False).head(50), use_container_width=True)
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
