# PPE Safety & Risk Analytics for Manufacturing (SH17 + Apache Spark)

Computer-vision PPE compliance monitoring with a Spark data platform around it: ingestion and
data-quality checks, leakage-safe splits, distributed inference, person-level compliance logic,
plant risk analytics with MLlib, and a Structured Streaming layer for live alerts.

## Layout

```
configs/config.yaml          every threshold, path and hyper-parameter (single source of truth)
safety/                      shared library (imported by all stages and by Spark workers)
  config.py                  SH17 taxonomy, config loading/validation, report helpers
  spark.py                   SparkSession tuned for image workloads
  geometry.py                NULL-safe box IoU / IoA / overlap as Spark Column expressions
  io_images.py               worker functions: metadata, pHash, resize/export, CCTV degradation
  detector.py                YOLO inference inside mapInPandas (one model per worker)
  compliance.py              person <-> PPE association, coverage, status, evaluation vs GT
  risk.py                    plant-context simulation, risk scoring, rate tables
pipelines/
  p01_ingest.py              raw -> bronze (flag, never drop)
  p02_quality_split.py       bronze -> silver: cleaning, EDA, near-duplicates, stratified split
  p03_export_yolo.py         silver -> YOLO dataset, repeat-factor sampling, CCTV test set
  p04_train.py               YOLO training + clean/CCTV evaluation (+ MLflow)
  p05_batch_inference.py     Spark distributed inference -> gold/detections (+ throughput benchmark)
  p06_compliance.py          detections -> person compliance; violation precision/recall vs GT
  p07_risk_analytics.py      risk scores, zone/shift KPIs, anomaly windows, KMeans profiles
  p08_stream_simulator.py    simulated plant cameras (atomic JSON event files)
  p08_streaming.py           Structured Streaming: scoring + alerts + watermarked windows
dashboard/app.py             Streamlit dashboard over the gold tables
```

## Setup

1. Java 17 (or 11) and `JAVA_HOME` set. Spark 3.5's Arrow path can fail on some Java 21 builds.
2. `pip install -r requirements.txt`
3. Download SH17 from https://github.com/ahmadmughees/SH17dataset into `data/SH17/` so that you
   have `data/SH17/images/*.jpg` and `data/SH17/labels/*.txt`.
4. Run `./run_all.sh`, or run the stages one by one.

On Colab/Kaggle, run stages 1–3 and 5–7 on the same machine as training, or train there and copy
`models/best.pt` back. For CPU-only inference, set `inference.device: "cpu"`, `half: false`, and
raise `concurrency`.

## How the design follows from the SH17 data

| Dataset fact | Design response |
|---|---|
| Images are 1920–8192 px, several MB each | Spark moves **paths, not bytes**. Header-only metadata reads, JPEG `draft()` downscaling, a one-time export to 1600 px, and small Arrow batches. |
| ~40k of ~76k boxes are < 1% of the image area (ears, glasses, gloves) | Train at `imgsz=1280`. EDA reports the share of boxes under 8 px at that resolution, per class. |
| Strong class imbalance (face-guard, medical-suit, earmuffs are rare) | Splits stratified on each image's rarest class; LVIS repeat-factor sampling; per-class AP reported. |
| Images scraped from Pexels may contain near-duplicates | 64-bit pHash with 8-band LSH (exact for Hamming ≤ 7), union-find groups; splits are made **per group** to avoid leakage. |
| Some images carry EXIF rotation | Detected in bronze and baked into exported images (Ultralytics convention). Spot-check these. |
| Annotations include **body parts** (head, face, ear, hands, foot) as well as PPE | Compliance is anchored on visible body parts: "head visible and no helmet on it" is a violation, and a hidden head counts as `not_visible`, not a violation. |
| Stock photos, not CCTV | A degraded "CCTV" copy of the test set (blur, low light, noise, JPEG) measures the robustness gap. |
| No cameras, zones or timestamps | Plant context is **simulated** with a planted hotspot, and stage 7 checks whether the analytics recover it. |

## Compliance statuses (per person × requirement)

`worn` · `partial` (some parts covered, e.g. one glove) · `not_worn` (PPE carried, not on the body part)
· `missing` (part visible, no PPE) · `not_visible` (no evidence, never counted as a violation).

The same logic runs on ground-truth boxes, so stage 6 reports **violation-detection precision and
recall per requirement**. Missed persons count as false negatives and hallucinated persons as false
positives. This is the metric that matters to a safety officer, not mAP.

## Risk score

`risk = zone_hazard × Σ_{requirements mandated in the zone} severity × status_weight × person_confidence`

Levels (`medium`, `high`) and all weights live in `config.yaml`.

## Suggested experiments for the report

1. Ablations: `p04_train.py --name img640 --imgsz 640` vs 1280; with vs without repeat-factor sampling (point `train:` at `train/images`).
2. Clean vs CCTV per-class AP drop (`reports/04_per_class_ap_*.csv`).
3. Spark scaling: `p05_batch_inference.py --device cpu --concurrency {1,2,4,8}` → `reports/05_inference_benchmark.csv`.
4. Sensitivity of compliance F1 to `cover_overlap` and `person_ioa`.
5. Planted-pattern recovery (`reports/07_planted_pattern_check.json`): lift of the planted cell vs `strength`.
