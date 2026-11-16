#!/usr/bin/env bash
# End-to-end batch pipeline. Stage 4 needs a GPU; run it on Colab/Kaggle if your machine has none
# and copy models/best.pt back before stage 5.
set -euo pipefail
cd "$(dirname "$0")"
python pipelines/p01_ingest.py
python pipelines/p02_quality_split.py
python pipelines/p03_export_yolo.py
python pipelines/p04_train.py
python pipelines/p05_batch_inference.py
python pipelines/p06_compliance.py
python pipelines/p07_risk_analytics.py
echo "Done. Dashboard: streamlit run dashboard/app.py"
echo "Streaming demo: python pipelines/p08_streaming.py  &  python pipelines/p08_stream_simulator.py"
