"""Stage 4 — train the PPE detector (GPU; Spark is not used for training) and evaluate it.

Outputs
  models/best.pt
  reports/04_detector_eval.json     overall metrics on clean test and CCTV-degraded test
  reports/04_per_class_ap.csv       per-class AP, clean vs CCTV, and the robustness drop
Usage
  python pipelines/p04_train.py                       # train + evaluate
  python pipelines/p04_train.py --eval-only --weights models/best.pt
  python pipelines/p04_train.py --name ablation_640 --imgsz 640   # quick ablations
"""
import _bootstrap  # noqa: F401
import os
import shutil
from pathlib import Path

import pandas as pd

from safety.config import load_config, parse_args, report_path, write_json


def _args(p):
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--model", default=None)


def evaluate(weights: str, data_yaml: str, imgsz: int, device, tag: str) -> tuple[dict, pd.DataFrame]:
    from ultralytics import YOLO
    m = YOLO(weights)
    met = m.val(data=data_yaml, split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
    names = m.names
    per = pd.DataFrame({
        "class": [names[int(c)] for c in met.box.ap_class_index],
        f"ap50_{tag}": [float(x) for x in met.box.ap50],
        f"ap50_95_{tag}": [float(x) for x in met.box.ap],
    })
    summary = {"map50": float(met.box.map50), "map50_95": float(met.box.map),
               "precision": float(met.box.mp), "recall": float(met.box.mr)}
    return summary, per


def main():
    args = parse_args(_args)
    cfg = load_config(args.config)
    t, paths = dict(cfg["train"]), cfg["paths"]
    if args.imgsz:
        t["imgsz"] = args.imgsz
    if args.epochs:
        t["epochs"] = args.epochs
    if args.model:
        t["model"] = args.model
    root = Path(paths["yolo_root"])
    data_yaml, cctv_yaml = str(root / "data.yaml"), str(root / "data_cctv.yaml")
    run_name = args.name or f"{Path(t['model']).stem}_img{t['imgsz']}"

    os.environ.setdefault("MLFLOW_TRACKING_URI", Path(paths["runs"], "mlruns").resolve().as_uri())
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", t["mlflow_experiment"])
    os.environ.setdefault("MLFLOW_RUN", run_name)

    if args.eval_only:
        weights = args.weights or cfg["inference"]["weights"]
    else:
        from ultralytics import YOLO, settings
        try:
            settings.update({"mlflow": True})  # Ultralytics' built-in MLflow callback logs params/metrics/artifacts
        except Exception:
            pass
        model = YOLO(t["model"])
        model.train(
            data=data_yaml, imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"], patience=t["patience"],
            workers=t["workers"], device=t["device"], close_mosaic=t["close_mosaic"], mosaic=t["mosaic"],
            mixup=t["mixup"], scale=t["scale"], degrees=t["degrees"], fliplr=t["fliplr"], hsv_v=t["hsv_v"],
            cos_lr=t["cos_lr"], seed=cfg["data"]["seed"], deterministic=True,
            project=paths["runs"], name=run_name, exist_ok=True, plots=True,
        )
        weights = str(model.trainer.best)
        Path(paths["models"]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, Path(paths["models"], "best.pt"))
        shutil.copy2(weights, Path(paths["models"], f"{run_name}.pt"))

    clean, per_clean = evaluate(weights, data_yaml, t["imgsz"], t["device"], "clean")
    result = {"run": run_name, "weights": weights, "imgsz": t["imgsz"], "test_clean": clean}
    per = per_clean
    if Path(root, "test_cctv", "images").exists() and any(Path(root, "test_cctv", "images").iterdir()):
        cctv, per_cctv = evaluate(weights, cctv_yaml, t["imgsz"], t["device"], "cctv")
        result["test_cctv"] = cctv
        result["map50_95_drop_under_cctv"] = clean["map50_95"] - cctv["map50_95"]
        per = per_clean.merge(per_cctv, on="class", how="outer")
        per["ap50_95_drop"] = per["ap50_95_clean"] - per["ap50_95_cctv"]

    per.sort_values("ap50_95_clean").to_csv(report_path(cfg, f"04_per_class_ap_{run_name}.csv"), index=False)
    write_json(result, report_path(cfg, f"04_detector_eval_{run_name}.json"))
    try:  # attach evaluation to the same MLflow experiment
        import mlflow
        mlflow.set_experiment(t["mlflow_experiment"])
        with mlflow.start_run(run_name=f"{run_name}_test_eval"):
            mlflow.log_params({"weights": weights, "imgsz": t["imgsz"]})
            mlflow.log_metrics({f"clean_{k}": v for k, v in clean.items()})
            if "test_cctv" in result:
                mlflow.log_metrics({f"cctv_{k}": v for k, v in result["test_cctv"].items()})
            mlflow.log_artifact(report_path(cfg, f"04_per_class_ap_{run_name}.csv"))
    except Exception as e:
        print(f"(mlflow logging skipped: {e})")
    print(result)
    print(per.to_string(index=False))


if __name__ == "__main__":
    main()
