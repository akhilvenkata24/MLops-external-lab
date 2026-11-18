import json
import pickle
import re
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


def pick_best_model(model_comparison_path: Path) -> dict:
    comparison = pd.read_csv(model_comparison_path)
    best_row = comparison.sort_values("MAE", ascending=True).iloc[0]
    return {
        "name": str(best_row["Model"]),
        "mae": float(best_row["MAE"]),
        "rmse": float(best_row["RMSE"]),
        "r2": float(best_row["R2_Score"]),
    }


def load_metadata(metadata_path: Path) -> dict:
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def sanitize_metric_token(value: str) -> str:
    token = value.strip().lower().replace(" ", "_")
    token = re.sub(r"[^a-z0-9_\-./]", "", token)
    return token


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "best_model.pkl"
    metadata_path = root / "models" / "metadata.json"
    comparison_path = root / "model_comparison.csv"
    weather_comparison_path = root / "weather_comparison.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {comparison_path}")

    best = pick_best_model(comparison_path)
    metadata = load_metadata(metadata_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = metadata.get("features", [])

    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Kalamkari_Drying_Time")

    with mlflow.start_run(run_name="ridge_best_model") as run:
        run_id = run.info.run_id

        mlflow.log_param("selected_model", best["name"])
        mlflow.log_param("model_file", str(model_path.name))
        mlflow.log_param("feature_count", len(features))

        mlflow.log_metric("mae", best["mae"])
        mlflow.log_metric("rmse", best["rmse"])
        mlflow.log_metric("r2_score", best["r2"])

        if weather_comparison_path.exists():
            weather_df = pd.read_csv(weather_comparison_path)
            for _, row in weather_df.iterrows():
                label = sanitize_metric_token(str(row["Model"]))
                mlflow.log_metric(f"weather_{label}_mae", float(row["MAE"]))
                mlflow.log_metric(f"weather_{label}_rmse", float(row["RMSE"]))
                # CSV column can be R2 or R2_Score depending on source.
                if "R2" in row:
                    mlflow.log_metric(f"weather_{label}_r2", float(row["R2"]))
                elif "R2_Score" in row:
                    mlflow.log_metric(f"weather_{label}_r2", float(row["R2_Score"]))

        mlflow.sklearn.log_model(model, artifact_path="model")

    model_name = "KalamkariDryingTimeBestModel"
    client = MlflowClient()

    try:
        client.create_registered_model(model_name)
    except Exception:
        # Model may already exist in local registry.
        pass

    model_source = f"runs:/{run_id}/model"
    model_version = client.create_model_version(name=model_name, source=model_source, run_id=run_id)

    metadata["mlflow"] = {
        "tracking_uri": tracking_uri,
        "experiment": "Kalamkari_Drying_Time",
        "run_id": run_id,
        "registered_model": model_name,
        "model_version": model_version.version,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"mlflow_run_id={run_id}")
    print(f"registered_model={model_name}")
    print(f"model_version={model_version.version}")


if __name__ == "__main__":
    main()
