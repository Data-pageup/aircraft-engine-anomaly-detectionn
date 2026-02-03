import logging
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG (relative to project root)
# ============================================================
DB_PATH = "engine_data.db"
TABLE_NAME = "engine_sensor_data"

MODEL_PATH = "isolation_forest_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    logger.info("Loading data from SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    logger.info(f"Loaded {len(df)} rows for {df['engine no'].nunique()} engines")
    return df

# ============================================================
# PREPROCESS + FEATURE ENGINEERING
# ============================================================
def preprocess_and_engineer(df):
    logger.info("Preprocessing data...")

    drop_cols = ["AP1", "AP2", "GWT", "Unnamed: 27"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["engine no", "datetime"])

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby("engine no")[col].ffill().bfill()

    logger.info("Engineering engine-level features...")

    engine_features = df.groupby("engine no").agg({
        "TGT": ["mean", "std", "max", "min", lambda x: x.quantile(0.95), lambda x: x.quantile(0.05)],
        "FF": ["mean", "std", "max"],
        "N1": ["mean", "std"],
        "N2": ["mean", "std"],
        "N3": ["mean", "std"],
        "T2": ["mean", "std"],
        "T3": ["mean", "std"],
        "T25": ["mean", "std"],
        "P3": ["mean", "std"],
        "P50": ["mean", "std"],
        "EPR": ["mean", "std"],
        "MN": ["mean"],
        "datetime": "count"
    }).reset_index()

    engine_features.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in engine_features.columns
    ]

    engine_features.columns = engine_features.columns.str.replace("<lambda_0>", "p95")
    engine_features.columns = engine_features.columns.str.replace("<lambda_1>", "p05")
    engine_features.columns = engine_features.columns.str.replace("datetime_count", "observation_count")

    # Derived features
    engine_features["TGT_per_FF"] = engine_features["TGT_mean"] / engine_features["FF_mean"]
    engine_features["TGT_per_N1"] = engine_features["TGT_mean"] / engine_features["N1_mean"]
    engine_features["TGT_per_N2"] = engine_features["TGT_mean"] / engine_features["N2_mean"]
    engine_features["TGT_range"] = engine_features["TGT_max"] - engine_features["TGT_min"]
    engine_features["TGT_cv"] = engine_features["TGT_std"] / engine_features["TGT_mean"]
    engine_features["FF_per_N1"] = engine_features["FF_mean"] / engine_features["N1_mean"]
    engine_features["T3_to_T2_ratio"] = engine_features["T3_mean"] / engine_features["T2_mean"]

    engine_features = engine_features.replace([np.inf, -np.inf], np.nan)
    engine_features = engine_features.fillna(engine_features.median())

    logger.info(f"Created features for {len(engine_features)} engines")
    return engine_features

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    logger.info("Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols

# ============================================================
# PREDICTION
# ============================================================
def predict(engine_features, model, scaler, feature_cols):
    X = engine_features[feature_cols]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)

    engine_features["anomaly_score"] = scores
    engine_features["is_anomalous"] = preds == -1
    return engine_features

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 80)
    logger.info("AIRCRAFT ENGINE ANOMALY DETECTION PIPELINE")
    logger.info("=" * 80)

    df = load_data()
    engine_features = preprocess_and_engineer(df)
    model, scaler, feature_cols = load_model()
    results = predict(engine_features, model, scaler, feature_cols)

    anomalies = results[results["is_anomalous"]]

    logger.info(f"Total engines: {len(results)}")
    logger.info(f"Anomalous engines: {len(anomalies)}")

    if len(anomalies) > 0:
        logger.info("Engines requiring inspection:")
        for eng in anomalies.sort_values("anomaly_score")["engine no"]:
            logger.info(f" - Engine {eng}")

    output_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
