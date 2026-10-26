"""Step 3 - Model training, evaluation and MLflow tracking.

Usage (from the project folder):
    python scripts/step3_train.py data/cleaned_housing.csv models
    python scripts/step3_train.py data/cleaned_housing.csv models --sample 50000   # quick test run

Tasks:  classification -> Good_Investment      regression -> Future_Price_5Y (lakhs)
Logs every run to MLflow (sqlite:///mlflow.db), registers the best model of each task,
saves models/classifier.joblib + models/regressor.joblib for the Streamlit app.
"""
import argparse, json, os, sys, time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES, PRICE_FEATURES, add_features, build_peer_table

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?", default="data/cleaned_housing.csv")
ap.add_argument("out", nargs="?", default="models")
ap.add_argument("--sample", type=int, default=0, help="use only N rows (quick test)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs("docs/models", exist_ok=True)
SEED = 42

raw = pd.read_csv(args.src)
if args.sample:
    raw = raw.sample(args.sample, random_state=SEED).reset_index(drop=True)
train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=SEED, stratify=raw["Good_Investment"])
peer = build_peer_table(train_raw)                 # peer medians learned from TRAIN rows only
train, test = add_features(train_raw, peer), add_features(test_raw, peer)
print(f"train {len(train):,}  test {len(test):,}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def make_pre(cat, num, scale):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("num", StandardScaler() if scale else "passthrough", num)])


def log_model(model, name):
    # cloudpickle: the models are trained by this script and loaded only by our own app (trusted source);
    # newer MLflow defaults to skops, which refuses tree models without an explicit trust list.
    try:
        return mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    except TypeError:  # older MLflow
        return mlflow.sklearn.log_model(model, artifact_path=name, serialization_format="cloudpickle")


def top_importances(pipe, n=20):
    names = pipe.named_steps["pre"].get_feature_names_out()
    m = pipe.named_steps["model"]
    imp = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(np.ravel(m.coef_))
    return pd.Series(imp, index=names).sort_values(ascending=False).head(n)


def clf_models():
    return {
        "LogisticRegression": (True, LogisticRegression(max_iter=1000)),
        "RandomForest": (False, RandomForestClassifier(n_estimators=100, max_depth=14, max_samples=0.5, n_jobs=-1, random_state=SEED)),
        "XGBoost_d4": (False, XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
        "XGBoost_d6": (False, XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
    }


def reg_models():
    return {
        "Ridge": (True, Ridge(alpha=1.0)),
        "RandomForest": (False, RandomForestRegressor(n_estimators=100, max_depth=14, max_samples=0.5, n_jobs=-1, random_state=SEED)),
        "XGBoost_d4": (False, XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
        "XGBoost_d6": (False, XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
    }


results = {"classification": [], "regression": [], "ablation": []}

# =================== CLASSIFICATION ===================
mlflow.set_experiment("RealEstate_Classification")
Xtr, Xte, ytr, yte = train[ALL_FEATURES], test[ALL_FEATURES], train["Good_Investment"], test["Good_Investment"]
best = {"f1": -1}
for name, (scale, est) in clf_models().items():
    t0 = time.time()
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("pre", make_pre(CAT_FEATURES, NUM_FEATURES, scale)), ("model", est)]).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        m = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
             "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)}
        mlflow.log_params({"model": name, "n_train": len(train), "n_features": len(ALL_FEATURES), "task": "classification"})
        mlflow.log_params({f"p_{k}": v for k, v in est.get_params().items() if k in ("max_depth", "n_estimators", "learning_rate", "C", "subsample")})
        mlflow.log_metrics(m)
        cm = confusion_matrix(yte, pred)
        fig, ax = plt.subplots(figsize=(4, 4)); ConfusionMatrixDisplay(cm, display_labels=["Not good", "Good"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} confusion matrix"); plt.tight_layout(); p = f"docs/models/cm_{name}.png"; plt.savefig(p, dpi=110); plt.close()
        mlflow.log_artifact(p)
        info = log_model(pipe, "model")
        print(f"[clf] {name:20s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} ({time.time()-t0:.0f}s)")
        results["classification"].append({"model": name, **{k: round(v, 4) for k, v in m.items()}})
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "name": name, "pipe": pipe, "uri": info.model_uri, "metrics": m, "cm": cm.tolist()}

reg_name = "RealEstate_GoodInvestment_Classifier"
mlflow.register_model(best["uri"], reg_name)
imp = top_importances(best["pipe"]); imp.to_csv("docs/models/classifier_feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 6)); imp[::-1].plot.barh(ax=ax, color="steelblue"); ax.set_title(f"Classifier feature importance ({best['name']})")
plt.tight_layout(); plt.savefig("docs/models/classifier_feature_importance.png", dpi=110); plt.close()
joblib.dump({"pipeline": best["pipe"], "peer": peer, "model_name": best["name"], "metrics": best["metrics"]}, f"{args.out}/classifier.joblib")
print("best classifier:", best["name"])

# ablation: remove price-based inputs -> how much of the accuracy came from price?
with mlflow.start_run(run_name="ABLATION_XGBoost_no_price", tags={"ablation": "true"}):
    feats = [f for f in ALL_FEATURES if f not in PRICE_FEATURES]
    cat, num = CAT_FEATURES, [f for f in NUM_FEATURES if f not in PRICE_FEATURES]
    est = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
    pipe = Pipeline([("pre", make_pre(cat, num, False)), ("model", est)]).fit(train[feats], ytr)
    pr = pipe.predict_proba(test[feats])[:, 1]; pd_ = (pr >= 0.5).astype(int)
    a = {"accuracy": accuracy_score(yte, pd_), "f1": f1_score(yte, pd_), "roc_auc": roc_auc_score(yte, pr)}
    mlflow.log_params({"task": "classification", "features_removed": ",".join(PRICE_FEATURES)}); mlflow.log_metrics(a)
    results["ablation"].append({"task": "classification (no price inputs)", **{k: round(v, 4) for k, v in a.items()}})
    print("[clf ablation] no-price inputs:", {k: round(v, 4) for k, v in a.items()})

# =================== REGRESSION ===================
mlflow.set_experiment("RealEstate_Regression")
ytr_r, yte_r = train["Future_Price_5Y"], test["Future_Price_5Y"]
rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
naive = test["Price_in_Lakhs"] * 1.08 ** 5
base = {"rmse": rmse(yte_r, naive), "mae": mean_absolute_error(yte_r, naive), "r2": r2_score(yte_r, naive)}
with mlflow.start_run(run_name="BASELINE_fixed_8pct", tags={"baseline": "true"}):
    mlflow.log_params({"task": "regression", "rule": "Price * 1.08^5"}); mlflow.log_metrics(base)
results["regression"].append({"model": "Baseline (Price x 1.08^5)", **{k: round(v, 4) for k, v in base.items()}})
print(f"[reg] baseline fixed 8%       rmse={base['rmse']:.3f} mae={base['mae']:.3f} r2={base['r2']:.5f}")
cands = {}
for name, (scale, est) in reg_models().items():
    t0 = time.time()
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("pre", make_pre(CAT_FEATURES, NUM_FEATURES, scale)), ("model", est)]).fit(Xtr, ytr_r)
        pred = pipe.predict(Xte)
        m = {"rmse": rmse(yte_r, pred), "mae": mean_absolute_error(yte_r, pred), "r2": r2_score(yte_r, pred)}
        mlflow.log_params({"model": name, "n_train": len(train), "n_features": len(ALL_FEATURES), "task": "regression"})
        mlflow.log_params({f"p_{k}": v for k, v in est.get_params().items() if k in ("max_depth", "n_estimators", "learning_rate", "alpha", "subsample")})
        mlflow.log_metrics(m)
        info = log_model(pipe, "model")
        print(f"[reg] {name:20s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.5f} ({time.time()-t0:.0f}s)")
        results["regression"].append({"model": name, **{k: round(v, 4) for k, v in m.items()}})
        cands[name] = {"rmse": m["rmse"], "name": name, "pipe": pipe, "uri": info.model_uri, "metrics": m}

# deployment choice: lowest RMSE overall, but prefer a compact XGBoost model if it is within 5% of the best
# (a 100-tree RandomForest is a ~220 MB file - too big for GitHub / Streamlit Cloud)
bestr = min(cands.values(), key=lambda c: c["rmse"])
xgb_best = min((c for c in cands.values() if c["name"].startswith("XGBoost")), key=lambda c: c["rmse"])
if bestr["name"] != xgb_best["name"] and xgb_best["rmse"] <= 1.05 * bestr["rmse"]:
    print(f"lowest RMSE is {bestr['name']} ({bestr['rmse']:.3f}); deploying {xgb_best['name']} ({xgb_best['rmse']:.3f}) - similar accuracy, much smaller file")
    bestr = xgb_best

mlflow.register_model(bestr["uri"], "RealEstate_FuturePrice_Regressor")
imp = top_importances(bestr["pipe"]); imp.to_csv("docs/models/regressor_feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 6)); imp[::-1].plot.barh(ax=ax, color="seagreen"); ax.set_title(f"Regressor feature importance ({bestr['name']})")
plt.tight_layout(); plt.savefig("docs/models/regressor_feature_importance.png", dpi=110); plt.close()
joblib.dump({"pipeline": bestr["pipe"], "peer": peer, "model_name": bestr["name"], "metrics": bestr["metrics"]}, f"{args.out}/regressor.joblib")
print("best regressor:", bestr["name"])

with mlflow.start_run(run_name="ABLATION_XGBoost_no_price_regression", tags={"ablation": "true"}):
    feats = [f for f in ALL_FEATURES if f not in PRICE_FEATURES]
    num = [f for f in NUM_FEATURES if f not in PRICE_FEATURES]
    est = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
    pipe = Pipeline([("pre", make_pre(CAT_FEATURES, num, False)), ("model", est)]).fit(train[feats], ytr_r)
    pr = pipe.predict(test[feats])
    a = {"rmse": rmse(yte_r, pr), "mae": mean_absolute_error(yte_r, pr), "r2": r2_score(yte_r, pr)}
    mlflow.log_params({"task": "regression", "features_removed": ",".join(PRICE_FEATURES)}); mlflow.log_metrics(a)
    results["ablation"].append({"task": "regression (no price inputs)", **{k: round(v, 4) for k, v in a.items()}})
    print("[reg ablation] no-price inputs:", {k: round(v, 4) for k, v in a.items()})

# ---- summary files ----
results["best_classifier"] = best["name"]; results["best_regressor"] = bestr["name"]; results["confusion_matrix_best"] = best["cm"]
json.dump(results, open(f"{args.out}/metrics.json", "w"), indent=2)
with open("docs/models/model_results.md", "w", encoding="utf-8") as f:
    f.write("# Model results (test set, 20% hold-out)\n\n## Classification: Good_Investment\n")
    f.write(pd.DataFrame(results["classification"]).to_markdown(index=False))
    f.write(f"\n\nBest: **{best['name']}**\n\n## Regression: Future_Price_5Y (lakhs)\n")
    f.write(pd.DataFrame(results["regression"]).to_markdown(index=False))
    f.write(f"\n\nBest: **{bestr['name']}**\n\n## Ablation: price inputs removed\n")
    f.write(pd.DataFrame(results["ablation"]).to_markdown(index=False))
print("\nDone. Models in", args.out, "| MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
