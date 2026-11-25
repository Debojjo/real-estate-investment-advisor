"""Step 3 - Model training, evaluation and MLflow tracking (revision 2).

Usage (from the project folder):
    python scripts/step3_train.py data/cleaned_housing.csv models
    python scripts/step3_train.py data/cleaned_housing.csv models --sample 50000   # quick test run

Classification: Good_Investment
  baselines  : majority class, stratified random guess, Logistic Regression
  models     : Random Forest, XGBoost depth 4 and 6
  selection  : F1 score (precision and recall are reported next to it; ROC-AUC and the confusion matrix too)

Regression: the model predicts the 5-YEAR GROWTH MULTIPLE (Future price / current price), then
  Future price = current price x predicted multiple.
  Predicting the multiple means the model cannot score well just by copying the price; it has to learn how
  infrastructure, transport, readiness, security, parking and building age change growth.
  baselines  : fixed 8% growth (1.08^5) and the training-set mean multiple
  models     : Ridge, Random Forest, XGBoost depth 4 and 6
  metrics    : on the multiple (rmse, mae, r2) and on the final price in lakhs

Extras: leakage / ablation checks, 5-fold cross-validation of the winners, MLflow registry,
models/classifier.joblib + models/regressor.joblib for the Streamlit app.
"""
import argparse, io, json, os, sys, time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
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
SEED, YEARS = 42, 5
MAX_MB = 25      # a deployable model file must stay small (GitHub limit 100 MB, Streamlit Cloud memory)
REVISION = "2-growth-multiple-and-baselines"

raw = pd.read_csv(args.src)
if args.sample:
    raw = raw.sample(args.sample, random_state=SEED).reset_index(drop=True)
train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=SEED, stratify=raw["Good_Investment"])
peer = build_peer_table(train_raw)                 # peer medians learned from TRAIN rows only
train, test = add_features(train_raw, peer), add_features(test_raw, peer)
for d in (train, test):
    d["Growth_Multiple"] = (1 + d["Growth_Rate"]) ** YEARS
print(f"train {len(train):,}  test {len(test):,}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def run(name, **tags):
    return mlflow.start_run(run_name=name, tags={"revision": REVISION, **tags})


def make_pre(cat, num, scale):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("num", StandardScaler() if scale else "passthrough", num)])


def log_model(model, name):
    # cloudpickle: models are trained by this script and loaded only by our own app (trusted source);
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


def size_mb(pipe):
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return buf.getbuffer().nbytes / 1e6


def xgb_params(depth, n):
    return dict(n_estimators=n, max_depth=depth, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", n_jobs=-1, random_state=SEED)


def clf_models():
    return {
        "LogisticRegression": (True, LogisticRegression(max_iter=1000)),
        "RandomForest": (False, RandomForestClassifier(n_estimators=100, max_depth=14, max_samples=0.5, n_jobs=-1, random_state=SEED)),
        "XGBoost_d4": (False, XGBClassifier(**xgb_params(4, 200))),
        "XGBoost_d6": (False, XGBClassifier(**xgb_params(6, 300))),
    }


def reg_models():
    return {
        "Ridge": (True, Ridge(alpha=1.0)),
        "RandomForest": (False, RandomForestRegressor(n_estimators=100, max_depth=14, max_samples=0.5, n_jobs=-1, random_state=SEED)),
        "XGBoost_d4": (False, XGBRegressor(**xgb_params(4, 200))),
        "XGBoost_d6": (False, XGBRegressor(**xgb_params(6, 300))),
    }


rd = lambda m: {k: round(float(v), 4) for k, v in m.items()}
results = {"classification": [], "regression": [], "ablation": [], "cross_validation": []}

# =================== CLASSIFICATION ===================
mlflow.set_experiment("RealEstate_Classification")
Xtr, Xte, ytr, yte = train[ALL_FEATURES], test[ALL_FEATURES], train["Good_Investment"], test["Good_Investment"]


def clf_metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


# baselines: does the model beat guessing?
for label, strat in [("Baseline: majority class", "most_frequent"), ("Baseline: random (stratified)", "stratified")]:
    with run("BASELINE_" + strat, baseline="true"):
        dummy = DummyClassifier(strategy=strat, random_state=SEED).fit(Xtr, ytr)
        m = clf_metrics(yte, dummy.predict(Xte), dummy.predict_proba(Xte)[:, 1])
        mlflow.log_params({"task": "classification", "rule": strat}); mlflow.log_metrics(m)
        results["classification"].append({"model": label, **rd(m)})
        print(f"[clf] {label:30s} acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

best = {"f1": -1}
for name, (scale, est) in clf_models().items():
    t0 = time.time()
    with run(name):
        pipe = Pipeline([("pre", make_pre(CAT_FEATURES, NUM_FEATURES, scale)), ("model", est)]).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        m = clf_metrics(yte, pred, proba)
        mlflow.log_params({"model": name, "n_train": len(train), "n_features": len(ALL_FEATURES), "task": "classification"})
        mlflow.log_params({f"p_{k}": v for k, v in est.get_params().items() if k in ("max_depth", "n_estimators", "learning_rate", "C", "subsample")})
        mlflow.log_metrics(m)
        cm = confusion_matrix(yte, pred)
        fig, ax = plt.subplots(figsize=(4, 4)); ConfusionMatrixDisplay(cm, display_labels=["Not good", "Good"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} confusion matrix"); plt.tight_layout(); p = f"docs/models/cm_{name}.png"; plt.savefig(p, dpi=110); plt.close()
        mlflow.log_artifact(p)
        info = log_model(pipe, "model")
        print(f"[clf] {name:30s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} ({time.time()-t0:.0f}s)")
        results["classification"].append({"model": name, **rd(m)})
        # selection: F1 first, ROC-AUC breaks ties; recall alone never decides
        key = (round(m["f1"], 4), m["roc_auc"])
        mb = size_mb(pipe)
        print(f"      model file size: {mb:.1f} MB" + ("  (too large to deploy)" if mb > MAX_MB else ""))
        if mb <= MAX_MB and (best["f1"] < 0 or key > best["key"]):
            best = {"f1": m["f1"], "key": key, "name": name, "pipe": pipe, "uri": info.model_uri, "metrics": m, "cm": cm.tolist()}

mlflow.register_model(best["uri"], "RealEstate_GoodInvestment_Classifier")
imp = top_importances(best["pipe"]); imp.to_csv("docs/models/classifier_feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 6)); imp[::-1].plot.barh(ax=ax, color="steelblue"); ax.set_title(f"Classifier feature importance ({best['name']})")
plt.tight_layout(); plt.savefig("docs/models/classifier_feature_importance.png", dpi=110); plt.close()
joblib.dump({"pipeline": best["pipe"], "peer": peer, "model_name": best["name"], "metrics": best["metrics"]}, f"{args.out}/classifier.joblib")
print("best classifier (by F1):", best["name"])

with run("ABLATION_XGBoost_no_price", ablation="true"):
    feats = [f for f in ALL_FEATURES if f not in PRICE_FEATURES]
    num = [f for f in NUM_FEATURES if f not in PRICE_FEATURES]
    pipe = Pipeline([("pre", make_pre(CAT_FEATURES, num, False)), ("model", XGBClassifier(**xgb_params(6, 300)))]).fit(train[feats], ytr)
    pr = pipe.predict_proba(test[feats])[:, 1]
    a = {k: v for k, v in clf_metrics(yte, (pr >= 0.5).astype(int), pr).items() if k in ("accuracy", "f1", "roc_auc")}
    mlflow.log_params({"task": "classification", "features_removed": ",".join(PRICE_FEATURES)}); mlflow.log_metrics(a)
    results["ablation"].append({"task": "classification: no price inputs", **rd(a)})
    print("[clf ablation] no price inputs:", rd(a))

# 5-fold cross-validation of the chosen classifier (stability check on a 50,000-row sample of the training set)
sub = train.sample(min(50000, len(train)), random_state=SEED)
cv = cross_val_score(clone(best["pipe"]), sub[ALL_FEATURES], sub["Good_Investment"], scoring="f1",
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
with run("CV_best_classifier", cv="true"):
    mlflow.log_params({"model": best["name"], "folds": 5, "rows": len(sub)})
    mlflow.log_metrics({"cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std())})
results["cross_validation"].append({"task": "classification (F1)", "model": best["name"], "mean": round(float(cv.mean()), 4), "std": round(float(cv.std()), 4)})
print(f"[clf cv] {best['name']} F1 = {cv.mean():.4f} +/- {cv.std():.4f}")

# =================== REGRESSION ===================
mlflow.set_experiment("RealEstate_Regression")
ytr_m, yte_m = train["Growth_Multiple"], test["Growth_Multiple"]
price_te, fut_te = test["Price_in_Lakhs"], test["Future_Price_5Y"]
rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))


def reg_metrics(pred_mult):
    pred_mult = np.asarray(pred_mult, dtype=float)
    pred_price = price_te.values * pred_mult
    return {"rmse_multiple": rmse(yte_m, pred_mult), "mae_multiple": mean_absolute_error(yte_m, pred_mult), "r2_multiple": r2_score(yte_m, pred_mult),
            "rmse_lakhs": rmse(fut_te, pred_price), "mae_lakhs": mean_absolute_error(fut_te, pred_price), "r2_lakhs": r2_score(fut_te, pred_price)}


for label, mult in [("Baseline: fixed 8% growth", (1.08 ** YEARS)), ("Baseline: training-set mean growth", float(ytr_m.mean()))]:
    m = reg_metrics(np.full(len(test), mult))
    with run("BASELINE_" + label.split(": ")[1].replace(" ", "_"), baseline="true"):
        mlflow.log_params({"task": "regression", "multiple": round(mult, 4)}); mlflow.log_metrics(m)
    results["regression"].append({"model": label, **rd(m)})
    print(f"[reg] {label:36s} rmse_mult={m['rmse_multiple']:.4f} r2_mult={m['r2_multiple']:.4f} rmse_lakhs={m['rmse_lakhs']:.2f}")

cands = {}
for name, (scale, est) in reg_models().items():
    t0 = time.time()
    with run(name):
        pipe = Pipeline([("pre", make_pre(CAT_FEATURES, NUM_FEATURES, scale)), ("model", est)]).fit(Xtr, ytr_m)
        m = reg_metrics(pipe.predict(Xte))
        mlflow.log_params({"model": name, "n_train": len(train), "n_features": len(ALL_FEATURES), "task": "regression", "target": "growth_multiple"})
        mlflow.log_params({f"p_{k}": v for k, v in est.get_params().items() if k in ("max_depth", "n_estimators", "learning_rate", "alpha", "subsample")})
        mlflow.log_metrics(m)
        info = log_model(pipe, "model")
        print(f"[reg] {name:36s} rmse_mult={m['rmse_multiple']:.4f} r2_mult={m['r2_multiple']:.4f} rmse_lakhs={m['rmse_lakhs']:.2f} ({time.time()-t0:.0f}s)")
        results["regression"].append({"model": name, **rd(m)})
        mb = size_mb(pipe)
        print(f"      model file size: {mb:.1f} MB" + ("  (too large to deploy)" if mb > MAX_MB else ""))
        cands[name] = {"rmse": m["rmse_multiple"], "name": name, "pipe": pipe, "uri": info.model_uri, "metrics": m, "mb": mb}

# deployment choice: lowest RMSE among models whose file is small enough, and prefer a compact XGBoost model
# if it is within 5% of the best (a 100-tree RandomForest can be a ~200 MB file)
eligible = [c for c in cands.values() if c["mb"] <= MAX_MB]
bestr = min(eligible, key=lambda c: c["rmse"])
xgb_best = min((c for c in eligible if c["name"].startswith("XGBoost")), key=lambda c: c["rmse"])
lowest = min(cands.values(), key=lambda c: c["rmse"])
if lowest["name"] != bestr["name"]:
    print(f"lowest RMSE is {lowest['name']} ({lowest['rmse']:.4f}) but its file is {lowest['mb']:.0f} MB; limiting to models under {MAX_MB} MB")
if bestr["name"] != xgb_best["name"] and xgb_best["rmse"] <= 1.05 * bestr["rmse"]:
    print(f"deploying {xgb_best['name']} ({xgb_best['rmse']:.4f}) instead of {bestr['name']} ({bestr['rmse']:.4f}) - similar accuracy, simpler model")
    bestr = xgb_best

mlflow.register_model(bestr["uri"], "RealEstate_FuturePrice_Regressor")
imp = top_importances(bestr["pipe"]); imp.to_csv("docs/models/regressor_feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 6)); imp[::-1].plot.barh(ax=ax, color="seagreen"); ax.set_title(f"Regressor feature importance ({bestr['name']})")
plt.tight_layout(); plt.savefig("docs/models/regressor_feature_importance.png", dpi=110); plt.close()
joblib.dump({"pipeline": bestr["pipe"], "peer": peer, "model_name": bestr["name"], "metrics": bestr["metrics"], "target": "growth_multiple"},
            f"{args.out}/regressor.joblib")
print("best regressor:", bestr["name"])

# ablations for the regressor
DRIVERS = ["Infrastructure_Score", "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility", "Availability_Status",
           "Parking_Flag", "Security_Flag", "Age_of_Property"]
for label, drop in [("regression: no price inputs", PRICE_FEATURES), ("regression: no growth drivers (infrastructure, transport, readiness, security, parking, age)", DRIVERS)]:
    with run("ABLATION_" + label.split(": ")[1][:22].replace(" ", "_"), ablation="true"):
        feats = [f for f in ALL_FEATURES if f not in drop]
        cat, num = [f for f in CAT_FEATURES if f in feats], [f for f in NUM_FEATURES if f in feats]
        pipe = Pipeline([("pre", make_pre(cat, num, False)), ("model", XGBRegressor(**xgb_params(6, 300)))]).fit(train[feats], ytr_m)
        a = reg_metrics(pipe.predict(test[feats]))
        a = {k: a[k] for k in ("rmse_multiple", "r2_multiple", "rmse_lakhs", "r2_lakhs")}
        mlflow.log_params({"task": "regression", "features_removed": ",".join(drop)}); mlflow.log_metrics(a)
        results["ablation"].append({"task": label, **rd(a)})
        print(f"[reg ablation] {label[:50]:50s}", rd(a))

# 5-fold cross-validation of the chosen regressor
cvr = cross_val_score(clone(bestr["pipe"]), sub[ALL_FEATURES], sub["Growth_Multiple"], scoring="r2", cv=KFold(5, shuffle=True, random_state=SEED))
with run("CV_best_regressor", cv="true"):
    mlflow.log_params({"model": bestr["name"], "folds": 5, "rows": len(sub)})
    mlflow.log_metrics({"cv_r2_multiple_mean": float(cvr.mean()), "cv_r2_multiple_std": float(cvr.std())})
results["cross_validation"].append({"task": "regression (R2 on growth multiple)", "model": bestr["name"], "mean": round(float(cvr.mean()), 4), "std": round(float(cvr.std()), 4)})
print(f"[reg cv] {bestr['name']} R2 = {cvr.mean():.4f} +/- {cvr.std():.4f}")

# ---- summary files ----
results["best_classifier"] = best["name"]; results["best_regressor"] = bestr["name"]; results["confusion_matrix_best"] = best["cm"]
results["selection"] = ("Classifier chosen on F1 (ROC-AUC breaks ties); precision, recall and the confusion matrix are reported next to it. "
                        "Regressor chosen on RMSE of the growth multiple among models under 25 MB, preferring XGBoost if within 5% of the best.")
json.dump(results, open(f"{args.out}/metrics.json", "w"), indent=2)
with open("docs/models/model_results.md", "w", encoding="utf-8") as f:
    f.write("# Model results (20% hold-out test set)\n\n## Classification: Good_Investment\n")
    f.write(pd.DataFrame(results["classification"]).to_markdown(index=False))
    f.write(f"\n\nSelected on F1 (ROC-AUC breaks ties): **{best['name']}**. Confusion matrix: `docs/models/cm_{best['name']}.png`. "
            "The two baselines show what guessing achieves.\n\n")
    f.write("## Regression: 5-year growth multiple, then price = current price x multiple\n")
    f.write(pd.DataFrame(results["regression"]).to_markdown(index=False))
    f.write(f"\n\nSelected: **{bestr['name']}**. `_multiple` columns score the growth multiple (the part the model has to learn); `_lakhs` columns score the final price.\n\n")
    f.write("## Ablation and leakage checks\n")
    f.write(pd.DataFrame(results["ablation"]).to_markdown(index=False))
    f.write("\n\n## 5-fold cross-validation of the selected models\n")
    f.write(pd.DataFrame(results["cross_validation"]).to_markdown(index=False))
    f.write("\n")
print("\nDone. Models in", args.out, "| MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
