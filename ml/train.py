"""Train and compare models, save the best one.   python -m ml.train [--csv data/BankChurners.csv] [--cost 25]"""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import from_csv, synthetic
from .explain import contributions
from .features import FEATURES, LABELS
try:
    from xgboost import XGBClassifier
except Exception:  # xgboost not installed
    XGBClassifier = None

VALUE_SAVED = 72.0  # 30% save rate x 240 yearly revenue per customer (assumption)

def candidates(seed):
    m = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    if XGBClassifier is not None:
        m["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1)
    else:
        print("WARNING: xgboost is not installed, skipping it.")
    return m

def sweep(p, y):
    out = []
    for t in np.arange(0.02, 0.91, 0.02):
        pred = p >= t
        out.append({"t": round(float(t), 2), "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum())})
    return out

def best_threshold(sw, cost):
    b = max(sw, key=lambda s: s["tp"] * VALUE_SAVED - (s["tp"] + s["fp"]) * cost)
    return float(min(.7, max(.1, b["t"])))

def train_from_frame(df, out_dir="artifacts", cost=25.0, seed=42, source="unknown"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES].astype(float), df["churn"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    results, fitted = [], {}
    for name, model in candidates(seed).items():
        t0 = time.time()
        scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc")
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        sw = sweep(p, yte); thr = best_threshold(sw, cost); pred = p >= thr
        results.append({
            "name": name, "roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)), "log_loss": float(log_loss(yte, p)),
            "cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std()), "threshold": thr,
            "precision": float(precision_score(yte, pred, zero_division=0)), "recall": float(recall_score(yte, pred)),
            "f1": float(f1_score(yte, pred)), "seconds": round(time.time() - t0, 1),
        })
        fitted[name] = (model, p, sw)
        print(f"{name:20s} ROC-AUC {results[-1]['roc_auc']:.3f}  PR-AUC {results[-1]['pr_auc']:.3f}")
    best = max(results, key=lambda r: (r["pr_auc"], r["roc_auc"]))["name"]
    model, p, sw = fitted[best]
    thr = next(r["threshold"] for r in results if r["name"] == best)
    fpr, tpr, _ = roc_curve(yte, p)
    keep = np.linspace(0, len(fpr) - 1, min(80, len(fpr))).astype(int)
    groups = np.array_split(np.argsort(p), 8)
    imp = np.abs(contributions(best, model, Xte)).mean(axis=0)
    pred = p >= thr
    metrics = {
        "selected": best, "models": results, "threshold": thr, "cost_per_contact": cost, "value_saved": VALUE_SAVED,
        "base_rate": float(y.mean()), "n_train": int(len(ytr)), "n_test": int(len(yte)), "n_test_positive": int(yte.sum()),
        "source": source,
        "curves": {"roc": [[float(fpr[i]), float(tpr[i])] for i in keep],
                   "calibration": [[float(p[g].mean()), float(yte[g].mean())] for g in groups],
                   "sweep": sw},
        "importance": sorted([{"feature": f, "label": LABELS[f], "value": float(v)} for f, v in zip(FEATURES, imp)],
                             key=lambda d: -d["value"]),
        "confusion": {"tp": int((pred & (yte == 1)).sum()), "fp": int((pred & (yte == 0)).sum()),
                      "fn": int((~pred & (yte == 1)).sum()), "tn": int((~pred & (yte == 0)).sum())},
    }
    joblib.dump({"name": best, "model": model, "features": FEATURES}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Selected: {best}  (threshold {thr:.2f})  saved to {out}/")
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="artifacts"); ap.add_argument("--cost", type=float, default=25.0)
    a = ap.parse_args()
    frame = from_csv(a.csv) if a.csv else synthetic()
    train_from_frame(frame, a.out, a.cost, source=a.csv or "synthetic")
