"""Dataset loading. Real data: Kaggle 'Credit Card customers' (BankChurners). Fallback: synthetic."""
import numpy as np
import pandas as pd
from .features import FEATURES

def from_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("naive_bayes")]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Expected the BankChurners layout.")
    if "attrition_flag" not in df.columns:
        raise ValueError("CSV needs an Attrition_Flag column.")
    out = pd.DataFrame({
        "customer_id": ("C-" + df["clientnum"].astype(str)) if "clientnum" in df else [f"C-{i+1}" for i in range(len(df))],
        "segment": df["card_category"].astype(str) if "card_category" in df else "Standard",
        "churn": df["attrition_flag"].astype(str).str.contains("attrited", case=False).astype(int),
    })
    for f in FEATURES:
        out[f] = pd.to_numeric(df[f], errors="coerce")
    return out.dropna().reset_index(drop=True)

def synthetic(n=6000, seed=7):
    """BankChurners-shaped data with a ~16% churn rate and non-linear signals."""
    r = np.random.default_rng(seed)
    e = r.normal(size=n)  # hidden engagement level
    seg = r.choice(["Blue", "Silver", "Gold", "Platinum"], n, p=[.6, .25, .1, .05])
    mult = pd.Series(seg).map({"Blue": 1, "Silver": 2.2, "Gold": 3, "Platinum": 4}).to_numpy()
    d = pd.DataFrame({"customer_id": [f"C-{720000 + i}" for i in range(n)], "segment": seg})
    d["customer_age"] = np.clip(np.round(r.normal(46, 8, n)), 26, 73)
    d["months_on_book"] = np.clip(np.round(r.normal(36, 8, n)), 13, 56)
    d["total_relationship_count"] = np.clip(np.round(3.8 + .6 * e + r.normal(0, 1.2, n)), 1, 6)
    d["months_inactive_12_mon"] = np.clip(np.round(2.3 - .5 * e + r.normal(0, .9, n)), 0, 6)
    d["contacts_count_12_mon"] = np.clip(np.round(2.4 - .4 * e + r.normal(0, 1, n)), 0, 6)
    d["credit_limit"] = np.clip(np.exp(r.normal(8.9, .8, n)) * mult, 1438, 34516).round()
    d["total_revolving_bal"] = np.clip(d.credit_limit * np.clip(r.normal(.28, .25, n) + .1 * e, 0, .95), 0, 2517).round()
    d["avg_utilization_ratio"] = (d.total_revolving_bal / d.credit_limit).round(3)
    d["total_trans_ct"] = np.clip(np.round(65 + 22 * e + r.normal(0, 10, n)), 10, 139)
    d["total_trans_amt"] = np.clip(d.total_trans_ct * r.normal(65, 25, n) + 600, 510, 18484).round()
    d["total_ct_chng_q4_q1"] = np.clip(r.normal(.71 + .12 * e, .2), 0, 3.7).round(3)
    d["total_amt_chng_q4_q1"] = np.clip(r.normal(.76 + .09 * e, .2), 0, 3.4).round(3)
    z = (-1.1 * (d.total_trans_ct - 65) / 22 + .9 * (d.months_inactive_12_mon >= 3) + .5 * (d.contacts_count_12_mon >= 4)
         - .3 * (d.total_relationship_count - 3.8) + 1.0 * (d.total_ct_chng_q4_q1 < .5) - .4 * (d.avg_utilization_ratio > .3)
         + r.normal(0, .6, n)).to_numpy()
    lo, hi = -8.0, 4.0
    for _ in range(40):  # find the intercept that gives a 16% churn rate
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + b)))).mean() > .16: hi = b
        else: lo = b
    d["churn"] = (r.random(n) < 1 / (1 + np.exp(-(z + b)))).astype(int)
    return d
