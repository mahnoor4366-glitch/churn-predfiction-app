import json
import joblib
import pandas as pd
from ml.explain import contributions
from ml.features import FEATURES, LABELS
from .config import settings

ACTIONS = {
    "total_trans_ct": "Spending has slowed. Offer cashback on everyday purchases.",
    "total_trans_amt": "Spend is falling. Offer a spending reward or a limit review.",
    "total_ct_chng_q4_q1": "Activity dropped versus last quarter. Send a re-engagement message.",
    "total_amt_chng_q4_q1": "Spend dropped versus last quarter. Offer a targeted promotion.",
    "months_inactive_12_mon": "Has been inactive for months. Send a re-engagement push or call.",
    "contacts_count_12_mon": "Contacted the bank often. Have a person follow up on their issues first.",
    "total_relationship_count": "Holds few products. Offer a savings or second product to deepen the relationship.",
    "avg_utilization_ratio": "Card use is low. Suggest a balance-transfer or rewards offer.",
    "total_revolving_bal": "Revolving balance is unusual. Review pricing and offer a payment plan.",
    "credit_limit": "Review the credit limit and offer an upgrade if eligible.",
    "months_on_book": "Newer customer. Run an onboarding journey to build the habit.",
    "customer_age": "Tailor the offer to this age group.",
}

class Service:
    def __init__(self):
        self.model = None
        self.name = ""
        self.metrics = {}

    def available(self) -> bool:
        return (settings.artifact_dir / "model.joblib").exists() and (settings.artifact_dir / "metrics.json").exists()

    def load(self):
        bundle = joblib.load(settings.artifact_dir / "model.joblib")
        self.model, self.name = bundle["model"], bundle["name"]
        self.metrics = json.loads((settings.artifact_dir / "metrics.json").read_text())

    @property
    def threshold(self) -> float:
        return float(self.metrics["threshold"])

    def tier(self, p: float) -> str:
        t = self.threshold
        return "High" if p >= t else "Medium" if p >= t / 2 else "Low"

    def score(self, df: pd.DataFrame):
        X = df[FEATURES].astype(float)
        return self.model.predict_proba(X)[:, 1], contributions(self.name, self.model, X)

SERVICE = Service()
