"""Start-up bootstrap: users, customers (CSV or synthetic), trained model, predictions."""
import logging
import pandas as pd
from sqlalchemy import delete, func, insert, select
from ml.data import from_csv, synthetic
from ml.features import FEATURES
from ml.train import train_from_frame
from .config import settings
from .ml_service import SERVICE
from .models import Customer, Prediction, User
from .security import hash_password

log = logging.getLogger("kestrel")

def ensure_users(db):
    if db.scalar(select(func.count(User.id))):
        return
    db.add_all([
        User(email=settings.demo_email, name="Demo Analyst", role="analyst", password_hash=hash_password(settings.demo_password)),
        User(email=settings.admin_email, name="Administrator", role="admin", password_hash=hash_password(settings.admin_password)),
    ])
    db.commit()
    log.info("Created demo and admin users")

def ensure_customers(db):
    if db.scalar(select(func.count(Customer.id))):
        return
    if settings.data_csv.exists():
        df, source = from_csv(settings.data_csv), str(settings.data_csv.name)
    else:
        df, source = synthetic(settings.synthetic_rows), "synthetic"
    rows = [{"id": r.customer_id, "segment": r.segment, "status": "Not contacted", "churned": int(r.churn),
             **{f: float(getattr(r, f)) for f in FEATURES}} for r in df.itertuples()]
    for i in range(0, len(rows), 2000):
        db.execute(insert(Customer), rows[i:i + 2000])
    db.commit()
    log.info("Loaded %d customers from %s", len(rows), source)

def customers_frame(db, labelled_only=False) -> pd.DataFrame:
    cols = [Customer.id, Customer.segment, Customer.churned] + [getattr(Customer, f) for f in FEATURES]
    stmt = select(*cols)
    if labelled_only:
        stmt = stmt.where(Customer.churned.is_not(None))
    return pd.read_sql(stmt, db.connection())

def retrain(db, source="database"):
    df = customers_frame(db, labelled_only=True).rename(columns={"churned": "churn"})
    train_from_frame(df, settings.artifact_dir, settings.cost_per_contact, source=f"SQL table customers ({len(df)} rows)")
    SERVICE.load()
    return rescore_all(db)

def rescore_all(db) -> int:
    df = customers_frame(db)
    p, c = SERVICE.score(df)
    top = c.argmax(axis=1)
    rows = [{"customer_id": cid, "probability": float(pi), "model_name": SERVICE.name,
             "top_factor": FEATURES[int(t)] if c[i, int(t)] > 0 else ""}
            for i, (cid, pi, t) in enumerate(zip(df["id"], p, top))]
    db.execute(delete(Prediction))
    for i in range(0, len(rows), 2000):
        db.execute(insert(Prediction), rows[i:i + 2000])
    db.commit()
    return len(rows)

def bootstrap(db):
    ensure_users(db)
    ensure_customers(db)
    if SERVICE.available():
        SERVICE.load()
    else:
        log.info("No trained model found: training from the customers table (first start takes about a minute)")
        retrain(db)
        return
    if not db.scalar(select(func.count(Prediction.customer_id))):
        rescore_all(db)
