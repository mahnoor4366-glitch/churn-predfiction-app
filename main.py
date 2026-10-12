"""Kestrel churn API (FastAPI). Run:  uvicorn app.main:app --reload   ->  http://localhost:8000  (API docs: /docs)"""
import io
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ml.features import FEATURES, LABELS
from .db import DIALECT, SessionLocal, engine, get_db
from .config import ROOT, settings
from .ml_service import ACTIONS, SERVICE
from .models import Action, Base, Customer, Prediction, User
from .agent import AgentUnavailable, run_agent
from .security import (clear_failures, current_user, hash_password, locked_seconds, make_token,
                       record_failure, require_role, verify_password)
from .seed import bootstrap, retrain

STATUSES = ["Not contacted", "Contacted", "Saved", "Lost"]

log = logging.getLogger("kestrel")
READY = threading.Event()
STARTUP_ERROR: list = []

def _boot():
    try:
        with SessionLocal() as db:
            bootstrap(db)
        READY.set()
    except Exception as e:  # shown to the user instead of a silent hang
        log.exception("Start-up failed")
        STARTUP_ERROR.append(f"The app failed to start: {e}"[:300])

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    if settings.background_start:  # free hosts: open the port now, train the model in the background
        threading.Thread(target=_boot, daemon=True).start()
    else:
        _boot()
        if STARTUP_ERROR:
            raise RuntimeError(STARTUP_ERROR[0])
    yield

app = FastAPI(title="Kestrel Churn API", version="1.0", lifespan=lifespan)

@app.middleware("http")
async def wait_until_ready(request, call_next):
    if request.url.path.startswith("/api") and request.url.path != "/api/health" and not READY.is_set():
        msg = STARTUP_ERROR[0] if STARTUP_ERROR else "The app is starting up and training its model. Please try again in about a minute."
        return JSONResponse({"detail": msg}, status_code=503)
    return await call_next(request)

# ---------- schemas
class LoginIn(BaseModel):
    email: str
    password: str

class UserIn(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=10)
    role: str = "analyst"

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=20)

class BatchIn(BaseModel):
    csv: str = Field(min_length=1, max_length=2_000_000)

class ActionIn(BaseModel):
    status: str
    note: str = Field("", max_length=500)

class CustomerIn(BaseModel):
    id: Optional[str] = Field(None, max_length=40)
    segment: str = Field("Standard", max_length=40)
    customer_age: float = Field(ge=18, le=100)
    months_on_book: float = Field(ge=0, le=600)
    total_relationship_count: float = Field(ge=0, le=20)
    months_inactive_12_mon: float = Field(ge=0, le=12)
    contacts_count_12_mon: float = Field(ge=0, le=50)
    credit_limit: float = Field(ge=0)
    total_revolving_bal: float = Field(ge=0)
    total_amt_chng_q4_q1: float = Field(ge=0)
    total_trans_amt: float = Field(ge=0)
    total_trans_ct: float = Field(ge=0)
    total_ct_chng_q4_q1: float = Field(ge=0)
    avg_utilization_ratio: float = Field(ge=0, le=1)

def _explain(body_or_customer):
    df = pd.DataFrame([{f: float(getattr(body_or_customer, f)) for f in FEATURES}])
    p, c = SERVICE.score(df)
    p, cs = float(p[0]), c[0]
    feats = sorted(
        [{"key": f, "label": LABELS[f], "value": float(getattr(body_or_customer, f)), "contribution": float(cs[i])}
         for i, f in enumerate(FEATURES)], key=lambda d: -abs(d["contribution"]))
    return p, feats

def _action(tier, feats):
    top = max(feats, key=lambda d: d["contribution"])
    title, text = {
        "High": ("Personal outreach this week", "Call or message from the relationship team within 48 hours, with a retention offer."),
        "Medium": ("Re-engagement nudge", "Send a push notification and an in-app tip. Hold the offer unless risk rises."),
        "Low": ("No action needed", "Keep the experience smooth and review again next cycle."),
    }[tier]
    why = ACTIONS.get(top["key"]) if top["contribution"] > 0 and tier != "Low" else None
    return {"title": title, "text": text, "why": why}

# ---------- health and auth
@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    sel = next((r for r in SERVICE.metrics.get("models", []) if r["name"] == SERVICE.name), None)
    return {"status": "ok", "ready": READY.is_set(), "database": {"mssql": "SQL Server", "sqlite": "SQLite"}.get(DIALECT, DIALECT),
            "customers": db.scalar(select(func.count(Customer.id))), "model": SERVICE.name,
            "roc_auc": sel["roc_auc"] if sel else None}

@app.post("/api/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    wait = locked_seconds(email)
    if wait:
        raise HTTPException(429, f"Too many attempts. Try again in {wait} seconds.")
    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(body.password, user.password_hash):
        record_failure(email)
        raise HTTPException(401, "Email or password is incorrect.")
    clear_failures(email)
    return {"access_token": make_token(user.email, user.role), "token_type": "bearer",
            "user": {"name": user.name, "email": user.email, "role": user.role}}

@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"name": user.name, "email": user.email, "role": user.role}

@app.get("/api/users")
def list_users(db: Session = Depends(get_db), admin: User = Depends(require_role("admin"))):
    return [{"email": u.email, "name": u.name, "role": u.role} for u in db.scalars(select(User))]

@app.post("/api/users", status_code=201)
def create_user(body: UserIn, db: Session = Depends(get_db), admin: User = Depends(require_role("admin"))):
    if body.role not in ("admin", "analyst", "viewer"):
        raise HTTPException(422, "Role must be admin, analyst or viewer")
    email = body.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "That email already exists")
    db.add(User(email=email, name=body.name.strip()[:120], role=body.role, password_hash=hash_password(body.password)))
    db.commit()
    return {"email": email, "role": body.role}

# ---------- dashboard data
@app.get("/api/overview")
def overview(db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = SERVICE.threshold
    ps = [float(x) for x in db.scalars(select(Prediction.probability))]
    n = len(ps)
    high = [p for p in ps if p >= t]
    med = sum(1 for p in ps if t / 2 <= p < t)
    hist = [0] * 10
    for p in ps:
        hist[min(9, int(p * 10))] += 1
    sel = next(r for r in SERVICE.metrics["models"] if r["name"] == SERVICE.name)
    return {"total": n, "high": len(high), "medium": med, "low": n - len(high) - med,
            "avg_risk": sum(ps) / n if n else 0, "threshold": t, "histogram": hist,
            "expected_churners_high": sum(high),
            "model": {"name": SERVICE.name, "roc_auc": sel["roc_auc"], "pr_auc": sel["pr_auc"]}}

@app.get("/api/customers")
def customers(q: str = "", tier: str = "", status: str = "", limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
              db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = SERVICE.threshold
    stmt = (select(Customer.id, Customer.segment, Customer.status, Prediction.probability, Prediction.top_factor)
            .join(Prediction, Prediction.customer_id == Customer.id))
    if q:
        stmt = stmt.where(Customer.id.like(f"%{q.strip()}%"))
    if status:
        stmt = stmt.where(Customer.status == status)
    if tier == "High":
        stmt = stmt.where(Prediction.probability >= t)
    elif tier == "Medium":
        stmt = stmt.where(Prediction.probability >= t / 2, Prediction.probability < t)
    elif tier == "Low":
        stmt = stmt.where(Prediction.probability < t / 2)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.execute(stmt.order_by(Prediction.probability.desc(), Customer.id).limit(limit).offset(offset)).all()
    return {"total": total, "items": [
        {"id": r.id, "segment": r.segment, "status": r.status, "probability": float(r.probability),
         "tier": SERVICE.tier(float(r.probability)), "top_factor": LABELS.get(r.top_factor, "")} for r in rows]}

@app.get("/api/customers/{cid}")
def customer(cid: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    c = db.get(Customer, cid)
    if not c:
        raise HTTPException(404, "Customer not found")
    p, feats = _explain(c)
    tier = SERVICE.tier(p)
    history = db.scalars(select(Action).where(Action.customer_id == cid).order_by(Action.created_at.desc()).limit(10))
    return {"id": c.id, "segment": c.segment, "status": c.status, "probability": p, "tier": tier,
            "features": feats, "action": _action(tier, feats),
            "history": [{"status": a.status, "note": a.note, "by": a.user_email, "at": a.created_at.isoformat()} for a in history]}

@app.post("/api/customers/{cid}/action")
def set_action(cid: str, body: ActionIn, db: Session = Depends(get_db),
               user: User = Depends(require_role("admin", "analyst"))):
    c = db.get(Customer, cid)
    if not c:
        raise HTTPException(404, "Customer not found")
    if body.status not in STATUSES:
        raise HTTPException(422, f"Status must be one of {STATUSES}")
    c.status = body.status
    db.add(Action(customer_id=cid, status=body.status, note=body.note, user_email=user.email))
    db.commit()
    return {"status": c.status}

@app.post("/api/customers", status_code=201)
def add_customer(body: CustomerIn, db: Session = Depends(get_db),
                 user: User = Depends(require_role("admin", "analyst"))):
    cid = (body.id or f"C-{uuid.uuid4().hex[:8]}").strip()
    if db.get(Customer, cid):
        raise HTTPException(409, "A customer with that id already exists")
    p, feats = _explain(body)
    db.add(Customer(id=cid, segment=body.segment, status="Not contacted", churned=None, **{f: getattr(body, f) for f in FEATURES}))
    db.add(Prediction(customer_id=cid, probability=p, model_name=SERVICE.name,
                      top_factor=max(feats, key=lambda d: d["contribution"])["key"]))
    db.commit()
    return {"id": cid, "probability": p, "tier": SERVICE.tier(p)}

@app.post("/api/predict")
def predict(body: CustomerIn, user: User = Depends(current_user)):
    """Score a customer without saving them."""
    p, feats = _explain(body)
    tier = SERVICE.tier(p)
    return {"probability": p, "tier": tier, "features": feats, "action": _action(tier, feats)}

def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

@app.post("/api/predict/batch")
def predict_batch(body: BatchIn, user: User = Depends(current_user)):
    """Score every row of an uploaded CSV. Nothing is saved to the database."""
    try:
        df = pd.read_csv(io.StringIO(body.csv))
    except Exception:
        raise HTTPException(422, "Could not read that file as a CSV.")
    df.columns = [_norm(c) for c in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(422, "Missing columns: " + ", ".join(missing))
    if len(df) > 5000:
        raise HTTPException(422, "Too many rows. The limit is 5,000.")
    idcol = next((c for c in ("customer_id", "id", "clientnum", "customer") if c in df.columns), None)
    ids = df[idcol].astype(str) if idcol else pd.Series([f"Row {i + 1}" for i in range(len(df))], index=df.index)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(axis=1) & (X >= 0).all(axis=1)
    if not ok.any():
        raise HTTPException(422, "No valid rows. Values must be numbers of zero or more.")
    probs, contribs = SERVICE.score(X[ok])
    items = []
    for cid, p, cs in zip(ids[ok], probs, contribs):
        j = int(max(range(len(FEATURES)), key=lambda k: cs[k]))
        items.append({"id": cid, "probability": float(p), "tier": SERVICE.tier(float(p)),
                      "top_factor": LABELS[FEATURES[j]] if cs[j] > 0 else ""})
    items.sort(key=lambda d: -d["probability"])
    n = len(items)
    count = lambda t: sum(1 for d in items if d["tier"] == t)
    return {"rows": int(len(df)), "scored": n, "skipped": int(len(df)) - n, "high": count("High"),
            "medium": count("Medium"), "low": count("Low"),
            "avg_risk": sum(d["probability"] for d in items) / n, "items": items}

@app.get("/api/segments")
def segments(db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = SERVICE.threshold
    j = Prediction.customer_id == Customer.id
    rows = db.execute(select(Customer.segment, func.count(), func.avg(Prediction.probability),
                             func.sum(case((Prediction.probability >= t, 1), else_=0)))
                      .join(Prediction, j).group_by(Customer.segment)).all()
    tops = db.execute(select(Customer.segment, Prediction.top_factor, func.count())
                      .join(Prediction, j).where(Prediction.top_factor != "")
                      .group_by(Customer.segment, Prediction.top_factor)).all()
    best = {}
    for seg, f, n in tops:
        if seg not in best or n > best[seg][1]:
            best[seg] = (f, n)
    return {"segments": [{"name": s, "customers": int(n), "avg_risk": float(a or 0), "high": int(h or 0),
                          "top_driver": LABELS.get(best.get(s, ("", 0))[0], "None")} for s, n, a, h in rows]}

@app.get("/api/model")
def model_info(user: User = Depends(current_user)):
    return SERVICE.metrics

@app.post("/api/model/retrain")
def retrain_model(db: Session = Depends(get_db), admin: User = Depends(require_role("admin"))):
    """Re-train every model on the labelled rows in the database and re-score all customers."""
    n = retrain(db)
    return {"rescored": n, "selected": SERVICE.name}

@app.post("/api/agent/chat")
def agent_chat(body: ChatIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Ask the LangGraph assistant a question. It can look up customers and (for non-viewers) update statuses."""
    try:
        return run_agent(db, user, body.history, body.message)
    except AgentUnavailable as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(502, "The AI assistant failed: " + str(e)[:200])

# The dashboard is served from the same server, so no CORS setup is needed.
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="ui")
