"""Run:  pytest -q      (uses a throw-away SQLite database and 1,500 synthetic customers)"""
import os, tempfile
_tmp = tempfile.mkdtemp()
os.environ.update(DATABASE_URL=f"sqlite:///{_tmp}/t.db", ARTIFACT_DIR=f"{_tmp}/art", SYNTHETIC_ROWS="1500", SECRET_KEY="test")

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def headers(client, email="demo@kestrel.io", pw="Demo@12345"):
    r = client.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}

def test_health_is_public(client):
    assert client.get("/api/health").json()["customers"] == 1500

def test_protected_routes_need_a_token(client):
    assert client.get("/api/overview").status_code == 401

def test_wrong_password_and_lockout(client):
    for _ in range(5):
        assert client.post("/api/auth/login", json={"email": "x@y.com", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "x@y.com", "password": "nope"}).status_code == 429

def test_overview_adds_up(client):
    o = client.get("/api/overview", headers=headers(client)).json()
    assert o["total"] == 1500 and o["high"] + o["medium"] + o["low"] == 1500 and len(o["histogram"]) == 10

def test_customer_list_detail_and_action(client):
    h = headers(client)
    items = client.get("/api/customers?limit=5", headers=h).json()["items"]
    assert len(items) == 5 and items[0]["probability"] >= items[-1]["probability"]
    d = client.get(f"/api/customers/{items[0]['id']}", headers=h).json()
    assert len(d["features"]) == 12 and d["tier"] in ("High", "Medium", "Low")
    r = client.post(f"/api/customers/{d['id']}/action", json={"status": "Contacted"}, headers=h)
    assert r.status_code == 200
    assert client.get(f"/api/customers/{d['id']}", headers=h).json()["status"] == "Contacted"

def test_viewer_cannot_change_status(client):
    admin = headers(client, "admin@kestrel.io", "Admin@12345")
    client.post("/api/users", json={"email": "v@kestrel.io", "name": "V", "password": "Viewer@12345", "role": "viewer"}, headers=admin)
    v = headers(client, "v@kestrel.io", "Viewer@12345")
    cid = client.get("/api/customers?limit=1", headers=v).json()["items"][0]["id"]
    assert client.post(f"/api/customers/{cid}/action", json={"status": "Saved"}, headers=v).status_code == 403

def test_predict_does_not_save(client):
    h = headers(client)
    body = dict(customer_age=45, months_on_book=36, total_relationship_count=2, months_inactive_12_mon=4,
                contacts_count_12_mon=4, credit_limit=5000, total_revolving_bal=300, total_amt_chng_q4_q1=.5,
                total_trans_amt=1500, total_trans_ct=25, total_ct_chng_q4_q1=.3, avg_utilization_ratio=.06)
    r = client.post("/api/predict", json=body, headers=h).json()
    assert 0 <= r["probability"] <= 1
    assert client.get("/api/health").json()["customers"] == 1500

def test_models_were_compared(client):
    m = client.get("/api/model", headers=headers(client)).json()
    assert len(m["models"]) >= 3 and m["selected"] in [x["name"] for x in m["models"]]
