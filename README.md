# Kestrel: bank customer churn prediction

A full-stack web application that predicts which bank customers are likely to leave, and shows why.

```
Browser dashboard  ──►  FastAPI backend  ──►  SQL Server (customers, predictions, users, actions)
 (HTML/CSS/JS)            │  JWT login, roles
                          └──►  ML: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
                                 compared on held-out data, best one served, SHAP-style explanations
```

## 1. Quick start with Docker (SQL Server included)

Requires Docker Desktop.

```bash
cp .env.example .env        # optional: change the passwords
docker compose up --build
```

Open http://localhost:8000. The first start takes a few minutes: SQL Server boots, the tables are created,
6,000 sample customers are loaded, all four models are trained and every customer is scored.

| Login | Role |
|---|---|
| `demo@kestrel.io` / `Demo@12345` | analyst (can view and update customer status) |
| `admin@kestrel.io` / `Admin@12345` | admin (can also retrain and create users) |

Interactive API documentation: http://localhost:8000/docs

## 2. Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
With no `DATABASE_URL` the app uses a local SQLite file, so it runs immediately. To use SQL Server instead, install
"ODBC Driver 18 for SQL Server" and set:
```
DATABASE_URL=mssql+pyodbc://sa:YOUR_PASSWORD@localhost:1433/kestrel?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
```
The `kestrel` database is created automatically if it does not exist.

## 3. Use the real dataset

1. Download **Credit Card customers** (BankChurners) from Kaggle and save it as `data/BankChurners.csv`.
2. Start with an empty database (`docker compose down -v`, then `docker compose up`). The CSV is loaded into the
   `customers` table, the models are trained from that table, and everyone is re-scored.

Until then the app uses synthetic data with the same columns, so everything works the same way.

## 4. Walkthrough for marking

1. **Login**: server-side sign-in (passwords hashed with PBKDF2, signed tokens, lock-out after 5 wrong tries, 15-minute auto sign-out).
2. **Overview**: risk counts, tier split, probability histogram, ROI estimate, searchable customer table (read from SQL Server).
3. **Customer lookup**: churn probability, per-customer feature contributions, recommended action, status history.
4. **Segments**: risk by card type.
5. **Model**: four models compared (ROC-AUC, PR-AUC, Brier, 5-fold CV), ROC curve, calibration, cut-off by business value, confusion matrix, feature importance. Admins can retrain.
6. **/docs**: every API endpoint, with try-it-out.

Train from the command line: `python -m ml.train --csv data/BankChurners.csv`

## 5. Deploying

* **Any Docker host** (Render, Railway, Azure Container Apps): deploy this `Dockerfile` and set `DATABASE_URL`, `SECRET_KEY`, `ADMIN_PASSWORD`.
* **Database**: Azure SQL Database works with the same connection string format (`mssql+pyodbc://user:pass@server.database.windows.net:1433/kestrel?driver=ODBC+Driver+18+for+SQL+Server`).
* Before going live: set a long random `SECRET_KEY`, change the admin password, serve over HTTPS, and add multi-factor authentication.

## 6. Tests

```bash
pytest -q
```

## Layout
```
app/       FastAPI backend (main.py, models.py, security.py, seed.py, ml_service.py, db.py)
ml/        data loading, training and comparison of models, explanations
frontend/  index.html, style.css, app.js
tests/     API tests
```

## Honest notes
* The synthetic data shows the pipeline works; it is not real bank data. Report results on BankChurners for real numbers.
* The ROI figures use assumptions (30% save rate, $240 yearly revenue per customer, adjustable contact cost).
* Production would add: monitoring for model drift, a fairness review, audit logging and a data-protection assessment.
