import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class Settings:
    # Leave DATABASE_URL empty to use a local SQLite file. For SQL Server use e.g.
    # mssql+pyodbc://sa:PASSWORD@localhost:1433/kestrel?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
    database_url = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{ROOT / 'kestrel.db'}"
    secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
    token_minutes = int(os.getenv("TOKEN_MINUTES", "60"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", str(ROOT / "artifacts")))
    data_csv = Path(os.getenv("DATA_CSV", str(ROOT / "data" / "BankChurners.csv")))
    synthetic_rows = int(os.getenv("SYNTHETIC_ROWS", "6000"))
    cost_per_contact = float(os.getenv("COST_PER_CONTACT", "25"))
    admin_email = os.getenv("ADMIN_EMAIL", "admin@kestrel.io")
    admin_password = os.getenv("ADMIN_PASSWORD", "Admin@12345")
    demo_email = "demo@kestrel.io"
    demo_password = "Demo@12345"

settings = Settings()
