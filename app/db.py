import time
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from .config import settings

def _ensure_sqlserver_database(url):
    """SQL Server starts empty: create the database if it does not exist (waits for the server to boot)."""
    u = make_url(url)
    name = u.database
    master = create_engine(u.set(database="master"), isolation_level="AUTOCOMMIT")
    last = None
    for _ in range(40):
        try:
            with master.connect() as c:
                c.execute(text(f"IF DB_ID(N'{name}') IS NULL CREATE DATABASE [{name}]"))
            master.dispose()
            return
        except Exception as e:  # server still starting
            last = e
            time.sleep(3)
    raise RuntimeError(f"SQL Server is not reachable: {last}")

def make_engine(url=None):
    url = url or settings.database_url
    if url.startswith("mssql"):
        _ensure_sqlserver_database(url)
        return create_engine(url, pool_pre_ping=True, fast_executemany=True)
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, **kw)

engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
DIALECT = engine.dialect.name

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
