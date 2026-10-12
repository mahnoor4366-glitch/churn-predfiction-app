from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="analyst")  # admin | analyst | viewer
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    segment: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(20), default="Not contacted")
    churned: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # known outcome, used for training
    customer_age: Mapped[float] = mapped_column(Float)
    months_on_book: Mapped[float] = mapped_column(Float)
    total_relationship_count: Mapped[float] = mapped_column(Float)
    months_inactive_12_mon: Mapped[float] = mapped_column(Float)
    contacts_count_12_mon: Mapped[float] = mapped_column(Float)
    credit_limit: Mapped[float] = mapped_column(Float)
    total_revolving_bal: Mapped[float] = mapped_column(Float)
    total_amt_chng_q4_q1: Mapped[float] = mapped_column(Float)
    total_trans_amt: Mapped[float] = mapped_column(Float)
    total_trans_ct: Mapped[float] = mapped_column(Float)
    total_ct_chng_q4_q1: Mapped[float] = mapped_column(Float)
    avg_utilization_ratio: Mapped[float] = mapped_column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    customer_id: Mapped[str] = mapped_column(String(40), ForeignKey("customers.id"), primary_key=True)
    probability: Mapped[float] = mapped_column(Float, index=True)
    top_factor: Mapped[str] = mapped_column(String(60), default="")
    model_name: Mapped[str] = mapped_column(String(40))
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Action(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(40), ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))
    note: Mapped[str] = mapped_column(String(500), default="")
    user_email: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
