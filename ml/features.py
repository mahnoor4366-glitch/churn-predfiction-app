"""Feature list shared by the trainer, the API and the database schema."""
FEATURES = [
    "customer_age", "months_on_book", "total_relationship_count",
    "months_inactive_12_mon", "contacts_count_12_mon", "credit_limit",
    "total_revolving_bal", "total_amt_chng_q4_q1", "total_trans_amt",
    "total_trans_ct", "total_ct_chng_q4_q1", "avg_utilization_ratio",
]
LABELS = {
    "customer_age": "Age", "months_on_book": "Months as customer",
    "total_relationship_count": "Products held", "months_inactive_12_mon": "Months inactive",
    "contacts_count_12_mon": "Contacts with bank", "credit_limit": "Credit limit",
    "total_revolving_bal": "Revolving balance", "total_amt_chng_q4_q1": "Spend change Q4/Q1",
    "total_trans_amt": "Spend (12 mo)", "total_trans_ct": "Transactions (12 mo)",
    "total_ct_chng_q4_q1": "Txn count change Q4/Q1", "avg_utilization_ratio": "Card utilisation",
}
