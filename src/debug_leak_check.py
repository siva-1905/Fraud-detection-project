import pandas as pd

X_train = pd.read_csv("data/processed/X_train_fe.csv")

# Keywords that indicate a feature derived AFTER fraud was known
LEAK_KEYWORDS = [
    "fraud", "label", "flag", "class", "target", "status",
    "is_fraud", "outcome", "result", "declined", "blocked",
    "chargeback", "dispute", "suspicious", "alert", "review",
    "score", "risk_tag", "decision"
]

print("=== ALL FEATURE NAMES ===")
for col in X_train.columns:
    hit = any(kw in col.lower() for kw in LEAK_KEYWORDS)
    marker = "  <-- INVESTIGATE" if hit else ""
    print(f"  {col}{marker}")