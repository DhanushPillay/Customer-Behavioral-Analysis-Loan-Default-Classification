"""Check the numbers the dashboard shows against the stored artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app

features = app.load_features()
scores = app.load_scores()
model = app.load_model()
columns = app.model_columns(model, features)


def close(value: float, expected: float, tolerance: float = 0.02) -> None:
    assert abs(value - expected) <= tolerance, f"{value:.4f} != {expected:.4f}"


assert len(features) == 682, len(features)
assert int(features["target"].sum()) == 76
close(features["target"].mean(), 0.1114, 0.0001)

early = app.cutoff_stats(scores, features, 0.10)
assert (early["flagged"], early["tp"]) == (152, 58), early
close(early["precision"], 0.382)
close(early["recall"], 0.763)

at_threshold = app.cutoff_stats(scores, features, 0.40)
assert (at_threshold["flagged"], at_threshold["tp"]) == (50, 35), at_threshold
close(at_threshold["precision"], 0.70)
close(at_threshold["recall"], 0.4605)
close(at_threshold["exposure"] / 1e6, 10.6, 0.1)

audit = app.leakage_audit()
mismatch = int((audit["n_txn"] != audit["used"]).sum())
assert mismatch == 0, f"{mismatch} loans use post-loan transactions"

row = app.applicant_row(
    app.median_row(features),
    {
        "amount": 116_928,
        "duration": 36,
        "balance_at_loan": 38_549.60,
        "min_balance": 600,
        "txn_per_month": 5.70,
    },
)
close(app.predict(model, row, columns), 0.0143, 0.002)
assert len(columns) == 18, len(columns)

print("numbers match the artifacts")
print(f"cutoff 0.40 -> {at_threshold['flagged']} flagged, {at_threshold['tp']}/76 caught")
print(f"screenshot applicant -> {app.predict(model, row, columns):.4f}")
