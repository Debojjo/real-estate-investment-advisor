"""Step 1b - Justify and stress-test the Good_Investment rule.

Usage (from the project folder):  python scripts/step1b_label_justification.py data/cleaned_housing.csv docs
Writes docs/label_justification.md: why each criterion is in the score, how often it is met,
and how the label changes if the weights or the cut-off change.
"""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_housing.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "docs"
df = pd.read_csv(src)

comp = pd.DataFrame({
    "peer": (df["Price_in_Lakhs"] <= df["City_BHK_Median_Price"]).astype(int),
    "bhk": (df["BHK"] >= 3).astype(int),
    "ready": df["Is_Ready"].astype(int),
    "infra": (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()).astype(int),
    "extras": ((df["Parking_Flag"] + df["Security_Flag"]) >= 1).astype(int),
})
score = 2 * comp["peer"] + comp[["bhk", "ready", "infra", "extras"]].sum(axis=1)
assert (score == df["Investment_Score"]).all(), "score does not match the cleaned file"

rationale = [
    ("Price at or below the median of the same city and BHK", "peer", 2,
     "Price compared with similar properties is the most direct value signal for an investor, and it is the only criterion about cost. It gets double weight. The benchmark uses city and BHK, not price per sqft, because price per sqft follows size in this data and would reward large properties."),
    ("BHK of 3 or more", "bhk", 1, "Larger homes suit families, who form the widest group of buyers and tenants, so resale and rental demand is broader (a domain assumption)."),
    ("Ready to move", "ready", 1, "No construction delay risk, and the owner can rent or sell immediately."),
    ("Infrastructure score at or above the median", "infra", 1, "Nearby schools, hospitals and public transport support demand and later appreciation."),
    ("Parking or security available", "extras", 1, "Facilities that buyers expect and pay for, which help resale."),
]
rows = [(name, w, f"{comp[k].mean():.1%}", why) for name, k, w, why in rationale]

dist = score.value_counts().sort_index()
share = {t: float((score >= t).mean()) for t in (3, 4, 5)}
equal = (comp.sum(axis=1) >= 3).astype(int)                     # every criterion worth 1 point, good if 3 of 5
agree_equal = float((equal == df["Good_Investment"]).mean())
no_price = ((comp[["bhk", "ready", "infra", "extras"]].sum(axis=1)) >= 3).astype(int)   # without the price criterion
agree_no_price = float((no_price == df["Good_Investment"]).mean())

with open(f"{out}/label_justification.md", "w", encoding="utf-8") as f:
    f.write("# Why the Good_Investment rule looks the way it does\n\n")
    f.write("The dataset has no sale history, so the brief's own option is used: a label from domain rules. "
            "Each property earns points out of 6, and 4 or more is a Good Investment. The weights and the cut-off are my judgement, "
            "not market facts, so this file also shows how sensitive the label is to them.\n\n")
    f.write("## 1. The criteria\n\n| Criterion | Points | Share of properties meeting it | Reason |\n|---|---|---|---|\n")
    f.writelines(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows)
    f.write("\n## 2. Why a cut-off of 4 out of 6\n\n")
    f.write("| Cut-off | Share labelled Good |\n|---|---|\n")
    f.writelines(f"| {t} or more | {v:.1%} |\n" for t, v in share.items())
    f.write("\nA cut-off of 4 means a property needs either good value (2 points) plus two other strengths, or all four of the other points "
            "without the value point. It gives a balanced split (about 47% good), so no class-imbalance technique is needed. "
            "A cut-off of 3 would label most properties good, and 5 would label very few.\n\n")
    f.write("Score distribution (points: share of properties): " + ", ".join(f"{k}: {v / len(score):.1%}" for k, v in dist.items()) + "\n\n")
    f.write("## 3. Sensitivity of the label\n\n| Alternative rule | Agreement with the chosen label |\n|---|---|\n")
    f.write(f"| All five criteria worth 1 point, Good if 3 or more | {agree_equal:.1%} |\n")
    f.write(f"| Without the price criterion (4 quality criteria, Good if 3 or more) | {agree_no_price:.1%} |\n\n")
    f.write("The label is a deterministic rule of the inputs. This is why the models score very high, and why removing the price inputs "
            "lowers accuracy to about 65%. The rule is documented so the limitation is visible, not hidden.\n")
print(open(f"{out}/label_justification.md", encoding="utf-8").read())
