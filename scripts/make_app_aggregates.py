"""Pre-compute the numbers behind the 'Market insights' tab from ALL 250,000 rows (small CSVs, safe for GitHub).

Usage (from the project folder):
    python scripts/make_app_aggregates.py data/cleaned_housing.csv data
Writes data/agg_heatmap.csv, agg_decade.csv, agg_state.csv, agg_infra.csv, agg_meta.json
"""
import json
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_housing.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "data"
df = pd.read_csv(src, usecols=["City", "State", "Property_Type", "BHK", "Furnished_Status", "Availability_Status",
                               "Year_Built", "Price_per_SqFt", "Good_Investment", "Future_Price_5Y", "Infrastructure_Score"])
df["Build_Decade"] = (df["Year_Built"] // 10 * 10).astype(int).astype(str) + "s"

# 1) city heatmap: one row per City x dimension x category
parts = []
for dim in ["BHK", "Property_Type", "Furnished_Status", "Availability_Status"]:
    g = df.groupby(["City", dim]).agg(n=("Good_Investment", "size"), avg_pps=("Price_per_SqFt", "mean"),
                                      good_rate=("Good_Investment", "mean"),
                                      avg_future_price=("Future_Price_5Y", "mean")).reset_index()
    g = g.rename(columns={dim: "Category"}); g["Category"] = g["Category"].astype(str); g.insert(1, "Dimension", dim)
    parts.append(g)
pd.concat(parts).round(4).to_csv(f"{out}/agg_heatmap.csv", index=False)

# 2) trend by build decade (sum + count so the app can re-group by city or property type)
d = df.groupby(["City", "Property_Type", "Build_Decade"]).agg(n=("Price_per_SqFt", "size"), sum_pps=("Price_per_SqFt", "sum")).reset_index()
d.round(2).to_csv(f"{out}/agg_decade.csv", index=False)

# 3) state and infrastructure
df.groupby("State").agg(n=("Good_Investment", "size"), good_rate=("Good_Investment", "mean")).reset_index().round(4).to_csv(f"{out}/agg_state.csv", index=False)
df["Infra_Band"] = pd.cut(df["Infrastructure_Score"], [-0.01, 0.35, 0.5, 0.65, 1.0], labels=["Low", "Mid", "High", "Top"])
df.groupby("Infra_Band", observed=True).agg(n=("Good_Investment", "size"), good_rate=("Good_Investment", "mean"),
                                            avg_future_price=("Future_Price_5Y", "mean")).reset_index().round(4).to_csv(f"{out}/agg_infra.csv", index=False)
json.dump({"rows": int(len(df)), "good_rate": round(float(df["Good_Investment"].mean()), 4)}, open(f"{out}/agg_meta.json", "w"))
print("saved aggregates for", f"{len(df):,}", "rows ->", out)
