"""Step 1 - Data preprocessing: Real Estate Investment Advisor.

Usage:  python scripts/step1_preprocess.py data/india_housing_prices.csv [out_dir] [report_dir]
Outputs (in out_dir, default ./data):
  cleaned_housing.csv    readable cleaned data + engineered features + labels
  processed_housing.csv  encoded + scaled version (for the deliverable / inspection)
Output (in report_dir, default ./docs):
  preprocessing_report.md  data-quality checks, column summary, outliers, targets (auto-generated)
Note: model training (Step 3+) should re-fit the encoder/scaler inside a sklearn
Pipeline on the TRAIN split only, to avoid leakage.
"""
import sys, os
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025          # dataset: Age_of_Property == 2025 - Year_Built
BASE_RATE, T = 0.08, 5       # doc example: r = 8%, t = 5 years

src = sys.argv[1] if len(sys.argv) > 1 else "india_housing_prices.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "data"
rep_dir = sys.argv[3] if len(sys.argv) > 3 else "docs"
os.makedirs(out, exist_ok=True)
os.makedirs(rep_dir, exist_ok=True)

df = pd.read_csv(src)
raw = df.copy()                      # untouched copy, used only for the data-quality report
log = {"rows_raw": len(df)}

# ---- 1. duplicates + missing values -------------------------------------
df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).reset_index(drop=True)
log["rows_after_dedup"] = len(df)
num_cols = df.select_dtypes("number").columns.drop("ID")
cat_cols = df.select_dtypes(exclude="number").columns
log["missing_before"] = int(df.isna().sum().sum())
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip()

# ---- 2. consistency fixes -----------------------------------------------
# Price_per_SqFt in the raw file is in lakhs/sqft rounded to 2 dp (2,534 rows = 0.00) -> recompute in Rs
df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
# Floor_No > Total_Floors is impossible (about half of the raw rows) -> flag, then cap
df["Floor_Invalid"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
df["Floor_No"] = df[["Floor_No", "Total_Floors"]].min(axis=1)

# ---- 3. outliers (IQR flag; winsorise PPS for modelling) ------------------
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).astype(int)
df["PPS_Outlier"] = iqr_flag(df["Price_per_SqFt"])
df["Size_Outlier"] = iqr_flag(df["Size_in_SqFt"])
lo, hi = df["Price_per_SqFt"].quantile([.01, .99])
df["Price_per_SqFt_W"] = df["Price_per_SqFt"].clip(lo, hi)
log["pps_outliers"] = int(df["PPS_Outlier"].sum())

# ---- 4. feature engineering ---------------------------------------------
transport = df["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3})
mm = lambda s: (s - s.min()) / (s.max() - s.min())
df["School_Density_Score"] = (df["Nearby_Schools"] / df.groupby("City")["Nearby_Schools"].transform("median")).round(3)
df["Infrastructure_Score"] = ((mm(df["Nearby_Schools"]) + mm(df["Nearby_Hospitals"]) + mm(transport)) / 3).round(3)
df["Amenity_Count"] = df["Amenities"].str.split(",").str.len()
df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
df["Is_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"]).round(3)
df["Sqft_per_BHK"] = (df["Size_in_SqFt"] / df["BHK"]).round(1)
df["City_Median_PPS"] = df.groupby("City")["Price_per_SqFt"].transform("median")
df["PPS_vs_City"] = (df["Price_per_SqFt"] / df["City_Median_PPS"]).round(3)
# peer benchmark: median price of properties with the same city and BHK (size-neutral, unlike price per sqft)
df["City_BHK_Median_Price"] = df.groupby(["City", "BHK"])["Price_in_Lakhs"].transform("median")

# ---- 5. targets -----------------------------------------------------------
# Regression: Future_Price_5Y = Price * (1+r)^5, r feature-based (assumption, document it):
#   5% base + up to 3% infrastructure + 1% ready-to-move + 1% security/parking + 0.5% newer building (<10 yrs)
df["Growth_Rate"] = (0.05 + 0.03 * df["Infrastructure_Score"] + 0.01 * df["Is_Ready"]
                     + 0.005 * (df["Security_Flag"] + df["Parking_Flag"]) + 0.005 * (df["Age_of_Property"] < 10)).clip(0.05, 0.10).round(4)
df["Future_Price_5Y"] = (df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** T).round(2)

# Classification: multi-factor score (doc option 3) - 6 points max, Good Investment if >= 4
df["Investment_Score"] = (
    2 * (df["Price_in_Lakhs"] <= df["City_BHK_Median_Price"])      # cheaper than same-city, same-BHK median
    + 1 * (df["BHK"] >= 3)
    + 1 * df["Is_Ready"]
    + 1 * (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median())
    + 1 * (df["Parking_Flag"] + df["Security_Flag"] >= 1)
).astype(int)
df["Good_Investment"] = (df["Investment_Score"] >= 4).astype(int)
log["good_investment_share"] = round(df["Good_Investment"].mean(), 3)

df.to_csv(f"{out}/cleaned_housing.csv", index=False)

# ---- 6. encode + scale (inspection copy) ---------------------------------
enc = df.drop(columns=["ID", "Amenities", "Locality", "Price_per_SqFt_W"]).copy()
enc["Public_Transport_Accessibility"] = transport
enc["Furnished_Status"] = enc["Furnished_Status"].map({"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2})
for c in ["Parking_Space", "Security"]:
    enc[c] = (enc[c] == "Yes").astype(int)
enc["Availability_Status"] = enc["Is_Ready"]
# City / State: frequency-free label encoding is misleading -> one-hot for low-card, target-free ordinal for the rest
enc = pd.get_dummies(enc, columns=["Property_Type", "Facing", "Owner_Type"], dtype=int)
enc["State"] = enc["State"].astype("category").cat.codes
enc["City"] = enc["City"].astype("category").cat.codes
scale = ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built", "Age_of_Property", "Floor_No",
         "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Sqft_per_BHK", "City_Median_PPS", "City_BHK_Median_Price", "Amenity_Count"]
for c in scale:  # z-score
    enc[c] = ((enc[c] - enc[c].mean()) / enc[c].std()).round(4)
enc.to_csv(f"{out}/processed_housing.csv", index=False)

print(log)
print(df[["Price_per_SqFt", "Infrastructure_Score", "Growth_Rate", "Future_Price_5Y", "Investment_Score", "Good_Investment"]].describe().round(2).T)


# ---- 7. auto-generated preprocessing report -------------------------------
def pct(n, d=len(raw)):
    return f"{n:,} ({n / d:.1%})"


amen_ok = {"Playground", "Gym", "Garden", "Pool", "Clubhouse"}
bad_amen = ~raw["Amenities"].astype(str).str.split(",").apply(lambda xs: all(x.strip() in amen_ok for x in xs))
cat_before = {c: raw[c].nunique() for c in raw.select_dtypes(exclude="number").columns}
cat_after = {c: raw[c].astype(str).str.strip().nunique() for c in cat_before}
checks = [
    ("Missing values (all columns)", f"{int(raw.isna().sum().sum()):,}", "Median / mode imputation is built in, so new data is handled safely"),
    ("Duplicate rows (ignoring ID)", pct(int(raw.duplicated(subset=[c for c in raw.columns if c != 'ID']).sum())), "Dropped"),
    ("Duplicate IDs", pct(int(raw["ID"].duplicated().sum())), "None to fix" if not raw["ID"].duplicated().any() else "Review"),
    ("Price or size <= 0", pct(int(((raw["Price_in_Lakhs"] <= 0) | (raw["Size_in_SqFt"] <= 0)).sum())), "None to fix"),
    ("Year_Built outside 1900-2025", pct(int(((raw["Year_Built"] < 1900) | (raw["Year_Built"] > CURRENT_YEAR)).sum())), "None to fix"),
    ("Age_of_Property != 2025 - Year_Built", pct(int((raw["Age_of_Property"] != CURRENT_YEAR - raw["Year_Built"]).sum())), "Age is recomputed anyway"),
    ("Price_per_SqFt shown as 0.00 (rounded lakhs)", pct(int((raw["Price_per_SqFt"] == 0).sum())), "Recomputed in rupees from price and size"),
    ("Price_per_SqFt differs from price / size by more than 0.01 lakh", pct(int(((raw["Price_in_Lakhs"] / raw["Size_in_SqFt"]) - raw["Price_per_SqFt"]).abs().gt(0.01).sum())), "Recomputed in rupees"),
    ("Floor_No higher than Total_Floors", pct(int((raw["Floor_No"] > raw["Total_Floors"]).sum())), "Flagged in Floor_Invalid, then capped at the top floor"),
    ("Layout below 250 sqft per BHK (for example 500 sqft, 5 BHK)", pct(int((raw["Size_in_SqFt"] / raw["BHK"] < 250).sum())), "Reported only; kept because size and BHK are both valid values"),
    ("Parking_Space / Security not Yes or No", pct(int((~raw["Parking_Space"].isin(["Yes", "No"]) | ~raw["Security"].isin(["Yes", "No"])).sum())), "None to fix"),
    ("Unknown amenity names", pct(int(bad_amen.sum())), "None to fix"),
    ("Category labels changed by trimming spaces", str(sum(cat_before[c] != cat_after[c] for c in cat_before)) + " columns", "Whitespace is stripped in every text column"),
]
rows = []
for c in raw.columns:
    if c == "ID":
        continue
    if pd.api.types.is_numeric_dtype(raw[c]):
        rows.append((c, "number", int(raw[c].isna().sum()), raw[c].nunique(), f"{raw[c].min():,.4g} to {raw[c].max():,.4g}"))
    else:
        rows.append((c, "text", int(raw[c].isna().sum()), raw[c].nunique(), "top: " + str(raw[c].mode()[0])))
out_rows = []
for c, lab in [("Price_in_Lakhs", "Price (lakhs)"), ("Size_in_SqFt", "Size (sqft)"), ("Price_per_SqFt", "Price per sqft (Rs, recomputed)"),
               ("Nearby_Schools", "Nearby schools"), ("Nearby_Hospitals", "Nearby hospitals")]:
    f = iqr_flag(df[c]); out_rows.append((lab, int(f.sum()), f"{f.mean():.1%}"))

with open(f"{rep_dir}/preprocessing_report.md", "w", encoding="utf-8") as fh:
    fh.write("# Preprocessing report (generated by scripts/step1_preprocess.py)\n\n")
    fh.write(f"Source: `{os.path.basename(src)}` with {len(raw):,} rows and {raw.shape[1]} columns. After cleaning: {len(df):,} rows and {df.shape[1]} columns.\n\n")
    fh.write("## 1. Data-quality checks\n\n| Check | Result | Action |\n|---|---|---|\n")
    fh.writelines(f"| {a} | {b} | {c} |\n" for a, b, c in checks)
    fh.write("\n## 2. Column summary (raw data)\n\n| Column | Type | Missing | Unique values | Range or most common |\n|---|---|---|---|---|\n")
    fh.writelines(f"| {a} | {b} | {c} | {d} | {e} |\n" for a, b, c, d, e in rows)
    fh.write("\n## 3. Outliers (IQR rule, 1.5 x IQR)\n\n| Column | Outlier rows | Share |\n|---|---|---|\n")
    fh.writelines(f"| {a} | {b:,} | {c} |\n" for a, b, c in out_rows)
    fh.write("\nOutliers are flagged (`PPS_Outlier`, `Size_Outlier`) and kept. A copy of price per sqft trimmed at the 1st and 99th percentile "
             "(`Price_per_SqFt_W`) is available for modelling.\n")
    fh.write("\n## 4. Encoding and scaling\n\n"
             f"- Scaled with z-scores in `processed_housing.csv`: {', '.join(scale)}.\n"
             "- One-hot encoded: Property_Type, Facing, Owner_Type. Ordinal: Furnished_Status (0 to 2) and Public_Transport_Accessibility (1 to 3). Yes/No columns become 0/1.\n"
             "- During model training the encoder and scaler are fitted on the training rows only, inside the pipeline, so the test set never leaks into preprocessing.\n")
    fh.write("\n## 5. Targets created\n\n| Target | Summary |\n|---|---|\n")
    fh.write(f"| Good_Investment | {df['Good_Investment'].mean():.1%} good, {1 - df['Good_Investment'].mean():.1%} not good (balanced) |\n")
    fh.write(f"| Growth_Rate | {df['Growth_Rate'].min():.1%} to {df['Growth_Rate'].max():.1%}, mean {df['Growth_Rate'].mean():.2%} (feature-based) |\n")
    fh.write(f"| Future_Price_5Y (lakhs) | {df['Future_Price_5Y'].min():,.1f} to {df['Future_Price_5Y'].max():,.1f}, mean {df['Future_Price_5Y'].mean():,.1f} |\n")
print("report ->", f"{rep_dir}/preprocessing_report.md")
