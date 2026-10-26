"""Step 1 - Data preprocessing: Real Estate Investment Advisor.

Usage:  python step1_preprocess.py india_housing_prices.csv [out_dir]
Outputs (in out_dir, default ./data):
  cleaned_housing.csv    readable cleaned data + engineered features + labels
  processed_housing.csv  encoded + scaled version (for the deliverable / inspection)
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
os.makedirs(out, exist_ok=True)

df = pd.read_csv(src)
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
# Price_per_SqFt in the raw file is in lakhs/sqft rounded to 2 dp (59 rows = 0.00) -> recompute in Rs
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

# Classification: multi-factor score (doc option 3) - 7 points, Good Investment if >= 4
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
