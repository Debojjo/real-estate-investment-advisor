"""Shared feature code for training (step3_train.py) and the Streamlit app (Step 4).
Both call add_features() so the model sees identical inputs in training and in the app."""
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025
CAT_FEATURES = ["State", "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
                "Facing", "Owner_Type", "Availability_Status"]
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Price_vs_Peer", "Age_of_Property",
                "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
                "Infrastructure_Score", "Amenity_Count", "Parking_Flag", "Security_Flag"]
PRICE_FEATURES = ["Price_in_Lakhs", "Price_per_SqFt", "Price_vs_Peer"]   # used in the leakage / ablation check
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def build_peer_table(df):
    """Median price (lakhs) per City + BHK, plus fallbacks for unseen combinations."""
    peer = df.groupby(["City", "BHK"])["Price_in_Lakhs"].median().rename("Peer_Median_Price").reset_index()
    return {"table": peer, "bhk": df.groupby("BHK")["Price_in_Lakhs"].median().to_dict(),
            "global": float(df["Price_in_Lakhs"].median())}


def add_features(df, peer):
    """df needs the raw columns of the dataset (Amenities as text, or Amenity_Count)."""
    d = df.copy()
    d["Floor_No"] = np.minimum(d["Floor_No"], d["Total_Floors"])
    d["Age_of_Property"] = CURRENT_YEAR - d["Year_Built"]
    d["Price_per_SqFt"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    d["Floor_Ratio"] = d["Floor_No"] / d["Total_Floors"]
    transport = d["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3})
    d["Infrastructure_Score"] = ((d["Nearby_Schools"] - 1) / 9 + (d["Nearby_Hospitals"] - 1) / 9 + (transport - 1) / 2) / 3
    if "Amenities" in d.columns:
        d["Amenity_Count"] = d["Amenities"].str.split(",").str.len()
    d["Parking_Flag"] = (d["Parking_Space"] == "Yes").astype(int)
    d["Security_Flag"] = (d["Security"] == "Yes").astype(int)
    d = d.merge(peer["table"], on=["City", "BHK"], how="left")
    fallback = d["BHK"].map(peer["bhk"]).fillna(peer["global"])
    d["Peer_Median_Price"] = d["Peer_Median_Price"].fillna(fallback)
    d["Price_vs_Peer"] = d["Price_in_Lakhs"] / d["Peer_Median_Price"]
    return d
