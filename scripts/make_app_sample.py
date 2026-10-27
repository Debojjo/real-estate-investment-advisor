"""Make the small dataset used by the Streamlit app (committed to GitHub; the full CSV is not).

Usage (from the project folder):
    python scripts/make_app_sample.py data/cleaned_housing.csv data/sample_app.csv
Takes 20,000 random rows (about 480 per city on average, same seed every time).
"""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_housing.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/sample_app.csv"
keep = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
        "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities", "Facing",
        "Owner_Type", "Availability_Status", "Infrastructure_Score", "Good_Investment", "Future_Price_5Y"]
df = pd.read_csv(src, usecols=keep)
df.sample(20000, random_state=42).sort_values("ID")[keep].to_csv(dst, index=False)
print("saved", dst)
