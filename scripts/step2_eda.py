"""Step 2 - EDA: Real Estate Investment Advisor (20 questions from the brief + 2 extras).

Usage (from project folder):  python scripts/step2_eda.py data/cleaned_housing.csv docs/eda
Outputs: one PNG per question (q01_*.png ...) and eda_insights.md (numbers computed from the data).
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

src = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_housing.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "docs/eda"
os.makedirs(out, exist_ok=True)
sns.set_theme(style="whitegrid")
df = pd.read_csv(src)
df["City_Locality"] = df["City"] + " / " + df["Locality"]
md = ["# EDA insights (auto-generated from the data)\n"]
PPS = "Price_per_SqFt"  # Rs per sqft


def save(name):
    plt.tight_layout()
    plt.savefig(f"{out}/{name}.png", dpi=110)
    plt.close()


def note(q, title, text):
    md.append(f"## {q}. {title}\n{text}\n")


# ---------------- 1-5 Price & size ----------------
q = 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Price_in_Lakhs"], bins=40, ax=ax[0]); ax[0].set_title("Price (lakhs)")
sns.histplot(np.log10(df[PPS]), bins=40, ax=ax[1]); ax[1].set_title("log10 price per sqft")
save("q01_price_distribution")
s = df["Price_in_Lakhs"]
note(1, "Distribution of property prices",
     f"Price ranges {s.min():.1f}-{s.max():.1f} lakhs, mean {s.mean():.1f}, median {s.median():.1f}, "
     f"skew {s.skew():.2f}. The shape is close to uniform, not the right-skewed shape real markets show, "
     f"which fits a synthetic dataset.")

fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(df["Size_in_SqFt"], bins=40, ax=ax); ax.set_title("Size (sqft)")
save("q02_size_distribution")
s = df["Size_in_SqFt"]
note(2, "Distribution of property sizes",
     f"Sizes run {s.min():.0f}-{s.max():.0f} sqft, median {s.median():.0f}, skew {s.skew():.2f} (again near-uniform).")

fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(data=df, x="Property_Type", y=PPS, showfliers=False, ax=ax)
ax.set_title("Price per sqft (Rs) by property type")
save("q03_pps_by_type")
g = df.groupby("Property_Type")[PPS].agg(["mean", "median"]).round(0)
note(3, "Price per sqft by property type",
     "Mean / median Rs per sqft:\n" + g.to_markdown() + "\n\nThe gaps between types are small.")

samp = df.sample(5000, random_state=1)
fig, ax = plt.subplots(figsize=(6, 4.5)); sns.scatterplot(data=samp, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.3, s=10, ax=ax)
ax.set_title("Size vs price (5,000-row sample)")
save("q04_size_vs_price")
r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
note(4, "Size vs price", f"Correlation = {r:.3f}. There is essentially no relationship: bigger properties are not priced higher. "
     "In real data this would be strongly positive, so say this explicitly in your report.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df[PPS], ax=ax[0]); ax[0].set_title("Price per sqft (Rs)")
sns.boxplot(x=df["Size_in_SqFt"], ax=ax[1]); ax[1].set_title("Size (sqft)")
save("q05_outliers")
note(5, "Outliers in price per sqft and size",
     f"{int(df['PPS_Outlier'].sum()):,} rows ({df['PPS_Outlier'].mean():.1%}) are IQR outliers on price per sqft "
     f"(max Rs {df[PPS].max():,.0f}); {int(df['Size_Outlier'].sum())} on size. Price per sqft is skewed because it is "
     "price divided by size, so small flats with high prices give very high values. They are flagged and a winsorised copy "
     "(`Price_per_SqFt_W`) is available for modelling.")

# ---------------- 6-10 Location ----------------
st = df.groupby("State")[PPS].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6)); sns.barplot(x=st.values, y=st.index, color="steelblue", ax=ax)
ax.set_title("Average price per sqft (Rs) by state"); save("q06_pps_by_state")
note(6, "Average price per sqft by state",
     f"Highest: {st.index[0]} (Rs {st.iloc[0]:,.0f}); lowest: {st.index[-1]} (Rs {st.iloc[-1]:,.0f}). "
     f"Spread between top and bottom is only {st.iloc[0] / st.iloc[-1] - 1:.0%}.")

ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 8)); sns.barplot(x=ct.values, y=ct.index, color="seagreen", ax=ax)
ax.set_xlim(ct.min() * 0.9, ct.max() * 1.02); ax.set_title("Average price (lakhs) by city (axis truncated)")
save("q07_price_by_city")
note(7, "Average property price by city",
     f"Top: {ct.index[0]} ({ct.iloc[0]:.1f} lakhs), {ct.index[1]} ({ct.iloc[1]:.1f}), {ct.index[2]} ({ct.iloc[2]:.1f}); "
     f"bottom: {ct.index[-1]} ({ct.iloc[-1]:.1f}). City means differ by only ~{ct.max() / ct.min() - 1:.0%}.")

loc_age = df.groupby("City_Locality").agg(median_age=("Age_of_Property", "median"), n=("ID", "size"))
loc_age = loc_age[loc_age["n"] >= 10]
fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(loc_age["median_age"], bins=25, ax=ax)
ax.set_title("Median property age across localities (n>=10 listings)"); save("q08_median_age_by_locality")
note(8, "Median age of properties by locality",
     f"Across {len(loc_age):,} city-localities, median age ranges {loc_age['median_age'].min():.0f}-{loc_age['median_age'].max():.0f} "
     f"years (overall median {df['Age_of_Property'].median():.0f}). Oldest: {loc_age['median_age'].idxmax()}; "
     f"newest: {loc_age['median_age'].idxmin()}. Locality names repeat across cities, so city + locality is used as the key.")

bc = pd.crosstab(df["City"], df["BHK"], normalize="index")
fig, ax = plt.subplots(figsize=(8, 9)); sns.heatmap(bc, annot=False, cmap="YlGnBu", ax=ax)
ax.set_title("BHK share by city"); save("q09_bhk_by_city")
note(9, "BHK distribution across cities",
     f"Overall BHK mix: {df['BHK'].value_counts(normalize=True).sort_index().round(3).to_dict()}. Every city has a near-even split "
     f"(largest deviation from 20% in any cell: {(bc - 0.2).abs().max().max():.1%}).")

top5 = df.groupby("City_Locality").agg(m=(PPS, "mean"), n=("ID", "size"))
top5 = top5[top5["n"] >= 10].sort_values("m", ascending=False).head(5).index
df["Build_Decade"] = (df["Year_Built"] // 10 * 10).astype(int).astype(str) + "s"
tr = df[df["City_Locality"].isin(top5)].groupby(["City_Locality", "Build_Decade"])[PPS].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5)); sns.lineplot(data=tr, x="Build_Decade", y=PPS, hue="City_Locality", marker="o", ax=ax)
ax.set_title("Top-5 most expensive localities: price per sqft by build decade"); ax.legend(fontsize=7)
save("q10_top5_locality_trends")
note(10, "Price trends for the top 5 most expensive localities",
     "Top 5 (n>=10 listings, by mean price per sqft): " + ", ".join(top5) + ". The dataset has no sale dates, so build decade "
     "is used as the time axis. State this limitation clearly; the lines are noisy because each locality has few listings.")

# ---------------- 11-15 Relationships ----------------
numc = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", PPS, "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
        "Floor_No", "Total_Floors", "Amenity_Count", "Infrastructure_Score", "Growth_Rate", "Future_Price_5Y"]
cm = df[numc].corr()
fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation matrix"); save("q11_correlation")
raw = ["BHK", "Size_in_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count"]
mx = cm.loc[raw, raw].where(~np.eye(len(raw), dtype=bool)).abs().max().max()
note(11, "Correlation between numeric features",
     f"The raw features are essentially uncorrelated with each other (largest |r| among {', '.join(raw)} = {mx:.2f}) and with price "
     f"(Size {cm.loc['Size_in_SqFt', 'Price_in_Lakhs']:.2f}, BHK {cm.loc['BHK', 'Price_in_Lakhs']:.2f}, Schools "
     f"{cm.loc['Nearby_Schools', 'Price_in_Lakhs']:.2f}). The strong correlations are mechanical: Price vs Price per sqft = "
     f"{cm.loc['Price_in_Lakhs', PPS]:.2f} (PPS is derived from price), Price vs Future price = {cm.loc['Price_in_Lakhs', 'Future_Price_5Y']:.2f}, "
     f"and Floor_No vs Total_Floors = {cm.loc['Floor_No', 'Total_Floors']:.2f} (because Floor_No was capped at Total_Floors in Step 1).")


def bar_by(col, q, name, title, fname, label):
    g = df.groupby(col)[PPS].mean()
    fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(x=g.index.astype(str), y=g.values, color="indianred", ax=ax)
    ax.set_title(title); ax.set_ylim(g.min() * 0.9, g.max() * 1.05); save(fname)
    r = df[col].corr(df[PPS]) if np.issubdtype(df[col].dtype, np.number) else np.nan
    corr_txt = f" Correlation with price per sqft = {r:.3f}." if not np.isnan(r) else ""
    note(q, name, f"Mean price per sqft ranges Rs {g.min():,.0f} to Rs {g.max():,.0f} across {label}.{corr_txt}")


bar_by("Nearby_Schools", 12, "Nearby schools vs price per sqft", "Avg price per sqft by # nearby schools (axis truncated)", "q12_schools_vs_pps", "school counts")
bar_by("Nearby_Hospitals", 13, "Nearby hospitals vs price per sqft", "Avg price per sqft by # nearby hospitals (axis truncated)", "q13_hospitals_vs_pps", "hospital counts")

fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", showfliers=False, ax=ax)
ax.set_title("Price (lakhs) by furnished status"); save("q14_price_by_furnished")
g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(1)
note(14, "Price by furnished status", f"Mean price (lakhs): {g.to_dict()}. Differences are tiny.")

fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(data=df, x="Facing", y=PPS, errorbar=None, color="slateblue", ax=ax)
ax.set_title("Avg price per sqft by facing"); save("q15_pps_by_facing")
g = df.groupby("Facing")[PPS].mean().round(0)
note(15, "Price per sqft by facing direction", f"Mean Rs per sqft: {g.to_dict()}.")

# ---------------- 16-20 Investment / amenities / ownership ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["Owner_Type"].value_counts().plot.bar(ax=ax[0], color="teal"); ax[0].set_title("Owner type")
df["Availability_Status"].value_counts().plot.bar(ax=ax[1], color="orange"); ax[1].set_title("Availability status")
save("q16_17_owner_availability")
note(16, "Properties per owner type", f"{df['Owner_Type'].value_counts().to_dict()}")
note(17, "Properties per availability status", f"{df['Availability_Status'].value_counts().to_dict()}")

fig, ax = plt.subplots(figsize=(5, 4)); sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", showfliers=False, ax=ax)
ax.set_title("Price (lakhs) by parking"); save("q18_parking_vs_price")
g = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1)
note(18, "Does parking affect price?",
     f"Mean price with parking {g['Yes']} vs without {g['No']} lakhs. `Parking_Space` is Yes/No in this file, not a count.")

am = {a: df["Amenities"].str.contains(a).astype(int) for a in ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]}
rows = [(a, df.loc[v == 1, PPS].mean(), df.loc[v == 0, PPS].mean()) for a, v in am.items()]
ad = pd.DataFrame(rows, columns=["Amenity", "With", "Without"]).set_index("Amenity")
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ad.plot.bar(ax=ax[0]); ax[0].set_title("Price per sqft: with vs without amenity")
ax[0].set_ylim(ad.values.min() * 0.9, ad.values.max() * 1.05)
sns.barplot(data=df, x="Amenity_Count", y=PPS, errorbar=None, color="gray", ax=ax[1]); ax[1].set_title("By amenity count")
ax[1].set_ylim(df.groupby("Amenity_Count")[PPS].mean().min() * 0.9, None); save("q19_amenities_vs_pps")
note(19, "Amenities vs price per sqft",
     "Mean Rs per sqft (with / without):\n" + ad.round(0).to_markdown() +
     f"\n\nCorrelation of Amenity_Count with price per sqft = {df['Amenity_Count'].corr(df[PPS]):.3f}.")

g = df.assign(PT=df["Public_Transport_Accessibility"]).groupby("PT").agg(pps=(PPS, "mean"), good=("Good_Investment", "mean"))
g = g.loc[["Low", "Medium", "High"]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(x=g.index, y=g["pps"], color="navy", ax=ax[0]); ax[0].set_title("Avg price per sqft"); ax[0].set_ylim(g["pps"].min() * 0.9, None)
sns.barplot(x=g.index, y=g["good"], color="green", ax=ax[1]); ax[1].set_title("Good-investment rate"); ax[1].set_ylim(0, 1)
save("q20_transport")
note(20, "Public transport vs price per sqft and investment potential",
     "\n" + g.round(3).to_markdown() + "\n\nBetter transport raises the good-investment rate partly by construction, "
     "because transport is part of the Infrastructure_Score used in the label. Mention this circularity.")

# ---------------- extras from the original task wording ----------------
df["Size_Band"] = pd.qcut(df["Size_in_SqFt"], 5, labels=["XS", "S", "M", "L", "XL"])
g = df.groupby("Size_Band", observed=True).agg(good=("Good_Investment", "mean"), gain=("Future_Price_5Y", "sum"), p=("Price_in_Lakhs", "sum"))
g["return_pct"] = (g["gain"] / g["p"] - 1) * 100
fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(x=g.index, y=g["good"], color="peru", ax=ax)
ax.set_title("Good-investment rate by size band"); ax.set_ylim(0, 1); save("x1_area_vs_return")
note("X1", "Area vs investment return",
     f"5-year modelled return by size band: {g['return_pct'].round(1).to_dict()} (%). Good-investment rate by band: "
     f"{g['good'].round(3).to_dict()}. Modelled return is flat because it comes from the growth formula, not from size. "
     "But the good-investment rate climbs sharply with size: price does not depend on size in this data, so bigger properties get a "
     "lower price per sqft and win the 'cheaper than city median' test. The label is therefore largely a size effect, "
     "which must be disclosed in the report.")

df["Infra_Band"] = pd.cut(df["Infrastructure_Score"], [-.01, .35, .5, .65, 1.0], labels=["Low", "Mid", "High", "Top"])
g = df.groupby("Infra_Band", observed=True).agg(price=("Price_in_Lakhs", "mean"), future=("Future_Price_5Y", "mean"), pps=(PPS, "mean"))
fig, ax = plt.subplots(figsize=(7, 4)); g[["price", "future"]].plot.bar(ax=ax); ax.set_title("Avg current vs 5-year price by infrastructure band")
ax.set_ylim(g.values.min() * 0.0, None); save("x2_infra_vs_value")
note("X2", "Infrastructure score vs resale value",
     "\n" + g.round(1).to_markdown() + "\n\nCurrent price barely changes with infrastructure; modelled future price rises "
     "only because infrastructure feeds the growth rate. The brief's 'crime rate' has no matching column in this dataset, so it is skipped.")

with open(f"{out}/eda_insights.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("done:", len([p for p in os.listdir(out) if p.endswith('.png')]), "charts ->", out)
