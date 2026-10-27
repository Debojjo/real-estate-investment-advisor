"""Real Estate Investment Advisor - Streamlit app.

Run from the project folder:   streamlit run app/app.py
Needs: models/classifier.joblib, models/regressor.joblib, models/metrics.json,
       data/sample_app.csv (python scripts/make_app_sample.py), docs/models/*.csv|png
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import ALL_FEATURES, CAT_FEATURES, add_features  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_models():
    return joblib.load(ROOT / "models" / "classifier.joblib"), joblib.load(ROOT / "models" / "regressor.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "sample_app.csv")
    df["Build_Decade"] = (df["Year_Built"] // 10 * 10).astype(int).astype(str) + "s"
    return df


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def load_importance(kind):
    s = pd.read_csv(ROOT / "docs" / "models" / f"{kind}_feature_importance.csv", index_col=0).iloc[:, 0]
    s.index = [i.split("__", 1)[-1] for i in s.index]
    return s.sort_values()


try:
    clf, reg = load_models()
    data = load_data()
    metrics = load_metrics()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}\n\nRun Steps 1-3 and `python scripts/make_app_sample.py` first.")
    st.stop()

CITY_STATE = data.drop_duplicates("City").set_index("City")["State"].to_dict()
STATES = sorted(set(CITY_STATE.values()))
AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]


# ------------------------------------------------------------------ helpers
def predict(raw: pd.DataFrame):
    X = add_features(raw, clf["peer"])
    Xf = X[ALL_FEATURES]
    return X, clf["pipeline"].predict_proba(Xf)[:, 1], reg["pipeline"].predict(Xf)


def explain_classifier(Xf: pd.DataFrame, top=8):
    """Per-prediction feature contributions (log-odds) from the XGBoost booster; None if not available."""
    try:
        import xgboost as xgb
        pre, model = clf["pipeline"].named_steps["pre"], clf["pipeline"].named_steps["model"]
        contrib = model.get_booster().predict(xgb.DMatrix(pre.transform(Xf)), pred_contribs=True)[0][:-1]
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        base = [next((c for c in CAT_FEATURES if n.startswith(c + "_")), n) for n in names]
        s = pd.Series(contrib, index=base).groupby(level=0).sum()
        return s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    except Exception:
        return None


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        title={"text": "Probability of Good Investment"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2e8b57" if p >= 0.5 else "#c0392b"},
               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
    fig.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20))
    return fig


# ------------------------------------------------------------------ header / sidebar
st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a Good Investment and its estimated price after 5 years.")
with st.sidebar:
    st.header("About")
    st.write(f"**Classifier:** {clf['model_name']}  \n**Regressor:** {reg['model_name']}")
    st.write(f"Explore/insight tabs use a {len(data):,}-row sample of the 250,000-row dataset.")
    st.warning("The dataset appears synthetic and both targets are computed from the inputs, so treat results as a "
               "modelling demo, not financial advice.")

tab_pred, tab_explore, tab_insights, tab_model = st.tabs(
    ["🔮 Predict", "🔎 Explore properties", "📊 Market insights", "🧪 Model details"])

# ================================================================== TAB 1: PREDICT
with tab_pred:
    st.subheader("Enter property details")
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", STATES, index=STATES.index("Maharashtra") if "Maharashtra" in STATES else 0)
    cities = sorted(c for c, s in CITY_STATE.items() if s == state)
    city = c2.selectbox("City", cities)

    with st.form("property_form"):
        a, b, c = st.columns(3)
        ptype = a.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        bhk = b.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
        size = c.number_input("Size (sqft)", 500, 5000, 1200, step=50, help="Training data covers 500-5,000 sqft")
        price = a.number_input("Current price (lakhs)", 10.0, 500.0, 150.0, step=5.0, help="Training data covers 10-500 lakhs")
        year = b.number_input("Year built", 1990, 2023, 2015)
        furnished = c.selectbox("Furnished status", ["Unfurnished", "Semi-furnished", "Furnished"])
        floor = a.number_input("Floor number", 0, 30, 3)
        total_floors = b.number_input("Total floors", 1, 30, 10)
        facing = c.selectbox("Facing", ["East", "North", "South", "West"])
        schools = a.slider("Nearby schools", 1, 10, 5)
        hospitals = b.slider("Nearby hospitals", 1, 10, 5)
        transport = c.selectbox("Public transport accessibility", ["Low", "Medium", "High"], index=1)
        parking = a.selectbox("Parking space", ["Yes", "No"])
        security = b.selectbox("Security", ["Yes", "No"])
        owner = c.selectbox("Owner type", ["Owner", "Builder", "Broker"])
        availability = a.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        amen = st.multiselect("Amenities", AMENITIES, default=["Gym", "Garden"])
        submitted = st.form_submit_button("Analyse this property", type="primary")

    if submitted and not amen:
        st.error("Select at least one amenity (every property in the training data has 1 to 5).")
    elif submitted:
        st.session_state["inp"] = {
            "State": state, "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
            "Price_in_Lakhs": price, "Year_Built": year, "Furnished_Status": furnished, "Floor_No": floor,
            "Total_Floors": total_floors, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Amenities": ", ".join(amen), "Facing": facing, "Owner_Type": owner,
            "Availability_Status": availability}

    if "inp" in st.session_state:
        inp = st.session_state["inp"]
        if inp["Floor_No"] > inp["Total_Floors"]:
            st.warning("Floor number is higher than total floors; it will be capped at the top floor.")
        X, p, fut = predict(pd.DataFrame([inp]))
        p, fut = float(p[0]), float(fut[0])
        good = p >= 0.5
        growth = (fut / inp["Price_in_Lakhs"]) ** (1 / 5) - 1
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("### Is this a Good Investment?")
            (st.success if good else st.error)(f"**{'YES - Good Investment' if good else 'NO - Not a Good Investment'}**")
            st.metric("Model confidence", f"{max(p, 1 - p):.1%}")
            st.plotly_chart(gauge(p), width="stretch")
        with r2:
            st.markdown("### Estimated price after 5 years")
            st.metric("Estimated price", f"₹ {fut:,.1f} lakhs",
                      f"{fut - inp['Price_in_Lakhs']:+,.1f} lakhs ({growth:+.1%} per year)")
            rmse = reg["metrics"]["rmse"]
            st.caption(f"Typical model error on test data: about ±{rmse:.1f} lakhs (RMSE).")
            yrs = list(range(6))
            fig = go.Figure()
            fig.add_scatter(x=yrs, y=[inp["Price_in_Lakhs"] * (1 + growth) ** t for t in yrs], mode="lines+markers",
                            name="Model estimate")
            fig.add_scatter(x=yrs, y=[inp["Price_in_Lakhs"] * 1.08 ** t for t in yrs], mode="lines",
                            name="Fixed 8% reference", line=dict(dash="dash"))
            fig.update_layout(height=300, xaxis_title="Years from now", yaxis_title="Price (lakhs)",
                              margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, width="stretch")
            st.caption("Curve assumes steady yearly growth between today's price and the 5-year estimate.")

        row = X.iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Price per sqft", f"₹ {row['Price_per_SqFt']:,.0f}")
        d2.metric(f"Median price, {inp['City']} {inp['BHK']} BHK", f"{row['Peer_Median_Price']:,.1f} lakhs")
        d3.metric("Vs peer median", f"{row['Price_vs_Peer'] - 1:+.0%}")
        d4.metric("Infrastructure score", f"{row['Infrastructure_Score']:.2f} / 1")

        st.markdown("#### What drove this prediction?")
        ex = explain_classifier(X[ALL_FEATURES])
        if ex is not None:
            fig = go.Figure(go.Bar(x=ex.values, y=ex.index, orientation="h",
                                   marker_color=["#2e8b57" if v > 0 else "#c0392b" for v in ex.values]))
            fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="Pushes toward Not good (−)  /  Good (+)")
            st.plotly_chart(fig, width="stretch")
        imp = load_importance("classifier").tail(10)
        with st.expander("Overall feature importance (classifier)"):
            st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "importance", "index": ""}).update_layout(
                height=350, showlegend=False), width="stretch")
    else:
        st.info("Fill in the form and click **Analyse this property**.")

# ================================================================== TAB 2: EXPLORE
with tab_explore:
    st.subheader("Filter properties")
    f1, f2, f3 = st.columns(3)
    sel_cities = f1.multiselect("City (empty = all)", sorted(data["City"].unique()))
    sel_types = f2.multiselect("Property type (empty = all)", sorted(data["Property_Type"].unique()))
    sel_bhk = f3.multiselect("BHK (empty = all)", sorted(data["BHK"].unique()))
    g1, g2, g3 = st.columns(3)
    pmin, pmax = float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())
    price_rng = g1.slider("Price (lakhs)", pmin, pmax, (pmin, pmax))
    smin, smax = int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())
    size_rng = g2.slider("Area (sqft)", smin, smax, (smin, smax))
    sel_avail = g3.multiselect("Availability (empty = all)", sorted(data["Availability_Status"].unique()))
    sort_by = st.radio("Sort by", ["Highest chance of Good Investment", "Lowest price", "Highest estimated 5-year price"],
                       horizontal=True)

    f = data[data["Price_in_Lakhs"].between(*price_rng) & data["Size_in_SqFt"].between(*size_rng)]
    if sel_cities: f = f[f["City"].isin(sel_cities)]
    if sel_types: f = f[f["Property_Type"].isin(sel_types)]
    if sel_bhk: f = f[f["BHK"].isin(sel_bhk)]
    if sel_avail: f = f[f["Availability_Status"].isin(sel_avail)]

    if f.empty:
        st.warning("No properties match these filters.")
    else:
        _, pg, pf = predict(f.drop(columns=["Good_Investment", "Future_Price_5Y"]).reset_index(drop=True))
        out = f.reset_index(drop=True).assign(P_Good=pg, Est_Price_5Y=pf)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matching properties", f"{len(out):,}")
        m2.metric("Average price", f"{out['Price_in_Lakhs'].mean():,.0f} lakhs")
        m3.metric("Avg price per sqft", f"₹ {out['Price_per_SqFt'].mean():,.0f}")
        m4.metric("Predicted good investments", f"{(out['P_Good'] >= 0.5).mean():.0%}")
        key = {"Highest chance of Good Investment": ("P_Good", False), "Lowest price": ("Price_in_Lakhs", True),
               "Highest estimated 5-year price": ("Est_Price_5Y", False)}[sort_by]
        show = out.sort_values(key[0], ascending=key[1]).head(200)
        cols = ["ID", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                "Availability_Status", "P_Good", "Est_Price_5Y"]
        st.dataframe(show[cols], hide_index=True, width="stretch", column_config={
            "P_Good": st.column_config.ProgressColumn("Chance good investment", min_value=0.0, max_value=1.0, format="percent"),
            "Est_Price_5Y": st.column_config.NumberColumn("Est. price in 5 yrs (lakhs)", format="%.1f"),
            "Price_per_SqFt": st.column_config.NumberColumn("Price per sqft (₹)", format="%.0f")})
        st.caption(f"Showing the top {len(show)} of {len(out):,} matches.")
        st.download_button("Download all matches (CSV)", out[cols].to_csv(index=False).encode("utf-8"),
                           "filtered_properties.csv", "text/csv")

# ================================================================== TAB 3: INSIGHTS
with tab_insights:
    st.subheader("Location-wise heatmap")
    h1, h2 = st.columns(2)
    metric = h1.selectbox("Metric", ["Average price per sqft (₹)", "Good-investment rate", "Average 5-year price (lakhs)"])
    across = h2.selectbox("Compare across", ["BHK", "Property_Type", "Furnished_Status", "Availability_Status"])
    col, fmt = {"Average price per sqft (₹)": ("Price_per_SqFt", ".0f"), "Good-investment rate": ("Good_Investment", ".2f"),
                "Average 5-year price (lakhs)": ("Future_Price_5Y", ".0f")}[metric]
    pv = data.pivot_table(index="City", columns=across, values=col, aggfunc="mean")
    fig = px.imshow(pv, aspect="auto", color_continuous_scale="YlGnBu", text_auto=fmt, labels={"color": metric})
    fig.update_layout(height=900, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Values are averages in the app sample. Differences between cities are small because the data is synthetic.")

    st.subheader("Price trend by build decade")
    t1, t2 = st.columns([1, 3])
    mode = t1.radio("Group by", ["Property_Type", "City"])
    if mode == "City":
        defaults = [c for c in ["Mumbai", "Delhi", "Bangalore"] if c in data["City"].unique()]
        picks = t1.multiselect("Cities", sorted(data["City"].unique()), default=defaults)
        tr = data[data["City"].isin(picks)]
    else:
        tr = data
    tr = tr.groupby(["Build_Decade", mode])["Price_per_SqFt"].mean().reset_index()
    t2.plotly_chart(px.line(tr, x="Build_Decade", y="Price_per_SqFt", color=mode, markers=True,
                            labels={"Price_per_SqFt": "Avg price per sqft (₹)", "Build_Decade": "Decade built"}),
                    width="stretch")
    st.caption("The dataset has no sale dates, so build decade is used as the time axis.")

    i1, i2 = st.columns(2)
    sg = data.groupby("State")["Good_Investment"].mean().sort_values()
    i1.markdown("**Good-investment rate by state**")
    i1.plotly_chart(px.bar(sg, orientation="h", labels={"value": "rate", "State": ""}).update_layout(
        showlegend=False, height=500), width="stretch")
    data["Infra_Band"] = pd.cut(data["Infrastructure_Score"], [-0.01, 0.35, 0.5, 0.65, 1.0], labels=["Low", "Mid", "High", "Top"])
    ib = data.groupby("Infra_Band", observed=True).agg(Good_rate=("Good_Investment", "mean"),
                                                         Future_Price=("Future_Price_5Y", "mean")).reset_index()
    i2.markdown("**Infrastructure score vs investment outcome**")
    i2.plotly_chart(px.bar(ib, x="Infra_Band", y="Good_rate", labels={"Good_rate": "Good-investment rate", "Infra_Band": "Infrastructure band"}
                           ).update_layout(height=500), width="stretch")

# ================================================================== TAB 4: MODEL
with tab_model:
    st.subheader("Model performance (20% hold-out test set)")
    st.markdown(f"**Classification - best: {metrics['best_classifier']}**")
    st.dataframe(pd.DataFrame(metrics["classification"]), hide_index=True, width="stretch")
    st.markdown(f"**Regression - best: {metrics['best_regressor']}** (price after 5 years, lakhs)")
    st.dataframe(pd.DataFrame(metrics["regression"]), hide_index=True, width="stretch")
    st.markdown("**Leakage check: price inputs removed**")
    st.dataframe(pd.DataFrame(metrics["ablation"]), hide_index=True, width="stretch")
    st.info("Scores are very high because both targets are calculated from the inputs. Without price inputs, classification "
            "accuracy drops to about 65% and regression R² to about 0.")
    m1, m2 = st.columns(2)
    cm_path = ROOT / "docs" / "models" / f"cm_{metrics['best_classifier']}.png"
    if cm_path.exists():
        m1.markdown("**Confusion matrix (best classifier)**")
        m1.image(str(cm_path))
    m2.markdown("**Classifier feature importance**")
    m2.plotly_chart(px.bar(load_importance("classifier").tail(12), orientation="h",
                           labels={"value": "importance", "index": ""}).update_layout(height=380, showlegend=False),
                    width="stretch")
    st.markdown("**Regressor feature importance**")
    st.plotly_chart(px.bar(load_importance("regressor").tail(12), orientation="h",
                           labels={"value": "importance", "index": ""}).update_layout(height=380, showlegend=False),
                    width="stretch")
