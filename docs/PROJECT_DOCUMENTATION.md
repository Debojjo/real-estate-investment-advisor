# Real Estate Investment Advisor: Project Documentation

**Predicting property profitability and future value** | Python, scikit-learn, XGBoost, MLflow, Streamlit

Live app: `<add Streamlit link here>` | Code: https://github.com/Debojjo/real-estate-investment-advisor | MLflow logs: `<https://drive.google.com/drive/folders/1-a0epP92o1qCvBXUQV6qqaloZAC16O_R?usp=sharing>`

---

## 1. Domain
Real estate is one of the largest asset classes for Indian households, and most buyers decide using price and gut feeling.
Investment quality depends on price relative to the market, location quality, infrastructure and readiness to move in.
Data-driven tools can make these comparisons fast, consistent and transparent.

## 2. Problem statement
Build a machine-learning application that tells an investor whether a property is a "Good Investment" and what it may be worth after 5 years.

## 3. Objective
Clean and analyse 250,000 Indian property listings, train a classifier (Good Investment) and a regressor (price after 5 years), track every experiment in MLflow, and ship a Streamlit app with predictions, confidence scores, feature importance and market insights.

## 4. Data cleaning and preprocessing
| Issue | What the data showed | Technique used |
|---|---|---|
| Missing values | 0 in all 23 columns | Median (numeric) / mode (categorical) imputation is in the script, so new data is handled safely |
| Duplicates | 0 duplicate rows | Dropped on all columns except ID |
| `Price_per_SqFt` | Stored in lakhs per sqft rounded to 2 decimals (59 rows show 0.00) | Recomputed in rupees per sqft = Price x 100,000 / Size |
| `Floor_No` > `Total_Floors` | 46.5% of rows (impossible) | Flagged in `Floor_Invalid`, then capped at the top floor |
| Outliers | 19,723 rows (7.9%) are IQR outliers on price per sqft; none on size | Flagged (`PPS_Outlier`); a 1st-99th percentile winsorised copy is kept for modelling. Not deleted because they are valid small, expensive flats |
| Encoding | Text categories (State, City, Property_Type, Furnished, Facing, Owner, Availability...) | One-hot encoding inside the model pipeline (unknown categories are ignored); ordinal mapping for furnished status and transport in the inspection file |
| Scaling | Numeric columns on very different ranges | StandardScaler (z-score) for the linear models; tree models do not need it |

The cleaned dataset (`cleaned_housing.csv`, 42 columns) and an encoded and scaled copy (`processed_housing.csv`) are produced by `scripts/step1_preprocess.py`.
In the training pipeline the encoder and scaler are fitted on the training rows only, so test data never leaks into preprocessing.

## 5. Exploratory data analysis (20 questions + 2 extras)
All charts are in `docs/eda/`; numbers are in `docs/eda/eda_insights.md`.

**Univariate (Q1, 2, 5, 16, 17)**
- Prices run 10-500 lakhs (median 254) and sizes 500-5,000 sqft, both close to uniform (skew about 0), unlike real markets that are right-skewed.
- Owner type (Broker, Owner, Builder) and availability (Ready, Under construction) are split almost exactly evenly.

**Bivariate (Q3, 4, 6, 7, 9, 12-15, 18, 19)**
- Price vs size: correlation -0.003, so bigger properties are not more expensive.
- Average price per sqft differs by only about 2% between states (Karnataka highest at Rs 13,252, Punjab lowest at Rs 12,931) and about 3% between cities (Bangalore 258.5 lakhs vs Cuttack 250.8).
- Property type, furnishing, facing, parking, amenities, nearby schools and hospitals all show negligible differences.
- Every city has an even BHK mix (1-5 BHK at about 20% each).

**Multivariate (Q8, 10, 11, 20)**
- Raw features are essentially uncorrelated with one another and with price; the strong correlations are mechanical (price vs price per sqft 0.56, price vs future price 1.00).
- Good-investment rate rises with transport accessibility (Low 37.5%, Medium 47.5%, High 55.0%) because transport is part of the rule used to build the label.
- Price trends for the top 5 localities use build decade as the time axis because the dataset has no sale dates; the lines are noisy because each locality has few listings.

**Requested analyses that depend on missing data:** the brief mentions crime rate, RERA and appreciation rate, but the dataset has no such columns. Infrastructure is built from schools, hospitals and transport instead, and appreciation is modelled (section 9).

## 6. Feature engineering
| Feature | Definition |
|---|---|
| `Price_per_SqFt` | Rs per sqft, recomputed |
| `Infrastructure_Score` | Average of scaled nearby schools, hospitals and transport (0 to 1) |
| `School_Density_Score` | Nearby schools relative to the city median |
| `Amenity_Count`, `Parking_Flag`, `Security_Flag`, `Is_Ready` | Count and Yes/No flags from the text columns |
| `Floor_Ratio`, `Sqft_per_BHK`, `Age_of_Property` | Position in building, room size, building age |
| `Peer_Median_Price`, `Price_vs_Peer` | Median price of the same city and BHK (computed from training rows only) and the property's price divided by it |
| `Growth_Rate`, `Future_Price_5Y`, `Investment_Score`, `Good_Investment` | The two targets and their building blocks (section 9) |

## 7. Statistical significance
Full table with effect sizes: `docs/stats_tests.md` (`scripts/step2b_stats.py`).
- **Pearson and Spearman** (size vs price): r = -0.0025, p = 0.21, so no relationship. Spearman checks the same without assuming normality.
- **Kruskal-Wallis** (price per sqft by property type): chosen because price per sqft is skewed. p = 0.019 but the effect size is about 0.00002, so it is not practically important.
- **One-way ANOVA** (price by city and by furnished status): p = 0.22 and 0.61, no evidence of a difference.
- **Chi-square** (Good_Investment vs transport, availability, property type): the first two are strongly associated (Cramer's V 0.14 and 0.25) because those features are in the rule; property type (V = 0.003) is a control and shows no association.
- With 250,000 rows tiny differences become significant, so effect sizes are reported next to p-values.

## 8. Class imbalance
Good Investment is 46.7% and Not good is 53.3%, so the target is balanced and no resampling or class weights were needed.

## 9. Target definitions (assumptions)
The dataset has no sale history and no appreciation column, so both targets are built from documented rules, following the options in the brief.

**Good Investment** (multi-factor score, max 6 points, Good if score >= 4):
- price at or below the median price of the same city and BHK: 2 points
- BHK of 3 or more: 1 point
- ready to move: 1 point
- infrastructure score at or above the median: 1 point
- parking or security available: 1 point

The price rule compares price with the same-city, same-BHK median and not price per sqft. Price per sqft is mostly a function of size here (price does not depend on size), and that version made the label largely a size effect (23% good in the smallest fifth vs 70% in the largest). The final label is about 46.5-47% good in every size band.

**Future price after 5 years** = Price x (1 + r)^5, where r is feature-based: 5% base, plus up to 3% for infrastructure, 1% if ready to move, 0.5% each for security and parking, 0.5% if the building is under 10 years old, capped between 5% and 10%. The brief offers a fixed rate, a rate per location or type, and a feature-based rate; the feature-based option was chosen so the model has something to learn from property features.

## 10. Model building
- **Split:** 80/20 train/test, stratified on the label, random_state 42 (200,000 train, 50,000 test).
- **Base models:** Logistic Regression (classification) and Ridge (regression), as simple linear baselines. For regression there is also a "fixed 8%" rule baseline.
- **Main models and why:** Random Forest (non-linear, robust, little tuning) and XGBoost (gradient boosting, usually the strongest on tabular data). Two XGBoost depths (4 and 6) were compared as light hyperparameter tuning (learning rate 0.1, 200 or 300 trees, subsample 0.8).
- **Tracking:** every run logs its parameters, metrics, confusion matrix and model to MLflow (experiments `RealEstate_Classification` and `RealEstate_Regression`); the best model of each task is registered (`RealEstate_GoodInvestment_Classifier`, `RealEstate_FuturePrice_Regressor`). Screenshots: `docs/mlflow/`.
- **Code structure:** `scripts/common.py` holds the feature code shared by training and the app, so the model sees identical inputs in both.

## 11. Evaluation metrics and why
- **Classification:** accuracy (overall hit rate), F1 (balances precision and recall), confusion matrix (shows the type of mistakes), plus ROC-AUC. The classes are balanced, so accuracy is meaningful.
- **Regression:** RMSE (penalises large errors, in lakhs), MAE (average error in lakhs), R² (share of variance explained).

| Classification (test set) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.858 | 0.851 | 0.844 | 0.847 | 0.944 |
| Random Forest | 0.989 | 0.981 | 0.996 | 0.988 | 0.999 |
| XGBoost depth 4 | 0.997 | 0.997 | 0.996 | 0.996 | 1.000 |
| **XGBoost depth 6 (final)** | **0.9975** | **0.9971** | **0.9976** | **0.9973** | **1.000** |

Confusion matrix of the final classifier: 26,598 true negatives, 67 false positives, 57 false negatives, 23,278 true positives.

| Regression (lakhs) | RMSE | MAE | R² |
|---|---|---|---|
| Fixed 8% rule (baseline) | 18.77 | 13.35 | 0.9916 |
| Ridge | 8.75 | 6.40 | 0.9982 |
| Random Forest | 1.74 | 1.17 | 0.9999 |
| XGBoost depth 4 | 1.31 | 1.04 | 0.99996 |
| **XGBoost depth 6 (final)** | **1.17** | **0.93** | **0.99997** |

## 12. Final models and feature importance
XGBoost depth 6 won both tasks: highest F1 for classification and lowest RMSE for regression, with a small model file (about 1 MB) that is easy to deploy.
- **Classifier:** the most important inputs are parking, price relative to the peer median, availability status, security, BHK and public transport. These match the label rule, which shows the model recovered it.
- **Regressor:** current price accounts for almost all of the importance (about 96%); the other inputs adjust the growth rate.

## 13. Honest limitations
- **The scores are very high by construction.** Both targets are calculated from the inputs, so the models mostly rediscover the rules. This is why the app's confidence is close to 0% or 100% for most properties; only properties sitting on a threshold (for example, price within about 1% of the city/BHK median) get values like 50-80%.
- **Leakage check:** with the three price inputs removed, classification accuracy falls to 0.649 (ROC-AUC 0.751) and regression R² falls to about 0 (-0.004), so almost all the skill comes from price.
- **Regression gain:** the fixed 8% rule already gives R² 0.9916; the model's improvement is lower error (RMSE 1.17 vs 18.77 lakhs), not a new signal.
- **The dataset looks synthetic:** price is unrelated to size, BHK, location and amenities, and categories are evenly split. Findings describe this dataset and should not be read as real market facts.
- **Missing real-world inputs:** crime rate, RERA registration and sale dates are not in the data; growth rates are assumed, not observed.
- **Floor data:** 46.5% of floor numbers were invalid and were capped.

## 14. Streamlit application
Run: `streamlit run app/app.py` (live link at the top).
- **Predict:** form for all property details, answer to "Is this a Good Investment?" with a confidence score and gauge, the estimated price after 5 years with a trend line against the fixed 8% reference, comparison with the city/BHK median, and a chart of what drove the decision.
- **Explore properties:** filters for city, property type, BHK, price range, area and availability; every match is scored by both models, sortable and downloadable as CSV.
- **Market insights:** city heatmaps (price per sqft, good-investment rate, 5-year price) computed from all 250,000 rows, price trend by build decade, good-investment rate by state and by infrastructure band.
- **Model details:** metrics tables, confusion matrix, feature importance and the leakage check.
- The models are trained on all 250,000 rows. The Explore tab uses a random 20,000-row sample so the app stays light for deployment.

## 15. Business use cases and suggestions
- **Investors** can screen listings quickly: the app shows how a price compares with similar properties and gives a projected value.
- **Buyers** can filter by city, budget, BHK and area and rank matches by predicted investment quality.
- **Real estate companies** can run the same scoring on every new listing to automate first-pass investment analysis.
- **Platforms** build trust by showing the reasoning (peer-median comparison, key drivers) next to every prediction.
- **Suggestion:** before using this for real decisions, retrain on real transaction history with sale dates, RERA status and crime and infrastructure data, and define "Good Investment" from observed appreciation instead of a rule.

## 16. Reproducing the project
```
pip install -r requirements.txt
python scripts/step1_preprocess.py data/india_housing_prices.csv data
python scripts/step2_eda.py data/cleaned_housing.csv docs/eda
python scripts/step2b_stats.py data/cleaned_housing.csv docs
python scripts/step3_train.py data/cleaned_housing.csv models
python scripts/make_app_sample.py data/cleaned_housing.csv data/sample_app.csv
python scripts/make_app_aggregates.py data/cleaned_housing.csv data
streamlit run app/app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Deliverable in the brief | Where it is |
|---|---|
| Cleaned and processed dataset | Drive link in README (generated by `step1_preprocess.py`) |
| Python scripts for EDA, training, evaluation | `scripts/` |
| MLflow logs and registered models | `docs/mlflow/` screenshots, `mlflow.db` link, `scripts/step3_train.py` |
| Streamlit application | `app/app.py`, live link |
| Documentation (methodology, findings, use cases) | This file, `README.md`, `docs/eda/eda_insights.md`, `docs/stats_tests.md`, `docs/models/model_results.md` |
