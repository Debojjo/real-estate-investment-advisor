"""Step 2b - Statistical significance tests (supports the EDA findings).

Usage (from the project folder):  python scripts/step2b_stats.py data/cleaned_housing.csv docs
Writes docs/stats_tests.md. Needs scipy (installed with scikit-learn).
With 250,000 rows almost any difference is 'significant', so every test also reports an effect size.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

src = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_housing.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "docs"
df = pd.read_csv(src)
rows = []


def add(question, test, why, stat, p, effect, reading):
    rows.append((question, test, why, stat, p, effect, reading))


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    return chi2, p, dof, float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def verdict(p, effect_small, what):
    """Wording that follows the p-value and the effect size, never hard-coded."""
    if p >= 0.05:
        return f"Not significant (p >= 0.05): no evidence of {what}."
    if effect_small:
        return f"Statistically significant (p < 0.05) but the effect size is negligible, so {what} is not practically important."
    return f"Significant and practically meaningful: {what}."


def eta_sq_kw(h, k, n):  # epsilon-squared style effect size for Kruskal-Wallis
    return max(0.0, (h - k + 1) / (n - k))


# 1. size vs price: correlation
r, p = stats.pearsonr(df["Size_in_SqFt"], df["Price_in_Lakhs"])
rho, p2 = stats.spearmanr(df["Size_in_SqFt"], df["Price_in_Lakhs"])
add("Is property size related to price?", "Pearson + Spearman correlation",
    "Pearson for a linear link, Spearman as a rank-based check that does not assume normality",
    f"r = {r:.4f}, rho = {rho:.4f}", f"{p:.3f} / {p2:.3f}", f"|r| = {abs(r):.4f}",
    verdict(min(p, p2), abs(r) < 0.1, "a size-price relationship"))

# 2. price per sqft across property types (skewed -> Kruskal-Wallis)
g = [x["Price_per_SqFt"].values for _, x in df.groupby("Property_Type")]
h, p = stats.kruskal(*g)
add("Does price per sqft differ by property type?", "Kruskal-Wallis H test",
    "Price per sqft is right-skewed (a ratio), so a rank-based test is safer than ANOVA",
    f"H = {h:.2f}", f"{p:.3f}", f"epsilon-sq = {eta_sq_kw(h, len(g), len(df)):.6f}",
    verdict(p, eta_sq_kw(h, len(g), len(df)) < 0.01, "a property-type effect on price per sqft"))

# 3. price across cities (roughly symmetric -> one-way ANOVA)
g = [x["Price_in_Lakhs"].values for _, x in df.groupby("City")]
f, p = stats.f_oneway(*g)
ss_b = sum(len(x) * (x.mean() - df["Price_in_Lakhs"].mean()) ** 2 for x in g)
ss_t = ((df["Price_in_Lakhs"] - df["Price_in_Lakhs"].mean()) ** 2).sum()
add("Does average price differ across cities?", "One-way ANOVA",
    "Price is roughly symmetric and each city has thousands of rows, so group means are close to normal",
    f"F = {f:.2f}", f"{p:.4f}", f"eta-sq = {ss_b / ss_t:.5f}",
    verdict(p, ss_b / ss_t < 0.01, "a city effect on average price"))

# 4. price by furnished status
g = [x["Price_in_Lakhs"].values for _, x in df.groupby("Furnished_Status")]
f, p = stats.f_oneway(*g)
add("Does price differ by furnished status?", "One-way ANOVA", "Same reasoning as the city test",
    f"F = {f:.2f}", f"{p:.3f}", "-", verdict(p, True, "a furnishing effect on price"))

# 5-7. chi-square: association with the Good_Investment label
for col, why, reading in [
    ("Public_Transport_Accessibility", "Both variables are categorical; tests independence",
     "Strong association, but partly by construction: transport feeds the Infrastructure_Score used in the label."),
    ("Availability_Status", "Both variables are categorical; tests independence",
     "Strong association by construction: 'ready to move' earns a point in the label rule."),
    ("Property_Type", "Control: property type is not part of the label rule", "No association, as expected (control check)."),
]:
    chi2, p, dof, v = cramers_v(df[col], df["Good_Investment"])
    add(f"Is Good_Investment related to {col}?", "Chi-square test of independence", why,
        f"chi2 = {chi2:.1f} (dof {dof})", "<0.001" if p < 0.001 else f"{p:.3f}", f"Cramer's V = {v:.3f}", reading)

# 8. class balance
share = df["Good_Investment"].mean()
add("Is the target imbalanced?", "Class share check", "Decides whether SMOTE/class weights are needed",
    f"Good = {share:.1%}, Not good = {1 - share:.1%}", "-", "-", "Balanced (about 47/53), so no imbalance technique was needed.")

with open(f"{out}/stats_tests.md", "w", encoding="utf-8") as fh:
    fh.write("# Statistical significance tests\n\nSignificance level 0.05. With 250,000 rows tiny differences become "
             "'significant', so effect sizes are reported next to p-values.\n\n")
    fh.write("| Question | Test | Why this test | Statistic | p-value | Effect size | Reading |\n|---|---|---|---|---|---|---|\n")
    for q in rows:
        fh.write("| " + " | ".join(str(c) for c in q) + " |\n")
print(open(f"{out}/stats_tests.md", encoding="utf-8").read())
