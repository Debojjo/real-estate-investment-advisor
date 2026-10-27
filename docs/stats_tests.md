# Statistical significance tests

Significance level 0.05. With 250,000 rows tiny differences become 'significant', so effect sizes are reported next to p-values.

| Question | Test | Why this test | Statistic | p-value | Effect size | Reading |
|---|---|---|---|---|---|---|
| Is property size related to price? | Pearson + Spearman correlation | Pearson for a linear link, Spearman as a rank-based check that does not assume normality | r = -0.0025, rho = -0.0025 | 0.206 / 0.206 | |r| = 0.0025 | Not significant (p >= 0.05): no evidence of a size-price relationship. |
| Does price per sqft differ by property type? | Kruskal-Wallis H test | Price per sqft is right-skewed (a ratio), so a rank-based test is safer than ANOVA | H = 7.97 | 0.019 | epsilon-sq = 0.000024 | Statistically significant (p < 0.05) but the effect size is negligible, so a property-type effect on price per sqft is not practically important. |
| Does average price differ across cities? | One-way ANOVA | Price is roughly symmetric and each city has thousands of rows, so group means are close to normal | F = 1.16 | 0.2232 | eta-sq = 0.00019 | Not significant (p >= 0.05): no evidence of a city effect on average price. |
| Does price differ by furnished status? | One-way ANOVA | Same reasoning as the city test | F = 0.50 | 0.609 | - | Not significant (p >= 0.05): no evidence of a furnishing effect on price. |
| Is Good_Investment related to Public_Transport_Accessibility? | Chi-square test of independence | Both variables are categorical; tests independence | chi2 = 5167.6 (dof 2) | <0.001 | Cramer's V = 0.144 | Strong association, but partly by construction: transport feeds the Infrastructure_Score used in the label. |
| Is Good_Investment related to Availability_Status? | Chi-square test of independence | Both variables are categorical; tests independence | chi2 = 15664.2 (dof 1) | <0.001 | Cramer's V = 0.250 | Strong association by construction: 'ready to move' earns a point in the label rule. |
| Is Good_Investment related to Property_Type? | Chi-square test of independence | Control: property type is not part of the label rule | chi2 = 2.6 (dof 2) | 0.275 | Cramer's V = 0.003 | No association, as expected (control check). |
| Is the target imbalanced? | Class share check | Decides whether SMOTE/class weights are needed | Good = 46.7%, Not good = 53.3% | - | - | Balanced (about 47/53), so no imbalance technique was needed. |
