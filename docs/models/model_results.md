# Model results (test set, 20% hold-out)

## Classification: Good_Investment
| model              |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:-------------------|-----------:|------------:|---------:|-------:|----------:|
| LogisticRegression |     0.858  |      0.851  |   0.8435 | 0.8472 |    0.9441 |
| RandomForest       |     0.9889 |      0.9808 |   0.9958 | 0.9882 |    0.9991 |
| XGBoost_d4         |     0.9966 |      0.9965 |   0.9962 | 0.9964 |    0.9998 |
| XGBoost_d6         |     0.9975 |      0.9971 |   0.9976 | 0.9973 |    1      |

Best: **XGBoost_d6**

## Regression: Future_Price_5Y (lakhs)
| model                     |    rmse |     mae |     r2 |
|:--------------------------|--------:|--------:|-------:|
| Baseline (Price x 1.08^5) | 18.7736 | 13.3532 | 0.9916 |
| Ridge                     |  8.7494 |  6.4031 | 0.9982 |
| RandomForest              |  1.7447 |  1.1713 | 0.9999 |
| XGBoost_d4                |  1.3113 |  1.0432 | 1      |
| XGBoost_d6                |  1.1695 |  0.9286 | 1      |

Best: **XGBoost_d6**

## Ablation: price inputs removed
| task                             |   accuracy |       f1 |   roc_auc |    rmse |     mae |      r2 |
|:---------------------------------|-----------:|---------:|----------:|--------:|--------:|--------:|
| classification (no price inputs) |      0.649 |   0.6219 |    0.7514 | nan     | nan     | nan     |
| regression (no price inputs)     |    nan     | nan      |  nan      | 205.234 | 177.258 |  -0.004 |