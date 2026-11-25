# Model results (20% hold-out test set)

## Classification: Good_Investment
| model                         |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:------------------------------|-----------:|------------:|---------:|-------:|----------:|
| Baseline: majority class      |     0.5333 |      0      |   0      | 0      |    0.5    |
| Baseline: random (stratified) |     0.5023 |      0.4666 |   0.4651 | 0.4659 |    0.4999 |
| LogisticRegression            |     0.858  |      0.851  |   0.8435 | 0.8472 |    0.9441 |
| RandomForest                  |     0.9889 |      0.9808 |   0.9958 | 0.9882 |    0.9991 |
| XGBoost_d4                    |     0.9966 |      0.9965 |   0.9962 | 0.9964 |    0.9998 |
| XGBoost_d6                    |     0.9975 |      0.9971 |   0.9976 | 0.9973 |    1      |

Selected on F1 (ROC-AUC breaks ties): **XGBoost_d6**. Confusion matrix: `docs/models/cm_XGBoost_d6.png`. The two baselines show what guessing achieves.

## Regression: 5-year growth multiple, then price = current price x multiple
| model                              |   rmse_multiple |   mae_multiple |   r2_multiple |   rmse_lakhs |   mae_lakhs |   r2_lakhs |
|:-----------------------------------|----------------:|---------------:|--------------:|-------------:|------------:|-----------:|
| Baseline: fixed 8% growth          |          0.0645 |         0.0525 |       -0.1799 |      18.7736 |     13.3532 |     0.9916 |
| Baseline: training-set mean growth |          0.0594 |         0.0482 |       -0      |      17.2813 |     12.2558 |     0.9929 |
| Ridge                              |          0.0097 |         0.0085 |        0.9731 |       2.8381 |      2.161  |     0.9998 |
| RandomForest                       |          0      |         0      |        1      |       0.003  |      0.0025 |     1      |
| XGBoost_d4                         |          0.0012 |         0.0009 |        0.9996 |       0.3556 |      0.2329 |     1      |
| XGBoost_d6                         |          0.0005 |         0.0003 |        0.9999 |       0.1327 |      0.0863 |     1      |

Selected: **XGBoost_d6**. `_multiple` columns score the growth multiple (the part the model has to learn); `_lakhs` columns score the final price.

## Ablation and leakage checks
| task                                                                                         |   accuracy |       f1 |   roc_auc |   rmse_multiple |   r2_multiple |   rmse_lakhs |   r2_lakhs |
|:---------------------------------------------------------------------------------------------|-----------:|---------:|----------:|----------------:|--------------:|-------------:|-----------:|
| classification: no price inputs                                                              |      0.649 |   0.6219 |    0.7514 |        nan      |      nan      |     nan      |   nan      |
| regression: no price inputs                                                                  |    nan     | nan      |  nan      |          0.0005 |        0.9999 |       0.1399 |     1      |
| regression: no growth drivers (infrastructure, transport, readiness, security, parking, age) |    nan     | nan      |  nan      |          0.0596 |       -0.007  |      17.3494 |     0.9928 |

## 5-fold cross-validation of the selected models
| task                               | model      |   mean |    std |
|:-----------------------------------|:-----------|-------:|-------:|
| classification (F1)                | XGBoost_d6 | 0.9962 | 0.0006 |
| regression (R2 on growth multiple) | XGBoost_d6 | 0.9998 | 0      |
