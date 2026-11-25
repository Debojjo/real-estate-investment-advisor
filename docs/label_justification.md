# Why the Good_Investment rule looks the way it does

The dataset has no sale history, so the brief's own option is used: a label from domain rules. Each property earns points out of 6, and 4 or more is a Good Investment. The weights and the cut-off are my judgement, not market facts, so this file also shows how sensitive the label is to them.

## 1. The criteria

| Criterion | Points | Share of properties meeting it | Reason |
|---|---|---|---|
| Price at or below the median of the same city and BHK | 2 | 50.0% | Price compared with similar properties is the most direct value signal for an investor, and it is the only criterion about cost. It gets double weight. The benchmark uses city and BHK, not price per sqft, because price per sqft follows size in this data and would reward large properties. |
| BHK of 3 or more | 1 | 60.0% | Larger homes suit families, who form the widest group of buyers and tenants, so resale and rental demand is broader (a domain assumption). |
| Ready to move | 1 | 50.0% | No construction delay risk, and the owner can rent or sell immediately. |
| Infrastructure score at or above the median | 1 | 51.8% | Nearby schools, hospitals and public transport support demand and later appreciation. |
| Parking or security available | 1 | 74.9% | Facilities that buyers expect and pay for, which help resale. |

## 2. Why a cut-off of 4 out of 6

| Cut-off | Share labelled Good |
|---|---|
| 3 or more | 71.7% |
| 4 or more | 46.7% |
| 5 or more | 22.8% |

A cut-off of 4 means a property needs either good value (2 points) plus two other strengths, or all four of the other points without the value point. It gives a balanced split (about 47% good), so no class-imbalance technique is needed. A cut-off of 3 would label most properties good, and 5 would label very few.

Score distribution (points: share of properties): 0: 1.2%, 1: 8.0%, 2: 19.1%, 3: 25.1%, 4: 23.9%, 5: 17.0%, 6: 5.8%

## 3. Sensitivity of the label

| Alternative rule | Agreement with the chosen label |
|---|---|
| All five criteria worth 1 point, Good if 3 or more | 82.9% |
| Without the price criterion (4 quality criteria, Good if 3 or more) | 64.9% |

The label is a deterministic rule of the inputs. This is why the models score very high, and why removing the price inputs lowers accuracy to about 65%. The rule is documented so the limitation is visible, not hidden.
