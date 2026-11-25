# EDA insights (auto-generated from the data)

## 1. Distribution of property prices
Price ranges 10.0-500.0 lakhs, mean 254.6, median 253.9, skew 0.01. The shape is close to uniform, not the right-skewed shape real markets show, which fits a synthetic dataset.

**Insight:** Prices are spread almost evenly from 10 to 500 lakhs, so a model sees cheap and expensive properties equally often.

## 2. Distribution of property sizes
Sizes run 500-5000 sqft, median 2747, skew 0.00 (again near-uniform).

**Insight:** Sizes are also spread evenly, so every size band has enough properties to learn from.

## 3. Price per sqft by property type
Mean / median Rs per sqft:
| Property_Type     |   mean |   median |
|:------------------|-------:|---------:|
| Apartment         |  13047 |     9245 |
| Independent House |  13102 |     9305 |
| Villa             |  13026 |     9190 |

The gaps between types are small.

**Insight:** Property type is not a price driver: the highest and lowest type differ by only 0.6% in price per sqft.

## 4. Size vs price
Correlation = -0.003. There is essentially no relationship: bigger properties are not priced higher. In real data this would be strongly positive, so say this explicitly in your report.

**Insight:** Size does not explain price (r = -0.003). So price per sqft simply falls as size grows, which is why the Good Investment rule compares price with a same-city, same-BHK peer and not price per sqft.

## 5. Outliers in price per sqft and size
19,723 rows (7.9%) are IQR outliers on price per sqft (max Rs 99,182); 0 on size. Price per sqft is skewed because it is price divided by size, so small flats with high prices give very high values. They are flagged and a winsorised copy (`Price_per_SqFt_W`) is available for modelling.

**Insight:** 7.9% of rows are price-per-sqft outliers. They come from small, high-priced properties, so I flagged them and kept them.

## 6. Average price per sqft by state
Highest: Karnataka (Rs 13,252); lowest: Punjab (Rs 12,931). Spread between top and bottom is only 2%.

**Insight:** Location barely moves price per sqft: the top and bottom state differ by 2.5%.

## 7. Average property price by city
Top: Bangalore (258.5 lakhs), Surat (258.1), Kochi (257.4); bottom: Cuttack (250.8). City means differ by only ~3%.

**Insight:** City is not a price driver: the top and bottom city differ by 3.1% in average price.

## 8. Median age of properties by locality
Across 15,270 city-localities, median age ranges 3-33 years (overall median 18). Oldest: Jaipur / Locality_249; newest: Vishakhapatnam / Locality_133. Locality names repeat across cities, so city + locality is used as the key.

**Insight:** Property age is similar across localities, so building age is not hiding a location effect.

## 9. BHK distribution across cities
Overall BHK mix: {1: 0.201, 2: 0.199, 3: 0.2, 4: 0.199, 5: 0.201}. Every city has a near-even split (largest deviation from 20% in any cell: 2.0%).

**Insight:** Every city has the same BHK mix, so no city is biased towards small or large homes.

## 10. Price trends for the top 5 most expensive localities
Top 5 (n>=10 listings, by mean price per sqft): Mysore / Locality_381, Amritsar / Locality_436, Durgapur / Locality_164, Coimbatore / Locality_147, Guwahati / Locality_142. The dataset has no sale dates, so build decade is used as the time axis. State this limitation clearly; the lines are noisy because each locality has few listings.

**Insight:** No locality shows a steady trend by build decade. The lines move up and down because each locality has few listings, so they should not be read as market history.

## 11. Correlation between numeric features
The raw features are essentially uncorrelated with each other (largest |r| among BHK, Size_in_SqFt, Age_of_Property, Nearby_Schools, Nearby_Hospitals, Amenity_Count = 0.00) and with price (Size -0.00, BHK -0.00, Schools 0.00). The strong correlations are mechanical: Price vs Price per sqft = 0.56 (PPS is derived from price), Price vs Future price = 1.00, and Floor_No vs Total_Floors = 0.58 (because Floor_No was capped at Total_Floors in Step 1).

**Insight:** The raw features are independent of price, so price cannot be predicted from them. The investment label and the future price therefore depend on the rules defined in Step 1, which is why I report leakage checks and baselines.

## 12. Nearby schools vs price per sqft
Mean price per sqft ranges Rs 12,888 to Rs 13,292 across school counts. Correlation with price per sqft = -0.000.

**Insight:** More nearby schools do not raise price per sqft (correlation -0.000).

## 13. Nearby hospitals vs price per sqft
Mean price per sqft ranges Rs 12,917 to Rs 13,257 across hospital counts. Correlation with price per sqft = -0.000.

**Insight:** More nearby hospitals do not raise price per sqft (correlation -0.000).

## 14. Price by furnished status
Mean price (lakhs): {'Furnished': 254.4, 'Semi-furnished': 254.3, 'Unfurnished': 255.0}. Differences are tiny.

**Insight:** Furnishing does not change price: the highest and lowest group differ by 0.3%.

## 15. Price per sqft by facing direction
Mean Rs per sqft: {'East': 13024.0, 'North': 13025.0, 'South': 13044.0, 'West': 13140.0}.

**Insight:** Facing direction has no price effect: the highest and lowest group differ by 0.9%.

## 16. Properties per owner type
{'Broker': 83479, 'Owner': 83268, 'Builder': 83253}

**Insight:** Each owner type holds about a third of the listings, so no seller type dominates the data.

## 17. Properties per availability status
{'Under_Construction': 125035, 'Ready_to_Move': 124965}

**Insight:** About half of the properties are ready to move and half are under construction. Ready-to-move earns a point in the investment score, so the label is balanced across both groups.

## 18. Does parking affect price?
Mean price with parking 254.7 vs without 254.4 lakhs. `Parking_Space` is Yes/No in this file, not a count.

**Insight:** Parking does not change price (gap 0.1%), so its point in the investment score is a design choice, not a market finding.

## 19. Amenities vs price per sqft
Mean Rs per sqft (with / without):
| Amenity    |   With |   Without |
|:-----------|-------:|----------:|
| Playground |  13049 |     13072 |
| Gym        |  13097 |     12999 |
| Garden     |  13065 |     13049 |
| Pool       |  13073 |     13037 |
| Clubhouse  |  13070 |     13041 |

Correlation of Amenity_Count with price per sqft = 0.002.

**Insight:** Amenities do not change price per sqft (amenity count correlation 0.002).

## 20. Public transport vs price per sqft and investment potential

| PT     |     pps |   good |
|:-------|--------:|-------:|
| Low    | 13085.5 |  0.375 |
| Medium | 13062.2 |  0.475 |
| High   | 13027.2 |  0.55  |

Better transport raises the good-investment rate partly by construction, because transport is part of the Infrastructure_Score used in the label. Mention this circularity.

**Insight:** Better transport goes with a higher good-investment rate (Low 37.5%, Medium 47.5%, High 55.0%), but this is partly built in because transport is part of the infrastructure score used in the label.

## X1. Area vs investment return
5-year modelled return by size band: {'XS': 44.4, 'S': 44.4, 'M': 44.5, 'L': 44.5, 'XL': 44.5} (%). Good-investment rate by band: {'XS': 0.465, 'S': 0.465, 'M': 0.47, 'L': 0.467, 'XL': 0.467}. Modelled return is flat because it comes from the growth formula, not from size. The good-investment rate is almost flat across size bands, so the label is not just a size effect (it compares price with the same-city, same-BHK median).

**Insight:** Modelled return does not depend on size, and the label is not a size effect, so the models are not simply rewarding large properties.

## X2. Infrastructure score vs resale value

| Infra_Band   |   price |   future |     pps |
|:-------------|--------:|---------:|--------:|
| Low          |   254.6 |    354.1 | 13053.2 |
| Mid          |   254.8 |    364.5 | 13099   |
| High         |   254.6 |    371.7 | 13063.1 |
| Top          |   254.3 |    381.1 | 13010.4 |

Current price barely changes with infrastructure; modelled future price rises only because infrastructure feeds the growth rate. The brief's 'crime rate' has no matching column in this dataset, so it is skipped.

**Insight:** Current price is the same in every infrastructure band. Infrastructure raises the modelled future price only through the growth rate, which is the relationship the regressor can learn.
