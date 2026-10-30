# Customer Segmentation with K-Means

Grouping shoppers by spend and visit frequency using scikit-learn.

## Problem
A shop wants to send the right offer to the right customer. I grouped customers by how much they spend and how often they visit, so each group can get a different offer.

## Data
A hand-made sample of 10 customers (for practice, not real shop data):
- `annual_spend`: annual spend in thousands of rupees
- `visits_month`: number of visits per month

## Approach
1. Used `annual_spend` and `visits_month` as features. `customer_id` is only a label, so it is not used.
2. Fitted `KMeans` from scikit-learn with k = 2.
3. Checked the final centres by hand-computing the averages of each group.

## Results
| Group | Avg annual spend | Avg visits per month | Meaning | Suggested action |
|---|---|---|---|---|
| Low-spend | about 13K | about 2.2 | Occasional shoppers | Discount offers |
| High-spend | about 50.4K | about 9.4 | Regular shoppers | New-collection alerts |

Cost (inertia): 59.2, the sum of squared distances from each customer to its group centre. A lower value means tighter groups.

## Limitations and Improvements
- Only 10 sample customers, so results are illustrative.
- Try several values of k and use the elbow method to choose k.
- Add features such as product category, and scale the features.
- Test on real shop data.

## How to run
```
pip install pandas scikit-learn
python kmeans_segmentation.py
```
