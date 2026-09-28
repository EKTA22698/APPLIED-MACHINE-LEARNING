# Applied Machine Learning Lab --- Assignment 1

**Course:** CSAI2017P --- Applied Machine Learning (Lab)\
**Lab Assignment:** LA01 --- Environment Setup, Preprocessing and
Exploratory Data Analysis
**Student:** Ekta Bhardwaj
**SAP ID:** 590022698

## 1. Overview

This assignment establishes a reproducible machine-learning environment,
performs an initial inspection of the California Housing dataset, and
evaluates how feature scaling affects a distance-based regression model.

The following components were attempted according to the assignment
choice rule:

-   **A1 --- Environment Proof**
-   **A2 --- First Look at the Data**
-   **B1 --- Scaling Changes the Answer**
-   **C1 --- Reproducibility Check**

## 2. What Was Attempted

### A1 --- Environment Proof

A fresh Python virtual environment was created and the required
libraries were installed: NumPy, Pandas, Scikit-learn, Matplotlib,
Seaborn, and Jupyter.

The environment was verified by recording the Python and library
versions in one notebook cell.

### A2 --- First Look at the Data

The California Housing dataset was loaded using the specified
Scikit-learn loader:

``` python
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
```

The resulting DataFrame was inspected using:

-   `head()`
-   `info()`
-   `describe()`
-   `isna().sum()`

The analysis covered dataset dimensions, target information, missing
values, and differences in feature scale.

### B1 --- Scaling Changes the Answer

The data was split into training and testing sets using an 80/20 split
with `random_state=0`.

Two KNN regression approaches were compared:

1.  `KNeighborsRegressor(n_neighbors=5)` on the raw features.
2.  `StandardScaler` followed by `KNeighborsRegressor(n_neighbors=5)`
    inside a `Pipeline`.

Test **Mean Absolute Error (MAE)** was used as the evaluation metric.

### C1 --- Reproducibility Check

The notebook kernel was restarted and the complete notebook was executed
from top to bottom. The reported numerical results were then compared
with the original execution.

## 3. Environment Results

  Component        Version
  -------------- ---------
  Python            3.14.2
  NumPy              2.4.4
  Pandas             3.0.2
  Scikit-learn       1.9.0
  Matplotlib        3.11.1
  Seaborn           0.13.2

**Observation:** Recording exact software versions makes the
computational environment easier to reproduce. Pinning versions is
useful when work is shared because changes between library releases can
affect dependencies, behaviour, or numerical results.

## 4. A2 --- Dataset Inspection

### Dataset Profile

The California Housing dataset contains:

-   **20,640 rows**
-   **8 numeric features**
-   **Target:** `MedHouseVal`
-   **Target units:** \$100,000

The DataFrame contains 9 columns after the target is added to the
feature set.

### Missing-Value Check

`isna().sum()` returned zero for every column, while `info()` showed
20,640 non-null entries for each column.

**Observation:** No missing values were detected, so no missing-value
treatment was necessary for this analysis.

### Feature-Scale Observation

`Population` was identified as a feature on a markedly different
numerical scale.

-   `Population`: **3 to 35,682**
-   `MedInc`: **0.4999 to 15.0001**

This difference is relevant for distance-based algorithms because
variables with larger numerical ranges can disproportionately affect
distance calculations.

## 5. B1 --- Effect of Feature Scaling on KNN

### Train-Test Split

-   **Training samples:** 16,512
-   **Testing samples:** 4,128
-   **Random state:** 0

### Model Comparison

  Approach                                    Test MAE
  --------------------------- ------------------------
  KNN without scaling           **0.8141586656976745**
  KNN with `StandardScaler`      **0.430758769379845**

### Observation

KNN determines predictions from distances between observations rather
than from learned model coefficients. With the raw features, high-range
variables such as `Population` can disproportionately influence those
distances and therefore affect which observations are selected as
neighbours. Standardization places the features on a comparable scale
before the neighbourhoods are determined. In this experiment, scaling
reduced the test MAE from **0.8142** to **0.4308**, showing a
substantial improvement in predictive performance.

**Key interpretation:** For a distance-based model such as KNN, feature
scaling changes the geometry of the feature space itself. Therefore,
preprocessing can directly change the model's neighbourhood structure
and its final predictions.

## 6. C1 --- Reproducibility Check

After restarting the kernel and executing the notebook from top to
bottom, the numerical results reproduced exactly:

-   Raw KNN MAE: **0.8141586656976745**
-   Scaled KNN MAE: **0.430758769379845**

**Observation:** No numerical discrepancies were observed. The identical
results under the same environment and fixed `random_state=0` confirm
that the reported results are reproducible and do not depend on the
previous kernel state or execution order.

## 7. Key Findings

1.  The California Housing dataset contains **20,640 observations and 8
    numeric predictors**, with `MedHouseVal` as the target.
2.  No missing values were detected in the inspected data.
3.  Feature ranges are substantially different; `Population` provides a
    clear example of a high-range feature.
4.  Standardization substantially improved KNN performance:
    -   Raw features: **MAE = 0.8142**
    -   Standardized features: **MAE = 0.4308**
5.  The reproducibility run produced identical numerical results.

## 8. Conclusion

This experiment demonstrates that data preparation can directly
influence model performance. The initial exploration revealed a
substantial difference in feature scales, and the KNN comparison showed
the practical effect of addressing that imbalance. Standardizing the
features reduced the test MAE considerably because it prevented
large-range variables from dominating the distance calculation.

The final reproducibility check confirmed that the reported results
could be reproduced by restarting the kernel and executing the notebook
from start to finish. The complete workflow therefore provides a
consistent path from environment setup and data inspection to model
evaluation and verification.

## 9. Dataset Source

California Housing was loaded through Scikit-learn's built-in
`fetch_california_housing()` dataset loader, as specified in the
assignment's anchor-dataset material.


