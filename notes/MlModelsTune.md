# Hyperparameter Tuning in scikit-learn (Regression) - Notes

> One `Pipeline` + `GridSearchCV` that searches across many models at once, plus the tuning knowledge that actually matters.

---

## Table of Contents
1. [Core Idea](#1-core-idea)
2. [Imports](#2-imports)
3. [Param Grids per Model](#3-param-grids-per-model)
4. [Scaling: Which Models Need It?](#4-scaling-which-models-need-it)
5. [Pipeline + GridSearchCV](#5-pipeline--gridsearchcv)
6. [Reading the Results](#6-reading-the-results)
7. [Important Hyperparameters Cheat Sheet](#7-important-hyperparameters-cheat-sheet)
8. [Search Strategies](#8-search-strategies-grid-vs-random-vs-halving)
9. [Best Practices and Common Mistakes](#9-best-practices-and-common-mistakes)
10. [Quick Reference](#10-quick-reference)

---

## 1. Core Idea

- A `Pipeline` chains steps: `scaler -> model`.
- `GridSearchCV` tries every combination of hyperparameters using cross-validation and picks the best.
- `param_grid` can be a **list of dicts**. Each dict is an independent grid. This lets us swap **entire models** (and even the scaler) in one search.
- Parameter naming: `stepname__parameter`, e.g. `model__alpha`.
- The `"model"` key (without `__`) replaces the whole estimator in that pipeline step.

```
Pipeline([("scaler", ...), ("model", ...)])
                 ^                 ^
        "scaler": [...]     "model": [...]
                            "model__alpha": [...]
```

---

## 2. Imports

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
```

> Note: `RandomForestRegressor` was accidentally imported twice in the original snippet. Import it once.

---

## 3. Param Grids per Model

We include a `"scaler"` key in each grid so that **each model gets the right preprocessing**:
- `StandardScaler()` for models that need scaling
- `"passthrough"` (skip the step) for tree-based models

```python
scaler_on  = [StandardScaler()]
scaler_off = ["passthrough"]

param_grid = []
```

### 3.1 Linear Models

```python
param_grid += [
    # Plain Linear Regression (no hyperparameters to tune)
    {
        "scaler": scaler_off,
        "model": [LinearRegression()],
    },

    # Ridge (L2 regularization)
    {
        "scaler": scaler_on,
        "model": [Ridge()],
        "model__alpha": [0.01, 0.1, 1, 10, 100],
    },

    # Lasso (L1 regularization, does feature selection)
    {
        "scaler": scaler_on,
        "model": [Lasso(max_iter=10000)],
        "model__alpha": [0.001, 0.01, 0.1, 1, 10],
    },

    # ElasticNet (L1 + L2 mix)
    {
        "scaler": scaler_on,
        "model": [ElasticNet(max_iter=10000)],
        "model__alpha": [0.001, 0.01, 0.1, 1, 10],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    },
]
```

**Notes**
- `alpha` = regularization strength. Bigger = simpler model = more bias, less variance.
- `l1_ratio`: `0` = pure Ridge, `1` = pure Lasso.
- Search `alpha` on a **log scale** (0.001, 0.01, 0.1, 1, 10, 100).

### 3.2 KNN

```python
param_grid += [
    {
        "scaler": scaler_on,
        "model": [KNeighborsRegressor()],
        "model__n_neighbors": [3, 5, 7, 11],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }
]
```

**Notes**
- `p=1` -> Manhattan distance, `p=2` -> Euclidean distance.
- Small `n_neighbors` = overfits (high variance). Large = underfits (high bias).
- KNN is distance based, so **scaling is mandatory**.

### 3.3 Decision Tree

```python
param_grid += [
    {
        "scaler": scaler_off,
        "model": [DecisionTreeRegressor(random_state=42)],
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": [None, "sqrt", "log2"],
    }
]
```

**Notes**
- Unconstrained trees overfit badly. Control complexity with `max_depth`, `min_samples_leaf`, `min_samples_split`.

### 3.4 Random Forest

```python
param_grid += [
    {
        "scaler": scaler_off,
        "model": [RandomForestRegressor(random_state=42, n_jobs=-1)],
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": [1.0, "sqrt"],
    }
]
```

**Notes**
- More trees (`n_estimators`) never hurts accuracy much, it only costs time. Treat it as a "set high and forget" parameter.
- `max_features=1.0` uses all features per split (the regression default). `"sqrt"` adds more randomness/decorrelation.

### 3.5 Gradient Boosting

```python
param_grid += [
    {
        "scaler": scaler_off,
        "model": [GradientBoostingRegressor(random_state=42)],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [2, 3, 5],
        "model__subsample": [0.8, 1.0],
    }
]
```

**Notes**
- `learning_rate` and `n_estimators` are **coupled**: lower learning rate needs more estimators.
- Shallow trees (`max_depth` 2-5) work best for boosting.
- `subsample < 1.0` = stochastic gradient boosting, reduces overfitting.

### 3.6 Extra Trees

```python
param_grid += [
    {
        "scaler": scaler_off,
        "model": [ExtraTreesRegressor(random_state=42, n_jobs=-1)],
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": [1.0, "sqrt"],
    }
]
```

**Notes**
- Like Random Forest but picks split thresholds randomly. Faster, often lower variance.

### 3.7 SVR

Split into **two grids**, because `gamma` is ignored by the linear kernel (otherwise we waste compute on duplicate fits).

```python
param_grid += [
    # Linear kernel (no gamma)
    {
        "scaler": scaler_on,
        "model": [SVR(kernel="linear")],
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2],
    },
    # RBF kernel
    {
        "scaler": scaler_on,
        "model": [SVR(kernel="rbf")],
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2],
        "model__gamma": ["scale", "auto", 0.01, 0.1],
    },
]
```

**Notes**
- `C`: regularization (higher = fit training data harder, risk overfit).
- `epsilon`: width of the "no penalty" tube around predictions.
- `gamma` (RBF): how far one sample's influence reaches. High = wiggly/overfit, low = smooth.
- SVR is **slow on large datasets** (roughly O(n^2) to O(n^3)). Consider skipping it beyond ~50k rows.
- SVR is sensitive to target scale too. See the `TransformedTargetRegressor` tip in section 9.

### 3.8 HistGradientBoosting

```python
param_grid += [
    {
        "scaler": scaler_off,
        "model": [HistGradientBoostingRegressor(random_state=42)],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [15, 31],
        "model__l2_regularization": [0, 0.1, 1],
    }
]
```

**Notes**
- Very fast on large data (LightGBM-style histograms). Handles `NaN` natively.
- `max_iter` is the number of boosting rounds (the equivalent of `n_estimators`).
- Supports built-in early stopping: `early_stopping=True`, `n_iter_no_change=10`.

---

## 4. Scaling: Which Models Need It?

| Model | Scale? | Why |
|---|---|---|
| Linear Regression | Usually not needed | Predictions are scale-invariant (but helps numerics/interpretation) |
| Ridge | Yes | Penalty depends on coefficient size |
| Lasso | Yes | Same as above |
| ElasticNet | Yes | Same as above |
| KNN | Yes | Distance-based |
| SVR | Yes | Kernel/distance-based |
| Decision Tree | No | Splits are order-based |
| Random Forest | No | Same |
| Gradient Boosting | No | Same |
| Extra Trees | No | Same |
| HistGradientBoosting | No | Same |

**Trick used above:** put `"scaler": [StandardScaler()]` or `"scaler": ["passthrough"]` inside each grid dict. One pipeline, correct preprocessing per model. No need for separate pipelines.

---

## 5. Pipeline + GridSearchCV

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge()),          # placeholder, replaced by the grid
])

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    refit=True,                  # retrain best model on full train set
)

grid.fit(X_train, y_train)
```

### Evaluate on the untouched test set

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

best = grid.best_estimator_
pred = best.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2  :", r2_score(y_test, pred))
```

---

## 6. Reading the Results

```python
print(grid.best_params_)      # winning hyperparameters (incl. which model)
print(grid.best_score_)       # mean CV score (NEGATIVE MSE, closer to 0 is better)
print(grid.best_estimator_)   # the refit pipeline
```

### Convert score to RMSE

```python
rmse = (-grid.best_score_) ** 0.5
```

### Compare all candidates in a DataFrame

```python
import pandas as pd

results = pd.DataFrame(grid.cv_results_)
results["model_name"] = results["param_model"].apply(lambda m: type(m).__name__)

summary = (
    results.groupby("model_name")["mean_test_score"]
    .max()
    .sort_values(ascending=False)
)
print(summary)
```

Useful columns in `cv_results_`:
- `mean_test_score`, `std_test_score` -> performance and stability
- `mean_train_score` (needs `return_train_score=True`) -> compare with test to spot **overfitting**
- `mean_fit_time` -> cost of each config
- `rank_test_score`

---

## 7. Important Hyperparameters Cheat Sheet

The 1-3 parameters that matter most for each model. Tune these first.

| Model | Most important | Also useful | Direction |
|---|---|---|---|
| Ridge / Lasso | `alpha` | `max_iter` (Lasso convergence) | Higher alpha = simpler |
| ElasticNet | `alpha`, `l1_ratio` | `max_iter` | Higher `l1_ratio` = more sparsity |
| KNN | `n_neighbors` | `weights`, `p` | Higher k = smoother |
| Decision Tree | `max_depth`, `min_samples_leaf` | `min_samples_split`, `ccp_alpha` | Lower depth = simpler |
| Random Forest | `n_estimators`, `max_features`, `min_samples_leaf` | `max_depth`, `max_samples` | More trees = stabler |
| Extra Trees | `n_estimators`, `max_features`, `min_samples_leaf` | `max_depth` | Same as RF |
| Gradient Boosting | `learning_rate`, `n_estimators`, `max_depth` | `subsample`, `min_samples_leaf`, `max_features` | Low LR + many trees |
| HistGradientBoosting | `learning_rate`, `max_iter`, `max_leaf_nodes` | `l2_regularization`, `min_samples_leaf`, `max_depth` | Low LR + more iter |
| SVR | `C`, `gamma` (rbf), `epsilon` | `kernel` | Higher C/gamma = more complex |

### Bias-Variance quick guide

| Symptom | Diagnosis | Fix |
|---|---|---|
| Train score high, CV score low | Overfitting (high variance) | More regularization, shallower trees, larger `min_samples_leaf`, lower `C`/`gamma`, more data |
| Both scores low | Underfitting (high bias) | Less regularization, deeper trees, more estimators, more features/features engineering |

### Good search ranges (rules of thumb)

| Parameter | Scale | Typical range |
|---|---|---|
| `alpha` (Ridge/Lasso) | log | 1e-4 to 1e2 |
| `C` (SVR) | log | 1e-2 to 1e3 |
| `gamma` (SVR) | log | 1e-4 to 1e1 |
| `learning_rate` (boosting) | log | 0.01 to 0.3 |
| `max_depth` (boosting) | linear | 2 to 8 |
| `max_depth` (RF) | linear | 5 to 30 or None |
| `n_estimators` | linear | 100 to 1000 |
| `min_samples_leaf` | linear | 1 to 20 |

---

## 8. Search Strategies: Grid vs Random vs Halving

Grid size explodes: 5 params x 4 values each = 1024 combos x 5 folds = **5120 fits**.

### GridSearchCV
- Tries every combination. Exhaustive but expensive.
- Best for **small** grids or a final fine-tuning pass.

### RandomizedSearchCV
- Samples `n_iter` random combos from lists **or distributions**. Usually finds nearly as good a result in a fraction of the time.
- Best for **large** search spaces (Bergstra & Bengio showed random beats grid for the same budget when only a few params really matter).

```python
from scipy.stats import loguniform, randint

rand_grid = [
    {
        "scaler": ["passthrough"],
        "model": [GradientBoostingRegressor(random_state=42)],
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 7),
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    },
    {
        "scaler": [StandardScaler()],
        "model": [SVR(kernel="rbf")],
        "model__C": loguniform(1e-2, 1e3),
        "model__gamma": loguniform(1e-4, 1e1),
        "model__epsilon": loguniform(1e-3, 1),
    },
]

rand = RandomizedSearchCV(
    pipe, rand_grid,
    n_iter=50, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, random_state=42,
)
rand.fit(X_train, y_train)
```

### HalvingGridSearchCV / HalvingRandomSearchCV
- Starts many candidates on a small amount of data, keeps the best ones, gives them more data. Much faster for big grids.
- Still experimental in scikit-learn, so it needs an explicit import:

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

halving = HalvingGridSearchCV(pipe, param_grid, cv=5, factor=3,
                              scoring="neg_mean_squared_error", n_jobs=-1)
```

### Recommended workflow: coarse -> fine

1. **Coarse**: `RandomizedSearchCV` with wide ranges to find promising models/regions.
2. **Fine**: `GridSearchCV` in a narrow band around the best values.
3. **Final**: evaluate once on the held-out test set.

### Beyond scikit-learn
- **Optuna** / **Hyperopt** / **scikit-optimize** (Bayesian optimization): smarter than random, learns from previous trials. Worth learning for serious tuning.

---

## 9. Best Practices and Common Mistakes

1. **Never scale outside the pipeline.** Fitting the scaler on all data before CV = **data leakage**. Putting it in the `Pipeline` makes it refit on each training fold only.
2. **Keep a final test set** that is never used in tuning. CV score from `best_score_` is optimistically biased because you picked the max.
3. **Choose the right metric.**
   - `neg_mean_squared_error` -> penalizes big errors heavily
   - `neg_root_mean_squared_error` -> same, in target units
   - `neg_mean_absolute_error` -> robust to outliers
   - `r2` -> relative measure
   - Remember sklearn scorers are "higher is better", hence the `neg_`.
4. **Use `random_state`** on models and CV splitters for reproducibility.
5. **Use `KFold(shuffle=True)`** if your data is sorted or grouped. For **time series** use `TimeSeriesSplit`. For grouped data use `GroupKFold`.
6. **Don't over-tune.** Big gains come from data quality, feature engineering, and picking the right model family. Tuning usually gives smaller gains.
7. **Watch `n_jobs`.** `n_jobs=-1` in GridSearchCV **and** inside RandomForest can oversubscribe CPUs. Parallelize at one level only (usually the search).
8. **Tune the important params first**, freeze the rest at defaults.
9. **Start with defaults as a baseline** so you know if tuning actually helped.
10. **Log-scale for scale-type params** (`alpha`, `C`, `gamma`, `learning_rate`).
11. **Skewed target?** Wrap the model so the target is transformed (helps linear models and SVR):

    ```python
    from sklearn.compose import TransformedTargetRegressor
    import numpy as np

    model = TransformedTargetRegressor(
        regressor=Ridge(),
        func=np.log1p, inverse_func=np.expm1,
    )
    # Params then become: "model__regressor__alpha"
    ```
12. **Categorical features?** Use `ColumnTransformer` (`OneHotEncoder` for categoricals, `StandardScaler` for numerics) as the first pipeline step instead of a bare scaler.
13. **Cost check:** total fits = (number of combinations) x `cv`. Compute it before hitting run.

---

## 10. Quick Reference

### Naming rules
```python
"model"              # swap the estimator in the pipeline step named "model"
"model__alpha"       # a hyperparameter of that estimator
"scaler"             # swap the scaler (or "passthrough")
"model__regressor__alpha"   # nested (e.g. inside TransformedTargetRegressor)
```

### Full minimal template

```python
pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print("CV RMSE :", -grid.best_score_)
print("Test R2 :", grid.score(X_test, y_test))   # R2 by default for regressors
```

### Useful `GridSearchCV` arguments

| Argument | Purpose |
|---|---|
| `cv` | Number of folds or a splitter object |
| `scoring` | Metric string, callable, or dict for multiple metrics |
| `n_jobs` | Parallel workers (`-1` = all cores) |
| `refit` | Retrain best model on the full train set (default `True`) |
| `return_train_score` | Also record train scores (to detect overfitting) |
| `error_score` | What to do if a fit fails (`"raise"` or `np.nan`) |
| `verbose` | Progress logging |

### Saving the best model

```python
import joblib
joblib.dump(grid.best_estimator_, "best_model.joblib")
model = joblib.load("best_model.joblib")
```

---
*End of notes.*