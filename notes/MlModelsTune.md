# Scikit-learn Hyperparameter Tuning Notes

Pattern used for every model:

```
Pipeline -> param_grid (keys = "stepname__param") -> GridSearchCV -> .fit(X_train, y_train) -> best_params_
```

**How each model section is laid out**

1. **Param table**: what the important params do
2. **`*_PARAMS` dict**: EVERY constructor param with its default value (use as `Model(**PARAMS)`, or as a checklist of what you can change)
3. **`*_GRID` dict**: the practical search space for `GridSearchCV`
4. **One `tune(...)` call**: runs the search and prints results

> **Conventions**
> - `random_state` is set to `42` in the dicts for reproducibility. Everything else is the library default.
> - `None` inside an XGBoost dict means "use XGBoost's internal default" (real value shown in the comment).
> - Params get added or deprecated between sklearn versions (e.g. `monotonic_cst`, `LogisticRegression.penalty` / `multi_class`). If `Model(**PARAMS)` raises `TypeError: unexpected keyword argument`, just delete that key.
> - Source of truth for YOUR installed version: `Model().get_params()` returns the full param dict.

---

## Table of contents

0. [Setup and helpers](#0-setup-and-helpers)
1. [Building blocks: Pipeline, GridSearchCV](#1-building-blocks)
2. [Regression models](#2-regression-models)
3. [Classification models](#3-classification-models)
4. [Cheat sheet](#4-cheat-sheet)

---

## 0. Setup and helpers

Assumes `X_train, y_train, X_test, y_test` already exist.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score,
)

# pip install xgboost lightgbm   (only needed for the XGBoost / LightGBM sections)


def evaluate_regression(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")


def evaluate_classification(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"{name}: Accuracy={acc:.2f}, F1={f1:.2f}")


def tune(name, pipe, grid, task="reg", scoring=None, cv=5,
         search=GridSearchCV, **search_kw):
    """Run a search, print best params + test metrics, return the fitted search object.

    task: "reg" or "clf"
    search: GridSearchCV (default) or RandomizedSearchCV (pass n_iter=30 via search_kw)
    """
    scoring = scoring or ("r2" if task == "reg" else "accuracy")
    gs = search(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, **search_kw)
    gs.fit(X_train, y_train)
    print(f"{name} best: {gs.best_params_}  |  cv {scoring} = {gs.best_score_:.4f}")
    pred = gs.predict(X_test)
    (evaluate_regression if task == "reg" else evaluate_classification)(name, y_test, pred)
    return gs
```

**Look up every param of any model (your installed version):**

```python
from sklearn.linear_model import Ridge
print(Ridge().get_params())        # dict of ALL params + current values
print(Ridge().get_params().keys()) # just the names
```

---

## 1. Building blocks

### `Pipeline()`

```python
Pipeline(steps, memory=None, verbose=False)
```

| Param | What it does |
|---|---|
| `steps` | List of `(name, transformer/estimator)` tuples. The only one you've used. |
| `memory` | Cache intermediate step results to disk (avoids recomputing on repeated runs, useful with slow preprocessing). |
| `verbose` | Prints timing info for each step when fitting. |

### `GridSearchCV()`

```python
GridSearchCV(
    estimator,
    param_grid,
    scoring=None,
    n_jobs=None,
    refit=True,
    cv=None,
    verbose=0,
    error_score=np.nan,
    return_train_score=False,
)
```

| Param | Default | What it does |
|---|---|---|
| `estimator` | n/a | The pipeline/model to tune. |
| `param_grid` | n/a | Dict or list of dicts of hyperparameters to search. |
| `scoring` | `None` | Metric to optimize (defaults to the estimator's own `.score()`; r2 for regressors, accuracy for classifiers). |
| `n_jobs` | `None` | Parallelism (`-1` = all cores). |
| `refit` | `True` | Automatically refits the best combo on the full training set. This is why `grid.predict()` works directly. |
| `cv` | `None` | Number of folds or a CV splitter object. Default is 5 if left `None`. |
| `verbose` | `0` | Prints progress while searching; higher = more detail. |
| `error_score` | `np.nan` | What to report if a fit fails on some combo instead of crashing the whole search. |
| `return_train_score` | `False` | If `True`, also reports training-set scores in `cv_results_`. Useful for spotting overfitting. |

```python
GRIDSEARCH_PARAMS = {
    "estimator": None,          # required: your pipeline
    "param_grid": None,         # required: dict or list of dicts
    "scoring": None,
    "n_jobs": None,
    "refit": True,
    "cv": None,                 # 5-fold default
    "verbose": 0,
    "pre_dispatch": "2*n_jobs",
    "error_score": np.nan,
    "return_train_score": False,
}
```

**`RandomizedSearchCV`** takes the same arguments, but `param_grid` becomes `param_distributions` and you add `n_iter=` (how many random combos to try) and `random_state=`. Use it when a grid is too big to search exhaustively.

### Useful things to read after fitting

```python
grid.best_params_          # winning combo
grid.best_score_           # mean CV score of the winner
grid.best_estimator_       # the refit pipeline
grid.cv_results_           # full results (wrap in pd.DataFrame)
grid.best_estimator_.named_steps["model"]   # the fitted model inside the pipeline
```

---

## 2. Regression models

### 2.1 `LinearRegression()`

```python
LinearRegression(fit_intercept=True, copy_X=True, tol=1e-6, n_jobs=None, positive=False)
```

| Param | Default | What it does |
|---|---|---|
| `fit_intercept` | `True` | Whether to calculate an intercept (bias term). Set `False` only if you know data passes through the origin. |
| `copy_X` | `True` | Copies X before fitting instead of overwriting it in place. |
| `tol` | `1e-6` | Precision tolerance for the solver's stopping criterion (newer sklearn only). |
| `n_jobs` | `None` | CPU cores to use (`-1` = all). Only helps for multi-target or large sparse problems. |
| `positive` | `False` | If `True`, forces all coefficients to be non-negative. |

There is no `alpha`. Plain linear regression has no regularization, so nothing to tune. This is why `GridSearchCV` on `LinearRegression` alone is pointless.

```python
from sklearn.linear_model import LinearRegression

LINREG_PARAMS = {
    "fit_intercept": True,
    "copy_X": True,
    "tol": 1e-6,          # remove if your sklearn is older than 1.7
    "n_jobs": None,
    "positive": False,
}

# Only two things to toggle (optional):
LINREG_GRID = {
    "model__fit_intercept": [True, False],
    "model__positive": [False, True],
}

linreg_pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(**LINREG_PARAMS))])
linreg_grid = tune("LinearRegression", linreg_pipe, LINREG_GRID, task="reg")
```

---

### 2.2 `Ridge()`

```python
Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4,
      solver='auto', positive=False, random_state=None)
```

| Param | Default | What it does |
|---|---|---|
| `alpha` | `1.0` | L2 penalty strength. Higher = more shrinkage. **The main one you tune.** |
| `fit_intercept` | `True` | Same as LinearRegression. |
| `copy_X` | `True` | Same as LinearRegression. |
| `max_iter` | `None` | Max iterations for iterative solvers (only used by some solver choices). |
| `tol` | `1e-4` | Solver precision / stopping tolerance. |
| `solver` | `'auto'` | `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`, `'sparse_cg'`, `'sag'`, `'saga'`, `'lbfgs'`. `'auto'` picks based on data. |
| `positive` | `False` | Forces non-negative coefficients (only supported by the `'lbfgs'` solver). |
| `random_state` | `None` | Seed, only relevant for `'sag'`/`'saga'` (stochastic). |

```python
from sklearn.linear_model import Ridge

RIDGE_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "copy_X": True,
    "max_iter": None,
    "tol": 1e-4,
    "solver": "auto",
    "positive": False,
    "random_state": 42,
}

RIDGE_GRID = {"model__alpha": [10, 1.0, 0.1, 0.01]}

ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(**RIDGE_PARAMS))])
ridge_grid = tune("Ridge", ridge_pipe, RIDGE_GRID, task="reg")
```

---

### 2.3 `Lasso()`

```python
Lasso(alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000,
      tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic')
```

| Param | Default | What it does |
|---|---|---|
| `alpha` | `1.0` | L1 penalty strength. Main tuning param. Can zero out coefficients entirely. |
| `fit_intercept` | `True` | Same as above. |
| `precompute` | `False` | Use a precomputed Gram matrix to speed up fitting on large datasets. |
| `copy_X` | `True` | Same as above. |
| `max_iter` | `1000` | Max iterations of the coordinate descent solver. Default often isn't enough, so bump to `10000`. |
| `tol` | `1e-4` | Convergence tolerance. |
| `warm_start` | `False` | Reuse the previous fit's solution as a starting point (useful when refitting repeatedly). |
| `positive` | `False` | Forces non-negative coefficients. |
| `random_state` | `None` | Seed for `selection='random'` only. |
| `selection` | `'cyclic'` | Which feature coordinate descent updates each step: `'cyclic'` (in order) or `'random'` (often converges faster). |

```python
from sklearn.linear_model import Lasso

LASSO_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "precompute": False,
    "copy_X": True,
    "max_iter": 10000,        # default is 1000; raised to avoid ConvergenceWarning
    "tol": 1e-4,
    "warm_start": False,
    "positive": False,
    "random_state": 42,
    "selection": "cyclic",
}

LASSO_GRID = {"model__alpha": [10, 1.0, 0.1, 0.01]}

lasso_pipe = Pipeline([("scaler", StandardScaler()), ("model", Lasso(**LASSO_PARAMS))])
lasso_grid = tune("Lasso", lasso_pipe, LASSO_GRID, task="reg")
```

---

### 2.4 `ElasticNet()`

Same params as Lasso, plus `l1_ratio`.

| Param | Default | What it does |
|---|---|---|
| `l1_ratio` | `0.5` | Blend between L1 and L2. `0` = pure Ridge, `1` = pure Lasso. Second main tuning param alongside `alpha`. |

```python
from sklearn.linear_model import ElasticNet

ELASTIC_PARAMS = {
    "alpha": 1.0,
    "l1_ratio": 0.5,
    "fit_intercept": True,
    "precompute": False,
    "max_iter": 10000,        # default is 1000
    "copy_X": True,
    "tol": 1e-4,
    "warm_start": False,
    "positive": False,
    "random_state": 42,
    "selection": "cyclic",
}

ELASTIC_GRID = {
    "model__alpha": [10, 1.0, 0.1, 0.01],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

elastic_pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(**ELASTIC_PARAMS))])
elastic_grid = tune("ElasticNet", elastic_pipe, ELASTIC_GRID, task="reg")
```

---

### 2.5 Ridge / Lasso / ElasticNet, all in ONE search

Swapping the whole `model` step by putting estimators inside the grid:

```python
pipe_all = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])  # placeholder

PARAM_GRID_ALL = [
    {"model": [Ridge()], "model__alpha": [10, 1.0, 0.1, 0.01]},
    {"model": [Lasso(max_iter=10000)], "model__alpha": [10, 1.0, 0.1, 0.01]},
    {"model": [ElasticNet(max_iter=10000)],
     "model__alpha": [10, 1.0, 0.1, 0.01],
     "model__l1_ratio": [0.1, 0.5, 0.9]},
]

grid_all = tune("Best linear model", pipe_all, PARAM_GRID_ALL, task="reg")
```

---

### 2.6 `DecisionTreeRegressor()`

```python
DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features=None, random_state=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, ccp_alpha=0.0)
```

| Param | Default | What it does |
|---|---|---|
| `criterion` | `'squared_error'` | Split quality measure: `'squared_error'`, `'friedman_mse'`, `'absolute_error'`, `'poisson'`. |
| `splitter` | `'best'` | `'best'` = optimal split each time; `'random'` = random splits (faster, more regularized). |
| `max_depth` | `None` | Max tree depth. `None` = grows until leaves are pure, which usually overfits. **Your #1 tuning knob.** |
| `min_samples_split` | `2` | Min samples a node needs before it may split. |
| `min_samples_leaf` | `1` | Min samples in a leaf. Raising this smooths the model. |
| `min_weight_fraction_leaf` | `0.0` | Same as above but as a fraction of total sample weight. |
| `max_features` | `None` | Features considered per split. `None` = all. |
| `random_state` | `None` | Seed for reproducibility. |
| `max_leaf_nodes` | `None` | Caps total number of leaves. |
| `min_impurity_decrease` | `0.0` | Split only happens if it reduces impurity by at least this much. |
| `ccp_alpha` | `0.0` | Cost-complexity pruning. Higher = prune more branches after the tree is built. |

```python
from sklearn.tree import DecisionTreeRegressor

TREE_REG_PARAMS = {
    "criterion": "squared_error",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": None,
    "random_state": 42,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,     # newer sklearn only
}

TREE_REG_GRID = {
    "model__max_depth": [3, 5, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

tree_pipe = Pipeline([("model", DecisionTreeRegressor(**TREE_REG_PARAMS))])   # no scaler needed
tree_grid = tune("Decision Tree", tree_pipe, TREE_REG_GRID, task="reg")
```

---

### 2.7 `RandomForestRegressor()`

All DecisionTree params apply per-tree, plus:

| Param | Default | What it does |
|---|---|---|
| `n_estimators` | `100` | Number of trees. More = generally better but slower, diminishing returns. |
| `bootstrap` | `True` | Each tree trains on a random resampled subset of rows (bagging). |
| `oob_score` | `False` | If `True` (needs `bootstrap=True`), uses out-of-bag samples to estimate performance without a validation set. |
| `n_jobs` | `None` | Parallelize tree-building across cores. |
| `verbose` | `0` | Print progress while fitting. |
| `warm_start` | `False` | Add more trees to an existing forest instead of starting over. |
| `max_samples` | `None` | If `bootstrap=True`, how many samples each tree draws. |
| `max_features` | `1.0` | Regressor default = all features (classifier default is `'sqrt'`). |

```python
from sklearn.ensemble import RandomForestRegressor

RF_REG_PARAMS = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 42,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None,     # newer sklearn only
}

RF_REG_GRID = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", "log2", None],
}

rf_pipe = Pipeline([("model", RandomForestRegressor(**RF_REG_PARAMS))])
rf_grid = tune("Random Forest", rf_pipe, RF_REG_GRID, task="reg")
```

---

### 2.8 `GradientBoostingRegressor()`

| Param | Default | What it does |
|---|---|---|
| `loss` | `'squared_error'` | `'squared_error'`, `'absolute_error'`, `'huber'`, `'quantile'`. |
| `learning_rate` | `0.1` | Shrinks each tree's contribution. Lower = needs more trees but often generalizes better. Trade-off with `n_estimators`. |
| `n_estimators` | `100` | Number of boosting stages (sequential trees). |
| `subsample` | `1.0` | Fraction of rows per tree (`< 1.0` adds randomness, like bagging). |
| `max_depth` | `3` | Depth per tree. Kept shallow since boosting builds many weak trees. |
| `validation_fraction` | `0.1` | Portion of training data held out for early stopping (only if `n_iter_no_change` is set). |
| `n_iter_no_change` | `None` | Stop early if validation score doesn't improve for this many rounds. |
| `tol` | `1e-4` | Tolerance for early stopping. |
| `alpha` | `0.9` | Only for `loss='huber'` or `'quantile'`. Sets the quantile. |
| `ccp_alpha` | `0.0` | Pruning, same as decision tree. |
| `init` | `None` | Estimator (or `'zero'`) for the initial prediction. |

```python
from sklearn.ensemble import GradientBoostingRegressor

GB_REG_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "criterion": "friedman_mse",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 3,
    "min_impurity_decrease": 0.0,
    "init": None,
    "random_state": 42,
    "max_features": None,
    "alpha": 0.9,
    "verbose": 0,
    "max_leaf_nodes": None,
    "warm_start": False,
    "validation_fraction": 0.1,
    "n_iter_no_change": None,
    "tol": 1e-4,
    "ccp_alpha": 0.0,
}

GB_REG_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__max_depth": [3, 5],
}

gb_pipe = Pipeline([("model", GradientBoostingRegressor(**GB_REG_PARAMS))])
gb_grid = tune("Gradient Boosting", gb_pipe, GB_REG_GRID, task="reg")
```

---

### 2.9 `SVR()` (Support Vector Regressor)

| Param | Default | What it does |
|---|---|---|
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`. |
| `degree` | `3` | Polynomial degree (only `kernel='poly'`). |
| `gamma` | `'scale'` | Kernel coefficient for rbf/poly/sigmoid. Controls how far one point's influence reaches. |
| `coef0` | `0.0` | Independent term, only for poly/sigmoid. |
| `C` | `1.0` | Regularization strength, **inverse** (lower = more regularization). |
| `epsilon` | `0.1` | Width of the "no penalty" margin around predictions. |
| `shrinking` | `True` | Shrinking heuristic to speed up training. |
| `max_iter` | `-1` | `-1` = no iteration limit. |

```python
from sklearn.svm import SVR

SVR_PARAMS = {
    "kernel": "rbf",
    "degree": 3,
    "gamma": "scale",
    "coef0": 0.0,
    "tol": 1e-3,
    "C": 1.0,
    "epsilon": 0.1,
    "shrinking": True,
    "cache_size": 200,
    "verbose": False,
    "max_iter": -1,
}

SVR_GRID = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"],
}

svr_pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR(**SVR_PARAMS))])
svr_grid = tune("SVR", svr_pipe, SVR_GRID, task="reg")
```

---

### 2.10 `KNeighborsRegressor()`

| Param | Default | What it does |
|---|---|---|
| `n_neighbors` | `5` | How many nearest points to average. Main tuning param. |
| `weights` | `'uniform'` | `'uniform'` = equal vote; `'distance'` = closer points count more. |
| `algorithm` | `'auto'` | Neighbor search structure: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`. |
| `leaf_size` | `30` | Affects speed/memory of tree-based algorithms, not results. |
| `p` | `2` | Minkowski power (`1` = Manhattan, `2` = Euclidean). |
| `metric` | `'minkowski'` | Distance metric. |

```python
from sklearn.neighbors import KNeighborsRegressor

KNN_REG_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": None,
}

KNN_REG_GRID = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
}

knn_pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(**KNN_REG_PARAMS))])
knn_grid = tune("KNN", knn_pipe, KNN_REG_GRID, task="reg")
```

---

### 2.11 NEW: `ExtraTreesRegressor()`

Like Random Forest, but split thresholds are chosen **randomly** rather than optimally. It is faster and often smoother. Also, `bootstrap=False` by default (each tree sees all rows).

| Param | Default | What it does |
|---|---|---|
| `n_estimators` | `100` | Number of trees. |
| `max_depth` | `None` | Depth per tree. |
| `min_samples_leaf` | `1` | Min samples in a leaf. Raising it smooths predictions. |
| `max_features` | `1.0` | Features considered per split. |
| `bootstrap` | `False` | Unlike RF, off by default. |

```python
from sklearn.ensemble import ExtraTreesRegressor

ET_REG_PARAMS = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": False,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 42,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None,
}

ET_REG_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [1.0, "sqrt", 0.5],
}

et_pipe = Pipeline([("model", ExtraTreesRegressor(**ET_REG_PARAMS))])
et_grid = tune("Extra Trees", et_pipe, ET_REG_GRID, task="reg")
```

---

### 2.12 NEW: `AdaBoostRegressor()`

Boosting that re-weights hard samples. By default the weak learner is a depth-3 decision tree.

| Param | Default | What it does |
|---|---|---|
| `estimator` | `None` | Base learner (default `DecisionTreeRegressor(max_depth=3)`). |
| `n_estimators` | `50` | Max number of boosting rounds. |
| `learning_rate` | `1.0` | Shrinks each learner's contribution. Trade-off with `n_estimators`. |
| `loss` | `'linear'` | `'linear'`, `'square'`, `'exponential'`. How sample weights are updated. |

```python
from sklearn.ensemble import AdaBoostRegressor

ADA_REG_PARAMS = {
    "estimator": None,
    "n_estimators": 50,
    "learning_rate": 1.0,
    "loss": "linear",
    "random_state": 42,
}

ADA_REG_GRID = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.1, 1.0],
    "model__loss": ["linear", "square", "exponential"],
    "model__estimator": [DecisionTreeRegressor(max_depth=d) for d in (1, 3, 5)],
}

ada_pipe = Pipeline([("model", AdaBoostRegressor(**ADA_REG_PARAMS))])
ada_grid = tune("AdaBoost", ada_pipe, ADA_REG_GRID, task="reg")
```

---

### 2.13 NEW: `HistGradientBoostingRegressor()`

sklearn's fast, LightGBM-style boosting (bins features into histograms). Handles missing values natively and is much faster than `GradientBoostingRegressor` on larger data. It uses **`max_iter`** (not `n_estimators`) for number of trees.

| Param | Default | What it does |
|---|---|---|
| `learning_rate` | `0.1` | Shrinkage per iteration. |
| `max_iter` | `100` | Number of boosting iterations (trees). |
| `max_leaf_nodes` | `31` | Max leaves per tree. Main complexity control. |
| `max_depth` | `None` | Optional depth cap. |
| `min_samples_leaf` | `20` | Min samples per leaf. |
| `l2_regularization` | `0.0` | L2 penalty on leaf values. |
| `early_stopping` | `'auto'` | On automatically when there are more than 10,000 samples. |
| `n_iter_no_change` | `10` | Patience for early stopping. |

```python
from sklearn.ensemble import HistGradientBoostingRegressor

HGB_REG_PARAMS = {
    "loss": "squared_error",
    "quantile": None,
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "max_bins": 255,
    "categorical_features": "from_dtype",
    "monotonic_cst": None,
    "interaction_cst": None,
    "warm_start": False,
    "early_stopping": "auto",
    "scoring": "loss",
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 1e-7,
    "verbose": 0,
    "random_state": 42,
}

HGB_REG_GRID = {
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_iter": [100, 300],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 50],
    "model__l2_regularization": [0.0, 1.0],
}

hgb_pipe = Pipeline([("model", HistGradientBoostingRegressor(**HGB_REG_PARAMS))])
hgb_grid = tune("HistGradientBoosting", hgb_pipe, HGB_REG_GRID, task="reg")
```

---

### 2.14 NEW: `XGBRegressor()` (`pip install xgboost`)

| Param | Effective default | What it does |
|---|---|---|
| `n_estimators` | `100` | Number of boosting rounds. |
| `learning_rate` | `0.3` | Shrinkage (eta). Lower + more trees generalizes better. |
| `max_depth` | `6` | Tree depth. |
| `min_child_weight` | `1` | Min sum of instance weight in a child. Higher = more conservative. |
| `subsample` | `1.0` | Row fraction per tree. |
| `colsample_bytree` | `1.0` | Column fraction per tree. |
| `gamma` | `0` | Min loss reduction to make a split. |
| `reg_alpha` / `reg_lambda` | `0` / `1` | L1 / L2 penalties on leaf weights. |
| `early_stopping_rounds` | `None` | Needs `eval_set` in `.fit()`, awkward inside GridSearchCV, so skip it there. |

```python
from xgboost import XGBRegressor

XGB_REG_PARAMS = {
    "n_estimators": None,            # effective 100
    "max_depth": None,               # effective 6
    "max_leaves": None,              # effective 0 (no limit)
    "max_bin": None,                 # effective 256
    "grow_policy": None,             # effective 'depthwise'
    "learning_rate": None,           # effective 0.3
    "verbosity": None,
    "objective": "reg:squarederror",
    "booster": None,                 # 'gbtree' | 'gblinear' | 'dart'
    "tree_method": None,             # 'auto' | 'exact' | 'approx' | 'hist'
    "n_jobs": None,
    "gamma": None,                   # effective 0
    "min_child_weight": None,        # effective 1
    "max_delta_step": None,          # effective 0
    "subsample": None,               # effective 1.0
    "sampling_method": None,
    "colsample_bytree": None,        # effective 1.0
    "colsample_bylevel": None,       # effective 1.0
    "colsample_bynode": None,        # effective 1.0
    "reg_alpha": None,               # effective 0
    "reg_lambda": None,              # effective 1
    "scale_pos_weight": None,        # classification only, ignored here
    "base_score": None,
    "random_state": 42,
    "missing": np.nan,
    "num_parallel_tree": None,       # effective 1
    "monotone_constraints": None,
    "interaction_constraints": None,
    "importance_type": None,
    "device": None,                  # 'cpu' | 'cuda'
    "validate_parameters": None,
    "enable_categorical": False,
    "feature_types": None,
    "max_cat_to_onehot": None,
    "max_cat_threshold": None,
    "multi_strategy": None,
    "eval_metric": None,
    "early_stopping_rounds": None,
    "callbacks": None,
}

XGB_REG_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1, 0.3],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

xgb_pipe = Pipeline([("model", XGBRegressor(**XGB_REG_PARAMS))])
xgb_grid = tune("XGBoost", xgb_pipe, XGB_REG_GRID, task="reg")
```

---

### 2.15 NEW: `LGBMRegressor()` (`pip install lightgbm`)

| Param | Default | What it does |
|---|---|---|
| `num_leaves` | `31` | Max leaves per tree. **Main complexity knob** (LightGBM grows leaf-wise). |
| `max_depth` | `-1` | `-1` = no limit. Cap it if overfitting. |
| `learning_rate` | `0.1` | Shrinkage. |
| `n_estimators` | `100` | Number of boosting rounds. |
| `min_child_samples` | `20` | Min samples in a leaf. Raise to fight overfitting on small data. |
| `subsample` + `subsample_freq` | `1.0` + `0` | Row bagging. **`subsample` only works if `subsample_freq > 0`.** |
| `colsample_bytree` | `1.0` | Column fraction per tree. |
| `reg_alpha` / `reg_lambda` | `0.0` / `0.0` | L1 / L2 penalties. |

```python
from lightgbm import LGBMRegressor

LGBM_REG_PARAMS = {
    "boosting_type": "gbdt",          # 'gbdt' | 'dart' | 'rf'
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample_for_bin": 200000,
    "objective": None,                # None -> 'regression'
    "min_split_gain": 0.0,
    "min_child_weight": 1e-3,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": None,
    "importance_type": "split",
    "verbose": -1,                    # extra kwarg: silences LightGBM's console spam
}

LGBM_REG_GRID = {
    "model__n_estimators": [100, 300],
    "model__num_leaves": [15, 31, 63],
    "model__learning_rate": [0.05, 0.1],
    "model__min_child_samples": [10, 20, 50],
    "model__subsample": [0.8, 1.0],
    "model__subsample_freq": [1],     # required for subsample < 1.0 to take effect
}

lgbm_pipe = Pipeline([("model", LGBMRegressor(**LGBM_REG_PARAMS))])
lgbm_grid = tune("LightGBM", lgbm_pipe, LGBM_REG_GRID, task="reg")
```

---

### 2.16 NEW: `MLPRegressor()` (small neural net)

Needs scaling. Watch for `ConvergenceWarning` and raise `max_iter`.

| Param | Default | What it does |
|---|---|---|
| `hidden_layer_sizes` | `(100,)` | Neurons per hidden layer, e.g. `(100, 50)` = two layers. |
| `activation` | `'relu'` | `'identity'`, `'logistic'`, `'tanh'`, `'relu'`. |
| `solver` | `'adam'` | `'lbfgs'` (small data), `'sgd'`, `'adam'`. |
| `alpha` | `0.0001` | L2 penalty. Raise it to fight overfitting. |
| `learning_rate_init` | `0.001` | Initial step size (adam/sgd). |
| `max_iter` | `200` | Epochs. Often needs raising. |
| `early_stopping` | `False` | Hold out `validation_fraction` and stop when it stops improving. |

```python
from sklearn.neural_network import MLPRegressor

MLP_REG_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",      # only used by solver='sgd'
    "learning_rate_init": 0.001,
    "power_t": 0.5,                   # only used by sgd + 'invscaling'
    "max_iter": 1000,                 # default is 200; raised to avoid ConvergenceWarning
    "shuffle": True,
    "random_state": 42,
    "tol": 1e-4,
    "verbose": False,
    "warm_start": False,
    "momentum": 0.9,                  # sgd only
    "nesterovs_momentum": True,       # sgd only
    "early_stopping": False,
    "validation_fraction": 0.1,
    "beta_1": 0.9,                    # adam only
    "beta_2": 0.999,                  # adam only
    "epsilon": 1e-8,                  # adam only
    "n_iter_no_change": 10,
    "max_fun": 15000,                 # lbfgs only
}

MLP_REG_GRID = {
    "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [0.001, 0.01],
}

mlp_pipe = Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(**MLP_REG_PARAMS))])
mlp_grid = tune("MLP", mlp_pipe, MLP_REG_GRID, task="reg")
```

---

### 2.17 NEW: `HuberRegressor()` (linear model robust to outliers)

| Param | Default | What it does |
|---|---|---|
| `epsilon` | `1.35` | Threshold between "inlier" (squared loss) and "outlier" (linear loss). Lower = more robust to outliers. Must be `>= 1`. |
| `alpha` | `0.0001` | L2 penalty. |
| `max_iter` | `100` | Often needs raising. |

```python
from sklearn.linear_model import HuberRegressor

HUBER_PARAMS = {
    "epsilon": 1.35,
    "max_iter": 1000,            # default is 100
    "alpha": 0.0001,
    "warm_start": False,
    "fit_intercept": True,
    "tol": 1e-5,
}

HUBER_GRID = {
    "model__epsilon": [1.1, 1.35, 1.5, 2.0],
    "model__alpha": [1e-4, 1e-3, 1e-2, 0.1],
}

huber_pipe = Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(**HUBER_PARAMS))])
huber_grid = tune("Huber", huber_pipe, HUBER_GRID, task="reg")
```

---

### 2.18 NEW: `BayesianRidge()`

Ridge-like, but it learns the regularization strength from the data. Fast, few things to tune.

| Param | Default | What it does |
|---|---|---|
| `max_iter` | `300` | Max iterations. |
| `alpha_1`, `alpha_2` | `1e-6` | Gamma prior on the noise precision. |
| `lambda_1`, `lambda_2` | `1e-6` | Gamma prior on the weight precision. |
| `compute_score` | `False` | Track log marginal likelihood at each iteration. |

```python
from sklearn.linear_model import BayesianRidge

BAYES_RIDGE_PARAMS = {
    "max_iter": 300,
    "tol": 1e-3,
    "alpha_1": 1e-6,
    "alpha_2": 1e-6,
    "lambda_1": 1e-6,
    "lambda_2": 1e-6,
    "alpha_init": None,
    "lambda_init": None,
    "compute_score": False,
    "fit_intercept": True,
    "copy_X": True,
    "verbose": False,
}

BAYES_RIDGE_GRID = {
    "model__alpha_1": [1e-6, 1e-4],
    "model__alpha_2": [1e-6, 1e-4],
    "model__lambda_1": [1e-6, 1e-4],
    "model__lambda_2": [1e-6, 1e-4],
}

br_pipe = Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge(**BAYES_RIDGE_PARAMS))])
br_grid = tune("BayesianRidge", br_pipe, BAYES_RIDGE_GRID, task="reg")
```

---

## 3. Classification models

`tune(..., task="clf")` defaults to `scoring="accuracy"`. Other scorers: `'f1'` (binary only), `'f1_weighted'` (multiclass), `'roc_auc'` (binary), `'balanced_accuracy'` (imbalanced data).

### 3.1 `LogisticRegression()`

| Param | Default | What it does |
|---|---|---|
| `penalty` | `'l2'` | `'l1'`, `'l2'`, `'elasticnet'`, `None`. Which solver supports which matters (see grid). |
| `C` | `1.0` | Inverse regularization strength (lower = stronger). |
| `solver` | `'lbfgs'` | `'lbfgs'`, `'liblinear'`, `'newton-cg'`, `'sag'`, `'saga'`. Only `saga` supports all penalties. |
| `l1_ratio` | `None` | Only for `penalty='elasticnet'`. |
| `class_weight` | `None` | `'balanced'` auto-adjusts for imbalanced classes. |
| `max_iter` | `100` | Often needs raising if you get a convergence warning. |
| `fit_intercept` | `True` | Same as linear models. |

```python
from sklearn.linear_model import LogisticRegression

LOGREG_PARAMS = {
    "penalty": "l2",           # deprecated in newest sklearn; delete this key if it errors
    "dual": False,
    "tol": 1e-4,
    "C": 1.0,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "class_weight": None,
    "random_state": 42,
    "solver": "lbfgs",
    "max_iter": 10000,         # default is 100
    "verbose": 0,
    "warm_start": False,
    "n_jobs": None,
    "l1_ratio": None,
    # "multi_class": "auto",   # removed in newer sklearn, add only on older versions
}

LOGREG_GRID = [
    {"model__penalty": ["l2"], "model__C": [10, 1.0, 0.1], "model__solver": ["lbfgs"]},
    {"model__penalty": ["l1"], "model__C": [10, 1.0, 0.1], "model__solver": ["saga"]},
    {"model__penalty": ["elasticnet"], "model__C": [10, 1.0, 0.1],
     "model__solver": ["saga"], "model__l1_ratio": [0.1, 0.5, 0.9]},
]

logreg_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**LOGREG_PARAMS))])
logreg_grid = tune("Logistic Regression", logreg_pipe, LOGREG_GRID, task="clf")
```

---

### 3.2 `DecisionTreeClassifier()`

Same params as the regressor, except `criterion` is `'gini'` (other: `'entropy'`, `'log_loss'`) and there's an added `class_weight`.

```python
from sklearn.tree import DecisionTreeClassifier

TREE_CLF_PARAMS = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": None,
    "random_state": 42,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
}

TREE_CLF_GRID = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 5, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

tree_clf_pipe = Pipeline([("model", DecisionTreeClassifier(**TREE_CLF_PARAMS))])
tree_clf_grid = tune("Decision Tree Classifier", tree_clf_pipe, TREE_CLF_GRID, task="clf")
```

---

### 3.3 `RandomForestClassifier()`

Same as the regressor, except `criterion='gini'`, `max_features='sqrt'`, plus `class_weight` (`'balanced'`, `'balanced_subsample'`).

```python
from sklearn.ensemble import RandomForestClassifier

RF_CLF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 42,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None,
}

RF_CLF_GRID = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", "log2", None],
    "model__class_weight": [None, "balanced"],
}

rf_clf_pipe = Pipeline([("model", RandomForestClassifier(**RF_CLF_PARAMS))])
rf_clf_grid = tune("Random Forest Classifier", rf_clf_pipe, RF_CLF_GRID, task="clf")
```

---

### 3.4 `GradientBoostingClassifier()`

Same as the regressor, except `loss='log_loss'` (or `'exponential'`) and no `alpha`.

```python
from sklearn.ensemble import GradientBoostingClassifier

GB_CLF_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "criterion": "friedman_mse",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 3,
    "min_impurity_decrease": 0.0,
    "init": None,
    "random_state": 42,
    "max_features": None,
    "verbose": 0,
    "max_leaf_nodes": None,
    "warm_start": False,
    "validation_fraction": 0.1,
    "n_iter_no_change": None,
    "tol": 1e-4,
    "ccp_alpha": 0.0,
}

GB_CLF_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__max_depth": [3, 5],
}

gb_clf_pipe = Pipeline([("model", GradientBoostingClassifier(**GB_CLF_PARAMS))])
gb_clf_grid = tune("Gradient Boosting Classifier", gb_clf_pipe, GB_CLF_GRID, task="clf")
```

---

### 3.5 `SVC()`

Same as SVR, plus `class_weight`, minus `epsilon` (regression-only). Adds `probability=False` (set `True` for `.predict_proba()`, but it slows training).

```python
from sklearn.svm import SVC

SVC_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "degree": 3,
    "gamma": "scale",
    "coef0": 0.0,
    "shrinking": True,
    "probability": False,
    "tol": 1e-3,
    "cache_size": 200,
    "class_weight": None,
    "verbose": False,
    "max_iter": -1,
    "decision_function_shape": "ovr",
    "break_ties": False,
    "random_state": 42,          # only used when probability=True
}

SVC_GRID = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"],
}

svc_pipe = Pipeline([("scaler", StandardScaler()), ("model", SVC(**SVC_PARAMS))])
svc_grid = tune("SVC", svc_pipe, SVC_GRID, task="clf")
```

---

### 3.4b `KNeighborsClassifier()`

Identical params to the regressor.

```python
from sklearn.neighbors import KNeighborsClassifier

KNN_CLF_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": None,
}

KNN_CLF_GRID = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
}

knn_clf_pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(**KNN_CLF_PARAMS))])
knn_clf_grid = tune("KNN Classifier", knn_clf_pipe, KNN_CLF_GRID, task="clf")
```

---

### 3.6 NEW: `GaussianNB()` (Naive Bayes)

Essentially one param. A very fast baseline. Assumes features are roughly Gaussian and independent.

| Param | Default | What it does |
|---|---|---|
| `priors` | `None` | Class prior probabilities (default = learned from data). |
| `var_smoothing` | `1e-9` | Portion of the largest feature variance added to all variances for stability. Raise it if the model is overconfident. |

```python
from sklearn.naive_bayes import GaussianNB

GNB_PARAMS = {
    "priors": None,
    "var_smoothing": 1e-9,
}

GNB_GRID = {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

gnb_pipe = Pipeline([("scaler", StandardScaler()), ("model", GaussianNB(**GNB_PARAMS))])
gnb_grid = tune("GaussianNB", gnb_pipe, GNB_GRID, task="clf")
```

---

### 3.7 NEW: `ExtraTreesClassifier()`

```python
from sklearn.ensemble import ExtraTreesClassifier

ET_CLF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": False,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 42,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None,
}

ET_CLF_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5],
    "model__class_weight": [None, "balanced"],
}

et_clf_pipe = Pipeline([("model", ExtraTreesClassifier(**ET_CLF_PARAMS))])
et_clf_grid = tune("Extra Trees Classifier", et_clf_pipe, ET_CLF_GRID, task="clf")
```

---

### 3.8 NEW: `AdaBoostClassifier()`

```python
from sklearn.ensemble import AdaBoostClassifier

ADA_CLF_PARAMS = {
    "estimator": None,           # default: DecisionTreeClassifier(max_depth=1) (a "stump")
    "n_estimators": 50,
    "learning_rate": 1.0,
    "random_state": 42,
    # "algorithm": "SAMME",      # deprecated/removed in newer sklearn, add only if your version needs it
}

ADA_CLF_GRID = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.1, 1.0],
    "model__estimator": [DecisionTreeClassifier(max_depth=d) for d in (1, 2, 3)],
}

ada_clf_pipe = Pipeline([("model", AdaBoostClassifier(**ADA_CLF_PARAMS))])
ada_clf_grid = tune("AdaBoost Classifier", ada_clf_pipe, ADA_CLF_GRID, task="clf")
```

---

### 3.9 NEW: `HistGradientBoostingClassifier()`

Same as the regressor, except `loss='log_loss'` and added `class_weight`.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

HGB_CLF_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "max_bins": 255,
    "categorical_features": "from_dtype",
    "monotonic_cst": None,
    "interaction_cst": None,
    "warm_start": False,
    "early_stopping": "auto",
    "scoring": "loss",
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 1e-7,
    "verbose": 0,
    "random_state": 42,
    "class_weight": None,
}

HGB_CLF_GRID = {
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_iter": [100, 300],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 50],
    "model__l2_regularization": [0.0, 1.0],
    "model__class_weight": [None, "balanced"],
}

hgb_clf_pipe = Pipeline([("model", HistGradientBoostingClassifier(**HGB_CLF_PARAMS))])
hgb_clf_grid = tune("HistGradientBoosting Classifier", hgb_clf_pipe, HGB_CLF_GRID, task="clf")
```

---

### 3.10 NEW: `XGBClassifier()` (`pip install xgboost`)

Same params as `XGBRegressor`, but `objective` defaults to `binary:logistic` (or multi-class, chosen automatically) and `scale_pos_weight` becomes useful for imbalanced binary data (set to `n_negative / n_positive`).

> Labels must be integers `0..n_classes-1`. Use `LabelEncoder` first if your target is strings.

```python
from xgboost import XGBClassifier

XGB_CLF_PARAMS = {
    "n_estimators": None,            # effective 100
    "max_depth": None,               # effective 6
    "max_leaves": None,
    "max_bin": None,
    "grow_policy": None,
    "learning_rate": None,           # effective 0.3
    "verbosity": None,
    "objective": None,               # auto: 'binary:logistic' / 'multi:softprob'
    "booster": None,
    "tree_method": None,
    "n_jobs": None,
    "gamma": None,
    "min_child_weight": None,
    "max_delta_step": None,
    "subsample": None,
    "sampling_method": None,
    "colsample_bytree": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "reg_alpha": None,
    "reg_lambda": None,
    "scale_pos_weight": None,        # binary imbalance: n_neg / n_pos
    "base_score": None,
    "random_state": 42,
    "missing": np.nan,
    "num_parallel_tree": None,
    "monotone_constraints": None,
    "interaction_constraints": None,
    "importance_type": None,
    "device": None,
    "validate_parameters": None,
    "enable_categorical": False,
    "feature_types": None,
    "max_cat_to_onehot": None,
    "max_cat_threshold": None,
    "multi_strategy": None,
    "eval_metric": None,
    "early_stopping_rounds": None,
    "callbacks": None,
}

XGB_CLF_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1, 0.3],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

xgb_clf_pipe = Pipeline([("model", XGBClassifier(**XGB_CLF_PARAMS))])
xgb_clf_grid = tune("XGBoost Classifier", xgb_clf_pipe, XGB_CLF_GRID, task="clf")
```

---

### 3.11 NEW: `LGBMClassifier()` (`pip install lightgbm`)

Same as `LGBMRegressor`, plus `class_weight` (`'balanced'`).

```python
from lightgbm import LGBMClassifier

LGBM_CLF_PARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample_for_bin": 200000,
    "objective": None,                # None -> 'binary' or 'multiclass' automatically
    "class_weight": None,
    "min_split_gain": 0.0,
    "min_child_weight": 1e-3,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": None,
    "importance_type": "split",
    "verbose": -1,
}

LGBM_CLF_GRID = {
    "model__n_estimators": [100, 300],
    "model__num_leaves": [15, 31, 63],
    "model__learning_rate": [0.05, 0.1],
    "model__min_child_samples": [10, 20, 50],
    "model__class_weight": [None, "balanced"],
}

lgbm_clf_pipe = Pipeline([("model", LGBMClassifier(**LGBM_CLF_PARAMS))])
lgbm_clf_grid = tune("LightGBM Classifier", lgbm_clf_pipe, LGBM_CLF_GRID, task="clf")
```

---

### 3.12 NEW: `MLPClassifier()`

Same params as `MLPRegressor` (see 2.16). Needs scaling.

```python
from sklearn.neural_network import MLPClassifier

MLP_CLF_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "power_t": 0.5,
    "max_iter": 1000,                 # default is 200
    "shuffle": True,
    "random_state": 42,
    "tol": 1e-4,
    "verbose": False,
    "warm_start": False,
    "momentum": 0.9,
    "nesterovs_momentum": True,
    "early_stopping": False,
    "validation_fraction": 0.1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-8,
    "n_iter_no_change": 10,
    "max_fun": 15000,
}

MLP_CLF_GRID = {
    "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [0.001, 0.01],
}

mlp_clf_pipe = Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(**MLP_CLF_PARAMS))])
mlp_clf_grid = tune("MLP Classifier", mlp_clf_pipe, MLP_CLF_GRID, task="clf")
```

---

## 4. Cheat sheet

### Actually worth tuning (per model)

| Model family | Tune these |
|---|---|
| Ridge / Lasso | `alpha` |
| ElasticNet | `alpha`, `l1_ratio` |
| Logistic Regression | `C`, `penalty`, `solver`, `l1_ratio`, `class_weight` |
| Huber | `epsilon`, `alpha` |
| Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split` |
| Random Forest / Extra Trees | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| GradientBoosting / AdaBoost | `n_estimators`, `learning_rate`, `max_depth` |
| HistGradientBoosting | `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` |
| LightGBM | `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples` |
| SVM (SVR / SVC) | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights` |
| MLP | `hidden_layer_sizes`, `alpha`, `learning_rate_init`, `activation` |
| GaussianNB | `var_smoothing` |
| Everything else | Leave at default unless you have a specific reason |

### Which models need `StandardScaler`

| Needs it (distance / gradient / penalty based) | Doesn't need it (split based) |
|---|---|
| Ridge, Lasso, ElasticNet, Huber, BayesianRidge, LogReg, SVR/SVC, KNN, MLP, GaussianNB (helps) | Decision Tree, Random Forest, Extra Trees, GradientBoosting, HistGB, AdaBoost, XGBoost, LightGBM |

### Notes

- Every block follows the same shape: `Pipeline` -> param_grid (keys as `'stepname__param'`) -> `GridSearchCV` -> `.fit(X_train, y_train)` -> `best_params_`.
- Regression scoring: `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'`, `'neg_root_mean_squared_error'`.
- Classification scoring: `'accuracy'`, `'f1'`, `'f1_weighted'`, `'roc_auc'`, `'balanced_accuracy'`.
- `neg_*` scorers are negated so that "higher is better". A `best_score_` of `-4.2` means an error of `4.2`.
- Grid size = product of the list lengths x `cv`. A grid of `3 x 3 x 3 x 2` with `cv=5` is 54 x 5 = 270 fits. Swap to `RandomizedSearchCV(..., n_iter=30)` when this gets large:

  ```python
  rs = tune("XGBoost (random)", xgb_pipe, XGB_REG_GRID, task="reg",
            search=RandomizedSearchCV, n_iter=30, random_state=42)
  ```

- To go from "defaults dict" to "tuned dict", copy the winner over:

  ```python
  best = {k.replace("model__", ""): v for k, v in xgb_grid.best_params_.items()}
  final_params = {**XGB_REG_PARAMS, **best}
  final_model = XGBRegressor(**final_params)
  ```

- To compare many models in one shot, loop over `(name, pipe, grid)` triples and call `tune(...)` on each, then compare test metrics.