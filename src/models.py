"""Training-only hyperparameter selection with a purged time-series split."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train(x, y, model_type="ridge"):
    if len(x) < 40:
        raise ValueError("At least 40 training samples required")
    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge())
        grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
        grid = {"min_samples_leaf": [5, 15]}
    else:
        raise ValueError("Supported models are ridge and rf")
    search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=3, gap=1),
                          scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, -float(search.best_score_)
