import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from pathlib import Path
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
    log_loss,
    brier_score_loss,
)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

FEATURES = [
    "elo_diff",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "h2h_matches", "h2h_home_avg_goals", "h2h_away_avg_goals", "h2h_home_win_rate",
    "home_wc_participations",
    "away_wc_participations",
    "is_home_home", "is_neutral",
    "competition_friendly",
    "competition_qualifier",
]

TARGET_HOME = "home_score"
TARGET_AWAY = "away_score"

RUN_TUNING = True
N_ITER_TUNING = 25
N_SPLITS = 5
MAX_GOALS = 8

DEFAULT_PARAMS = {
    "objective": "count:poisson",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

PARAM_DISTRIBUTIONS = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "n_estimators": [200, 300, 500, 700],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8],
    "reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    train = df[df[TARGET_HOME].notna()].copy()
    train["date"] = pd.to_datetime(train["date"])
    train = train.sort_values("date").reset_index(drop=True)

    id_cols = train[["date", "home_team", "away_team", TARGET_HOME, TARGET_AWAY]].copy()

    for col in [
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "away_avg_goals_scored", "away_avg_goals_conceded",
    ]:
        train[col] = train[col].fillna(train[col].mean())

    train["h2h_home_avg_goals"] = train["h2h_home_avg_goals"].fillna(train["home_avg_goals_scored"])
    train["h2h_away_avg_goals"] = train["h2h_away_avg_goals"].fillna(train["away_avg_goals_scored"])
    train["h2h_home_win_rate"] = train["h2h_home_win_rate"].fillna(0.33)

    X = train[FEATURES]
    y_home = train[TARGET_HOME].astype(int)
    y_away = train[TARGET_AWAY].astype(int)

    return X, y_home, y_away, id_cols


def result_label(home_goals: pd.Series, away_goals: pd.Series) -> np.ndarray:
    return np.select(
        [
            home_goals.values > away_goals.values,
            home_goals.values == away_goals.values,
            home_goals.values < away_goals.values,
        ],
        [0, 1, 2],
    )


def scoreline_probabilities(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    max_goals: int = 8,
) -> np.ndarray:
    probs = np.zeros((len(lambda_home), 3))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

            if i > j:
                probs[:, 0] += p
            elif i == j:
                probs[:, 1] += p
            else:
                probs[:, 2] += p

    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(row_sums, 1e-12, None)

    return probs


def evaluate_outcome_metrics(
    y_home_true: pd.Series,
    y_away_true: pd.Series,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    label: str,
) -> dict:
    probs = scoreline_probabilities(lambda_home, lambda_away, max_goals=MAX_GOALS)
    y_result = result_label(y_home_true, y_away_true)

    pred_result = np.argmax(probs, axis=1)

    brier_home = brier_score_loss((y_result == 0).astype(int), probs[:, 0])
    brier_draw = brier_score_loss((y_result == 1).astype(int), probs[:, 1])
    brier_away = brier_score_loss((y_result == 2).astype(int), probs[:, 2])

    return {
        f"logloss_1x2_{label}": log_loss(y_result, probs, labels=[0, 1, 2]),
        f"accuracy_1x2_{label}": np.mean(pred_result == y_result),
        f"brier_home_win_{label}": brier_home,
        f"brier_draw_{label}": brier_draw,
        f"brier_away_win_{label}": brier_away,
        f"brier_mean_1x2_{label}": np.mean([brier_home, brier_draw, brier_away]),
    }


def evaluate_goal_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 5,
    label: str = "home",
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    maes, rmses, poisson_devs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = np.clip(model.predict(X_val), 1e-6, None)

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        poisson_dev = mean_poisson_deviance(y_val, preds)

        maes.append(mae)
        rmses.append(rmse)
        poisson_devs.append(poisson_dev)

        print(
            f"  Fold {fold + 1} [{label}] "
            f"MAE={mae:.4f} | RMSE={rmse:.4f} | PoissonDev={poisson_dev:.4f}"
        )

    return {
        f"mae_{label}": float(np.mean(maes)),
        f"rmse_{label}": float(np.mean(rmses)),
        f"poisson_deviance_{label}": float(np.mean(poisson_devs)),
    }


def evaluate_double_poisson(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    params_home: dict,
    params_away: dict,
    n_splits: int = 5,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        yh_tr, yh_val = y_home.iloc[train_idx], y_home.iloc[val_idx]
        ya_tr, ya_val = y_away.iloc[train_idx], y_away.iloc[val_idx]

        model_home = xgb.XGBRegressor(**params_home)
        model_away = xgb.XGBRegressor(**params_away)

        model_home.fit(X_tr, yh_tr, eval_set=[(X_val, yh_val)], verbose=False)
        model_away.fit(X_tr, ya_tr, eval_set=[(X_val, ya_val)], verbose=False)

        pred_home = np.clip(model_home.predict(X_val), 1e-6, None)
        pred_away = np.clip(model_away.predict(X_val), 1e-6, None)

        metrics = {
            "mae_home": mean_absolute_error(yh_val, pred_home),
            "rmse_home": np.sqrt(mean_squared_error(yh_val, pred_home)),
            "poisson_deviance_home": mean_poisson_deviance(yh_val, pred_home),

            "mae_away": mean_absolute_error(ya_val, pred_away),
            "rmse_away": np.sqrt(mean_squared_error(ya_val, pred_away)),
            "poisson_deviance_away": mean_poisson_deviance(ya_val, pred_away),
        }

        metrics.update(
            evaluate_outcome_metrics(
                y_home_true=yh_val,
                y_away_true=ya_val,
                lambda_home=pred_home,
                lambda_away=pred_away,
                label="val",
            )
        )

        fold_metrics.append(metrics)

        print(
            f"  Fold {fold + 1} [1X2] "
            f"LogLoss={metrics['logloss_1x2_val']:.4f} | "
            f"Brier={metrics['brier_mean_1x2_val']:.4f} | "
            f"Acc={metrics['accuracy_1x2_val']:.4f}"
        )

    return {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0].keys()
    }


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    base_params: dict,
    param_distributions: dict,
    n_iter: int = 25,
    n_splits: int = 5,
    label: str = "home",
) -> dict:
    print(f"\nTuning temporal — {label}")

    sampler = list(
        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=42,
        )
    )

    best_score = np.inf
    best_params = None

    for i, sampled_params in enumerate(sampler, start=1):
        params = {**base_params, **sampled_params}
        metrics = evaluate_goal_model(
            X=X,
            y=y,
            params=params,
            n_splits=n_splits,
            label=f"{label}_tune",
        )

        score = metrics[f"poisson_deviance_{label}_tune"]

        print(f"  Trial {i}/{n_iter} [{label}] PoissonDev={score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params

    print(f"\nMelhores params — {label}:")
    print(best_params)
    print(f"Best PoissonDev — {label}: {best_score:.4f}")

    return best_params


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def predict_scoreline(
    model_home: xgb.XGBRegressor,
    model_away: xgb.XGBRegressor,
    X: pd.DataFrame,
    max_goals: int = 8,
) -> pd.DataFrame:
    lambda_home = np.clip(model_home.predict(X), 0.1, None)
    lambda_away = np.clip(model_away.predict(X), 0.1, None)

    probs = scoreline_probabilities(lambda_home, lambda_away, max_goals=max_goals)

    results = X.copy()
    results["lambda_home"] = lambda_home
    results["lambda_away"] = lambda_away
    results["prob_home_win"] = probs[:, 0]
    results["prob_draw"] = probs[:, 1]
    results["prob_away_win"] = probs[:, 2]

    return results


def save_feature_importance(model: xgb.XGBRegressor, name: str, output_dir: str = "models") -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax)
    ax.set_title(f"Feature importance - {name.upper()} model")

    path = f"{output_dir}/feature_importance_{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return path


def save_shap_summary(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    name: str,
    output_dir: str = "models",
    sample_size: int = 1000,
) -> str:
    print(f"\nA gerar SHAP values — {name.upper()} model...")

    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)

    path = f"{output_dir}/shap_summary_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    return path


if __name__ == "__main__":
    df = pd.read_csv("data/processed/training_df.csv")

    X, y_home, y_away, id_cols = prepare_data(df)

    print(f"Treino: {X.shape[0]} jogos | Features: {X.shape[1]}")

    artifacts_dir = Path("artifacts") / RUN_TS
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    MLFLOW_DB = Path("mlflow.db").absolute()
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

    mlflow.set_experiment("wc2026")

    with mlflow.start_run(run_name=f"xgb_poisson_{RUN_TS}"):

        mlflow.log_param("features", FEATURES)
        mlflow.log_param("n_train", len(X))
        mlflow.log_param("run_tuning", RUN_TUNING)
        mlflow.log_param("n_iter_tuning", N_ITER_TUNING)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("max_goals", MAX_GOALS)

        if RUN_TUNING:
            best_params_home = tune_model(
                X=X, y=y_home,
                base_params=DEFAULT_PARAMS,
                param_distributions=PARAM_DISTRIBUTIONS,
                n_iter=N_ITER_TUNING,
                n_splits=N_SPLITS,
                label="home",
            )
            best_params_away = tune_model(
                X=X, y=y_away,
                base_params=DEFAULT_PARAMS,
                param_distributions=PARAM_DISTRIBUTIONS,
                n_iter=N_ITER_TUNING,
                n_splits=N_SPLITS,
                label="away",
            )
        else:
            best_params_home = DEFAULT_PARAMS
            best_params_away = DEFAULT_PARAMS

        mlflow.log_params({f"home_{k}": v for k, v in best_params_home.items()})
        mlflow.log_params({f"away_{k}": v for k, v in best_params_away.items()})

        print("\nValidação temporal conjunta — Double Poisson:")
        metrics = evaluate_double_poisson(
            X=X, y_home=y_home, y_away=y_away,
            params_home=best_params_home,
            params_away=best_params_away,
            n_splits=N_SPLITS,
        )
        mlflow.log_metrics(metrics)

        print("\nMétricas médias:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nTreino final...")
        model_home = train_model(X, y_home, best_params_home)
        model_away = train_model(X, y_away, best_params_away)

        mlflow.xgboost.log_model(model_home, "model_home")
        mlflow.xgboost.log_model(model_away, "model_away")

        model_home_path = Path("models") / f"model_home_{RUN_TS}.json"
        model_away_path = Path("models") / f"model_away_{RUN_TS}.json"
        model_home.save_model(model_home_path)
        model_away.save_model(model_away_path)
        mlflow.log_artifact(str(model_home_path))
        mlflow.log_artifact(str(model_away_path))

        fi_home = save_feature_importance(model_home, "home", artifacts_dir)
        fi_away = save_feature_importance(model_away, "away", artifacts_dir)
        mlflow.log_artifact(str(fi_home))
        mlflow.log_artifact(str(fi_away))

        print("\nA gerar previsões de auditoria...")
        audit = predict_scoreline(model_home, model_away, X, max_goals=MAX_GOALS)
        audit = pd.concat([id_cols.reset_index(drop=True), audit], axis=1)
        audit_path = artifacts_dir / f"audit_predictions_{RUN_TS}.csv"
        audit.to_csv(audit_path, index=False)
        mlflow.log_artifact(str(audit_path))

        print("\nA calcular SHAP...")
        shap_home = save_shap_summary(model_home, X, "home", artifacts_dir)
        shap_away = save_shap_summary(model_away, X, "away", artifacts_dir)
        mlflow.log_artifact(str(shap_home))
        mlflow.log_artifact(str(shap_away))

        print(f"\nArtifacts em: {artifacts_dir}")
        print("Concluído.")
