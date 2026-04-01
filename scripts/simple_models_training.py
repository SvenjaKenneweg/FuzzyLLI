import json
import statistics
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

from . import config

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_data(events, events_nl, classification = True) -> pd.DataFrame:
    records = []
    for event, event_nl in zip(events, events_nl):
        with open(f"{config.DATA_DIR}/{event}/cleanedData_minutes.json", "r") as f:
            adverbial_data = json.load(f)
        with open(f"{config.DATA_DIR}/event_properties.json", "r", encoding="utf-8") as fh:
            properties_list = json.load(fh)

        event_to_search = event_nl.replace("Tom", "A friend").replace("You", "I")
        frequency = properties_list[event_to_search]["Frequency"]
        richness = properties_list[event_to_search]["Richness"]
        importance = properties_list[event_to_search]["Importance"]

        for adverbial, time_dict in adverbial_data.items():
            for minutes_str, votes in time_dict.items():
                vote_values = list(map(float, votes))

                vote_median = statistics.median(vote_values)
                minutes_ago = int(minutes_str)
                if vote_median > 0.49 and classification:
                    records.append({
                        "event": event,
                        "adverbial": adverbial,
                        "frequency": frequency,
                        "richness": richness,
                        "importance": importance,
                        "minutes_ago": minutes_ago
                    })
                else:
                    records.append({
                        "event": event,
                        "adverbial": adverbial,
                        "frequency": frequency,
                        "richness": richness,
                        "importance": importance,
                        "minutes_ago": minutes_ago,
                        "vote": vote_median
                    })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_classifier(events, events_nl, *args):
    df = load_data(events, events_nl, classification=False)

    df['log_minutes_ago'] = np.log1p(df['minutes_ago'])
    X = df[["frequency", "richness", "importance", "log_minutes_ago"]]
    y = df["adverbial"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, config.RESULTS_SIMPLE_FILE_PATH / 'label_encoder.pkl')

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X, y_encoded)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, config.RESULTS_SIMPLE_FILE_PATH / 'gradient_boosting_classifier.pkl')

    return model


def fit_regression(events, events_nl, *args):
    df = load_data(events, events_nl, classification=False)

    ohe = OneHotEncoder(sparse_output=False)
    adverbial_encoded = ohe.fit_transform(df[['adverbial']])
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    adverbial_df = pd.DataFrame(adverbial_encoded, columns=adverbial_cols, index=df.index)

    features_df = pd.concat([df[["frequency", "richness", "importance", "minutes_ago"]], adverbial_df], axis=1)
    X = features_df
    y = df["vote"]

    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X, y)
    joblib.dump(model, config.RESULTS_SIMPLE_FILE_PATH / f"XGBRegressor.pkl")
    joblib.dump(ohe, config.RESULTS_SIMPLE_FILE_PATH / "onehotencoder.pkl")


def fit_non_factorized_gauss(events, events_nl, *args):
    df = load_data(events, events_nl, classification=False)

    fit_rows = []
    fitted_groups = []

    for (event, adverbial), g in df.groupby(["event", "adverbial"], sort=False):
        g = g.sort_values("minutes_ago").copy()

        x = g["minutes_ago"].to_numpy(dtype=float)
        y = g["vote"].to_numpy(dtype=float)

        # Sensible initialization:
        # mu starts near the weighted center of mass of the votes,
        # sigma starts from the weighted spread.
        if np.sum(y) > 0:
            w = np.clip(y, 1e-8, None)
            mu0 = float(np.average(x, weights=w))
            sigma0 = float(np.sqrt(np.average((x - mu0) ** 2, weights=w)))
            sigma0 = max(sigma0, 1.0)
        else:
            mu0 = float(np.median(x))
            sigma0 = max(float(np.std(x)), 1.0)

        fit_success = True

        try:
            # Fit only mu and sigma; peak is fixed to 1
            popt, _ = curve_fit(
                config.normalized_gaussian,
                x,
                y,
                p0=(mu0, sigma0),
                bounds=([x.min(), 1e-6], [x.max(), np.inf]),
                maxfev=50000,
            )
            mu_hat, sigma_hat = map(float, popt)
        except Exception:
            fit_success = False
            mu_hat, sigma_hat = mu0, sigma0

        y_hat = config.normalized_gaussian(x, mu_hat, sigma_hat)
        y_hat = np.clip(y_hat, 0.0, 1.0)

        g["fit_vote"] = y_hat
        g["fit_residual"] = g["vote"] - g["fit_vote"]
        fitted_groups.append(g)

        errors = y - y_hat
        mae = float(np.mean(np.abs(errors)))
        mdse = float(np.median(errors ** 2))

        fit_rows.append(
            {
                "event": event,
                "adverbial": adverbial,
                "mu_minutes_ago": mu_hat,
                "sigma_minutes": sigma_hat,
                "n_points": len(g),
                "mae": mae,
                "mdse": mdse,
                "fit_success": fit_success,
            }
        )

    params_df = pd.DataFrame(fit_rows)
    fitted_df = pd.concat(fitted_groups, ignore_index=True)

    avg_metrics = {
        "mean_mae_over_fits": float(params_df["mae"].mean()) if not params_df.empty else np.nan,
        "mean_mdse_over_fits": float(params_df["mdse"].mean()) if not params_df.empty else np.nan,
        "n_fitted_functions": int(len(params_df)),
    }

    print(avg_metrics)

    return avg_metrics