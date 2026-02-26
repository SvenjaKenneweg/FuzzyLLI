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
                        "adverbial": adverbial,
                        "frequency": frequency,
                        "richness": richness,
                        "importance": importance,
                        "minutes_ago": minutes_ago
                    })
                else:
                    records.append({
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