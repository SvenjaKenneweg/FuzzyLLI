import json
import statistics
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/with_event_properties")
RESULTS_FILE_PATH = Path("results/fits/simple_models/")


def load_data(events, classification = True) -> pd.DataFrame:
    records = []
    for event in events:
        try:
            with open(f"{DATA_DIR}/{event}/cleanedData_minutes.json", "r") as f:
                adverbial_data = json.load(f)
            with open(f"{DATA_DIR}/{event}/event_properties.json", "r") as f:
                properties_list = json.load(f)

            if not isinstance(properties_list, list):
                continue

            frequency = int(statistics.median([int(p["Frequency"]) for p in properties_list if "Frequency" in p]))
            duration = int(statistics.median([int(p["Duration"]) for p in properties_list if "Duration" in p]))

            for adverbial, time_dict in adverbial_data.items():
                for minutes_str, votes in time_dict.items():
                    try:
                        vote_values = list(map(float, votes))
                        if not vote_values:
                            continue
                        vote_median = statistics.median(vote_values)
                        minutes_ago = int(minutes_str)
                        if vote_median > 0.49 and classification:
                            records.append({
                                "adverbial": adverbial,
                                "frequency": frequency,
                                "duration": duration,
                                "minutes_ago": minutes_ago
                            })
                        else:
                            records.append({
                                "adverbial": adverbial,
                                "frequency": frequency,
                                "duration": duration,
                                "minutes_ago": minutes_ago,
                                "vote": vote_median
                            })
                    except:
                        continue
        except:
            continue
    return pd.DataFrame(records)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_classifier(events) -> GradientBoostingClassifier:
    df = load_data(events, classification=False)

    df['log_minutes_ago'] = np.log1p(df['minutes_ago'])
    X = df[["frequency", "duration", "log_minutes_ago"]]
    y = df["adverbial"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, RESULTS_FILE_PATH / 'label_encoder.pkl')

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X, y_encoded)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, RESULTS_FILE_PATH / 'gradient_boosting_classifier.pkl')

    return model


def fit_regression(events) -> None:
    df = load_data(events, classification=False)

    ohe = OneHotEncoder(sparse_output=False)
    adverbial_encoded = ohe.fit_transform(df[['adverbial']])
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    adverbial_df = pd.DataFrame(adverbial_encoded, columns=adverbial_cols, index=df.index)

    features_df = pd.concat([df[["frequency", "duration", "minutes_ago"]], adverbial_df], axis=1)
    X = features_df
    y = df["vote"]

    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBRegressor": XGBRegressor(objective="reg:squarederror", random_state=42)
    }

    for name, model in models.items():
        model.fit(X, y)
        joblib.dump(model, RESULTS_FILE_PATH / f"{name}.pkl")
    joblib.dump(ohe, RESULTS_FILE_PATH / "onehotencoder.pkl")