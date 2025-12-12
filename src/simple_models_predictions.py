import numpy as np
import pandas as pd
import joblib
import json
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor

import src.config as config



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_event_properties(event_properties):
    """
    Accept DataFrame, dict, or list of dicts and return a DataFrame with
    Frequency/Richness/Importance columns. Returns None if input is None.
    """
    if event_properties is None:
        return None
    if isinstance(event_properties, pd.DataFrame):
        return event_properties
    if isinstance(event_properties, dict):
        event_properties = [event_properties]
    if isinstance(event_properties, list):
        normalized = []
        for item in event_properties:
            if not isinstance(item, dict):
                raise ValueError("Each event_properties item must be a dict.")
            normalized.append({
                "Frequency": item.get("Frequency", item.get("frequency")),
                "Richness": item.get("Richness", item.get("richness")),
                "Importance": item.get("Importance", item.get("importance")),
            })
        return pd.DataFrame(normalized)
    raise ValueError("event_properties must be a DataFrame, dict, list of dicts, or None.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_adverbial_classifier(event_nl, minutes_ago, event_properties=None, *args):
    event_properties_df = _normalize_event_properties(event_properties)

    if event_properties_df is not None and not event_properties_df.empty:
        values_dict = event_properties_df.iloc[0].to_dict()
        frequency = values_dict.get("Frequency", values_dict.get("frequency"))
        richness = values_dict.get("Richness", values_dict.get("richness"))
        importance = values_dict.get("Importance", values_dict.get("importance"))
    else:
        # Get the previously saved event properties
        event_to_find = event_nl.replace("Tom", "A friend").replace("You", "I")
        file_path = f"{config.DATA_DIR}/event_properties.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            properties = json.load(fh)
        if event_to_find not in properties:
            file_path = f"{config.DATASET_TEST_PATH}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                properties = json.load(fh)
        frequency = properties[event_to_find]["Frequency"]
        richness = properties[event_to_find]["Richness"]
        importance = properties[event_to_find]["Importance"]

    x_input = pd.DataFrame(
        [[frequency, richness, importance, np.log1p(minutes_ago)]],
        columns=["frequency", "richness", "importance", "log_minutes_ago"])

    model = joblib.load(config.RESULTS_SIMPLE_FILE_PATH / config.GRADIENT_BOOSTING_FILE)
    le = joblib.load(config.RESULTS_SIMPLE_FILE_PATH / config.LABEL_ENCODER_FILE)

    # Get class probabilities
    probas = model.predict_proba(x_input)[0]  # shape: (num_classes,)
    class_labels = le.inverse_transform(np.arange(len(probas)))

    return dict(zip(class_labels, probas))


def predict_adverbial_regression(event_nl, minutes_ago, event_properties=None, *args):
    event_properties_df = _normalize_event_properties(event_properties)

    if event_properties_df is not None and not event_properties_df.empty:
        values_dict = event_properties_df.iloc[0].to_dict()
        frequency = values_dict.get("Frequency", values_dict.get("frequency"))
        richness = values_dict.get("Richness", values_dict.get("richness"))
        importance = values_dict.get("Importance", values_dict.get("importance"))
    else:
        # Get the previously saved event properties
        event_to_find = event_nl.replace("Tom", "A friend").replace("You", "I")
        file_path = f"{config.DATA_DIR}/event_properties.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            properties = json.load(fh)
        if event_to_find not in properties:
            file_path = f"{config.DATASET_TEST_PATH}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                properties = json.load(fh)
        frequency = properties[event_to_find]["Frequency"]
        richness = properties[event_to_find]["Richness"]
        importance = properties[event_to_find]["Importance"]

    rf_model = joblib.load(config.RESULTS_SIMPLE_FILE_PATH / config.XGB_REGRESSOR_FILE)
    ohe = joblib.load(config.RESULTS_SIMPLE_FILE_PATH / config.ONE_HOT_ENCODER_FILE)

    results = {}
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    numeric_cols = ["frequency", "richness", "importance", "minutes_ago"]

    for adv in config.VAGUE_ADVERBIALS:
        # One-hot encode the adverbial
        adverbial_encoded = ohe.transform(pd.DataFrame([[adv]], columns=['adverbial']))
        # Build DataFrame with numeric columns first, then onehot columns, exactly like during training
        input_df = pd.DataFrame(
            data=np.hstack([
                [frequency, richness, importance, minutes_ago],
                adverbial_encoded.flatten()
            ]).reshape(1, -1),
            columns=numeric_cols + list(adverbial_cols)
        )

        # Predict votes
        rf_pred = rf_model.predict(input_df)[0]
        results[adv] = float(rf_pred)

    return results
