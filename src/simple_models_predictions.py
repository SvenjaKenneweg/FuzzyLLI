import numpy as np
import pandas as pd
import joblib
import json
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor

from src.config import (VAGUE_ADVERBIALS,
                        RESULTS_SIMPLE_FILE_PATH,
                        GRADIENT_BOOSTING_FILE,
                        DATA_DIR,
                        LABEL_ENCODER_FILE,
                        DATA_EVALUATION_SURVEY_PATH,
                        XGB_REGRESSOR_FILE,
                        ONE_HOT_ENCODER_FILE, EVALUATION_FILE_PATH)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_adverbial_classifier(event_nl, minutes_ago, event_properties=None):
    if event_properties is not None and not event_properties.empty:
        values_dict = event_properties.iloc[0].to_dict()
        frequency = values_dict["Frequency"]
        duration = values_dict["Duration"]
        importance = values_dict["Importance"]
    else:
        # Get the previously saved event properties
        event_to_find = event_nl.replace("Tom", "A friend").replace("You", "I")
        file_path = f"{DATA_DIR}/event_properties.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            properties = json.load(fh)
        if event_to_find not in properties:
            file_path = f"{DATA_EVALUATION_SURVEY_PATH}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                properties = json.load(fh)
        frequency = properties[event_to_find]["Frequency"]
        duration = properties[event_to_find]["Duration"]
        importance = properties[event_to_find]["Importance"]

    x_input = pd.DataFrame(
        [[frequency, duration, importance, np.log1p(minutes_ago)]],
        columns=["frequency", "duration", "importance", "log_minutes_ago"])

    model = joblib.load(RESULTS_SIMPLE_FILE_PATH / GRADIENT_BOOSTING_FILE)
    le = joblib.load(RESULTS_SIMPLE_FILE_PATH / LABEL_ENCODER_FILE)

    # Get class probabilities
    probas = model.predict_proba(x_input)[0]  # shape: (num_classes,)
    class_labels = le.inverse_transform(np.arange(len(probas)))

    return dict(zip(class_labels, probas))


def predict_adverbial_regression(event_nl, minutes_ago, event_properties=None):
    if event_properties is not None and not event_properties.empty:
        values_dict = event_properties.iloc[0].to_dict()
        frequency = values_dict["Frequency"]
        duration = values_dict["Duration"]
        importance = values_dict["Importance"]
    else:
        # Get the previously saved event properties
        event_to_find = event_nl.replace("Tom", "A friend").replace("You", "I")
        file_path = f"{DATA_DIR}/event_properties.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            properties = json.load(fh)
        if event_to_find not in properties:
            file_path = f"{DATA_EVALUATION_SURVEY_PATH}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                properties = json.load(fh)
        frequency = properties[event_to_find]["Frequency"]
        duration = properties[event_to_find]["Duration"]
        importance = properties[event_to_find]["Importance"]

    rf_model = joblib.load(RESULTS_SIMPLE_FILE_PATH / XGB_REGRESSOR_FILE)
    ohe = joblib.load(RESULTS_SIMPLE_FILE_PATH / ONE_HOT_ENCODER_FILE)

    results = {}
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    numeric_cols = ["frequency", "duration", "importance", "minutes_ago"]

    for adv in VAGUE_ADVERBIALS:
        # One-hot encode the adverbial
        adverbial_encoded = ohe.transform(pd.DataFrame([[adv]], columns=['adverbial']))
        # Build DataFrame with numeric columns first, then onehot columns, exactly like during training
        input_df = pd.DataFrame(
            data=np.hstack([
                [frequency, duration, importance, minutes_ago],
                adverbial_encoded.flatten()
            ]).reshape(1, -1),
            columns=numeric_cols + list(adverbial_cols)
        )

        # Predict votes
        rf_pred = rf_model.predict(input_df)[0]
        results[adv] = float(rf_pred)

    return results