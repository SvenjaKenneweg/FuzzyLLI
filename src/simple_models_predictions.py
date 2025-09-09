import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor

from src.predictions import get_event_properties_gpt

from src.config import (VAGUE_ADVERBIALS,
                        DURATION_ORDER,
                        FREQUENCY_ORDER,
                        RESULTS_SIMPLE_FILE_PATH,
                        GRADIENT_BOOSTING_FILE,
                        LABEL_ENCODER_FILE,
                        XGB_REGRESSOR_FILE,
                        RANDOM_FOREST_REGRESSOR_FILE,
                        ONE_HOT_ENCODER_FILE)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_adverbial_gpt_classifier(event_nl, minutes_ago, max_retries=5):
    duration, frequency = None, None
    #Get the event properties with gpt
    for _ in range(max_retries):
        gpt_result = get_event_properties_gpt(event_nl)
        duration = (gpt_result.get("Duration")).replace(" ", "")
        frequency = (gpt_result.get("Frequency")).replace(" ", "")
        if frequency == "OnceinLife": frequency = "Once in Life"

    model = joblib.load(RESULTS_SIMPLE_FILE_PATH / GRADIENT_BOOSTING_FILE)
    le = joblib.load(RESULTS_SIMPLE_FILE_PATH / LABEL_ENCODER_FILE)

    X_input = pd.DataFrame([[FREQUENCY_ORDER.index(frequency), DURATION_ORDER.index(duration), np.log1p(minutes_ago)]],
                           columns=["frequency", "duration", "log_minutes_ago"])
    # Get class probabilities
    probas = model.predict_proba(X_input)[0]  # shape: (num_classes,)
    class_labels = le.inverse_transform(np.arange(len(probas)))

    return dict(zip(class_labels, probas))


def predict_adverbial_gpt_regression(event_nl, minutes_ago, max_retries=5):
    duration, frequency = None, None
    #Get the event properties with gpt
    for _ in range(max_retries):
        gpt_result = get_event_properties_gpt(event_nl)
        duration = (gpt_result.get("Duration")).replace(" ", "")
        frequency = (gpt_result.get("Frequency")).replace(" ", "")
        if frequency == "OnceinLife": frequency = "Once in Life"

    rf_model = joblib.load(RESULTS_SIMPLE_FILE_PATH / RANDOM_FOREST_REGRESSOR_FILE)
    ohe = joblib.load(RESULTS_SIMPLE_FILE_PATH / ONE_HOT_ENCODER_FILE)

    results = {}
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    numeric_cols = ["frequency", "duration", "minutes_ago"]

    for adv in VAGUE_ADVERBIALS:
        # One-hot encode the adverbial
        adverbial_encoded = ohe.transform(pd.DataFrame([[adv]], columns=['adverbial']))
        # Build DataFrame with numeric columns first, then onehot columns, exactly like during training
        input_df = pd.DataFrame(
            data=np.hstack([
                [FREQUENCY_ORDER.index(frequency), DURATION_ORDER.index(duration), minutes_ago],
                adverbial_encoded.flatten()
            ]).reshape(1, -1),
            columns=numeric_cols + list(adverbial_cols)
        )

        # Predict votes
        rf_pred = rf_model.predict(input_df)[0]
        results[adv] = rf_pred

    return results