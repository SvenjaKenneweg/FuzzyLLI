import json
import statistics
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

# Predictions for the simple models are only possible for the best fitting vague adverbial

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/with_event_properties")
RESULTS_FILE_PATH = Path("results/fits/simple_models/")

GRADIENT_BOOSTING_FILE = 'gradient_boosting_classifier.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
ONE_HOT_ENCODER_FILE = 'onehotencoder.pkl'
RANDOM_FOREST_REGRESSOR_FILE = 'RandomForestRegressor.pkl'
XGB_REGRESSOR_FILE = 'XGBRegressor.pkl'

VAGUE_ADVERBIALS: List[str] = [
    "recently",
    "just",
    "some time ago",
    "long time ago",
]

duration_order = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years', 'Decades']
frequency_order = ['Daily', 'Monthly', 'Yearly', 'Decadal', 'Once in Life']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_adverbial_classifier(duration, frequency, minutes_ago):
    model = joblib.load(RESULTS_FILE_PATH / GRADIENT_BOOSTING_FILE)
    le = joblib.load(RESULTS_FILE_PATH / LABEL_ENCODER_FILE)

    log_minutes_ago = np.log1p(minutes_ago)
    X_input = pd.DataFrame([[frequency_order.index(frequency), duration_order.index(duration), log_minutes_ago]],
                           columns=["frequency", "duration", "log_minutes_ago"])

    # Get class probabilities
    probas = model.predict_proba(X_input)[0]  # shape: (num_classes,)
    class_labels = le.inverse_transform(np.arange(len(probas)))

    return dict(zip(class_labels, probas))


def predict_adverbial_regression(duration, frequency, minutes_ago):
    rf_model = joblib.load(RESULTS_FILE_PATH / RANDOM_FOREST_REGRESSOR_FILE)
    xgb_model = joblib.load(RESULTS_FILE_PATH / XGB_REGRESSOR_FILE)
    ohe = joblib.load(RESULTS_FILE_PATH / ONE_HOT_ENCODER_FILE)

    results = {}
    adverbial_cols = ohe.get_feature_names_out(['adverbial'])
    numeric_cols = ["frequency", "duration", "minutes_ago"]

    for adv in VAGUE_ADVERBIALS:
        # One-hot encode the adverbial
        adverbial_encoded = ohe.transform(pd.DataFrame([[adv]], columns=['adverbial']))
        # Build DataFrame with numeric columns first, then onehot columns, exactly like during training
        input_df = pd.DataFrame(
            data=np.hstack([
                [frequency_order.index(frequency), duration_order.index(duration), minutes_ago],
                adverbial_encoded.flatten()
            ]).reshape(1, -1),
            columns=numeric_cols + list(adverbial_cols)
        )

        # Predict votes
        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]

        results[adv] = {
            "RandomForestRegressor": rf_pred,
            "XGBRegressor": xgb_pred
        }

    return results