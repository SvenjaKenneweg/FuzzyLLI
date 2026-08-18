import numpy as np
import math
import json
import re
import os
import random
from openai import OpenAI
from pathlib import Path
from typing import Dict
from joblib import load
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cache
from sentence_transformers import SentenceTransformer

from . import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = config.RESULTS_FILE_PATH / "event_adverbials") -> Dict[str, dict]:
    path = path.with_suffix('.json')
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _safe_round(value):
    return round(value) if math.isfinite(value) else value

def _properties_dataframe(properties):
    """
    Normalize properties input to a DataFrame with columns ordered as config.properties_to_use.
    Accepts dict, list of dicts, or DataFrame. Raises if required columns missing.
    """
    if isinstance(properties, pd.DataFrame):
        df = properties.copy()
    elif isinstance(properties, dict):
        df = pd.DataFrame([properties])
    else:
        df = pd.DataFrame(properties)

    # Normalize column casing to expected names
    expected = {col.lower(): col for col in config.properties_to_use}
    for col in list(df.columns):
        col_lower = str(col).lower()
        if col_lower in expected and col != expected[col_lower]:
            df = df.rename(columns={col: expected[col_lower]})

    missing = [col for col in config.properties_to_use if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required property columns: {', '.join(missing)}")

    return df[config.properties_to_use]

def get_all_event_properties_gpt(events_nl, file_path, model_engine=config.GPT_VERSION):
    def format_scale(scale_dict):
        return "\n".join([f"{k} – {v}" for k, v in scale_dict.items()])

    instruction = f"""You are an assistant that evaluates events based on three dimensions: richness, frequency, and importance.
    - Richness → How vivid, detailed, and contextually rich the event is.  
      Scale: {format_scale(config.RICHNESS_DESCRIPTIONS)}

    - Frequency → How often the event typically occurs.  
      Scale: {format_scale(config.FREQUENCY_DESCRIPTIONS)}

    - Importance → How important or meaningful the event is to the individual.  
      Scale: {format_scale(config.IMPORTANCE_DESCRIPTIONS)}

    Instructions:
    - Read the event description carefully.
    - Determine whether the event was performed by the user or someone else.
    - Use general human knowledge to estimate richness, frequency, and importance.
    - Always respond in the following exact format:
    Richness: <1–5>
    Frequency: <1–5>
    Importance: <1–5>
    """
    results = {}
    for event_nl in events_nl:
        event_clean = event_nl.replace("You", "I").replace("Tom", "A friend")
        prompt = f"Event: {event_clean}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=0
        )

        content = completion.choices[0].message.content
        richness_match = re.search(r"Richness:\s*(\d)", content)
        frequency_match = re.search(r"Frequency:\s*(\d)", content)
        importance_match = re.search(r"Importance:\s*(\d)", content)

        if richness_match and frequency_match and importance_match:
            richness = int(richness_match.group(1))
            frequency = int(frequency_match.group(1))
            importance = int(importance_match.group(1))

            results[event_clean] = {
                "Richness": richness,
                "Richness_desc": config.RICHNESS_DESCRIPTIONS.get(richness),
                "Frequency": frequency,
                "Frequency_desc": config.FREQUENCY_DESCRIPTIONS.get(frequency),
                "Importance": importance,
                "Importance_desc": config.IMPORTANCE_DESCRIPTIONS.get(importance)
            }
        else:
            print(f"Could not parse response for event: {event_nl}")
            results[event_clean] = {
                "Richness": None,
                "Richness_desc": None,
                "Frequency": None,
                "Frequency_desc": None,
                "Importance": None,
                "Importance_desc": None,
                "raw_response": content
            }
    # Save to JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_time_frame_embedding(event, adverbial, min_prob=0.6):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    ridge_model = load(config.RESULTS_FILE_PATH / config.EMBEDDING_RIDGE_FILE)

    vecc = embedding_model.encode(event)
    log_pred = ridge_model.predict(vecc.reshape(1, -1))[0]
    event_std = int(max(0, np.expm1(log_pred)))

    lower_adverbial, higher_adverbial = config.gauss_inverse(min_prob, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adverbial, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adverbial, event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_embedding(event_nl, minutes_ago, *args):
    params = _load_packed()
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    ridge_model = load(config.RESULTS_FILE_PATH / config.EMBEDDING_RIDGE_FILE)

    vecc = embedding_model.encode(event_nl)
    log_pred = ridge_model.predict(vecc.reshape(1, -1))[0]
    event_std = int(max(0, np.expm1(log_pred)))

    adverbial_probs = {}
    for adverbial in config.VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = config.adverbial_specific_function(config.event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs

def predict_time_frame_random_forest(properties, adverbial, min_prob=0.6):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    random_forest = load(config.RESULTS_FILE_PATH / config.RANDOM_FOREST_FILE)
    properties_df = _properties_dataframe(properties)
    event_std = random_forest.predict(properties_df)[0]

    lower_adverbial, higher_adverbial = config.gauss_inverse(min_prob, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adverbial, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adverbial, event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_random_forest(properties, minutes_ago, *args):
    params = _load_packed()

    random_forest = load(config.RESULTS_FILE_PATH / config.RANDOM_FOREST_FILE)
    properties_df = _properties_dataframe(properties)
    event_std = random_forest.predict(properties_df)[0]

    adverbial_probs = {}
    for adverbial in config.VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = config.adverbial_specific_function(config.event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs


def predict_adverbial_functions(properties, minutes_ago, function_to_predict=config.powerlaw, *args):
    params = _load_packed()

    model = load( f"{config.RESULTS_FILE_PATH}/{function_to_predict.__name__}.pkl")
    function_params = list(model["params"].values())
    properties_df = _properties_dataframe(properties)
    property_values = [properties_df.iloc[0][prop] for prop in config.properties_to_use]
    event_std = function_to_predict(property_values, *function_params)

    adverbial_probs = {}
    for adverbial in config.VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = config.adverbial_specific_function(config.event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial
    return adverbial_probs



def block(i, o, dropout=None):
    x = [nn.Linear(i, o), nn.LayerNorm(o), nn.GELU()]
    return x + ([] if dropout is None else [nn.Dropout(dropout)])


class TimeEncoder(nn.Module):
    def __init__(self, dim, n_freq):
        super().__init__()
        self.register_buffer("freqs", torch.empty(n_freq))
        self.proj = nn.Sequential(*block(2*n_freq+1, dim), *block(dim, dim))

    def forward(self, t):
        x = torch.log1p(t.float().clamp_min(0))
        a = x[:, None] * self.freqs
        return self.proj(torch.cat((x[:, None], a.sin(), a.cos()), 1))


class DNN(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, n, td, nf, p = c["hidden_dim"], c["n_layers"], c["time_dim"], c["n_freq"], c["dropout"]
        self.time_encoder = TimeEncoder(td, nf)
        self.event_proj = nn.Sequential(*block(c["event_dim"], h))
        self.time_proj = nn.Sequential(*block(td, h))

        dims = [2*h] + [h]*(n-1) + [h//2]
        self.fusion_mlp = nn.Sequential(
            *(m for a, b in zip(dims, dims[1:]) for m in block(a, b, p))
        )
        self.output_head = nn.Linear(dims[-1], 4)

    def forward(self, e, t):
        x = torch.cat((self.event_proj(e), self.time_proj(self.time_encoder(t))), 1)
        return self.output_head(self.fusion_mlp(x))


class MOC(nn.Module):
    def __init__(self, c):
        super().__init__()
        e, h = c["event_dim"], c["hidden_dim"]
        self.x_scale_raw = nn.Parameter(torch.tensor(0.))
        self.x_bias = nn.Parameter(torch.tensor(0.))
        self.thresh_net = nn.Sequential(*block(e, h, 0), nn.Linear(h, 3))
        self.k_net = nn.Sequential(*block(e, h, 0), nn.Linear(h, 1))

    def forward(self, e, t):
        x = F.softplus(self.x_scale_raw) * torch.log1p(t.float().clamp_min(0)) + self.x_bias
        r = self.thresh_net(e)
        th = torch.cat((r[:, :1], r[:, :1] + F.softplus(r[:, 1:]).cumsum(1)), 1)
        c = torch.sigmoid((F.softplus(self.k_net(e)) + 1e-6) * (x[:, None] - th))
        return torch.cat((1-c[:, :1], c[:, :-1]-c[:, 1:], c[:, -1:]), 1)


@cache
def load_model(filename):
    c = torch.load(config.RESULTS_FILE_PATH / filename, map_location=DEVICE, weights_only=True)
    model = MOC(c) if c["arch"] == "monotonic_ordinal" else DNN(c)
    model.load_state_dict(c["model_state"])
    return model.to(DEVICE).eval(), c


@cache
def embedder(name):
    return SentenceTransformer(name, device=str(DEVICE))


def predict(filename, event, minutes):
    model, c = load_model(filename)
    event = event.lower().strip()

    e = c["event_embeddings"].get(event)
    if e is None:
        e = embedder(c["embedding_model_name"]).encode(
            event, convert_to_tensor=True,
            normalize_embeddings=c["normalize_embeddings"]
        )

    e = e.to(DEVICE).float().unsqueeze(0)
    t = torch.tensor([minutes], device=DEVICE)

    with torch.no_grad():
        return config.VAGUE_ADVERBIALS[model(e, t).argmax(1).item()]


def predict_adverbial_moc(event, minutes):
    return predict(config.MOC_FILE, event, minutes)


def predict_adverbial_dnn(event, minutes):
    return predict(config.DNN_FILE, event, minutes)

