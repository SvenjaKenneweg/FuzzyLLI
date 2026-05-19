import json
from functools import lru_cache
from importlib import resources

import numpy as np
from joblib import load

from . import config


def _res(*parts: str):
    # points to src/fuzzylli/resources/...
    return resources.files("fuzzylli").joinpath("resources", *parts)


@lru_cache(maxsize=1)
def _load_event_adverbials() -> dict:
    with _res("event_adverbials.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_random_forest():
    rf_ref = _res(config.RANDOM_FOREST_FILE)
    # joblib needs a filesystem path -> as_file provides one
    with resources.as_file(rf_ref) as rf_path:
        return load(rf_path)


@lru_cache(maxsize=1)
def _load_embedding_regressor():
    ridge_ref = _res(config.EMBEDDING_RIDGE_FILE)
    with resources.as_file(ridge_ref) as ridge_path:
        return load(ridge_path)


@lru_cache(maxsize=1)
def _load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The embedding variant requires sentence-transformers. "
            'Install it with `pip install "fuzzylli[embeddings]"` or `pip install "fuzzylli[dev]"`.'
        ) from exc

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_kgqa_event_types() -> tuple[dict, ...]:
    with _res(config.KGQA_EVENT_TYPES_FILE).open("r", encoding="utf-8") as fh:
        return tuple(json.load(fh))


def _readable_event_type(event_type: str) -> str:
    name = event_type.split(":", 1)[-1]
    return name.replace("_", " ").strip()


def _candidate_texts(record: dict) -> list[str]:
    texts = [record.get("label") or _readable_event_type(record["event_type"])]
    texts.extend(record.get("examples", []))
    texts.append(_readable_event_type(record["event_type"]))
    return [text for text in texts if text]


def _query_text(event: str) -> str:
    return _readable_event_type(event) if ":" in event else event


@lru_cache(maxsize=1)
def _kgqa_event_type_index():
    encoder = _load_embedding_model()
    records = _load_kgqa_event_types()
    vectors = []

    for record in records:
        embeddings = encoder.encode(_candidate_texts(record), normalize_embeddings=True)
        vector = np.mean(np.asarray(embeddings), axis=0)
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        vectors.append(vector)

    return records, np.vstack(vectors)


def resolve_event_type_embedding(event: str, top_k: int = 3) -> dict:
    """
    Resolve a noisy or clean event mention to a KGQA event type using embedding similarity.

    This intentionally avoids keyword rules such as checking whether "bath" occurs in the
    input. The returned `clean_event_type` is selected by nearest-neighbor similarity in
    the sentence-embedding space.
    """
    records, matrix = _kgqa_event_type_index()
    encoder = _load_embedding_model()
    query = encoder.encode([_query_text(event)], normalize_embeddings=True)[0]
    scores = matrix @ np.asarray(query)

    ranked = np.argsort(scores)[::-1]
    top_k = max(1, min(top_k, len(ranked)))
    best = int(ranked[0])
    best_record = records[best]

    alternatives = []
    for idx in ranked[:top_k]:
        record = records[int(idx)]
        alternatives.append(
            {
                "clean_event_type": record["event_type"],
                "label": record.get("label") or _readable_event_type(record["event_type"]),
                "score": float(scores[int(idx)]),
            }
        )

    return {
        "clean_event_type": best_record["event_type"],
        "label": best_record.get("label") or _readable_event_type(best_record["event_type"]),
        "score": float(scores[best]),
        "alternatives": alternatives,
    }


def _predict_event_std_embedding(event: str, use_clean_event_type: bool = True) -> tuple[float, dict | None]:
    resolved = resolve_event_type_embedding(event) if use_clean_event_type else None
    embedding_text = resolved["label"] if resolved else event

    encoder = _load_embedding_model()
    ridge = _load_embedding_regressor()
    vector = encoder.encode(embedding_text)
    log_pred = ridge.predict(np.asarray(vector).reshape(1, -1))[0]
    event_std = float(max(0.0, np.expm1(log_pred)))

    return event_std, resolved


def predict_time_frame_random_forest(properties, adverbial, min_prob=0.6):
    params = _load_event_adverbials()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    rf = _load_random_forest()
    properties_df = config.properties_dataframe(properties)
    event_std = rf.predict(properties_df)[0]

    lower_adv, higher_adv = config.gauss_inverse(min_prob, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adv, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adv, event_std)

    upper = max(0, config.safe_round(upper_raw))
    lower = config.safe_round(lower_raw)
    return upper, lower


def predict_adverbial_random_forest(properties, minutes_ago):
    params = _load_event_adverbials()

    rf = _load_random_forest()
    properties_df = config.properties_dataframe(properties)
    event_std = rf.predict(properties_df)[0]

    adverbial_probs = {}
    for adv in config.VAGUE_ADVERBIALS:
        mean = params["adverbial_means"][adv]
        std = params["adverbial_stds"][adv]
        p = config.adverbial_specific_function(
            config.event_specific_function(minutes_ago, event_std),
            mean,
            std,
        )
        adverbial_probs[adv] = p

    return adverbial_probs


def predict_time_frame_embedding(
    event: str,
    adverbial: str,
    min_prob: float = 0.6,
    use_clean_event_type: bool = True,
):
    """
    Predict the interval for an event/adverbial pair using the word-embedding variant.

    By default the event is first resolved to a KGQA clean event type using embeddings,
    and the clean event label is used for FuzzyLLI's event-specific standard deviation.
    """
    params = _load_event_adverbials()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]
    event_std, _ = _predict_event_std_embedding(event, use_clean_event_type)

    lower_adv, higher_adv = config.gauss_inverse(min_prob, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adv, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adv, event_std)

    upper = max(0, config.safe_round(upper_raw))
    lower = config.safe_round(lower_raw)
    return upper, lower


def predict_adverbial_embedding(
    event: str,
    minutes_ago: float,
    use_clean_event_type: bool = True,
):
    """
    Predict vague-adverbial memberships using the word-embedding variant.
    """
    params = _load_event_adverbials()
    event_std, _ = _predict_event_std_embedding(event, use_clean_event_type)

    adverbial_probs = {}
    for adv in config.VAGUE_ADVERBIALS:
        mean = params["adverbial_means"][adv]
        std = params["adverbial_stds"][adv]
        p = config.adverbial_specific_function(
            config.event_specific_function(minutes_ago, event_std),
            mean,
            std,
        )
        adverbial_probs[adv] = p

    return adverbial_probs


def predict_kgqa_interval_embedding(
    event: str,
    adverbial: str,
    min_prob: float = 0.6,
    top_k: int = 3,
) -> dict:
    """
    KGQA-facing helper returning both the FuzzyLLI interval and clean event type.
    """
    resolved = resolve_event_type_embedding(event, top_k=top_k)
    event_std, _ = _predict_event_std_embedding(resolved["label"], use_clean_event_type=False)
    upper, lower = predict_time_frame_embedding(
        resolved["label"],
        adverbial,
        min_prob=min_prob,
        use_clean_event_type=False,
    )

    return {
        "input_event": event,
        "clean_event_type": resolved["clean_event_type"],
        "event_type_label": resolved["label"],
        "event_type_score": resolved["score"],
        "event_type_alternatives": resolved["alternatives"],
        "adverbial": adverbial,
        "min_prob": min_prob,
        "event_std": event_std,
        "interval": {
            "upper_minutes_ago": upper,
            "lower_minutes_ago": lower,
        },
    }
