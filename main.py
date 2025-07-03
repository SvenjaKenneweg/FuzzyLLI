import numpy as np
import math
from scipy.special import erf

from src.train import fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest
from src.plot import plot_allPersons_event_adverbials
from src.predictions import predict_time_frame_embedding, predict_adverbial_embedding, predict_time_frame_random_forest, predict_adverbial_random_forest


# ========================
# Fitting Functions
# ========================
def adverbial_specific_function(x, mean, std):
    # Normalized gaussian
    return (np.exp(-0.5 * ((x - mean) / std) ** 2))
def event_specific_function(temporal_distance, std):
    # Cumulative distribution function of a gaussian distribution
    return 1/2 * (erf(temporal_distance / (math.sqrt(2) * std))+1)

events = [
    # "brushing_teeth",
    # "birthday",  # Do not fit birthday as it is also in the events where the properties are measured
    # "vacation",  # Do not fit vacation as it is also in the events where the properties are measured
    # "sabbatical",
    # "year_abroad", # Do not fit year abroad as it is also in the events where the properties are measured
    # "marriage", # Do not fit marriage as it could be the whole marriage or only the celebration
    "tom_wedding_celebration",
    "own_year_abroad",
    "own_birthday",
    "own_vacation",
    "own_rent_payment",
    "own_shower",
    "tom_watching_film",
    "tom_eating_risotto",
    "tom_reading_book",
    "tom_dancing_salsa",
    "tom_storing_wineBottle",
    "tom_drinking_juice",
    "tom_chatting_friend",
    "own_wedding_celebration",
    "own_wallet_theft"
]

if __name__ == '__main__':
    # fit_event_adverbials(events, event_specific_function, adverbial_specific_function)
    # fit_event_specific_embeddings(events)
    # fit_event_specific_random_forest(events)

    event = "Tom had a meeting"
    adverbial = "just"
    minutes_ago = 144000

    print(predict_time_frame_embedding(event, adverbial))
    print(predict_adverbial_embedding(event, minutes_ago))

    print(predict_time_frame_random_forest(event, adverbial))
    print(predict_adverbial_random_forest(event, minutes_ago))

    # plot_allPersons_event_adverbials(events, event_specific_function, adverbial_specific_function)
