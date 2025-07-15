from src.train import fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest
from src.plot import plot_all_persons_event_adverbials
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding,
                             predict_time_frame_gpt_random_forest, predict_adverbial_gpt_random_forest,
                             predict_time_frame_random_forest, predict_adverbial_random_forest)
from src.evaluation import evaluate_embedding, evaluate_gpt_random_forest, evaluate_random_forest, evaluate_classifier, evaluate_regression
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_classifier, predict_adverbial_regression


events = [
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

    # simple models
    # fit_classifier(events)
    # fit_regression(events)

    # event = "Tom had a meeting"
    # duration = "Minutes"
    # frequency = "Monthly"
    # adverbial = "just"
    # minutes_ago = 144000

    # print(predict_adverbial_classifier(duration, frequency, minutes_ago))
    # print(predict_adverbial_regression(duration, frequency, minutes_ago))

    # print(predict_time_frame_embedding(event, adverbial))
    # print(predict_adverbial_embedding(event, minutes_ago))
    #
    # print(predict_time_frame_gpt_random_forest(event, adverbial))
    # print(predict_adverbial_gpt_random_forest(event, minutes_ago))
    #
    # print(predict_time_frame_random_forest(duration, frequency, adverbial))
    # print(predict_adverbial_random_forest(duration, frequency, minutes_ago))

    plot_all_persons_event_adverbials(events)

    # print("Embeddings+Regressor:")
    # print(evaluate_embedding(events, event_specific_function, adverbial_specific_function, metric="rmse"))

    # print("GPT+Random Forest:")
    # print(evaluate_gpt_random_forest(events, event_specific_function, adverbial_specific_function))

    # print("Random Forest:")
    # print(evaluate_random_forest(events, event_specific_function, adverbial_specific_function))

    # print("Baseline models:")
    # print(evaluate_classifier(events))
    # print(evaluate_regression(events))
