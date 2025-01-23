import joblib
import pyLDAvis.lda_model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from type import ModelType
from validate import *


def fit(tokenised_data):
    """
    Fits an LDA model to the tokenised data using GridSearchCV to find the best number of components.

    Parameters:
        tokenised_data (array-like or sparse matrix): The tokenised data to fit the LDA model on.

    Returns:
        GridSearchCV: The fitted GridSearchCV object with the best LDA model.
    """
    lda = LatentDirichletAllocation()
    grid_params = {'n_components': list(range(5, 14))}
    lda_model = GridSearchCV(lda, param_grid=grid_params)
    lda_model.fit(tokenised_data)
    return lda_model


def get_and_save_best(model, tokenised_data, model_type: ModelType = ModelType.LDA):
    """
    Retrieves the best LDA model from the GridSearchCV object, calculates log likelihood and perplexity,
    prints the results, and saves the model to a file.

    Parameters:
        model (GridSearchCV): The fitted GridSearchCV object containing the LDA models.
        tokenised_data (array-like or sparse matrix): The tokenised data used to fit the LDA model.
        model_type (ModelType, optional): The type of model being used. Defaults to ModelType.LDA.

    Returns:
        LatentDirichletAllocation: The best LDA model.
    """
    best_lda_model = model.best_estimator_
    log_likelihood_m1 = model.best_score_
    perplexity_m1 = best_lda_model.perplexity(tokenised_data)

    print(f"Parameters of Best LDA {model_type.value[2]}model", model.best_params_)
    print(f"Best log likelihood Score for the LDA {model_type.value[2]}model", log_likelihood_m1)
    print(f"Perplixity Score on the LDA {model_type.value[2]}model", perplexity_m1)

    joblib.dump(best_lda_model, f"./models/{model_type.value[1]}")
    return best_lda_model


def load(model_type: ModelType = ModelType.LDA):
    """
    Loads a saved model from a file.

    Parameters:
        model_type (ModelType, optional): The type of model to load. Defaults to ModelType.LDA.

    Returns:
        object: The loaded model.
    """
    return joblib.load(f"./models/{model_type.value[1]}")


def visualise(model, tokenised_data, tokeniser, model_type: ModelType = ModelType.LDA):
    """
    Visualises the LDA model using pyLDAvis and saves the visualization as an HTML file.

    Parameters:
        model (LatentDirichletAllocation): The LDA model to visualise.
        tokenised_data (array-like or sparse matrix): The tokenised data used to fit the LDA model.
        tokeniser (object): The tokeniser used to preprocess the data.
        model_type (ModelType, optional): The type of model being used. Defaults to ModelType.LDA.
    """
    lda_panel = pyLDAvis.lda_model.prepare(model, tokenised_data, tokeniser, mds='tsne')
    pyLDAvis.save_html(lda_panel, f"./static/html/{model_type.value[1]}")


def convert_labels(result):
    """
    Converts the topic labels in the result DataFrame to more meaningful labels based on the issue type.

    Parameters:
        result (pd.DataFrame): The DataFrame containing the results with 'issue_type' and 'Topic_LDA' columns.

    Returns:
        DataFrame: The DataFrame with updated 'Topic_LDA' labels.
    """
    to_map = result.groupby(['issue_type', 'Topic_LDA'], as_index=False)['cleaned_statement'].count()

    to_map = (
        to_map.groupby('issue_type', as_index=False).apply(lambda x: x['Topic_LDA'][x['cleaned_statement'].idxmax()])
        .rename(columns=str).rename(columns={'None': 'LDA'}))

    to_map_dict = dict(zip(to_map['LDA'], to_map['issue_type']))
    result['Topic_LDA'] = result['Topic_LDA'].map(to_map_dict)
    for category in to_map_dict:
        print(f'{category} is mapped to {to_map_dict[category]}')
    return result


def evaluate(data, tokenised, model, model_type: ModelType = ModelType.LDA):
    """
    Evaluates the LDA model by assigning topics, converting labels, and generating plots.

    Parameters:
        data (pd.DataFrame): The original data.
        tokenised (array-like or sparse matrix): The tokenised data used to fit the LDA model.
        model (LatentDirichletAllocation): The LDA model to evaluate.
        model_type (ModelType, optional): The type of model being used. Defaults to ModelType.LDA.
    """
    result = assign_topics(data, tokenised, model)
    print(result.head())
    print(result.groupby(['issue_type', 'Topic_LDA'])['cleaned_statement'].count())
    result_df = convert_labels(result)
    print(result_df.head())
    aggregate_plot(result_df)
    print(f'The LDA {model_type.value[2]} model has an accuracy of {round(accuracy(result_df), 2)}%')
    print(precision_recall_f1(result_df))


def top_word(tokeniser, model, n_top_words=50):
    """
    Extracts the top words for each topic in the LDA model and generates word clouds for visualization.

    Parameters:
        tokeniser (object): The tokeniser used to preprocess the data.
        model (LatentDirichletAllocation): The LDA model to extract top words from.
        n_top_words (int, optional): The number of top words to extract for each topic. Defaults to 50.
    """
    top_words = top_n_terms(tokeniser, model, n_top_words)
    for topic in top_words:
        plot_wordcloud(topic)
