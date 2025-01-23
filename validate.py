import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud


def assign_topics(df, tokenised, model):
    """
    Assigns topics to the given data using the provided model.

    Parameters:
        df (pd.DataFrame): The original data.
        tokenised: The tokenized representation of the data.
        model: The topic model used to transform the tokenized data.

    Returns:
        pd.DataFrame: The original data with an additional column 'Topic_LDA' indicating the assigned topic.
    """
    topic_dist = model.transform(tokenised)

    topics = []
    for doc in topic_dist:
        topics.append(np.argmax(np.abs(doc)) + 1)

    topics_df = pd.DataFrame(topics).rename(columns={0: 'Topic_LDA'})
    return pd.merge(df, topics_df, left_index=True, right_index=True)


def aggregate_plot(df):
    """
    Plots a bar chart of the counts of issue types and topics.

    Parameters:
        df (pd.DataFrame): DataFrame containing the results with 'issue_type' and 'Topic_LDA' columns.
    """
    aggregated_input = pd.DataFrame(df['issue_type'].value_counts())
    aggregated_LDA = pd.DataFrame(df['Topic_LDA'].value_counts())
    aggregated = pd.merge(aggregated_input, aggregated_LDA, left_index=True, right_index=True)
    aggregated.plot(kind='bar', figsize=(13, 7))
    plt.show()


def accuracy(df):
    """
    Calculates the accuracy of the topic assignment.

    Parameters:
        df (pd.DataFrame): DataFrame containing the results with 'issue_type', 'Topic_LDA', and 'statement' columns.

    Returns:
        float: The accuracy percentage of the topic assignment.
    """
    df1 = df.groupby(['issue_type', 'Topic_LDA'], as_index=False)['statement'].count().rename(
        columns={'statement': 'Count'})
    correct_preds = df1[df1['issue_type'] == df1['Topic_LDA']]['Count'].reset_index()['Count'].sum()
    total_accuracy = correct_preds / df.shape[0]
    return total_accuracy * 100


def precision(df):
    """
    Calculates the precision of the topic assignment.

    Parameters:
        df (pd.DataFrame): DataFrame containing the results with 'issue_type', 'Topic_LDA', and 'Count' columns.

    Returns:
        pd.DataFrame: DataFrame with 'Topic' and 'Precision' columns.
    """
    predicted_articles_total = df.groupby('Topic_LDA')['Count'].sum().reset_index()
    predicted_articles = df[df['issue_type'] == df['Topic_LDA']]['Count'].reset_index()['Count']
    predicted_articles = predicted_articles.fillna(0)
    predicted_articles_total['Count'] = (predicted_articles / predicted_articles_total['Count']) * 100
    return predicted_articles_total.rename(columns={'Count': 'Precision', 'Topic_LDA': 'Topic'})


def recall(df):
    """
    Calculates the recall of the topic assignment.

    Parameters:
        df (pd.DataFrame): DataFrame containing the results with 'issue_type', 'Topic_LDA', and 'Count' columns.

    Returns:
        pd.DataFrame: DataFrame with 'Topic' and 'Recall' columns.
    """
    predicted_articles_total = df.groupby('issue_type')['Count'].sum().reset_index()
    predicted_articles = df[df['issue_type'] == df['Topic_LDA']]['Count'].reset_index()['Count']
    predicted_articles = predicted_articles.fillna(0)
    predicted_articles_total['Count'] = (predicted_articles / predicted_articles_total['Count']) * 100
    return predicted_articles_total.rename(columns={'Count': 'Recall', 'issue_type': 'Topic'})


def precision_recall_f1(df):
    """
    Calculates the precision, recall, and F1 score for the topic assignment.

    Parameters:
        df (pd.DataFrame): DataFrame containing the results with 'issue_type', 'Topic_LDA', and 'label' columns.

    Returns:
        pd.DataFrame: DataFrame with 'Topic', 'Precision', 'Recall', and 'F1_Score' columns.
    """
    df_grouped = df.groupby(['issue_type', 'Topic_LDA'], as_index=False)['label'].count().rename(
        columns={'label': 'Count'})
    precisions = precision(df_grouped)
    recalls = recall(df_grouped)
    result = pd.merge(recalls, precisions, on='Topic')
    result['Precision'] = result['Precision'].fillna(0)
    result['Recall'] = result['Recall'].fillna(0)
    result['F1_Score'] = (2 * result['Precision'] * result['Recall']) / (result['Precision'] + result['Recall'])
    return result


def top_n_terms(vectorizer, model, n_top_words):
    """
    Extracts the top N terms for each topic from the topic model.

    Parameters:
        vectorizer: The vectorizer used to transform the data.
        model: The topic model containing the topic-term distributions.
        n_top_words (int): The number of top words to extract for each topic.

    Returns:
        list of dict: A list where each element is a dictionary of the top words and their weights for a topic.
    """
    words = vectorizer.get_feature_names_out()

    top_words_topic = []
    for topic, weights in enumerate(model.components_):
        word_weights = dict()

        indices = np.argsort(weights)[::-1][:n_top_words]

        top_words = words[indices]
        top_weights = weights[indices]

        for i in range(len(words[indices])):
            word_weights[top_words[i]] = top_weights[i]
        top_words_topic.append(word_weights)
    return top_words_topic


def plot_wordcloud(topic_term_freq):
    """
    Plots a word cloud for the given topic term frequencies.

    Parameters:
    topic_term_freq: A dictionary where keys are terms and values are their frequencies.
    """
    colours = ['black', 'darkslateblue', 'darkolivegreen', 'khaki']
    wordcloud = WordCloud(width=1300, height=800,
                          background_color=random.choice(colours),
                          min_font_size=10).generate(' '.join(list(topic_term_freq.keys())))
    wordcloud = wordcloud.generate_from_frequencies(frequencies=topic_term_freq)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
