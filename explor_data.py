import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import spacy
from datasets import load_dataset
from nltk.corpus import stopwords


def load():
    """
    Load the political ideologies dataset.

    Returns:
        dataset: The loaded dataset.
    """
    dataset = load_dataset("JyotiNayak/political_ideologies")
    return dataset


def convert_to_pandas(df):
    """
    Convert the dataset to a pandas DataFrame and process it.

    Args:
        df: The dataset to convert.

    Returns:
        df_train: The processed pandas DataFrame.
    """
    df_train = df['train'].to_pandas()
    labels = ['economic', 'environmental', 'family/gender', 'geo-political and foreign policy', 'political',
              'racial justice and immigration', 'religious', 'social, health and education']
    df_train['issue_type'] = df_train['issue_type'].apply(lambda x: labels[x])

    df_train['statement_length'] = df_train['statement'].apply(len)
    return df_train


def show_data_info(df):
    """
    Display basic information about the dataset.

    Args:
        df (pd.DataFrame): The dataset to display information about.

    Prints:
        - First few rows of the dataset.
        - Information about the dataset.
        - Value counts of the 'issue_type' column.
        - Number of missing values in each column.
        - Descriptive statistics of the 'statement_length' column.
    """
    print(df.head())
    print(df.info())
    print(df['issue_type'].value_counts())
    print(df.isnull().sum())
    print(df['statement_length'].describe())


def text_length_distribution(data):
    """
    Plot the distribution of text lengths.

    Args:
        data (pd.DataFrame): The dataset containing the 'statement_length' column.

    Displays:
        A histogram with a KDE plot showing the distribution of text lengths.
    """
    sns.histplot(data['statement_length'], bins=30, kde=True)
    plt.title("Rozkład długości tekstów")
    plt.xlabel("Długość tekstu")
    plt.ylabel("Liczba próbek")
    plt.show()


def issue_type_distribution(df):
    """
    Plot the distribution of issue types.

    Args:
        df (pd.DataFrame): The dataset containing the 'issue_type' column.

    Displays:
        A count plot showing the distribution of issue types.
    """
    sns.countplot(data=df, x='issue_type')
    plt.title("Rozkład etykiet")
    plt.xlabel("Etykiety")
    plt.xticks(rotation=90)
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.show()


def text_length_distribution_by_issue_type(df):
    """
    Plot the distribution of text lengths by issue type.

    Args:
        df (pd.DataFrame): The dataset containing the 'issue_type' and 'statement_length' columns.

    Displays:
        A KDE plot for each issue type showing the distribution of text lengths.
    """
    plt.figure(figsize=(12, 6))
    for issue in df['issue_type'].unique():
        subset = df[df['issue_type'] == issue]
        sns.kdeplot(subset['statement_length'], label=issue)

    plt.title("Dystrybucja długości tekstów według typów zagadnień")
    plt.xlabel("Długość tekstu (ilość słów)")
    plt.ylabel("Gęstość")
    plt.legend(title="Typ zagadnienia")
    plt.grid(True)
    plt.show()


def preprocess_text(text, stop_words, nlp):
    """
    Clean and preprocess the input text.

    Args:
        text (str): The text to be cleaned.
        stop_words (set): A set of stop words to be removed from the text.
        nlp (spacy.lang): The SpaCy language model for lemmatization.

    Returns:
        str: The cleaned and lemmatized text.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = text.lower()
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct])

    return cleaned_text


def clean_date(df):
    """
    Clean and preprocess the dataset.

    Args:
        df (pd.DataFrame): The dataset to be cleaned.

    Returns:
        pd.DataFrame: The cleaned dataset with an additional 'cleaned_statement' column.
    """
    # Pobranie stop-słów NLTK
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Ładowanie modelu językowego SpaCy
    nlp = spacy.load("en_core_web_sm")

    # Zastosowanie oczyszczania do kolumny 'statement'
    df['cleaned_statement'] = df['statement'].apply(lambda x: preprocess_text(x, stop_words, nlp))
    return df


def common_words(df):
    """
    Count and plot the 20 most common words in the cleaned statements.

    Args:
        df (pd.DataFrame): The dataset containing the 'cleaned_statement' column.

    Displays:
        A bar plot showing the 20 most common words in the cleaned statements.
    """
    # Zliczanie słów we wszystkich oczyszczonych tekstach
    all_words = " ".join(df['cleaned_statement']).split()
    word_counts = Counter(all_words)

    # Pobranie 20 najczęstszych słów
    most_common_words = word_counts.most_common(20)
    words, counts = zip(*most_common_words)

    # Wykres słupkowy dla najczęstszych słów
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title("20 najczęstszych słów w oczyszczonych danych")
    plt.xlabel("Liczba wystąpień")
    plt.ylabel("Słowa")
    plt.show()


def common_words_by_issue_type(df):
    """
    Count and plot the 5 most common words for each issue type.

    Args:
        df (pd.DataFrame): The dataset containing the 'cleaned_statement' and 'issue_type' columns.

    Displays:
        A bar plot showing the 5 most common words for each issue type.
    """
    # Zliczanie słów dla każdego 'issue_type'
    word_distribution = {}
    for issue in df['issue_type'].unique():
        issue_texts = " ".join(df[df['issue_type'] == issue]['cleaned_statement'])
        word_distribution[issue] = Counter(issue_texts.split())

    # Zliczanie wszystkich słów w danych
    all_words = " ".join(df['cleaned_statement']).split()
    most_common_words = [word for word, _ in Counter(all_words).most_common(5)]

    # Tworzenie DataFrame dla wizualizacji
    data = []
    for issue, counter in word_distribution.items():
        for word in most_common_words:
            data.append({'issue_type': issue, 'word': word, 'count': counter[word]})

    df_viz = pd.DataFrame(data)

    # Tworzenie wykresu
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_viz, x='word', y='count', hue='issue_type', palette='viridis')
    plt.title("Rozkład 5 najczęstszych słów w poszczególnych 'issue_type'")
    plt.xlabel("Słowa")
    plt.ylabel("Liczba wystąpień")
    plt.legend(title="Typ zagadnienia")
    plt.show()


if __name__ == "__main__":
    data = load()
    data_df = convert_to_pandas(data)

    print(data)
    show_data_info(data_df)

    text_length_distribution(data_df)
    issue_type_distribution(data_df)
    text_length_distribution_by_issue_type(data_df)

    cleaned_data = clean_date(data_df)
    common_words(cleaned_data)
    common_words_by_issue_type(cleaned_data)
