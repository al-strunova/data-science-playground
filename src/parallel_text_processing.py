"""
This script processes text data from a Parquet file using parallel computing.
It performs text cleaning (removing URLs, mentions, punctuation, and stopwords,
and lemmatizing words) and writes the processed data back to another Parquet file.
Designed for efficiency and scalability, it is well-suited for large text datasets
in natural language processing tasks.
"""

import re
from string import punctuation

import pandas as pd
from joblib import delayed
from joblib import Parallel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def clear_data(source_path: str, target_path: str, n_jobs: int):
    """
        Parallel process a dataframe to clean text data.

        Parameters
        ----------
        source_path : str
            Path to load the dataframe from.
        target_path : str
            Path to save the processed dataframe to.
        n_jobs : int
            Number of jobs to run in parallel.
        """

    # Read the data from a Parquet file
    data = pd.read_parquet(source_path)
    data = data.copy().dropna().reset_index(drop=True)

    # Initialize the lemmatizer and stopwords list
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # Compile regular expressions for URL, mentions, and extra spaces
    url_regex = re.compile(r"https?://[^,\s]+,?")
    mention_regex = re.compile(r"@[^,\s]+,?")
    space_regex = re.compile(" +")

    # Use job lib's Parallel and delayed to process text in parallel
    cleaned_text_list = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(process_text)(lemmatizer, stop_words, url_regex, mention_regex, space_regex, text)
        for text in data['text']  # Ensure 'text' is the correct column name
    )

    # Assign the processed texts back to the dataframe
    data["cleaned_text"] = cleaned_text_list

    # Save the processed data to a Parquet file
    data.to_parquet(target_path)


# Function to process each text entry
def process_text(lemmatizer, stop_words, url_regex, mention_regex, space_regex, text):
    """
    Process a single text entry.

    Parameters
    ----------
    lemmatizer : WordNetLemmatizer
        NLTK's lemmatizer object.
    stop_words : list
        List of stopwords.
    url_regex, mention_regex, space_regex : re.Pattern
        Compiled regular expressions for URLs, mentions, and extra spaces.
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.
    """

    # Convert to string in case of non-string types
    text = str(text)

    # Remove URLs and mentions using the compiled regex
    text = url_regex.sub("", text)
    text = mention_regex.sub("", text)

    # Remove punctuation and extra spaces
    transform_text = text.translate(str.maketrans("", "", punctuation))
    transform_text = space_regex.sub(" ", transform_text)

    # Tokenize the text
    text_tokens = word_tokenize(transform_text)

    # Lemmatize each word and remove stopwords
    lemma_text = [lemmatizer.lemmatize(word.lower()) for word in text_tokens]
    cleaned_text = " ".join(word for word in lemma_text if word not in stop_words)

    return cleaned_text
