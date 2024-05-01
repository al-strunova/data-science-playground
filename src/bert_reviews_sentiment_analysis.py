"""
This script leverages a pre-trained BERT model for sentiment classification of reviews from an external dataset.
The steps included are:

1. Extracting review data from a CSV dataset that was generated using an SQL query.
2. Exploring the Transformers BERT library, which is utilized for tasks such as text classification.
3. Processing text in batches, including tokenization and padding.
4. Generating sentence embeddings by extracting the [CLS] token from the BERT model's output for each input.
5. Training a logistic regression model on these embeddings to classify reviews based on their sentiment.
6. Evaluating the model's performance using Cross-Entropy Loss across multiple folds of cross-validation.

This exercise aims to familiarize with transformers and NLP techniques for sentiment analysis tasks."
"""

from dataclasses import dataclass
from transformers import PreTrainedTokenizer, DistilBertModel, DistilBertTokenizer
from typing import List, Generator, Tuple
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    """
    Create an attention mask to ignore padding tokens in sequences.

    Parameters:
    padded (List[List[int]]): A list of tokenized entries where padding tokens are represented by 0.

    Returns:
    List[List[int]]: A mask list where 1 represents a real token and 0 represents a padding token.
    """
    mask = []
    for line in padded:
        mask_line = [1 if token != 0 else 0 for token in line]
        mask.append(mask_line)
    return mask


def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """
    Generate embeddings for a batch of tokenized texts using a specified model.

    Parameters:
    tokens (List[List[int]]): Tokenized texts for which embeddings are to be generated.
    model: The BERT model used to generate embeddings.

    Returns:
    List[List[float]]: Embeddings corresponding to the [CLS] token of each input text.
    """
    mask = attention_mask(tokens)
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)
    with torch.no_grad():
        last_hidden_states = model(tokens, attention_mask=mask)
    return last_hidden_states[0][:, 0, :].tolist()


def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    """
    Evaluate a model using K-Fold cross-validation and calculate the Cross-Entropy Loss for each fold.

    Parameters:
    model: The model to be evaluated.
    embeddings (np.ndarray): The embeddings on which the model is to be trained.
    labels (np.ndarray): The target labels for the embeddings.
    cv (int): The number of folds to use for cross-validation.

    Returns:
    List[float]: The Cross-Entropy Loss for each fold.
    """
    log_losses = []

    # Set up the K-Fold cross-validator
    kf = KFold(n_splits=cv)

    for train_index, test_index in kf.split(embeddings):
        X_train, y_train = embeddings[train_index], labels[train_index]
        X_test, y_test = embeddings[test_index], labels[test_index]

        model.fit(X_train, y_train)
        fold_predict_y = model.predict_proba(X_test)
        fold_log_loss = log_loss(y_test, fold_predict_y)
        log_losses.append(fold_log_loss)

    return log_losses


@dataclass
class DataLoader:
    """
    DataLoader class for loading and tokenizing data in batches.

    Attributes:
    path (str): Path to the dataset file.
    tokenizer (PreTrainedTokenizer): Tokenizer to be used for processing text.
    batch_size (int): Number of entries per batch.
    max_length (int): Maximum length of tokenized sequences.
    padding (str): Strategy for padding sequences ('max_length', 'batch', or 'do_not_pad').
    """
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """
        Iterate over the dataset, yielding batches of tokenized texts.

        Yields:
        Generator[List[List[int]], None, None]: A generator that yields batches of tokenized text sequences.
        """
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """
        Calculates the number of batches in the dataset based on the batch size and dataset size.

        Returns:
        int: The number of batches.
        """
        with open(self.path, 'r', newline='', encoding='utf-8') as csv_file:
            row_cnt = sum(1 for row in csv_file) - 1
        return (row_cnt + self.batch_size - 1) // self.batch_size

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """
        Tokenizes a batch of text strings into token IDs, applying padding as specified.

        Parameters:
        batch (List[str]): The batch of text strings to tokenize.

        Returns:
        List[List[int]]: The tokenized and padded batch of text.
        """
        if self.padding == 'max_length':
            padding_strategy = 'max_length'
        elif self.padding == 'batch':
            padding_strategy = 'longest'
        else:
            padding_strategy = 'do_not_pad'
        return self.tokenizer(batch,
                              add_special_tokens=True,
                              truncation=True,
                              max_length=self.max_length,
                              padding=padding_strategy
                              )['input_ids']

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """
        Loads and returns a specific batch of data based on the batch index.

        Parameters:
        i (int): Index of the batch to load.

        Returns:
        Tuple[List[str], List[int]]: A tuple containing the batch of text strings and their corresponding labels.
        """
        with open(self.path, 'r', newline='', encoding='utf-8') as csv_file:
            next(csv_file)  # Skip the header
            start = i * self.batch_size
            end = start + self.batch_size
            text, label_text = [], []
            for index, line in enumerate(csv_file):
                if start <= index < end:
                    columns = line.strip().split(",", 4)
                    text.append(columns[4])
                    label_text.append(columns[3])
                elif index >= end:
                    break
        value_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        label = [value_mapping[value] for value in label_text]
        return text, label

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """
        Retrieves and tokenizes a specific batch of data based on the batch index.

        Parameters:
        i (int): Index of the batch to tokenize.

        Returns:
        Tuple[List[List[int]], List[int]]: A tuple of tokenized texts and their corresponding labels, ready for model input.
        """
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels
