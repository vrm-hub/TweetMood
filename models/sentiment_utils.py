# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE:


"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import Counter
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk import NaiveBayesClassifier, ConfusionMatrix
from sklearn.linear_model import LogisticRegression

# so that we can indicate a function in a type hint

nltk.download('punkt')
import string
from nltk.stem import WordNetLemmatizer


# Preprocessing the data
def preprocess(text):
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove punctuation
    cleaned_tokens = [token for token in lemmatized_tokens if token not in string.punctuation and token.isalpha()]

    return cleaned_tokens


def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    id\ttext\tlabel
    id\ttext\tlabel
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            preprocessed_review = preprocess(t[1])
            X.append(preprocessed_review)
            y.append(int(t[2]))
    f.close()
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def metrics_fun(train_data, dev_labels, dev_data, classifier_type=None, nn_model=None, nn_epochs=3, nn_batch_size=32):
    predictions = []
    # Train the classifier based on the specified type
    if classifier_type == 'Naive Bayes':
        classifier = NaiveBayesClassifier.train(train_data)
        predictions = [classifier.classify(feats) for feats in dev_data]
    elif classifier_type == 'Logistic Regression':
        train_x, train_y = zip(*train_data)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(train_x, train_y)
        predictions = lr.predict(dev_data)
    elif classifier_type == 'Neural Network':
        if nn_model is None:
            raise ValueError("Neural network model is not provided.")

        # Extract features and labels
        train_x, train_y = zip(*train_data)
        train_X = np.array(train_x)
        train_Y = np.array([int(label) for label in train_y])
        dev_X = np.array(dev_data.toarray())
        dev_Y = np.array([int(label) for label in dev_labels])
        nn_model.fit(train_X, train_Y, epochs=nn_epochs, batch_size=nn_batch_size, validation_data=(dev_X, dev_Y))

        # Predict on the dev set
        probs = nn_model.predict(dev_X)
        predictions = [str(1 if p[0] > 0.5 else 0) for p in probs]
    else:
        raise ValueError("Invalid classifier type")
    # Use the get_prfa function to calculate metrics
    return get_prfa(dev_labels, predictions)


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    preds = [str(p) for p in preds]
    dev_y = [str(y) for y in dev_y]

    cm = ConfusionMatrix(dev_y, preds)
    p = cm.precision('1')
    r = cm.recall('1')
    if p + r == 0:
        f1 = 0
    else:
        f1 = (2 * p * r) / (p + r)
    a = nltk.accuracy(dev_y, preds)
    if verbose:
        print(f"Precision: {p:.4f}")
        print(f"Recall: {r:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {a:.4f}")
    return p, r, f1, a


def create_training_graph(train_feats: list, training_tups:list, dev_feats: list, dev_labels:list, kind: str, savepath: str = None,
                          verbose: bool = False, model=None) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        dev_labels: a list of dev data labels
        training_tups: the tuple consists of the training labels and data
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}

    for size in train_sizes:
        subset_size = int(len(train_feats) * size / 100)
        if kind == 'Naive Bayes':
            train_subset = [(train_feats[i], str(training_tups[1][i])) for i in range(subset_size)]
            prec, rec, f1, acc = metrics_fun(train_subset, dev_labels, dev_feats, kind)
        elif kind == 'Neural Network':
            train_x, _ = zip(*train_feats)
            train_subset = [(train_x[i], str(training_tups[1][i])) for i in range(subset_size)]
            prec, rec, f1, acc = metrics_fun(train_subset, dev_labels, dev_feats, kind, model)
        else:
            train_x, _ = zip(*train_feats)
            train_subset = [(train_x[i], str(training_tups[1][i])) for i in range(subset_size)]
            prec, rec, f1, acc = metrics_fun(train_subset, dev_labels, dev_feats, kind)

        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['accuracy'].append(acc)

    plt.figure(figsize=(10, 6))
    for metric, values in metrics.items():
        plt.plot(train_sizes, values, label=metric)

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Performance')
    plt.title(f"Performance of {kind} Classifier")
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(savepath)
    plt.show()


def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    # Flatten the list of lists and then create a set to get unique words
    unique_words = set()
    for review in all_train_data_X:
        for word in review:
            unique_words.add(word)

    # Convert the set back to a list
    vocab = list(unique_words)

    return vocab


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    feature_data = []

    for review in data_to_be_featurized_X:
        word_counts = Counter(review)

        if binary:
            review_set = set(review)
            features = {word: (word in review_set) for word in vocab}
        else:
            features = {word: word_counts[word] if word in word_counts else 0 for word in vocab}
        feature_data.append(features)

    if verbose:
        print(feature_data)

    return feature_data
