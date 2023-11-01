"""
Filter Test
Used to test filter against particular training set
"""
import math

from spam_filter.naive_bayes import NaiveBayesFilter
import os


class FilterTest:
    def __init__(self, spam_path: str, ham_path: str):
        self._spam_folder = spam_path
        self._ham_folder = ham_path
        self._filter = NaiveBayesFilter()
        self._spam = []
        self._ham = []
        self._spam_train = []
        self._ham_train = []
        self._spam_test = []
        self._ham_test = []

    def extract_data(self):
        for root, dirs, files in os.walk(self._spam_folder):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        self._spam.append(f.read())
        print(f"*** Message: Successfully extracted all spam mail ({len(self._spam)} in total). ***")
        for root, dirs, files in os.walk(self._ham_folder):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        self._ham.append(f.read())
        print(f"*** Message: Successfully extracted all spam mail ({len(self._ham)} in total). ***")

    """
    Splits data into training set and test set. 
    Note: function shallow copies elements from spam and ham lists
    Args
    (float) ratio : what percentage to use as test set
    """
    def train_test_split(self, ratio=0.2):
        max_spam = math.floor(ratio * len(self._spam))
        max_ham = math.floor(ratio * len(self._ham))
        self._spam_test = self._spam[:max_spam]
        self._spam_train = self._spam[max_spam:]
        self._ham_test = self._ham[:max_ham]
        self._ham_train = self._ham[max_ham:]

    def train_filter(self):
        self._filter.train(self._spam_train, self._ham_train)
        print("*** Message: finished training. ***")

    def test_filter(self, print_result=True):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        # Test filter on spam data set
        for email_text in self._spam_test:
            if self._filter.classify(email_text):
                # spam classified as spam.
                true_pos += 1
            else:
                # spam classified as ham. false_negative
                false_neg += 1
        # Test filter on ham data set
        for email_text in self._ham_test:
            if self._filter.classify(email_text):
                # ham classified as spam. False pos
                false_pos += 1

        # Compute precision and recall
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        # If we want to print result, print it out
        if print_result:
            print("***************************")
            print(f"Test with {len(self._spam_test)} spam emails and {len(self._ham_test)} 'ham' emails.")
            print("Precision: ", precision)
            print("Recall: ", recall)
            print(true_pos, false_pos, false_neg)

        # return scores
        return precision, recall
