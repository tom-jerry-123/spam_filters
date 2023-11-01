"""
Naive Bayes Spam Filter
Expected Training Data Format
- Two folders (spam and ham)
- each containing only txt documents
"""
import math

from spam_filter import SpamFilter


class NaiveBayesFilter(SpamFilter):
    def __init__(self):
        super().__init__()
        self._freq_word_given_spam = dict()
        self._freq_word_given_ham = dict()
        self._total_spam = 0
        self._total_mail = 0

    def train(self, spam: list[str], ham: list[str]):
        self._total_spam = len(spam)
        self._total_mail = len(spam) + len(ham)
        for email in spam:
            vocab = self._get_vocab(email)
            for word in vocab:
                self._freq_word_given_spam[word] = self._freq_word_given_spam.get(word, 1) + 1
        for email in ham:
            vocab = self._get_vocab(email)
            for word in vocab:
                self._freq_word_given_ham[word] = self._freq_word_given_ham.get(word, 1) + 1

    def classify(self, email_text: str) -> bool:
        log_p_spam = 0
        log_p_ham = 0
        word_freq = dict()
        email_words = self._get_words(email_text)
        # get word count
        for word in email_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        # compute log of probabilities
        for word, freq in word_freq.items():
            p_spam = freq * self._freq_word_given_spam.get(word, 1) / (self._total_spam + 1)
            log_p_spam += math.log(p_spam)
            p_ham = freq * self._freq_word_given_ham.get(word, 1) / (self._total_mail - self._total_spam + 1)
            log_p_ham += math.log(p_ham)
        # Multiply probability by prior belief (i.e. add log of priors)
        log_p_spam += math.log(self._total_spam/self._total_mail)
        log_p_ham += math.log((self._total_mail - self._total_spam)/self._total_mail)
        # compare probabilities
        return log_p_spam >= log_p_ham

    """
    Helper Function
    Gets set of unique words in email
    """

    @staticmethod
    def _clean_email(email: str) -> str:
        cleaned_email = ""
        for char in email:
            if char.isalnum() or char.isspace():
                cleaned_email += char
        return cleaned_email

    def _get_vocab(self, email: str) -> set:
        # Remove all characters that are neither alphanumeric nor whitespace
        cleaned_email = self._clean_email(email)
        word_list = cleaned_email.split()
        return set(word_list)

    def _get_words(self, email: str) -> list[str]:
        cleaned_email = self._clean_email(email)
        return cleaned_email.split()

