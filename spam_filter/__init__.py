

from abc import abstractmethod


class SpamFilter:
    """
    Methods
    train       : trains spam filter model
    classify    : given email text, return whether it's spam (true = spam, false = not spam 'ham')
    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self, spam, ham):
        pass

    @abstractmethod
    def classify(self, email_text: str) -> bool:
        pass
