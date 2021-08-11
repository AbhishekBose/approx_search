from abc import ABC, abstractmethod


class Embeddings(ABC):

    @abstractmethod
    def generate(self, data):
        """
        Use this finction to generate embeddings using the model of choice
        :param data: Could contain list of strings
        :return: return a list of embeddings
        """
