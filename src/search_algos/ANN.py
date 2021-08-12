from abc import ABC, abstractmethod


class ANN(ABC):

    @abstractmethod
    def build(self):
        """
        This method is used to build the index for the ANN algo in concern
        :return: None
        """
        pass

    @abstractmethod
    def query(self, query_vec, k):
        """
        This method is used to query the index for a particular vector
        :param query_vec: search vector
        :param k: number of closest neighbors needed
        :return:
        """
        pass

    @abstractmethod
    def query_batch(self, query_vec, k=10):
        """
        This method can be used to search the nearest neighbors for a batch of vectors
        :param query_vec: vector batch
        :param k: number of nearest neighbors needed
        :return:
        """
        pass
