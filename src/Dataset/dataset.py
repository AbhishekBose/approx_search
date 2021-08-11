from abc import ABC, abstractmethod

class Dataset(ABC):

    @abstractmethod
    def download(self):
        """
        Use this function to download the dataset from a url

        """
        pass

    @abstractmethod
    def extract_files(self):
        """
        Use this method to extract files from a zip file
        :return:
        """
        pass

    @abstractmethod
    def parse_files(self):
        """
        Use this method to parse the content files of the zip file
        :return:
        """
        pass

    @abstractmethod
    def load(self):
        """
        Use this method to assimilate all methods defined above and the load the dataset onto memory
        :return:
        """
        pass

    def save(self):
        """
        Use this method to save the parsed dataset
        :return:
        """
        pass