from dataset import Dataset
import os
import wget
import logging
from zipfile import ZipFile
from glob import glob
import io
import pickle



class TweetData(Dataset):
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00438/Health-News-Tweets.zip"
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.data_file_ext = ".txt"
        self.dataset_path = None
        self.tweet_dict = {}

    def download(self):
        data_folder = "tweets_dataset"
        self.dataset_path = os.path.join(self.data_path, data_folder)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            wget.download(self.url, self.dataset_path)
            print("Dataset successfully downloaded. Check path :: {}".format(self.dataset_path))
        else:
            print("Dataset already present. Check path :: {}".format(self.dataset_path))

    def extract_files(self, path):
        path_to_extract = os.path.join(os.path.dirname(path), "data")
        if not os.path.exists(path_to_extract):
            os.makedirs(path_to_extract)
        with ZipFile(path, "r") as zipObj:
            zipObj.extractall(path_to_extract)
        return path_to_extract

    def parse_files(self, path):
        data_files = glob(os.path.join(path, "*" + self.data_file_ext))
        for filename in data_files:
            print("Filename is:: {}".format(filename))
            file_obj = io.open(filename, "r", errors="ignore")
            lines = file_obj.readlines()
            lines = [x.strip().split("|") for x in lines]
            for line in lines:
                try:
                    self.tweet_dict.update({int(line[0]): line[2].split("http")[0].strip()})
                except ValueError:
                    continue

    def save(self):
        filename = "saved_data"
        filepath = os.path.join(self.dataset_path,filename)
        filehandler = open(filepath, 'wb')
        pickle.dump(self.tweet_dict, filehandler)

    def load(self):
        self.download()
        files = os.listdir(self.dataset_path)
        print(files)
        for filename in files:
            if filename.endswith(".zip"):
                filepath = os.path.join(self.dataset_path, filename)
                extracted_files_path = self.extract_files(filepath)
                print("Extracted file path is ::: {}".format(extracted_files_path))
                break

        extracted_files = os.listdir(extracted_files_path)
        for folder in extracted_files:
            if folder == "__MACOSX":
                continue
            self.parse_files(os.path.join(extracted_files_path, folder))
        print("Parsed data is :: {}".format(list(self.tweet_dict.items())[0:3]))


if __name__ == "__main__":
    tweet = TweetData()
    tweet.load()
    tweet.save()
