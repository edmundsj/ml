import kaggle
import os
from ml import Dataset
from sklearn.datasets import fetch_openml
import json
import pandas as pd


class Downloader:
    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        self.data_file = os.path.join(data_dir, 'data.db')
        self.label_file = os.path.join(data_dir, 'labels.db')
        self.metadata_file = os.path.join(data_dir, 'metadata.db')
        self.dataset = Dataset()

    def fetch(self, db_name=None, version=None, meta_filename=None, data_filename=None, label_filename=None):
        """
        Looks for the MNIST database in a local database and returns it. If not present, fetches from server

        """
        if self.is_cached:
            dataset = self._fetch_local(meta_filename=meta_filename, data_filename=data_filename, label_filename=label_filename)
        else:
            print('Not found in local cache. Retrieving from server...')
            dataset = self._fetch_server(db_name=db_name, version=version)
            self.cache()

        self.dataset = dataset
        return self.dataset

    def _fetch_server(self, db_name, version=1):
        db = fetch_openml(db_name, version=version)
        data = db['data']
        labels = db['target']
        metadata = {k: v for k, v in db.items() if k not in ['data', 'target', 'frame']}
        dataset = Dataset(data=data, labels=labels, metadata=metadata)
        return dataset

    def _fetch_local(self, meta_filename=None, data_filename=None, label_filename=None):
        """
        Fetches the database from a local cache
        """
        if meta_filename is None:
            meta_filename = self.metadata_file
        if data_filename is None:
            data_filename = self.data_file
        if label_filename is None:
            label_filename = self.label_file

        with open(meta_filename, 'r') as meta_file:
            metadata = json.load(meta_file)
            data = pd.read_csv(data_filename)
            labels = pd.read_csv(label_filename).squeeze()
            dataset = Dataset(data=data, labels=labels, metadata=metadata)
            return dataset

    @property
    def is_cached(self):
        if os.path.exists(self.data_file) and os.path.exists(self.label_file) and os.path.exists(self.metadata_file):
            return True
        else:
            return False

    def cache(self):
        with open(self.metadata_file, 'w') as meta_file:
            json.dump(self.dataset.metadata, meta_file)
        self.dataset.data.to_csv(self.data_file, index=False)
        self.dataset.labels.to_csv(self.label_file, index=False)
