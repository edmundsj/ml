# TODO: Figure out how to efficiently save the MNIST database
from ml import Downloader
import os


class MNISTDownloader(Downloader):

    def fetch(self, db_name='mnist_784', version=1, meta_filename=None, data_filename=None, label_filename=None):
        """
        Fetches the MNIST database from a server and returns said value dictionary with the data, labels, and metadata
        """
        return super().fetch(db_name=db_name, version=version, meta_filename=None, data_filename=None, label_filename=None)

