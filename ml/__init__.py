import os
base_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(base_path, 'datasets')

from ml.datasets import Dataset
from ml.downloader import Downloader
from ml.chapter_notes.ch1.ch1_functions import fetch_housing_data, load_housing_data
from ml.mnist import MNISTDownloader
from ml.trainers import Trainer
