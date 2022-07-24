import sklearn
from sklearn.model_selection import train_test_split
from ml import Dataset
"""
This follows the generic pattern you should always follow:
- First, before doing anything, split data into test data and train data.
"""


class Trainer:

    def __init__(self, dataset):
        # First, split the dataset into "test" and "train" partitions
        self.input_dataset = dataset
        self.train_dataset = None
        self.test_dataset = None
        self.classifier = None

    def train_test_split(self, train_size=0.7, random_state=None, shuffle=True):
        x_train, x_test, y_train, y_test = train_test_split(self.input_dataset.data, self.input_dataset.labels,
                         train_size=train_size, random_state=random_state, shuffle=shuffle)

        self.train_dataset = Dataset(data=x_train, labels=y_train, metadata=self.input_dataset.metadata)
        self.test_dataset = Dataset(data=x_train, labels=y_train, metadata=self.input_dataset.metadata)

    def fit(self, classifier):
        self.classifier = classifier
        self.classifier.fit(self.train_dataset.data, self.train_dataset.labels)

    def predict(self, data=None):
        if data is None:
            data = self.train_dataset.data

        self.predictions = self.classifier.predict(data)
        return self.predictions
