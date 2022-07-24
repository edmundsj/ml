import os
from ml import datasets_path, MNISTDownloader, Trainer
from sklearn.neighbors import KNeighborsClassifier


data_path = os.path.join(datasets_path, 'mnist')
mnist = MNISTDownloader(data_path)
dataset = mnist.fetch()
trainer = Trainer(dataset)
trainer.train_test_split()

knn_clf = KNeighborsClassifier()
trainer.fit(knn_clf)
predictions = trainer.predict()
breakpoint()