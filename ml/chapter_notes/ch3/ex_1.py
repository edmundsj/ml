from ml import MNISTDatabase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

mnist_db = MNISTDatabase()
mnist_db.load()

x_train, x_test = mnist_db.data[:60000], mnist_db.data[60000:]
y_train, y_test = mnist_db.labels[:60000], mnist_db.labels[:60000]

knn_classifier = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
}

grid_search = GridSearchCV(knn_classifier, param_grid, cv=3)
grid_search.fit(x_train, y_train)

# Question: how do we try a bunch of different hyperparameters and find the score of that model?
# I know we start with something like a grid search, but how do we actually do this?

# Also, how do we evaluate a model on the training data directly, not using cross-validation?
# You can do this with the .score() method of a classifier, or just compute the error directly using .transform() and compute the number of errors
