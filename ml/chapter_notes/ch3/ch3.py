from ml.mnist import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from ch3_includes import plot_precision_recall

mnist_db = MNISTDatabase()
mnist_db.load()

x_train, x_test = mnist_db.data[:10000], mnist_db.data[10000:]
y_train, y_test = mnist_db.labels[:10000], mnist_db.labels[10000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#sgd_classifier = SGDClassifier(random_state=42)
#scores = cross_val_score(sgd_classifier, x_train, y_train_5, cv=3, scoring='accuracy')
#y_scores = cross_val_predict(sgd_classifier, x_train, y_train_5, cv=3, method='decision_function')
#conf = confusion_matrix(y_train_5, y_train_predict)
#precision = precision_score(y_train_5, y_train_predict)
#recall = recall_score(y_train_5, y_train_predict)

#fig, ax = plot_precision_recall(y_train_5, y_scores, method='ROC')
#plt.show()

# Try out an SVM classifier for multi-class classification
svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)
y_scores = cross_
