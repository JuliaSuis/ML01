import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from repository import Repository
from configuration import config
import math

repository = Repository(config)
dataset, labels = repository.get_dataset_and_labels()

from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
bag_classifier = BaggingClassifier(dt_classifier, n_jobs=7)

# Ensure that there are no NaNs
dataset=dataset.fillna(-85)

def use_PCA(dataset):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dataset.shape[1])
    dataset = pca.fit_transform(dataset)

use_PCA(dataset)

# Split the dataset into training (90 \%) and testing (10 \%)
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.1)

#run Decission Tree Classifier
dt_classifier.fit(X_train, [repository.locations.keys().index(tuple(i)) for i in y_train])
#run Bagging Classifier
bag_classifier.fit(X_train, [repository.locations.keys().index(tuple(i)) for i in y_train])

def find_accurancy(classifier):
    acc = classifier.score(X_test, [repository.locations.keys().index(tuple(i)) for i in y_test])
    print ("accurancy = ", acc)


def find_error(classifier):
    predictY = [repository.locations.keys()[i] for i in classifier.predict(X_test)]
    #print  ("predictY: ", predictY)
    for lat, long in predictY:
        predictLat = lat
        predictLong = long
    # print y_test
    for lat, long in y_test:
        RealLat = lat
        RealLong = long
    DiffLat = predictLat - RealLat
    DiffLong = predictLong - RealLong
    Cst = math.pi / 180
    R = 6378.1

    Em = R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2))
    print ("error = ", Em)

find_accurancy(dt_classifier)
find_error(dt_classifier)

find_accurancy(bag_classifier)
find_error(bag_classifier)

