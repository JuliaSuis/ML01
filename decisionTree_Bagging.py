import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from IPython.display import Image
import pydotplus

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

#principle component analysis
def use_PCA(dataset):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dataset.shape[1])
    dataset = pca.fit_transform(dataset)

#use_PCA(dataset)

# Split the dataset into training (90 \%) and testing (10 \%)
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.1)

#Decission Tree Classifier
dt_classifier.fit(X_train, [repository.locations.keys().index(tuple(i)) for i in y_train])

#Bagging Classifier
bag_classifier.fit(X_train, [repository.locations.keys().index(tuple(i)) for i in y_train])

def find_accurancy(classifier):
    acc = classifier.score(X_test, [repository.locations.keys().index(tuple(i)) for i in y_test])
    print ("accurancy = ", acc)


def find_error(classifier):
    predict_locations  = [repository.locations.keys()[i] for i in classifier.predict(X_test)]
    label_test = y_test

    sumOfErr = 0
    length_of_test_data = len(label_test)
    for i in range(0, length_of_test_data):
        real_loc = label_test[i]
        RealLat = real_loc[0]
        RealLong = real_loc[1]
        predict_loc = predict_locations[i]
        predictLat = predict_loc[0]
        predictLong = predict_loc[1]

        DiffLat = predictLat - RealLat
        DiffLong = predictLong - RealLong
        Cst = math.pi / 180
        R = 6378.1  # Radius of the Earth
        sumOfErr = sumOfErr + (R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2)))

    error = sumOfErr / length_of_test_data;
    print("error", error)  

find_accurancy(dt_classifier)
find_error(dt_classifier)

find_accurancy(bag_classifier)
find_error(bag_classifier)

