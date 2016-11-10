import math
from sklearn.neighbors import KNeighborsClassifier

__author__ = 'devil'
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV

from repository import Repository
from configuration import config
repository = Repository(config)
dataset, labels = repository.get_dataset_and_labels()


for i in range(1, 10):
    print "i is",i
    # Ensure that there are no NaNs
    dataset = dataset.fillna(-85)

    #dataset = dataset.truncate(after=19)
    #labels = labels[:20]

    #print("Dataset", dataset)

    neigh = KNeighborsClassifier(n_neighbors=i, weights="distance")



    # Split the dataset into training (90 \%) and testing (10 \%)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1)

    #print("X_train", X_train)
    #print("X_test", X_test)
    #print("y_train", y_train)
    #print("y_test", y_test)




    # Use Test dataset and use cross validation to find best hyper-parameters.

    neigh.fit(X_train, [repository.locations.keys().index(tuple(l)) for l in y_train])

    # Test final results with the testing dataset
    #print("y_Test", y_test)
    ac = neigh.score(X_test, [repository.locations.keys().index(tuple(l)) for l in y_test])
    print "accurancy= ", ac

    # calculating the error
    predictY = [repository.locations.keys()[i] for i in neigh.predict(X_test)]
    # print predictY
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
    R = 6378.1  # Radius of the Earth

    Em = R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2))
    print "error= ", Em