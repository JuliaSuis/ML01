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


for i in range(3, 10):
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
     # calculating the error
    predict_locations = [repository.locations.keys()[i] for i in neigh.predict(X_test)]
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
    print ("error= ", error)
