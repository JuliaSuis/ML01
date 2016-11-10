import math
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

__author__ = 'devil'

from repository import Repository
from configuration import config
repository = Repository(config)
dataset, labels = repository.get_dataset_and_labels()

# Ensure that there are no NaNs
dataset = dataset.fillna(-85)
#dataset = dataset.truncate(after=39)
#labels = labels[:40]

for i in range(1,50,4):
    print "Number of features selected for this iteration :",i
    # Split the dataset into training (90 \%) and testing (10 \%)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1)



    # This dataset is way too high-dimensional. Better do PCA:
    pca = PCA(n_components=i)

    X_features = pca.fit(X_train, [repository.locations.keys().index(tuple(l)) for l in y_train]).transform(X_train)

    #print(X_features)

    svm = SVC(kernel="linear")

    pipeline = Pipeline([("pca", pca), ("svm", svm)])


    param_grid = dict(svm__C=[0.1])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, cv=2)

    #print grid_search

    grid_search.fit(X_train, [repository.locations.keys().index(tuple(l)) for l in y_train])

    print("Printing the score")
    ac = grid_search.score(X_test, [repository.locations.keys().index(tuple(l)) for l in y_test])
    print "accurancy= ", ac

     # calculating the error
    predictY = [repository.locations.keys()[i] for i in grid_search.predict(X_test)]
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