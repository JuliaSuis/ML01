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

dataset = dataset.fillna(-85)
size = dataset.size

#PCA (Principal Components Analysis ) to improve acc
def use_PCA(dataset):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dataset.shape[1])
    dataset = pca.fit_transform(dataset)

use_PCA(dataset)
dataset = dataset.truncate(after=299)
labels= labels[:300]

print dataset


# Split the dataset into training (90 \%) and testing (10 \%)
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.1)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# Define the classifier to use
#estimator = SVC(kernel="linear",cache_size=310)
estimator = SVC(kernel="linear")

# Define parameter space.
gammas = np.logspace(-6, -1, 10)

# Use Test dataset and use cross validation to find best hyper-parameters.
CV_classifier = GridSearchCV(estimator=estimator, cv=cv ,param_grid=dict(gamma=gammas))
CV_classifier.fit(X_train, [repository.locations.keys().index(tuple(l)) for l in y_train])

# Test final results with the testing dataset

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

find_accurancy(CV_classifier)
find_error(CV_classifier)
