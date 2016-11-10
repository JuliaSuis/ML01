
from repository import Repository
from configuration import config
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import math


repository = Repository(config)
dataset, labels = repository.get_dataset_and_labels()

#labels [loc1,loc2,....,loc6540]
# loc = {lat,long}

#dataset [fingerpoints of loc1,fingerpoints of loc2,....,finger points of loc6540]
#fingerpoints = {mac-ap1 : level, mac-ap2 : level, mac-ap3: level ......}
#dataset = [ {mac-ap1 : NAN, mac-ap2: 73 ,mac-ap3: 23 ,mac-ap4: NAN,...},
#            {mac-ap1 : 76, mac-ap2: NAN ,mac-ap3: NAN ,mac-ap4: NAN,...},
#            {mac-ap1 : -76, mac-ap2: 109,mac-ap3: NAN ,mac-ap4: NAN,...},
#          ]

minOfLevels = -102
avgOflevels = -72
dataset = dataset.fillna(minOfLevels)#replace nan levels with -85,-72,102
clf = LogisticRegression()

#split training data and test data
dataset_train, dataset_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.1)

#learn model with train data
l_train = [list(repository.locations.keys()).index(tuple(l)) for l in label_train]

clf.fit(dataset_train, l_train)

#test model with test data and report the acuracy
l_test = [list(repository.locations.keys()).index(tuple(l)) for l in label_test]

ac = clf.score(dataset_test,l_test)

print("accurancy= ", ac) # 0.24


predictY = [list(repository.locations.keys())[i] for i in clf.predict(dataset_test)]


for lat, long in predictY: # get the location label of predicted data
    predictLat = lat
    predictLong = long

for lat, long in label_test: # get the label location of test data
    RealLat = lat
    RealLong = long

DiffLat = predictLat - RealLat
DiffLong = predictLong - RealLong
Cst = math.pi / 180
R = 6378.1  # Radius of the Earth

Em = R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2))
print("error= ", Em)
