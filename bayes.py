from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import math
from repository import Repository
from configuration import config

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
dataset = dataset.fillna(minOfLevels)#replace nan levels with ...
clf = GaussianNB()

#split training data and test data
dataset_train, dataset_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.1)

#learn model with train data
l_train = [list(repository.locations.keys()).index(tuple(l)) for l in label_train]
clf.fit(dataset_train, l_train)

#test model with test data and report the acuracy
l_test = [list(repository.locations.keys()).index(tuple(l)) for l in label_test]
ac = clf.score(dataset_test,l_test)

print("accurancy= ", ac)
#-85 => 0.139143730887
#missing data = avg of level => 0.16,0.14
#missing data = min of level => 0.165,0.149
#missing data = 0 => 0.14

predict_locations_indexes = clf.predict(dataset_test) # it gives the level information and returns the index of label information,
                                                        # the index of location
real_locations = list(repository.locations.keys())
predict_locations = [real_locations[i] for i in predict_locations_indexes] # we retrive the location by index

for lat, long in predict_locations: # get the location label of predicted data
    predictLat = lat
    predictLong = long

for lat, long in label_test: # get the label location of test data
    RealLat = lat
    RealLong = long



print (predictLat)
DiffLat = predictLat - RealLat
DiffLong = predictLong - RealLong
Cst = math.pi / 180
R = 6378.1  # Radius of the Earth

Em = R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2))
print("error= ", Em)

#
# label_test_index = 0
# for i in predict_locations_indexes: # get the location label of predicted data
#     predict_location = real_locations[i]
#     reallocation = label_test[label_test_index]
#     label_test_index = label_test_index + 1
#
# for lat, long in label_test: # get the label location of test data
#     RealLat = lat
#     RealLong = long
#
# DiffLat = predictLat - RealLat
# DiffLong = predictLong - RealLong
# Cst = math.pi / 180
# R = 6378.1  # Radius of the Earth
#
# Em = R * Cst * math.sqrt(math.pow(DiffLat, 2) + math.pow(DiffLong, 2))
# print("error= ", Em) #0.03721078098007929
