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



#error Calculation
predict_locations_indexes = clf.predict(dataset_test) # it gives the level information and returns the index of label information,                                                        # the index of location
real_locations = list(repository.locations.keys())
predict_locations = [real_locations[i] for i in predict_locations_indexes] # we retrive the location by index


#train set = {train set of level, train set of locations}
#test set = {test set of level, test set of locations}

#train a model with training set
#based on model predict locations for test set of levels, then compare the rsult with test test of locations

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
print ("error",error) # 0.009606909422648182
