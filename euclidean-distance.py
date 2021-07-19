import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1], s=100, color=i)
##
###in one line
#[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1])
#plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total warning groups!")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    #for i in sorted(distances)[:k]:
     #   i[1] or thus
    votes = [i[1] for i in sorted(distances) [:k]]
    print(Counter(votes).most_common (1))
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

#lets visualize
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result)
plt.show()


#lets apply read world dataset to our KNN algorithm
df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
#this converts the type to float in case there are quotes around the integers
full_data = df.astype(float).values.tolist()
print(full_data[:5])

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
#so this is everything up to the last 20%
train_data = full_data[:-int(test_size*len(full_data))]
#this will be the last 20%
test_data = full_data[-int(test_size*len(full_data)):]

#populate the dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])


#lets create a counter
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote - k_nearest_neighbors(train_set, daya, k=5)
        if group == vote:
            correct += 1
        total =+ 1
print("Accuracy: ", correct/total)
