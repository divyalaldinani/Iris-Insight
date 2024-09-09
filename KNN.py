import numpy as np # type: ignore
from collections import Counter


class KNN:
    def __init__(self, k = 5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test): #make prediction for a new point
        predictions = np.array([self._make_predictions(x_test) for x_test in X_test] )
        return predictions


    def euclidean_distance(self, x1, x2 ):
        return np.sqrt(np.sum(x1-x2)**2) #each x1 and x2 has a number of features so sqrt(summation (x1,fi - x2,fi))


    def _make_predictions(self, x):
    
        #calculate distance from all points
        distances = np.array([self.euclidean_distance(x, x_train) for x_train in self.X_train])

        #calculate k nearest distance
        #argsort returns the indices of points from a list with inc distances, i.e., instead of sorting the array it 
        #returns the indices of the elements that are there in sorted array from org array
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        majority_label = Counter(k_nearest_labels).most_common()


        return majority_label[0][0] 