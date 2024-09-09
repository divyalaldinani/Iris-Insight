from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees = 20, max_depth = 20, min_samples_split = 2, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth,
                            min_samples_split = self.min_samples_split,
                            n_features = self.n_features)
            
            X_samples, y_samples = self._bootstrap_samples(X, y)
            tree.fit(X_samples, y_samples) #call the decision tree to fit the samples 
            self.trees.append(tree)


    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0] #self n_samples number of random sample(with replacement) from the samples
        # np.random.choice(tota_samples, total-samples-to-be-selcted, replace = True/False)
        #replace = true means that a sample can be selcted more than once
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]
    


    def _most_common_label(self, y):
        counter = Counter(y)
        majority_label = counter.most_common(1)
        return majority_label[0][0]



    def predict(self, X):
        #This line iterates over each decision tree in self.trees and calls the predict method of each tree with the input data X.
        #there are n_trees number of rows and each row contains the label given by that tree for a particular sample
        predictions = np.array([tree.predict(X)for tree in self.trees])
        
        #transpose of matrix or you can try swapaxes
        predictions = np.transpose(predictions)
        final_predictions = np.array([self._most_common_label(pred) for pred in predictions])
        return final_predictions




