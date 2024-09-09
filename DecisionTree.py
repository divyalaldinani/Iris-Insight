#2 classes Node for each Node in the tree to define the struture of each node in the tree
#Decision Tree manages all connections bw nodes

#each node includes which feature it was divided with to future nodes(internal node), threshold for division, left and right nodes and value of node
import numpy as np # type: ignore
from collections import Counter

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split #min samples a node must have for further split
        self.max_depth = max_depth  #max depth of decision tree
        self.n_features = n_features # total number of features we can divide the tree upon
        self.root = None # root of DT
        



    def fit(self, X, y):
        #total number of features of DT <= number of features in data
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)


    
    def _grow_tree(self, X, y, depth = 0):

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y)) #diff possible values of y(labels)

    
        #split the nodes until one of the stopping criteria is reached or the node is pure
        if( depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            #create new leaf node with label = majority of labels
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        #find the best split among all
        features = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, features)


        #create child nodes and attach them
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)
            



    def _most_common_label(self, y):
        counter = Counter(y)
        #suppose y is a list with number of repeating labels so counter creates a dictionary that has label and its freq 
        most_frequent = counter.most_common(1) #returns a list with single tuple containing label with highest freq among the y labels
        return most_frequent[0][0]


    
    def _best_split(self, X, y, features ):
        best_gain = -1 #information gain
        split_idx, split_threshold = None, None #selects the feature and the value of that feature for which the best split occurs, X0 and threshold = 9, X0 <= 9 and X0 > 9, 2 splits will be created
        for feature in features:
            X_column = X[:, feature] #all possible values of that feature
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                #calculate the info gain and update best gain and split factors
                gain = self._information_gain(y, X_column, threshold)

                if (gain > best_gain):
                    best_gain = gain
                    split_idx, split_threshold = feature, threshold

        
        return split_idx, split_threshold



    def _information_gain(self, y, X, threshold):
        #IG = E(parent) -  ∑ prob(child) * E(Child)
        #∑ prob(child) * E(Child) = weighted avg of entropy of children
        #E = entropy = impurity in sample
        # entropy = - pi*log(pi) 
        #pi = probability of a particular label i in sample = sample with label i/total samples


        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X, threshold)
        #so now the samples in the current node will be split into 2 nodes that have feature's value <= threshold and feature's value > threshold

        if (len(left_idxs) == 0 or len(right_idxs) == 0) :
            return 0

        #calculate weight avg pf children's entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])
        #so here we have divided the samples based on their values of a particular features acc to threshold and for entropy we need labels or y values


        child_entropy = n_l/n * e_l + n_r/n * e_r


        #calculate the information gain 
        information_gain = parent_entropy - child_entropy
        return information_gain

        


    def _split(self, X_column, split_thres):
        left_idxs = np.argwhere(X_column <= split_thres).flatten()
        right_idxs = np.argwhere(X_column > split_thres).flatten()
        return left_idxs, right_idxs
        


        
    def _entropy(self, X):
        labels = np.unique(X)
        counter = Counter(X)
        entropy = 0
        for label in labels:
            if(counter[label] != 0 ) : #log 0 will give undefined
                pi = counter[label]/len(X)
                entropy += pi*np.log2(pi)

        return -entropy


    def predict(self, X):
         return np.array([self._traverse_tree(x, self.root) for x in X])




    def _traverse_tree(self, X, node):
        if (node.is_leaf_node()):
            return node.value


        #traversing the tree based on values in sample data
        if( X[node.feature] <= node.threshold):
            return self._traverse_tree(X, node.left)
        
        return self._traverse_tree(X, node.right)