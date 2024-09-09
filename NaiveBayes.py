import numpy as np # type: ignore

class NaiveBayesClassfier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probabilities = {}
        self.conditional_probabilities = {}
        self.class_counts = {}


    def fit(self, X, y):
        # P(y | x ) = P( x | y) P(y) /P(x)
        # P(x | y) - we have to calculate the prob of value of each feature given a certain label
        # P( Xk = xi | y = yj ), for each label, calculate the value of a certain feature 



        # P(y = yi) - prob of having a certain label

        #P(x) = nx + alpha/ ( total + alpha * nvals)
        #nx = number of ocurences of x in X
        #alpha = smoothing parameter
        #nvals = number of possible values of feature
        # total = total training examples
        self.class_labels = np.unique(y)

        self.features = X.shape[1]
        class_label_probabilities = {} #dictionary
        conditional_probability = {}
        for class_label in self.class_labels:
            total = np.sum( y == class_label ) 
            class_label_probabilities[class_label] = (total + self.alpha)/(len(y) + self.alpha * len(self.class_labels))
            
            #indices of examples with current class label
            samples_with_label = np.where( y == class_label)[0] 
            for feature_idx in range(self.features):
                feature_values = np.unique(X[:, feature_idx])
                for value in feature_values:
                    nx = np.sum(X[samples_with_label, feature_idx] == value)
                    conditional_probability[(class_label, feature_idx, value)] = (nx + self.alpha)/( total + self.alpha * len(feature_values))


        self.conditional_probabilities = conditional_probability
        self.class_label_probabilities = class_label_probabilities


        
    def predict(self, X):
        predicted_labels = []
        for test_example in X:
            probabilities = {}
            for class_label in self.class_labels:
                probability = self.class_label_probabilities[class_label]
                for feature_idx in range(self.features):
                    feature_value = test_example[feature_idx]

                    #calculate P( x | y ) if not given
                    probability = probability * self.conditional_probabilities.get(
                        (class_label, feature_idx, feature_value),  # Use parentheses for a tuple
                        self.alpha / (self.class_label_probabilities[class_label] + self.alpha * len(np.unique(X[:, feature_idx])))
                    )
                probabilities[class_label] = probability

            class_label_with_max_probability = max(probabilities, key = probabilities.get)
            predicted_labels.append(class_label_with_max_probability)
        
        return np.array(predicted_labels)

        