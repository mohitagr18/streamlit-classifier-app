import pandas as pd
import numpy as np
from collections import Counter

########################## NAIVE BAYES #########################################

class NaiveBayes():

    def __init__(self):
        pass


    def fit(self, X, y):
        '''
        1. Identify unique classes and their number in target variable

        2. Initialize mean, variance and prior probability vectors
            - Since mean and variance are calculated for each feature in each class,
              their shape will be (no. of classes, no. of features)
            - Since prior is just the probability of each class, it remains a 1-D array

        3. Iterate over each class and
            - Subset the data for class
            - Compute mean, variance and prior probability for each class
            - Add these values to initialized arrays at the same time
        '''

        # Get no. of records and features
        n_records, n_features = X.shape

        # Identify unique classes and their number in target variable
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, variance and prior probability vectors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._prior = np.zeros(n_classes, dtype = np.float64)

        # Iterate over each class to compute mean, variance and prior
        for c in self._classes:
            X_class = X[c==y]
            self._mean[c,:] = np.mean(X_class, axis=0)
            self._var[c,:] = np.var(X_class, axis=0)
            self._prior[c] = len(X_class) / n_records


    def predict(self, X):
        '''
        Returns the predicted class given test data
        '''
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self, x):

        '''
        1. Initialize a list of posterior prob for each class

        2. Calculate posterior probability for each class using argmax formula:
          - get the prior probability of class from fit method
          - calculate conditional probability
          - calculate posterior and append to list of posterior probabilities

        3. Select class with highest posterior
        '''
        # Initialize posteriors
        posteriors = []

        # Iterate over each class and calculate posterior using PDF function
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + conditional
            posteriors.append(posterior)

        # Select class with highest posterior
        final_cls = self._classes[np.argmax(posteriors)]
        return final_cls


    def _pdf(self, cls_idx, x):
        '''
        Calculates conditional probability using Gaussian PDF formula
        '''
        mean = self._mean[cls_idx]
        var = self._var[cls_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

########################## K NEAREST NEIGHBOR #########################################

class My_KNN():
    '''
    Fits and predicts a KNN model given data values and true labels
    '''

    def __init__(self):
        pass

    def fit(self, X, y, k=3, p=2):
        '''
        Initializes the:
        - training data and their class labels
        - k for no of neighbors
        - p for distance type: 1: Manhattan, 2: Euclidean
        '''
        self.X_train = X
        self.y_train = y
        self.k = k
        self.p = p

    def predict(self, X):
        '''
        Generates predictions given a set of new points
        '''
        # Iterate over each point and call _predict on each point
        pred_cls = [self._predict(x, self.p) for x in X]
        return np.array(pred_cls)

    def _predict(self, x, p):
        '''
        Given a new data point and distance type:
        1. Computes distance from new point to each point in the data
        2. Sorts the distances in descending order
        3. Subsets the indices for first `k` distances
        4. Using indices, get class labels for those `k` points
        5. Identify the most common class label out of the k data points selected
        '''
        # Compute distance
        dist = [self._compute_dist(x, train_x, p) for train_x in self.X_train]

        # Get k nearest sample indexes and their labels
        k_indices = [np.argsort(dist)[:self.k]]
        k_labels = [self.y_train[i] for i in k_indices[0]]


        # Compute majority vote
        maj_vote = Counter(k_labels).most_common(1)
        return maj_vote[0][0]


    def _compute_dist(self, x1, x2, p):
        '''
        Computes Manhattan (p=1) or Euclidean(p=2) distance between 2 points
        '''
        dist = np.sum(np.abs(x1 - x2)**p)
        return dist**(1/p)

################### SUPPORT VECTOR CLASSIFIER ###################################

class My_SVC():

    def __init__(self):
        self.b = None
        self.w = None


    def _cost(self,y,X,w,b,lambd):
        '''
        computes cost = hinge loss + max margin
        hinge loss = (1/n) * sum(max(0, 1 - y_i(w*x_i + b)))
        max margin = (lambd/2) * ||w||^2
        '''
        loss = 0
        for i in range(X.shape[0]):
            z = y[i] * (np.dot(self.w, X[i]) + self.b)
            loss += max(0, 1-z)
        hinge = loss/len(X)
        cost = lambd * 0.5 * np.dot(w, w.T) + hinge
        return cost


    def fit(self, X, y, learning_rate=0.01, epochs=100, lambd=0.1, verbose=False):
        n_obs, m_features = X.shape
        total_loss = []
        total_cost = []

        # convert y values to be -1 or 1
        y_new = np.where(y <= 0, -1, 1)

        # initialize weights and bias
        self.b = 0
        self.w = np.random.rand(m_features)

        # iterate over epochs
        for e in range(epochs):
            cost = self._cost(y_new,X,self.w,self.b,lambd)  # get cost

            # Calculate gradient and update weights, bias
            for i in range(X.shape[0]):
                z = y_new[i] * (np.dot(self.w, X[i]) + self.b)
                if z >= 1:
                    self.w -= learning_rate * (lambd * self.w)

                else:
                    self.b -= learning_rate * y[i]
                    self.w -= learning_rate * ((lambd * self.w) - np.dot(X[i], y_new[i]))


            total_loss.append(cost)

            if verbose == True and e%10 == 0:
                print(f'Epoch: {e}, Loss: {cost}')


        return self.b, self.w, total_loss


    def predict(self, X):
        pred = np.dot(self.w, X.T) + self.b
        return np.sign(pred)


def calc_entropy(y):
    '''
    Calculates the entropy for a single node
    '''
    # Get total count of each class in y
    y_classes = np.bincount(y)  # returns array([54, 89])

    # Divide class count by length
    y_hist = y_classes/len(y)

    # calculate entropy
    entropy = -np.sum([p * np.log2(p) for p in y_hist if p > 0])
    return entropy

class Node:
    """
    Stores information such as its feature, split threshold, left node, right node
    and value if node is a leaf node.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """
        determines if a node is a leaf node
        """
        return self.value is not None

########################## DECISION TREE #########################################

class My_Tree():
    """
    fits the decision tree model and makes predictions
    """
    def __init__(self):
        self.root = None

    def fit(self, X, y, min_split_samples=2, max_depth=100, n_feats=None, replace=False):
        """
        creates a root node and starts growing the tree
        """
        # Initialize
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.n_feats = n_feats

        # Subset number of features
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Checks for stopping criteria
        i. If met, returns the `_most_common()` value for node
        ii. If not met:
            a. Randomly choose features to split on
            b. Find the best feature and threshold value using `_best_criteria()`
            c. Partition the data on best feature and threshold using `_create_split()`
            d. Recursively grow the left and right trees using `_grow_tree()`
            e. Return `Node` with best feature, threshold, left and right value
        """
        n_records, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check for stopping criteria
        if (depth >= self.max_depth
           or n_records < self.min_split_samples
           or n_classes == 1):
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)

        # Randomly choose feature indices to split on
        feat_indxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Find the best feature and threshold value
        best_feat_idx, best_thresh = self._best_criteria(feat_indxs, X, y)

        # Split left and right indices
        left_idxs, right_idxs = self._create_split(X[:,best_feat_idx], best_thresh)

        # Grow left and right trees
        left_tree = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right_tree = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)

        # Return `Node` with best feature, threshold, left and right value
        return Node(best_feat_idx, best_thresh, left_tree, right_tree)

    def _best_criteria(self, feat_indxs, X, y):
        """
        i. Iterates over all features and unique feature values
        ii. Calculates `_information_gain()` using a feature and unique value as threhold
        iii. Identifies largest gain and returns best feature and threhold value
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        # Iterate over features and their unique values to
        # identify the best feature and its split threhold
        for idx in feat_indxs:
            X_col = X[:, idx]
            unique_vals = np.unique(X_col)
            for thresh in unique_vals:
                gain = self._information_gain(X_col, y, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, X_col, y, thresh):
        """
        i. Calculates parent entropy - entropy(y) for the feature using `_calc_entropy()`
        ii. Creates split using `_create_split()` based on feature and threshold
        iii. Compute entropy for indices on left and right splits using `_calc_entropy()`
        iv. Calculate weighted avg. of entropy for splits (left and right children)
        v. Return Information Gain = Parent entropy - Wt. avg of entropy for children
        """
        # Calculate parent entropy
        ent_parent = calc_entropy(y)

        # Create split
        left_idx, right_idx = self._create_split(X_col, thresh)

        # Calculate weighted avg. entropy of left and right indices
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        if n_l == 0 or n_r == 0:
            return 0
        left_ent, right_ent = calc_entropy(y[left_idx]), calc_entropy(y[right_idx])
        wt_avg_ent = (n_l/n) * left_ent + (n_r/n) * right_ent

        # Calculate information gain
        gain = ent_parent - wt_avg_ent
        return gain

    def _create_split(self, X_col, thresh):
        """
        i. Compute left indices - indices where feature value >= threshold
        ii. Compute right indices - indices where feature value < threshold
        iii. Return left and right indices
        """
        left_idx = np.argwhere(X_col >= thresh).flatten()
        right_idx = np.argwhere(X_col < thresh).flatten()
        return left_idx, right_idx

    def _most_common(self,y):
        """
        returns the most common value in a node
        """
        count = Counter(y)
        common = count.most_common(1)[0][0]
        return common

    def predict(self, X):
        '''
        iterates over test data and traverses the tree using `_traverse_tree()`
        '''
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        i. Checks if the node is leaf and returns value
        ii. If not, checks if the test data value of best feature for the node >= threshold
          a. If so, return `_traverse_tree()` again for that feature and left of Node
          b. Else, return `_traverse_tree()` again for that feature and right of Node
        """

        if node.is_leaf():
            return node.value

        if x[node.feature] >= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


########################## RANDOM FOREST #########################################

class My_RandomForest():
    def __init__(self):
        self.trees = []

    def fit(self, X, y, n_trees=100, min_split_samples=2, max_depth=100, n_feats=None, replace=False):
        """
        i. Initializes the following parameters:
            a. n_trees = no of trees to build
            b. min_split_samples = min. samples required by a node to create a split
            c. max_depth = max. no. of nodes in a tree
            d. n_feats = number of features to randomly sample and pass to each tree
            e. replace = when True randomly samples features with replacement
        ii. Iterates over each tree to:
            a. Randomly sample the data with replacement using `_bootstrap_agg()`
            b. Fit a tree using `My_Tree` class imported from Decision Tree
        """
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.replace = replace
        self.trees = []

        for _ in range(self.n_trees):
            tree = My_Tree()
            x_sub, y_sub = self._bootstrap_agg(X, y)
            tree.fit(x_sub, y_sub, min_split_samples=min_split_samples,
                    max_depth=max_depth, n_feats=n_feats, replace=replace)
            self.trees.append(tree)

    def _bootstrap_agg(self, X, y):
        """
        randomly samples the data with replacement and returns it
        """
        n_records = X.shape[0]
        idxs = np.random.choice(n_records, size=n_records, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """
        i. Iterates over different trees created
        ii. Uses `My_Tree.predict()` to generates predictions for each tree
        iii. Uses `_most_common()` to return the most common value as an array

        `tree_preds` returns an array of predictions generated by each tree for each record in test data.
        For e.g. a test data with 4 records when iterated over 3 trees will result in labels generated
        by each tree [1111 0000 1111]. The result we want is the most common label for each test record i.e.
        [101 101 101 101]. This shows that for 1st test record, 1st tree predicted 1, second: 0 and third tree: 1.
        To get the result in this form, we use `swapaxes()`.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common(pred) for pred in tree_preds]
        return np.array(y_pred)

    def _most_common(self, y):
        """
        returns the most common value in a node
        """
        count = Counter(y)
        common = count.most_common(1)[0][0]
        return common
