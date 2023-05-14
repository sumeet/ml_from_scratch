import numpy as np
from sklearn.utils import resample
from abc import abstractmethod
from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        for num in range(self.n_estimators):
            inb_index = resample(range(len(X)), n_samples=round(0.66 * len(X)))
            inb_X = X[inb_index]
            inb_y = y[inb_index]
            oob_index = np.array(list(set(range(len(X))) - set(list(inb_index))))
            oob_X = X[oob_index]
            oob_y = y[oob_index]
            T = self.tree()
            T.fit(inb_X,inb_y)
            self.trees.append(T)
            
            if self.oob_score:
                self.oob_score_ = self.calculate_oob(oob_X,oob_y)


            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        leaves = []
        for tree in self.trees:
            leaves.append(tree.predict(X_test))
        
        y_hat = []
        for row in range(len(X_test)):
            nobs = 0
            ysum = 0
            for leaf in leaves:
                nobs += leaf[row].n
                ysum += leaf[row].n * leaf[row].prediction
            y_hat.append(ysum/nobs)
        return np.array(y_hat)

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        yhat = self.predict(X_test)
        return r2_score(y_test, yhat)

    def tree(self):
        return RegressionTree621(self.min_samples_leaf, self.max_features)


    def calculate_oob(self, X, y):
        oob_counts = np.zeros(len(X))
        oob_predicts = np.zeros(len(X))

        for tree in self.trees:
            predicted_leaves = tree.predict(X)
            leaves_size = np.array([leaf.n for leaf in predicted_leaves])
            leaves_prediction = np.array([leaf.prediction for leaf in predicted_leaves])
            oob_predicts += leaves_size * leaves_prediction
            oob_counts += leaves_size
        
        yhat = oob_predicts / oob_counts

        self.oob_score_ = r2_score(y, yhat)
            

        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

    def predict(self, X_test) -> np.ndarray:
        leaves = []
        for tree in self.trees:
            leaves.append(tree.predict(X_test))
        yhat = []
        for row in range(len(X_test)):
            cat_counts = {}
            for leaf in leaves:
                cat = leaf[row].prediction[0]
                if cat in cat_counts:
                    cat_counts[cat] += 1
                else:
                    cat_counts[cat] = 1
            category_highest = (0,0)
            for key,value in cat_counts.items():
                if value >= category_highest[1]:
                    category_highest = (key,value)
            yhat.append(category_highest[0])
        return np.array(yhat)

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_hat = self.predict(X_test)
        return accuracy_score(y_test, y_hat)


    def tree(self):
        return ClassifierTree621(self.min_samples_leaf, self.max_features)
    
    def calculate_oob(self, X,y):
        unique_y = np.unique(y)
        oob_counts = np.zeros(len(X))
        oob_predicts = np.zeros((len(X), len(unique_y)))

        for tree in self.trees:
            predicted_leaves = tree.predict(X)
            leaves_size = np.array([leaf.n for leaf in predicted_leaves])
            leaves_prediction_index = np.array([list(unique_y).index(leaf.prediction) for leaf in predicted_leaves])
            oob_predicts[range(len(X)),leaves_prediction_index] += leaves_size
            oob_counts += leaves_size
        
        oob_votes = np.zeros(len(X))
        for i, _ in enumerate(oob_counts):
                oob_votes[i] = unique_y[np.argmax(oob_predicts[i])]

        self.oob_score_ = accuracy_score(y, oob_votes)
            