from sklearn.base import BaseEstimator
import numpy as np
from os import listdir
from os.path import join
from joblib import load

class GridOptimisedClassifier(BaseEstimator):

    def __init__(self, intValue=0, stringParam="Grid Optimised Injection Classifier", otherParam=None, gridPath=None, modelsPath=None):
        self.intValue = intValue
        self.stringParam = stringParam
        self.otherParam = otherParam
        self.name = 'Grid Optimised Injection Classifier'
        if gridPath is None:
            raise ValueError('Path to grids was not specified')
        if modelsPath is None:
            raise ValueError('Path to models was not specified')
        self.gridPath = gridPath
        self.modelsPath = modelsPath
        self.grids = None

    def getName(self):
        return self.name

    def fit(self, X, y):
        grids = []
        with open(self.gridPath, 'r') as fp:
            line = fp.readline()
            while line:
                splits = line.split(',')
                brackets = splits[0]
                brackets = brackets.replace(' ', '')
                brackets = brackets.replace('[', '')
                brackets = brackets.replace(']', '')
                nums = brackets.split('.')
                bbox = np.array([[int(nums[0]), int(nums[1])], [int(nums[2]), int(nums[3])]])
                grid = {"bbox": bbox, "classifier": splits[1]}
                grids.append(grid)
                line = fp.readline()
        self.grids = grids
        self.models = {
            'AdaBoostClassifier': load(join(self.modelsPath, 'AdaboostClassifier.joblib')),
            'BaggingClassifier': load(join(self.modelsPath, 'BaggingClassifier.joblib')),
            'DecisionTreeClassifier': load(join(self.modelsPath, 'DecisionTreeClassifier.joblib')),
            'GaussianNB': load(join(self.modelsPath, 'GaussianNB.joblib')),
            'KNeighborsClassifier': load(join(self.modelsPath, 'KNeighborsClassifier.joblib')),
            'MLPClassifier': load(join(self.modelsPath, 'MLPClassifier.joblib')),
            'RandomForestClassifier': load(join(self.modelsPath, 'RandomForestClassifier.joblib'))
        }

    def predict(self, X):
        rows = X.shape[0]
        predictions = np.ndarray(shape=(rows, 1), dtype='<U16')
        num = 0
        for row in X:
            for cov in self.grids:
                bbox = cov['bbox']
                if ((bbox[0,1] <= row[0]) & (bbox[1,1] >= row[0]) & (bbox[0,0] <= row[1]) & (bbox[1,0] >= row[1])):
                    train = row.reshape(1,-1)
                    try:
                        blerp = self.models[cov['classifier']].predict(train)
                        predictions[num] = blerp[0]
                    except KeyError:
                        blerp = self.models['RandomForestClassifier'].predict(train)
                        predictions[num] = blerp[0]
                    num = num + 1
                    print(predictions[num])
                    print(f'Predicted Point {num} of {rows}      \r', end='')
                    break
        return predictions