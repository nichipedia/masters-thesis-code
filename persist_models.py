from nrl_commons.mls.grids.readers import GGG_fs_grid_reader as reader
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from joblib import dump, load

def generateClassifiers():
    clfs = [
        #KNeighborsClassifier(3),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
            #GaussianProcessClassifier(1.0 * RBF(1.0)),
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #MLPClassifier(alpha=1, max_iter=1000),
        #AdaBoostClassifier(),
        #GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    return clfs


def generateEnsembles():
    norms = generateClassifiers()

    clfs = [
        #BaggingClassifier(RandomForestClassifier(max_depth=10, n_estimators=20, max_features=20, n_jobs=-1), max_samples=0.5, max_features=0.5, n_jobs=-1),
        #BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_jobs=-1),
        #RandomForestClassifier(max_depth=10, n_estimators=20, max_features=20, n_jobs=-1),
        #RandomForestClassifier(max_depth=20, n_estimators=20, max_features=30, n_jobs=-1),
        #GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=10, random_state=41),
        #VotingClassifier(estimators=[('knn',norms[0]), ('dtc',norms[1]), ('mlp',norms[2]), ('ada',norms[3]), ('gau',norms[4]), ('qua',norms[5])], voting='hard', n_jobs=-1)
    ]
    return clfs+norms

def splitData(df):
	#yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    #yLabel = ['GL_ELEVATION_GRADIENT_STEEP_MPKM_NGA.2m']
    classes_label =  ['Bathy_Bins']
    xLabels = list(df.columns.values)
    xLabels.remove(yLabel[0])
    xLabels.remove(classes_label[0])
    df = df.loc[df['GL_LAND_IS_1.0_ETOPO2v2.2m'] == 0]
    return df[xLabels].values, np.ravel(df[classes_label].values)

print('Generate ensembles')
clfs = generateEnsembles()
print('Read data')
data = reader.getDataBBOX('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m', -178.3, -71.4, -80.2, 20.7)
print('Split data')
X, y = splitData(data)
y = y.astype('str')
for clf in clfs:
    print(f'Fitting {type(clf).__name__}')
    clf.fit(X, y)
    print(f'Dumping {type(clf).__name__}')
    dump(clf, f'./models/{type(clf).__name__}.joblib')

