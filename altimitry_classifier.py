import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nrlmls.feature_selection.GeneticAlgorithms import FeatureGA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings

def readDataSet(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    #yLabel = ['GL_ELEVATION_GRADIENT_STEEP_MPKM_NGA.2m']
    classes_label =  ['Bathy_Bins']
    df = pd.read_csv(csvFile, index_col=0)
    print(df.head())
    print(df.tail())
    xLabels = list(df.columns.values)
    xLabels.remove(yLabel[0])
    xLabels.remove(classes_label[0])
    df = df.loc[df['GL_LAND_IS_1.0_ETOPO2v2.2m'] == 0]
    return df[xLabels].values, np.ravel(df[classes_label].values)

warnings.filterwarnings("ignore")
X, y = readDataSet('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/GOMFeaturesBinned2m.csv')

#clf = GaussianNB()
#dt = DecisionTreeClassifier(random_state=41)
#clf = BaggingClassifier(dt, max_samples=0.5, max_features=0.5)
clf = RandomForestClassifier(n_estimators=200)
#selector = FeatureGA(clf)
#selector.fit(X, y=y)
#dna = selector.transform()
#print(dna)


scores = cross_val_score(clf, X, y, cv=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Cross Value Score: {}'.format(scores.mean()))
print('Bal Acc: {}'.format(accuracy_score(y_test, pred)))
print(classification_report(y_test, pred))



#print('10 Fold Cross Validation Score: {}'.format(scores.mean()))
