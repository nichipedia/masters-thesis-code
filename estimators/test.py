import GridOptimisedClassifier
import numpy as np
from nrl_commons.mls.grids.readers import GGG_fs_grid_reader as reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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


data = reader.getDataBBOX('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m', -178.3, -71.4, -160.2, -50.7)
X, y = splitData(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)
clf = GridOptimisedClassifier.GridOptimisedClassifier(gridPath='C:/Users/nmoran/Documents/bathy-project/bathy-model/estimators/optgrid.csv', modelsPath='C:/Users/nmoran/Documents/bathy-project/bathy-model/models')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(pred)
print('Bal Acc: {}'.format(accuracy_score(y_test, pred)))
print(classification_report(y_test, pred))