from nrlmls.grids.readers import GGG_fs_grid_reader as reader
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np

def generateBBOXes():
	bboxs = []
	for i in np.arange(-90.0, 90.0, 4.0):
		for j in np.arange(-180.0, 180.0, 4.0):
			bbox = np.array([[i, j], [i+4.0, j+4.0]], dtype=float)
			bboxs.append(bbox)
	return bboxs

def generateClassifiers():
	clfs = [
		KNeighborsClassifier(3),
	    #SVC(kernel="linear", C=0.025),
	    #SVC(gamma=2, C=1),
	    #GaussianProcessClassifier(1.0 * RBF(1.0)),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    MLPClassifier(alpha=1, max_iter=1000),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()
    ]
	return clfs

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


clfs = generateClassifiers()
bboxs = generateBBOXes()
print('BBOX,CLF,SCORE')
data = reader.getData('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m')
for bbox in bboxs:
	#print('Reading Data...')
	bboxString = '{}'.format(bbox).replace('\n', '').replace(',', ':')
	if bbox[:,0].min() >= -74.0:
		#X, y = splitData(reader.getDataBBOX('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m', bbox[:,1].min(), bbox[:,0].min(), bbox[:,1].max(), bbox[:,0].max()))
		X, y = splitData(data.loc[(data['lat'] >= bbox[:,0].min()) & (data['lon'] >= bbox[:,1].min()) & (data['lat'] <= bbox[:,0].max()) & (data['lon'] <= bbox[:,1].max())])
		y = y.astype('str')
		clf_name = ''
		top_score = 0.0
		#print('Training....')
		if (X.shape[0] != 0):
			try:
				for clf in clfs:
					#print('Training: {}'.format(type(clf).__name__))
					score = cross_val_score(clf, X, y, cv=10).mean()
					if score > top_score:
						clf_name = type(clf).__name__
						top_score = score
				print('{},{},{}'.format(bboxString, clf_name, top_score))
			except ValueError:
				print('{},NULL,LAND'.format(bboxString))
		else:
			print('{},NULL,SINGLE_CLASS'.format(bboxString))
	else:
		print('{},NULL,LAND'.format(bboxString))
