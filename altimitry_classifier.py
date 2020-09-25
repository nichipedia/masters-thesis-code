import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
import warnings
import matplotlib.pyplot as plt
import itertools


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


    
def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()


warnings.filterwarnings("ignore")
X, y = readDataSet('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/GOMFeaturesBinned2m.csv')

#clf = GaussianNB()
dt = DecisionTreeClassifier(random_state=41)
#rfc = RandomForestClassifier(n_estimators=200)
clf = BaggingClassifier(dt, max_samples=0.5, max_features=0.5)
#selector = FeatureGA(clf)
#selector.fit(X, y=y)
#dna = selector.transform()
#print(dna)
#clf = MLPClassifier()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Bal Acc: {}'.format(accuracy_score(y_test, pred)))
sns.heatmap(pd.DataFrame(classification_report(y_test, pred, output_dict=True)).iloc[:-1, :].T, annot=True)
plt.show()


#print('10 Fold Cross Validation Score: {}'.format(scores.mean()))
