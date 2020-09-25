import pandas as pd
import numpy as np
import time
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

def getIterator(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    #yLabel = ['GL_ELEVATION_GRADIENT_STEEP_MPKM_NGA.2m']
    df = pd.read_csv(csvFile, index_col=0, nrows=2)
    itr = pd.read_csv(csvFile, index_col=0, iterator=True, chunksize=100000)
    return itr, list(df.columns.values)

def getChunk(itr, xLabels, yLabel):
    chunk = itr.get_chunk()
    df = pd.DataFrame(chunk)
    return df[xLabels].values, np.ravel(df[yLabel].values)


print('Reading Dataset')
csvFile = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv'
itr, xLabels = getIterator(csvFile)

yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
cols = xLabels
xLabels.remove(yLabel[0])
sgd = SGDRegressor()
count = 30
xTest = None
yTest = None
start = time.time()
try:
    while True:
        if (count == 0):
            xText, yTest = getChunk(itr, xLabels, yLabel)
        else:
            x, y = getChunk(itr, xLabels, yLabel)
            sgd.partial_fit(x,y)
        count = count-1
except StopIteration:
    print("Finished Training!")

end = time.time()

print('SGD Regression training time: {}'.format(end-start))
print('1/3 Holdout Split') # Split data for validation
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)

# Create Models

print('Training') # Model Training
print('Fitting SVM')
#start = time.time()
#svr.fit(x_train, y_train)
#end = time.time()
#print('SVM Regression training time: {}'.format(end-start))

# Model Predictions
sgdPred = sgd.predict(xTest)
print('SVM Results') # Show Results
r_squared = r2_score(yTest, sgdPred)
#medianAE = median_absolute_error(y_test, svrPred)
mse = mean_squared_error(yTest, sgdPred)
#mae = mean_absolute_error(y_test, svrPred)
#evs = explained_variance_score(y_test, svrPred)
print('SGD Regression R^2 score: {}'.format(r_squared))
#print('SVM Regression Median Absolute Error: {}'.format(medianAE))
print('SGD Regression Mean Squared Error: {}'.format(mse))
#print('SVM Regression Mean Absolute Error: {}'.format(mae))
#print('SVM Regression Explained Variance Score: {}'.format(evs))

