import pandas as pd
import numpy as np
import time
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

def readDataSet(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    #yLabel = ['GL_ELEVATION_GRADIENT_STEEP_MPKM_NGA.2m']
    df = pd.read_csv(csvFile, index_col=0, skiprows=range(1, 10000000), nrows=10000000)
    print(df.head())
    print(df.tail())
    labels = ['CM_MANTLE_DEN_KGM3_CRUST1.2m',
            'SC_CRUST_THICK_M_CRUST1.2m',
            'SC_MID_CRUST_DEN_KGM3_CRUST1.2m',
            'SF_CURRENT_EAST_MS_2012_12_HYCOMx.2m',
            'SF_CURRENT_MAG_MS_2012_12_HYCOMx.2m',
            'SF_CURRENT_NORTH_MS_2012_12_HYCOMx.2m',
            'SF_SEA_NITRATE_MCML_DECADAL_MEAN_woa13x.2m',
            'SF_SEA_OXYGEN_PCTSAT_DECADAL_MEAN_woa13x.2m',
            'SF_SEA_PHOSPHATE_MCML_DECADAL_MEAN_woa13x.2m',
            'SF_SEA_SALINITY_PSU_DECADAL_MEAN_woa13x.2m',
            'SF_SEA_SILICATE_MCML_DECADAL_MEAN_woa13x.2m',
            'SF_SEA_TEMPERATURE_C_DECADAL_MEAN_woa13x.2m',
            'SF_UP_SED_THICK_M_CRUST1.2m',
            'SL_GEOID_GRADIENT_MEAN_MPKM_NGA.2m',
            'SS_BIOMASS_FISH_LOG10_MGCM2_Wei2010x.2m',
            'SS_BIOMASS_MACROFAUNA_LOG10_MGCM2_Wei2010x.2m',
            'GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    df = df.loc[df['GL_COAST_FROM_LAND_IS_1.0_ETOPO2v2.2m'] == 0]
    df = df[labels]
    xLabels = list(df.columns.values)
    xLabels.remove(yLabel[0])
    return df[xLabels].values, np.ravel(df[yLabel].values)

print('Reading Dataset') # Read Data
csvFile = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv'
x, y = readDataSet(csvFile)

print('1/3 Holdout Split') # Split data for validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)

# Create Models
svr = LinearSVR(random_state=0, tol=1e-5)
bar = BayesianRidge()
reg = LinearRegression()

print('Training') # Model Training
print('Fitting SVM')
start = time.time()
svr.fit(x_train, y_train)
end = time.time()
print('SVM Regression training time: {}'.format(end-start))
print('Fitting Naive Bayes')
start = time.time()
bar.fit(x_train, y_train)
end = time.time()
print('Naive Bayes Regression training time: {}'.format(end-start))
print('Fitting Linear Regression')
start = time.time()
reg.fit(x_train, y_train)
end = time.time()
print('Linear Regression training time: {}'.format(end-start))

# Model Predictions
svrPred = svr.predict(x_test)
barPred = bar.predict(x_test)
regPred = reg.predict(x_test)

print('SVM Results') # Show Results
r_squared = r2_score(y_test, svrPred)
medianAE = median_absolute_error(y_test, svrPred)
mse = mean_squared_error(y_test, svrPred)
mae = mean_absolute_error(y_test, svrPred)
evs = explained_variance_score(y_test, svrPred)
print('SVM Regression R^2 score: {}'.format(r_squared))
print('SVM Regression Median Absolute Error: {}'.format(medianAE))
print('SVM Regression Mean Squared Error: {}'.format(mse))
print('SVM Regression Mean Absolute Error: {}'.format(mae))
print('SVM Regression Explained Variance Score: {}'.format(evs))

print('Naive Bayes Results')
r_squared = r2_score(y_test, barPred)
medianAE = median_absolute_error(y_test, barPred)
mse = mean_squared_error(y_test, barPred)
mae = mean_absolute_error(y_test, barPred)
evs = explained_variance_score(y_test, barPred)
print('Naive Bayes Regression R^2 score: {}'.format(r_squared))
print('Naive Bayes Regression Median Absolute Error: {}'.format(medianAE))
print('Naive Bayes Regression Mean Squared Error: {}'.format(mse))
print('Naive Bayes Regression Mean Absolute Error: {}'.format(mae))
print('Naive Bayes Regression Explained Variance Score: {}'.format(evs))

print('Linear Regression Results')
r_squared = r2_score(y_test, regPred)
medianAE = median_absolute_error(y_test, regPred)
mse = mean_squared_error(y_test, regPred)
mae = mean_absolute_error(y_test, regPred)
evs = explained_variance_score(y_test, regPred)
print('Linear Regression R^2 score: {}'.format(r_squared))
print('Linear Regression Median Absolute Error: {}'.format(medianAE))
print('Linear Regression Mean Squared Error: {}'.format(mse))
print('Linear Regression Mean Absolute Error: {}'.format(mae))
print('Linear Regression Explained Variance Score: {}'.format(evs))
