import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readDataSet(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    df = pd.read_csv(csvFile, index_col=0, skiprows=range(1, 10000000), nrows=10000)
    xLabels = list(df.columns.values)
    xLabels.remove(yLabel[0])
    return df[xLabels], np.ravel(df[yLabel].values)

print('Reading Dataset') # Read Data
csvFile = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv'
x, y = readDataSet(csvFile)
head = x.head(1000)
init = head['lat'][0]
print(init)

