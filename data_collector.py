import numpy as np
from pathlib import Path
import pandas as pd
import time
import os

def gggReader(grid, xmin, ymin):
    split = grid.split('.')
    reolution = ''
    nx = 0
    ny = 0
    dx = 0
    dy = 0

    if len(split[-2]) == 2:
        resolution = split[-2]
    else:
        raise ValueError('Could not read resolution from ggg file: {}'.format(grid))

    if resolution == '1d':
        dx = 1
        dy = 1
        nx = 360
        ny = 180
    elif resolution == '5m':
        dx = 5/60
        dy = dx
        nx = int(360/dx)
        ny = int(180/dy)
    elif resolution == '2m':
        dx = 2/60
        print(dx)
        dy = dx
        nx = int(360/dx)
        print(nx)
        ny = int(180/dy)
        print(ny)
    else:
        raise ValueError('The resolution {} of gggfile {} is not currently supported'.format(resolution, grid))
    print(int(360/(5/60)))
    print(int(180/(5/60)))

    z = np.fromfile(grid, dtype=np.float32).reshape((-1,1))
    lons = np.linspace(xmin,xmin+(dx*(nx-1)),nx)
    lats = np.linspace(ymin,ymin+(dy*(ny-1)),ny)
    x, y = np.meshgrid(lons,lats)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    matrix = np.concatenate((x,y,z),axis=1)
    df = pd.DataFrame(matrix, columns=['lon', 'lat', Path(grid).stem])
    return df

def getData(dataDir):
    if os.path.isdir(dataDir) is not True:
        raise ValueError('Parameter {} was not a directory'.format(dataDir))
    files = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
    df = None
    for f in files:
        print('Reading file {}'.format(f))
        start = time.time()
        data = gggReader(f, -179.5, -89.5)
        end = time.time()
        print(end-start)
        if df is None:
            df = data
        else:
            col = Path(f).stem
            df[col] = pd.Series(data[col].values, index=df.index)
    print('Filling Nans')
    df = df.fillna(0)
    return df

dataDir = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m'
df = getData(dataDir)
print('Writing to CSV')
df.to_csv('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv')
