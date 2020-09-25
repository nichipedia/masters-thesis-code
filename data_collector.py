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
    bins = pd.IntervalIndex.from_tuples([(-10790, -10640), (-10640, -10490), (-10490, -10340), (-10340, -10190), (-10190, -10040), (-10040, -9890), (-9890, -9740), (-9740, -9590), (-9590, -9440), (-9440, -9290), (-9290, -9140), (-9140, -8990), (-8990, -8840), (-8840, -8690), (-8690, -8540), (-8540, -8390), (-8390, -8240), (-8240, -8090), (-8090, -7940), (-7940, -7790), (-7790, -7640), (-7640, -7490), (-7490, -7340), (-7340, -7190), (-7190, -7040), (-7040, -6890), (-6890, -6740), (-6740, -6590), (-6590, -6440), (-6440, -6290), (-6290, -6140), (-6140, -5990), (-5990, -5840), (-5840, -5690), (-5690, -5540), (-5540, -5390), (-5390, -5240), (-5240, -5090), (-5090, -4940), (-4940, -4790), (-4790, -4640), (-4640, -4490), (-4490, -4340), (-4340, -4190), (-4190, -4040), (-4040, -3890), (-3890, -3740), (-3740, -3590), (-3590, -3440), (-3440, -3290), (-3290, -3140), (-3140, -2990), (-2990, -2840), (-2840, -2690), (-2690, -2540), (-2540, -2390), (-2390, -2240), (-2240, -2090), (-2090, -1940), (-1940, -1790), (-1790, -1640), (-1640, -1490), (-1490, -1340), (-1340, -1190), (-1190, -1040), (-1040, -890), (-890, -740), (-740, -590), (-590, -440), (-440, -290), (-290, -140), (-140, 10), (10, 160), (160, 310), (310, 460), (460, 610), (610, 760), (760, 910), (910, 1060), (1060, 1210), (1210, 1360), (1360, 1510), (1510, 1660), (1660, 1810), (1810, 1960), (1960, 2110), (2110, 2260), (2260, 2410), (2410, 2560), (2560, 2710), (2710, 2860), (2860, 3010), (3010, 3160), (3160, 3310), (3310, 3460), (3460, 3610), (3610, 3760), (3760, 3910), (3910, 4060), (4060, 4210), (4210, 4360), (4360, 4510), (4510, 4660), (4660, 4810), (4810, 4960), (4960, 5110), (5110, 5260), (5260, 5410), (5410, 5560), (5560, 5710), (5710, 5860), (5860, 6010), (6010, 6160), (6160, 6310), (6310, 6460), (6460, 6610), (6610, 6760), (6760, 6910), (6910, 7060), (7060, 7210), (7210, 7360), (7360, 7510), (7510, 7660), (7660, 7810), (7810, 7960), (7960, 8110), (8110, 8260), (8260, 8410), (8410, 8560)])
    print('Get Range')
    binned_bathy = pd.cut(df['GL_ELEVATION_M_ASL_ETOPO2v2.2m'], bins)
    df['Bathy_Bins'] = binned_bathy
    return df

def getDataBBOX(dataDir, minX, minY, maxX, maxY):
    if os.path.isdir(dataDir) is not True:
        raise ValueError('Parameter {} was not a directory'.format(dataDir))
    files = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
    df = None
    for f in files:
        print('Reading file {}'.format(f))
        start = time.time()
        temp = gggReader(f, -179.5, -89.5)
        data = temp.loc[(temp['lat'] >= minY) & (temp['lon'] >= minX) & (temp['lat'] <= maxY) & (temp['lon'] <= maxX)]
        end = time.time()
        print(end-start)
        if df is None:
            df = data
        else:
            col = Path(f).stem
            df[col] = pd.Series(data[col].values, index=df.index)
    print('Filling Nans')
    df = df.fillna(0)
    bins = pd.IntervalIndex.from_tuples([(-10790, -10640), (-10640, -10490), (-10490, -10340), (-10340, -10190), (-10190, -10040), (-10040, -9890), (-9890, -9740), (-9740, -9590), (-9590, -9440), (-9440, -9290), (-9290, -9140), (-9140, -8990), (-8990, -8840), (-8840, -8690), (-8690, -8540), (-8540, -8390), (-8390, -8240), (-8240, -8090), (-8090, -7940), (-7940, -7790), (-7790, -7640), (-7640, -7490), (-7490, -7340), (-7340, -7190), (-7190, -7040), (-7040, -6890), (-6890, -6740), (-6740, -6590), (-6590, -6440), (-6440, -6290), (-6290, -6140), (-6140, -5990), (-5990, -5840), (-5840, -5690), (-5690, -5540), (-5540, -5390), (-5390, -5240), (-5240, -5090), (-5090, -4940), (-4940, -4790), (-4790, -4640), (-4640, -4490), (-4490, -4340), (-4340, -4190), (-4190, -4040), (-4040, -3890), (-3890, -3740), (-3740, -3590), (-3590, -3440), (-3440, -3290), (-3290, -3140), (-3140, -2990), (-2990, -2840), (-2840, -2690), (-2690, -2540), (-2540, -2390), (-2390, -2240), (-2240, -2090), (-2090, -1940), (-1940, -1790), (-1790, -1640), (-1640, -1490), (-1490, -1340), (-1340, -1190), (-1190, -1040), (-1040, -890), (-890, -740), (-740, -590), (-590, -440), (-440, -290), (-290, -140), (-140, 10), (10, 160), (160, 310), (310, 460), (460, 610), (610, 760), (760, 910), (910, 1060), (1060, 1210), (1210, 1360), (1360, 1510), (1510, 1660), (1660, 1810), (1810, 1960), (1960, 2110), (2110, 2260), (2260, 2410), (2410, 2560), (2560, 2710), (2710, 2860), (2860, 3010), (3010, 3160), (3160, 3310), (3310, 3460), (3460, 3610), (3610, 3760), (3760, 3910), (3910, 4060), (4060, 4210), (4210, 4360), (4360, 4510), (4510, 4660), (4660, 4810), (4810, 4960), (4960, 5110), (5110, 5260), (5260, 5410), (5410, 5560), (5560, 5710), (5710, 5860), (5860, 6010), (6010, 6160), (6160, 6310), (6310, 6460), (6460, 6610), (6610, 6760), (6760, 6910), (6910, 7060), (7060, 7210), (7210, 7360), (7360, 7510), (7510, 7660), (7660, 7810), (7810, 7960), (7960, 8110), (8110, 8260), (8260, 8410), (8410, 8560)
])
    print('Get Range')
    binned_bathy = pd.cut(df['GL_ELEVATION_M_ASL_ETOPO2v2.2m'], bins)
    df['Bathy_Bins'] = binned_bathy
    return df

if __name__ == "__main__":
    dataDir = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m'
    df = getData(dataDir)
    #print('Writing to CSV')
    df.to_csv('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features_binned.2m.csv')
