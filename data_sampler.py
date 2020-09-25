import data_collector

# GOM (17.65, -97.15) (30.66, -82.16)
# Mediterrainian (28.96, -4.89) (44.88, 36.42)
# Indian Gulf (6.66, 79.29) (22.16, 98.05)
# South Atlantic (54.8, -46.3) (-14.4, 6.2)
# The Trench (2.4, 135.6) (43.0, 177.8)
dataDir = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/2m'
#df = data_collector.getDataBBOX(dataDir, -97.65, 17.65, -82.16, 30.66)
#df = data_collector.getDataBBOX(dataDir, -4.89, 28.96, 36.42, 44.88)
#df = data_collector.getDataBBOX(dataDir, 79.29, 6.66, 98.05, 22.16)
df = data_collector.getDataBBOX(dataDir, -46.3, 54.8, 6.2, -14.4)
#df = data_collector.getDataBBOX(dataDir, 135.6, 2.4, 177.8, 43.0)

print('Writing to CSV')
df.to_csv('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/SouthAtlanticFeaturesBinned2m.csv')
