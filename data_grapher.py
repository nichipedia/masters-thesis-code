import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

def readDataSet(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    df = pd.read_csv(csvFile, index_col=0, skiprows=range(1, 10000000), nrows=100000)
    # xLabels = list(df.columns.values)
    # xLabels.remove(yLabel[0])
    # return df[xLabels], np.ravel(df[yLabel].values)
    return df

print('Reading Dataset') # Read Data
csvFile = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv'
x = readDataSet(csvFile)
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
newLabels = ['Mantle',
            'Crust Thickness',
            'Crust Density',
            'East Current',
            'Magnitude Current',
            'North Current',
            'Nitrate',
            'Oxygen',
            'Phosphate',
            'Salinity',
            'Silicate',
            'Temperature',
            'Sediment Thickness',
            'Geoid Gradient',
            'Fish Biomass',
            'MacroFauna Biomass',
            'Bathy']



x = x[labels]

x.columns = newLabels


plt = sns.pairplot(x)
plt.savefig('./plots/pairplot.png')

# for col in x.columns:
#     val = x[col].values
#     title = '{}_X_{}'.format('Bathymetry', col)
#     # plt.title(title)
#     # plt.xlabel(col)
#     # plt.ylabel('Altimetry')
#     # plt.scatter(val,y,c='blue')
#     # plt.savefig(
#     # plt.clf()
#     data = pd.DataFrame({col.casefold(): val, 'bathymetry': y})
#     plt = sns.scatterplot(x=col.casefold(), y='bathymetry', data=data).get_figure()
#     plt.savefig('./plots/{}.png'.format(title))
#     plt.clf()
