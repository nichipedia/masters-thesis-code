import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

rows = {
    0: 'lon',
    1: 'lat',
    2: 'CM_MANTLE_DEN_KGM3_CRUST1.2m',
    3: 'CM_MANTLE_VP_MS_CRUST1.2m',
    4: 'CM_MANTLE_VS_MS_CRUST1.2m',
    5: 'GL_COAST_FROM_LAND_IS_1.0_ETOPO2v2.2m',
    6: 'GL_COAST_FROM_SEA_IS_1.0_ETOPO2v2.2m',
    7: 'GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m',
    8: 'GL_ELEVATION_GRADIENT_STEEP_MPKM_NGA.2m',
    9: 'GL_LAND_IS_1.0_ETOPO2v2.2m',
    10: 'SC_CRUST_THICK_M_CRUST1.2m',
    11: 'SC_LOW_CRUST_DEN_KGM3_CRUST1.2m',
    12: 'SC_LOW_CRUST_THICK_M_CRUST1.2m',
    13: 'SC_LOW_CRUST_VP_MS_CRUST1.2m',
    14: 'SC_LOW_CRUST_VS_MS_CRUST1.2m',
    15: 'SC_MID_CRUST_DEN_KGM3_CRUST1.2m',
    16: 'SC_MID_CRUST_THICK_M_CRUST1.2m',
    17: 'SC_MID_CRUST_VP_MS_CRUST1.2m',
    18: 'SC_MID_CRUST_VS_MS_CRUST1.2m',
    19: 'SC_UP_CRUST_DEN_KGM3_CRUST1.2m',
    20: 'SC_UP_CRUST_THICK_M_CRUST1.2m',
    21: 'SC_UP_CRUST_VP_MS_CRUST1.2m',
    22: 'SC_UP_CRUST_VS_MS_CRUST1.2m',
    23: 'SF_AVG_SEA_DENSITY_KGM3_DECADAL_MEAN_woa13x.2m',
    24: 'SF_AVG_SEA_SOUNDSPEED_MS_DECADAL_MEAN_woa13x.2m',
    25: 'SF_CURRENT_EAST_MS_2012_12_HYCOMx.2m',
    26: 'SF_CURRENT_MAG_MS_2012_12_HYCOMx.2m',
    27: 'SF_CURRENT_NORTH_MS_2012_12_HYCOMx.2m',
    28: 'SF_LOW_SED_THICK_M_CRUST1.2m',
    29: 'SF_LOW_SED_VS_MS_CRUST1.2m',
    30: 'SF_MID_SED_DEN_KGM3_CRUST1.2m',
    31: 'SF_MID_SED_VP_MS_CRUST1.2m',
    32: 'SF_MID_SED_VS_MS_CRUST1.2m',
    33: 'SF_SEA_NITRATE_MCML_DECADAL_MEAN_woa13x.2m',
    34: 'SF_SEA_OXYGEN_MLL_DECADAL_MEAN_woa13x.2m',
    35: 'SF_SEA_OXYGEN_PCTSAT_DECADAL_MEAN_woa13x.2m',
    36: 'SF_SEA_PHOSPHATE_MCML_DECADAL_MEAN_woa13x.2m',
    37: 'SF_SEA_SALINITY_PSU_DECADAL_MEAN_woa13x.2m',
    38: 'SF_SEA_SILICATE_MCML_DECADAL_MEAN_woa13x.2m',
    39: 'SF_SEA_TEMPERATURE_C_DECADAL_MEAN_woa13x.2m',
    40: 'SF_SED_THICK_M_CRUST1.2m',
    41: 'SF_TOT_SED_THICK_M_CRUST1.2m',
    42: 'SF_UP_SED_DEN_KGM3_CRUST1.2m',
    43: 'SF_UP_SED_THICK_M_CRUST1.2m',
    44: 'SF_UP_SED_VP_MS_CRUST1.2m',
    45: 'SF_UP_SED_VS_MS_CRUST1.2m',
    46: 'SL_GEOID_GRADIENT_MEAN_MPKM_NGA.2m',
    47: 'SL_GEOID_GRADIENT_STEEP_MPKM_NGA.2m',
    48: 'SL_GEOID_M_ABOVE_WGS84_NGA_egm2008.2m',
    49: 'SS_BIOMASS_BACTERIA_LOG10_MGCM2_Wei2010x.2m',
    50: 'SS_BIOMASS_BACTERIA_LOG10_SDEV_MGCM2_Wei2010x.2m',
    51: 'SS_BIOMASS_FISH_LOG10_MGCM2_Wei2010x.2m',
    52: 'SS_BIOMASS_FISH_LOG10_SDEV_MGCM2_Wei2010x.2m',
    53: 'SS_BIOMASS_INVERTEBRATE_LOG10_MGCM2_Wei2010x.2m',
    54: 'SS_BIOMASS_INVERTEBRATE_LOG10_SDEV_MGCM2_Wei2010x.2m',
    55: 'SS_BIOMASS_MACROFAUNA_LOG10_MGCM2_Wei2010x.2m',
    56: 'SS_BIOMASS_MACROFAUNA_LOG10_SDEV_MGCM2_Wei2010x.2m',
    57: 'SS_BIOMASS_MEGAFAUNA_LOG10_MGCM2_Wei2010x.2m',
    58: 'SS_BIOMASS_MEGAFAUNA_LOG10_SDEV_MGCM2_Wei2010x.2m',
    59: 'SS_BIOMASS_MEIOFAUNA_LOG10_MGCM2_Wei2010x.2m',
    60: 'SS_BIOMASS_MEIOFAUNA_LOG10_SDEV_MGCM2_Wei2010x.2m',
    61: 'SS_BIOMASS_TOTAL_LOG10_MGCM2_Wei2010x.2m',
    62: 'SS_BIOMASS_TOTAL_LOG10_SDEV_MGCM2_Wei2010x.2m',
    63: 'SS_DENSITY_KGM-3_SACD_Aquarius_MISSION_MEANx.2m',
    64: 'SS_MIXED_LAYER_DEPTH_APR_M_Goyetx.2m',
    65: 'SS_MIXED_LAYER_DEPTH_AUG_M_Goyetx.2m',
    66: 'SS_MIXED_LAYER_DEPTH_DEC_M_Goyetx.2m',
    67: 'SS_MIXED_LAYER_DEPTH_FEB_M_Goyetx.2m',
    68: 'SS_MIXED_LAYER_DEPTH_JAN_M_Goyetx.2m',
    69: 'SS_MIXED_LAYER_DEPTH_JUL_M_Goyetx.2m',
    70: 'SS_MIXED_LAYER_DEPTH_JUN_M_Goyetx.2m',
    71: 'SS_MIXED_LAYER_DEPTH_MAR_M_Goyetx.2m',
    72: 'SS_MIXED_LAYER_DEPTH_MAX_M_Goyetx.2m',
    73: 'SS_MIXED_LAYER_DEPTH_MAY_M_Goyetx.2m',
    74: 'SS_MIXED_LAYER_DEPTH_MIN_M_Goyetx.2m',
    75: 'SS_MIXED_LAYER_DEPTH_NOV_M_Goyetx.2m',
    76: 'SS_MIXED_LAYER_DEPTH_OCT_M_Goyetx.2m',
    77: 'SS_MIXED_LAYER_DEPTH_SEP_M_Goyetx.2m',
    78: 'SS_WAVE_HEIGHT_M_2012_12_WAVEWATCH3x.2m',
    79: 'SS_WAVE_PERIOD_S_2012_12_WAVEWATCH3x.2m'
}

def readDataSet(csvFile):
    #yLabel = ['GL_ELEVATION_GRADIENT_MEAN_MPKM_NGA.2m']
    yLabel = ['GL_ELEVATION_M_ASL_ETOPO2v2.2m']
    df = pd.read_csv(csvFile, index_col=0, skiprows=range(1, 10000000), nrows=1000000)
    xLabels = list(df.columns.values)
    xLabels.remove(yLabel[0])
    return df[xLabels], np.ravel(df[yLabel].values)

print('Reading Dataset') # Read Data
csvFile = 'C:/Users/nmoran/Documents/bathy-project/bathy-model/data/features2m.csv'
x, y = readDataSet(csvFile)

temp = []
for col in x.columns:
    values = x[col].values
    coef = pearsonr(values, y)
    temp.append(coef[0])

pcc = pd.DataFrame(temp, columns=['pcc'])
pcc.rename(index=rows, inplace=True)
pcc.to_csv('C:/Users/nmoran/Documents/bathy-project/bathy-model/data/csv/pcc.csv')

