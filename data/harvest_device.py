#
import os

#
import numpy
import pandas

#
data_hub = []
for d in os.listdir('./RAW/'):
    data_spice = pandas.read_excel('./RAW/{0}'.format(d), sheet_name='Data')
    data_spice = data_spice.rename(columns={x: '{0}__{1}'.format(d.replace('.xlsx', ''), x)
                                            for x in data_spice.columns})
    data_spice = data_spice.rename(columns={data_spice.columns[0]: 'datetime'})
    data_spice['datetime'] = pandas.to_datetime(data_spice['datetime'], infer_datetime_format=True)
    data_spice = data_spice.set_index('datetime')
    data_spice = data_spice.sort_index()
    data_hub.append(data_spice)
data = pandas.concat(data_hub, axis=1)
# data = data.ffill().dropna()
data = data.drop(columns=['GBPX__Volume', 'JPYX__Volume'])

data['TRES10Y_JP__Price__REAL'] = data['TRES10Y_JP__Price'] - data['CPI_JP__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_JP__Open__REAL'] = data['TRES10Y_JP__Open'] - data['CPI_JP__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_JP__Max__REAL'] = data['TRES10Y_JP__Max'] - data['CPI_JP__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_JP__Min__REAL'] = data['TRES10Y_JP__Min'] - data['CPI_JP__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_GB__Price__REAL'] = data['TRES10Y_GB__Price'] - data['CPI_GB__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_GB__Open__REAL'] = data['TRES10Y_GB__Open'] - data['CPI_GB__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_GB__Max__REAL'] = data['TRES10Y_GB__Max'] - data['CPI_GB__CPI_YOY_STDZ_SA'].ffill()
data['TRES10Y_GB__Min__REAL'] = data['TRES10Y_GB__Min'] - data['CPI_GB__CPI_YOY_STDZ_SA'].ffill()

data['_TARGET'] = data['GBPX__Close'] / data['JPYX__Close']
# data['_TARGET'] = data['JPYX__Close'] / data['GBPX__Close']
data['_TARGET_ewm21'] = data['_TARGET'].ffill().ewm(span=21, adjust=False).mean()
data['_TARGET_ewm55'] = data['_TARGET'].ffill().ewm(span=55, adjust=False).mean()
# data['_TARGET_ewm21'] = data['_TARGET'].ffill().rolling(window=21).mean()
# data['_TARGET_ewm55'] = data['_TARGET'].ffill().rolling(window=55).mean()
data['TARGET_1M'] = data['_TARGET'].shift(-30)
data['TARGET_3M'] = data['_TARGET'].shift(-90)
data['TARGET_6M'] = data['_TARGET'].shift(-180)

# data = data.dropna()
start_date = numpy.array([data[x].dropna().index.min() for x in data.columns]).max()
end_date = numpy.array([data[x].dropna().index.max() for x in data.columns]).min()
data = data[(data.index >= start_date) * (data.index <= end_date)]

data.to_excel('./dataset_raw.xlsx')
