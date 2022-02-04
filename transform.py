#


#
import numpy
import pandas

#


#
data = pandas.read_excel('./data/dataset_raw.xlsx')
data = data.set_index('datetime')
cols = data.columns

tb = {'ACTUAL': ['TRADEB_GB__TRADEBALANCE_ACTUAL', 'TRADEB_JP__TRADEBALANCE_ACTUAL'],
      'POLL': ['TRADEB_GB__TRADEBALANCE_POLL', 'TRADEB_JP__TRADEBALANCE_POLL'],
      'MIN': ['TRADEB_GB__TRADEBALANCE_MIN', 'TRADEB_JP__TRADEBALANCE_MIN'],
      'MAX': ['TRADEB_GB__TRADEBALANCE_MAX', 'TRADEB_JP__TRADEBALANCE_MAX']}


for key in tb.keys():
    tb1, tb2 = tb[key]
    data['{0}_FLAG'.format(tb1)] = data[tb1] > 0
    data['{0}_FLAG'.format(tb2)] = data[tb2] > 0
    joint_ix = data[[tb1, tb2]].dropna().index
    data['TB_REL'] = numpy.nan
    data.loc[joint_ix, 'TB_REL'] = data.loc[joint_ix, tb1] / data.loc[joint_ix, tb2]
    # pct = data['TB_REL'].dropna().pct_change()
    pct = data['TB_REL'].dropna().diff()
    data['TB_REL_CH'] = numpy.nan
    data.loc[pct.index, 'TB_REL_CH'] = pct
    data['TB_REL_CH_FLAG'] = data['TB_REL_CH']
    data.loc[pct.index, 'TB_REL_CH_FLAG'] = data.loc[pct.index, 'TB_REL_CH_FLAG'] > 0

for col in cols:
    # pct = data[col].dropna().pct_change()
    pct = data[col].dropna().diff()
    data.loc[pct.index, col] = pct
    data[col + '_FLAG'] = data[col]
    data.loc[pct.index, col + '_FLAG'] = data.loc[pct.index, col + '_FLAG'] > 0

data = data.ffill().dropna()
data.to_excel('./data/dataset_refined.xlsx')
