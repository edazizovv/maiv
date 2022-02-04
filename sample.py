#


#
import pandas

#


#
data = pandas.read_excel('./data/dataset_raw.xlsx')
data = data.set_index('datetime')

print(data.index.min())
print(data.index[int(data.shape[0] * 0.5)])
print(data.index[int(data.shape[0] * 0.8)])
print(data.index.max())
