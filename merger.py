import pandas as pd


data = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/AllDataNormalizedFiltered.csv")

data = data.apply(abs)
cols = list(data.columns)
cols = [cols[-1]] + cols[:-1]
data = data[cols]
print(data)

data.to_csv("/home/local/ASUAD/psheth5/RA-Hydrological/AllDataNormalizedFiltered1.csv")