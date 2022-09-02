import pandas as pd
from haversine import haversine,Unit
import numpy as np

data = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/AllDataNormalizedFiltered1.csv")
#data = data.iloc[:,1:]
cols = data.columns
'''cols_i = cols
scores= [0]*len(cols_i)
for j in range(len(cols_i)):
    lonlat = (float(cols_i[j].split("  ")[0]), float(cols_i[j].split("  ")[1]))
    lonlat_1 = (float(cols_i[2].split("  ")[0]), float(cols_i[2].split("  ")[1]))
    dist = haversine(lonlat, lonlat_1, unit=Unit.MILES)
    if (float(cols_i[j].split("  ")[1]) >= float(cols_i[2].split("  ")[1])):
        scores[j] += 0.001 * (dist)
    else:
        scores[j] -= 1* (dist)
c=0
for i in scores:
    if(i>=0.0):
        c+=1
    elif(i<0.0 and i>=-3.0):
        c+=1
print(c)
#data.to_csv("/home/local/ASUAD/psheth5/RA-Hydrological/AllDataNormalizedFiltered1.csv",index=False)'''
distance = []
for j in range(len(cols)):
  temp = []
  for i in range(len(cols)):
    lonlat = (float(cols[i].split("  ")[0]), float(cols[i].split("  ")[1]))
    lonlat_1 = (float(cols[j].split("  ")[0]), float(cols[j].split("  ")[1]))
    dist = haversine(lonlat, lonlat_1, unit=Unit.MILES)
    temp.append(dist)
  print("Done for ",cols[j])
  distance.append(temp)
distance = np.array(distance)
print(distance.shape)
np.savetxt('test.txt', distance)   # X is an array