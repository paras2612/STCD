import numpy as np
import pandas as pd
import plotly.graph_objects as go

idx = np.loadtxt("/home/local/ASUAD/psheth5/Downloads/TCDF-master/validatedIndex.txt").astype(int)
scores = np.loadtxt("/home/local/ASUAD/psheth5/Downloads/TCDF-master/validatedScores.txt")
data = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/Yearly_Data_Normalized/FilteredNormalized2007.csv").columns

f=pd.DataFrame()
f['location'] = data[idx]
f['attention_score'] = scores
x = f['attention_score'].mean()

imp_points = pd.DataFrame()
imp_points['location'] = f.loc[f['attention_score'] >= x, 'location']
imp_points['attention_score'] = f.loc[f['attention_score'] >= x, 'attention_score']
imp_points.to_csv("ImportantPointsv2.csv",index=False)

f.to_csv("AllPointsScore.csv",index=False)
points = f.loc[f['attention_score'] >= x, 'location'].values
# points = data[idx]
# points = data[np.where(scores>np.mean(scores))]

latitude = []
longitude = []
for i in points:
  latitude.append(eval(i.split(" ")[0]))
  longitude.append(eval(i.split(" ")[2]))

print(len(latitude))

fig = go.Figure()
fig.add_trace(go.Scattermapbox(
    mode = "markers",
    lat = longitude,
    lon = latitude,
    #hovertext = df.name.tolist(),
    marker = {'color': "red",
              "size": 10},
))
fig.update_layout(margin ={'l':0,'t':0,'b':0,'r':0},
                  mapbox = {
                      'center': {'lon': 139, 'lat': 36.5},
                      'style': "stamen-terrain",
                      'zoom': 4.5},
                  width=1600,
                  height=900,)
fig.show()