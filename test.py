import pandas as pd
import haversine as hs
import networkx as nx
from IPython.display import Image, display
import plotly.graph_objects as go


data = pd.read_csv("/home/local/ASUAD/psheth5/Downloads/TCDF-master/causeEffectListAllDataNormalizedFiltered2007_Final.csv")
#data = pd.read_csv("/home/local/ASUAD/psheth5/Downloads/TCDF-master/causeEffectListLogNormalizedNoSpatial2007.csv")
#data = pd.read_csv("/home/local/ASUAD/psheth5/Downloads/TCDF-master/causeEffectListAllNormalized.csv")

def get_distance(r1,r2):
  if(len(r1.split(" "))==2):
    lon1 = eval(r1.split(" ")[0])
    lat1 = eval(r1.split(" ")[1])
    #print(lon1,lat1)
  elif(len(r1.split("  "))==2):
    lon1 = eval(r1.split("  ")[0])
    lat1 = eval(r1.split("  ")[1])
    #print(lon1,lat1)
  if(len(r2.split(" "))==2):
    lon2 = eval(r2.split(" ")[0])
    lat2 = eval(r2.split(" ")[1])
    #print(lon2,lat2)
  elif(len(r2.split("  "))==2):
    lon2 = eval(r2.split("  ")[0])
    lat2 = eval(r2.split("  ")[1])
    #print(lon2,lat2)
  return round(hs.haversine((lon1,lat1),(lon2,lat2)),2)

dist = []
for i,row in data.iterrows():
  dist.append(get_distance(row[0],row[1]))

data['distance'] = dist


def get_k_hop_causes(causes, x, k=1):
    froms = []
    tos = []
    for l in range(0, k):
        i = 0
        temp = len(causes)
        print("THIS IS INITIAL LENGTH: ", temp)
        while (i != temp):
            temp1 = x.get_group(causes[i])['cause'].values
            for j in temp1:
                if (j not in causes):
                    causes.append(j)
                    froms.append(j)
                    tos.append(causes[i])
                    '''if(len(causes[i].split(" "))==2):
                      lon1 = eval(causes[i].split(" ")[0])
                      lat1 = eval(causes[i].split(" ")[1])
                      tos.append((lon1,lat1))
                    elif(len(causes[i].split("  "))==2):
                      lon1 = eval(causes[i].split("  ")[0])
                      lat1 = eval(causes[i].split("  ")[1])
                      tos.append((lon1,lat1))
                    if(len(j.split(" "))==2):
                      lon2 = eval(j.split(" ")[0])
                      lat2 = eval(j.split(" ")[1])
                      froms.append((lon2,lat2))
                    elif(len(j.split("  "))==2):
                      lon2 = eval(j.split("  ")[0])
                      lat2 = eval(j.split("  ")[1])
                      froms.append((lon2,lat2))'''
            i += 1
            print("LENGTH UPDATED TO: ", len(causes), " PROCESSED: ", i, " K: ", l, " LENGTH OF FROMS: ", len(froms),
                  "LENGTH OF TOS: ", len(tos))
    print(len(causes))
    return causes, froms, tos
x = data.groupby("effect")
froms = []
tos = []
dists = []
all_causes = x.get_group("-95.35 28.85")['cause'].values
causes = []
for i in all_causes:
  #a = " ".join(i[1:-1].split("  "))
  if(i not in causes):
    causes.append(i)
    froms.append(i)
    tos.append("-95.35  28.85")
  '''if(a not in causes):
    causes.append(a)
    tos.append((95.35,28.85))
    if(len(i.split(" "))==2):
      lon1 = eval(i.split(" ")[0][1:])
      lat1 = eval(i.split(" ")[1][1:])
      froms.append((lon1,lat1))
    elif(len(i.split("  "))==2):
      lon1 = eval(i.split("  ")[0][1:])
      lat1 = eval(i.split("  ")[1][1:])
      froms.append((lon1,lat1))'''
print(len(causes))
latitude = []
longitude = []
causes.append("-95.35 28.85")
for i in causes:
  latitude.append(eval(i.split(" ")[0]))
  longitude.append(eval(i.split(" ")[1]))

print(latitude[0],longitude[0])

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
# d2 = x.get_group("-95.35 28.85")
# d2.to_csv("DirectCauses2.csv",index=False)
causes,fr,ts = get_k_hop_causes(causes,x,1)
froms.extend(fr)
tos.extend(ts)
d1 = pd.DataFrame([froms,tos]).transpose()
d1.columns = ['from','to']
print(d1.head(5))
#
tw_small = nx.from_pandas_edgelist(d1,source='from',target='to', edge_attr=None, create_using=nx.DiGraph())
p=nx.drawing.nx_pydot.to_pydot(tw_small)
def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

view_pydot(p)
p.write_png('./example2007.png')
#
# causes.append("-95.35 28.85")
# latitude = []
# longitude = []
# for i in causes:
#   latitude.append(eval(i.split(" ")[0]))
#   longitude.append(eval(i.split(" ")[1]))
#
# print(latitude[0],longitude[0])
#
# # fig = go.Figure()
# # fig.add_trace(go.Scattermapbox(
# #     mode = "markers",
# #     lat = longitude,
# #     lon = latitude,
# #     #hovertext = df.name.tolist(),
# #     marker = {'color': "red",
# #               "size": 10},
# # ))
# # fig.update_layout(margin ={'l':0,'t':0,'b':0,'r':0},
# #                   mapbox = {
# #                       'center': {'lon': 139, 'lat': 36.5},
# #                       'style': "stamen-terrain",
# #                       'zoom': 4.5},
# #                   width=1600,
# #                   height=900,)
# # fig.show()
#
# d1.to_csv("CausesofTargetNode2.csv",index=False)
