from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

match=sb.matches(competition_id=2,season_id=27)
sourceFile = open('LEI_V_LIV.txt', 'w')
print('Leicester VS Liverpool\n--------------------------------',file=sourceFile)

#print(match.to_markdown())

LEI_LIV_events=sb.events(match_id=3754341)
for i in range(380):
    if match['match_id'][i] == 3754275:
        print(match['home_team'][i], ':', match['home_score'][i], '-', match['away_team'][i], ':', match['away_score'][i],file=sourceFile)

print(LEI_LIV_events.head(10).to_markdown())

tact = LEI_LIV_events[LEI_LIV_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_LEI = tact[tact['team'] == 'Leicester City']
tact_LIV = tact[tact['team'] == 'Liverpool']
tact_LEI = tact_LEI['tactics']
tact_LIV = tact_LIV['tactics']
print(tact_LEI.to_markdown())
print(tact_LIV.to_markdown())

dict_LEI = tact_LEI[0]['lineup']
dict_LIV = tact_LIV[1]['lineup']

lineup_LEI = pd.DataFrame.from_dict(dict_LEI)
lineup_LIV=pd.DataFrame.from_dict(dict_LIV)
print(lineup_LEI.to_markdown())
print(lineup_LIV.to_markdown())

players_TOT = {}
for i in range(len(lineup_LEI)):
    key = lineup_LEI.player[i]['name']
    val = lineup_LEI.jersey_number[i]
    players_TOT[key] = str(val)
print("\n",players_TOT)

players_LIV = {}
for i in range(len(lineup_LIV)):
    key = lineup_LIV.player[i]['name']
    val = lineup_LIV.jersey_number[i]
    players_LIV[key] = str(val)
print("\n",players_LIV)