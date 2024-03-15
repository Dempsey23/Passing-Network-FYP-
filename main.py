from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis


comp = sb.competitions()
#Premier League 15/16 Season
match=sb.matches(competition_id=2,season_id=27)

print(match.to_markdown())

match_count=0
#Printing Leicester Matches
for i in range(380):
    if match['home_team'][i]=='Leicester City':
        print(match_count,' ',match['home_team'][i],":",match['home_score'][i],match['away_team'][i],":",match['away_score'][i],match['match_id'][i])
        match_count+=1

match_count=0;print('\n')
#Printing Tottenham Matches
for i in range(380):
    if match['home_team'][i]=='Tottenham Hotspur':
        print(match_count,' ',match['home_team'][i],":",match['home_score'][i],match['away_team'][i],":",match['away_score'][i],match['match_id'][i])
        match_count+=1
    if match['match_id'][i]==3754209:
        print(match['home_team'][i],':',match['home_score'][i],'-',match['away_team'][i],':',match['away_score'][i])




