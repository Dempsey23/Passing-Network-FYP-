from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

match=sb.matches(competition_id=2,season_id=27)
sourceFile = open('TOT_V_UNITED.txt', 'w')
print('Tottenham VS Manchester United\n--------------------------------',file=sourceFile)

#print(match.to_markdown())

TOT_UNI_events=sb.events(match_id=3754065)
for i in range(380):
    if match['match_id'][i] == 3754065:
        print(match['home_team'][i], ':', match['home_score'][i], '-', match['away_team'][i], ':', match['away_score'][i],file=sourceFile)

print(TOT_UNI_events.head(10).to_markdown())

tact = TOT_UNI_events[TOT_UNI_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_TOT = tact[tact['team'] == 'Tottenham Hotspur']
tact_UNI = tact[tact['team'] == 'Leicester']
tact_TOT = tact_TOT['tactics']
tact_UNI = tact_UNI['tactics']
print(tact_TOT.to_markdown())
print(tact_UNI.to_markdown())

dict_TOT = tact_TOT[0]['lineup']
dict_UNI = tact_UNI[1]['lineup']

lineup_TOT = pd.DataFrame.from_dict(dict_TOT)
lineup_UNI=pd.DataFrame.from_dict((dict_UNI))
print(lineup_TOT.to_markdown())
print(lineup_UNI.to_markdown())

players_TOT = {}
for i in range(len(lineup_TOT)):
    key = lineup_TOT.player[i]['name']
    val = lineup_TOT.jersey_number[i]
    players_TOT[key] = str(val)
print("\n",players_TOT)

players_UNI = {}
for i in range(len(lineup_UNI)):
    key = lineup_UNI.player[i]['name']
    val = lineup_UNI.jersey_number[i]
    players_UNI[key] = str(val)
print("\n",players_UNI)

#Events for TOT & UNITED

events_pn = TOT_UNI_events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player','pass_length']]
#print(TOT_UNI_events.columns)
events_TOT = events_pn[events_pn['team'] == 'Tottenham Hotspur']
events_UNI = events_pn[events_pn['team'] == 'Manchester United']

events_pn_UNI = events_UNI[events_UNI['type'] == 'Pass']
events_pn_TOT = events_TOT[events_TOT['type'] == 'Pass']

#Passing Events
events_pn_TOT['pass_maker'] = events_pn_TOT['player']
events_pn_TOT['pass_receiver'] = events_pn_TOT['player'].shift(-1)

events_pn_UNI['pass_maker'] = events_pn_UNI['player']
events_pn_UNI['pass_receiver'] = events_pn_UNI['player'].shift(-1)

print(events_pn_TOT.head(10).to_markdown())
print(events_pn_UNI.head(10).to_markdown())

events_pn_TOT2=events_pn_TOT
events_pn_UNI = events_pn_UNI[events_pn_UNI['pass_outcome'].isnull() == True].reset_index()
events_pn_TOT= events_pn_TOT[events_pn_TOT['pass_outcome'].isnull() == True].reset_index()
print(events_pn_TOT.head(10).to_markdown())

#Substitutes
substitution_UNI = events_UNI[events_UNI['type'] == 'Substitution']
substitution_TOT = events_TOT[events_TOT['type'] == 'Substitution']

substitution_TOT_minute = np.min(substitution_TOT['minute'])
substitution_TOT_minute_data = substitution_TOT[substitution_TOT['minute'] == substitution_TOT_minute]
substitution_TOT_second = np.min(substitution_TOT_minute_data['second'])
print("minute =", substitution_TOT_minute, "second =",  substitution_TOT_second)

substitution_UNI_minute = np.min(substitution_UNI['minute'])
substitution_UNI_minute_data = substitution_UNI[substitution_UNI['minute'] == substitution_UNI_minute]
substitution_UNI_second = np.min(substitution_UNI_minute_data['second'])
print("minute =", substitution_UNI_minute, "second =",  substitution_UNI_second)

events_pn_UNI = events_pn_UNI[(events_pn_UNI['minute'] <= substitution_UNI_minute)]

events_pn_TOT = events_pn_TOT[(events_pn_TOT['minute'] <= substitution_TOT_minute)]
events_pn_TOT2 = events_pn_TOT2[(events_pn_TOT2['minute'] <= substitution_TOT_minute)]

#Passing Locations (Splitting X & Y locations
Loc = events_pn_TOT['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_TOT['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_TOT['pass_maker_x'] = Loc['pass_maker_x']
events_pn_TOT['pass_maker_y'] = Loc['pass_maker_y']
events_pn_TOT['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_TOT['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_TOT = events_pn_TOT[['index','minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y','pass_length']]

print('hello\n',events_pn_TOT.head(10).to_markdown())

events_pn_TOT_PASS = events_pn_TOT2[events_pn_TOT2['type'] == 'Pass']
TOT_PASS_SUCCESS=events_pn_TOT_PASS[events_pn_TOT_PASS['pass_outcome'].isnull() == True]
TOT_PASS_INCOMPLETE=events_pn_TOT_PASS[events_pn_TOT_PASS['pass_outcome']=='Incomplete']





#print(events_pn_TOT_PASS.to_markdown())
print(TOT_PASS_SUCCESS.to_markdown())
print('\n Number of Complete Passes:',len(TOT_PASS_SUCCESS),file=sourceFile)
print(TOT_PASS_INCOMPLETE.to_markdown())
print('\n Number of Incomplete Passes:',len(TOT_PASS_INCOMPLETE),file=sourceFile)

#average Passing distance
avg_len_pass=TOT_PASS_SUCCESS.loc[:,'pass_length'].mean()
print('\n Average length of Passes:',avg_len_pass,file=sourceFile)

#Pass Success rate
pass_success_rate=100-((len(TOT_PASS_INCOMPLETE)/len(TOT_PASS_SUCCESS))*100)
print('\n Pass Success rate:',pass_success_rate,file=sourceFile)
print(len(events_pn_TOT_PASS))

#average passing location
av_loc_TOT = events_pn_TOT.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
av_loc_TOT.columns = ['pass_maker_x', 'pass_maker_y', 'count']
print(av_loc_TOT.to_markdown())

pass_TOT = events_pn_TOT.groupby(['pass_maker', 'pass_receiver']).index.count().reset_index()
pass_TOT.rename(columns = {'index':'number_of_passes'}, inplace = True)
pass_TOT = pass_TOT.merge(av_loc_TOT, left_on = 'pass_maker', right_index = True)
pass_TOT = pass_TOT.merge(av_loc_TOT, left_on = 'pass_receiver', right_index = True, suffixes = ['', '_receipt'])
pass_TOT.rename(columns = {'pass_maker_x_receipt':'pass_receiver_x', 'pass_maker_y_receipt':'pass_receiver_y', 'count_receipt':'number_of_passes_received'}, inplace = True)
pass_TOT = pass_TOT[pass_TOT['pass_maker'] != pass_TOT['pass_receiver']].reset_index()
print(pass_TOT.to_markdown())

#Using player Jersey Number
pass_TOT_new = pass_TOT.replace({"pass_maker": players_TOT, "pass_receiver": players_TOT})
print(pass_TOT_new.to_markdown())

#Passing graph on pitch
pitch = Pitch(pitch_color='grass', goal_type='box', line_color='white', stripe=True)
fig, ax = pitch.draw()
arrows = pitch.arrows(pass_TOT.pass_maker_x, pass_TOT.pass_maker_y,
                      pass_TOT.pass_receiver_x, pass_TOT.pass_receiver_y, lw=5,
                      color='black', zorder=1, ax=ax)
nodes = pitch.scatter(av_loc_TOT.pass_maker_x, av_loc_TOT.pass_maker_y,
                      s=350, color='white', edgecolors='black', linewidth=1, alpha=1, ax=ax)

for index, row in av_loc_TOT.iterrows():
    pitch.annotate(players_TOT[row.name], xy=(row.pass_maker_x, row.pass_maker_y), c='black', va='center', ha='center',
                   size=10, ax=ax)
plt.title("Pass network for Tottenham against Manchester United", size = 20)
plt.show()


#Graph of Passing Network
pass_TOT_new = pass_TOT_new[['pass_maker', 'pass_receiver', 'number_of_passes']]
print(pass_TOT_new.to_markdown())

L_TOT = pass_TOT_new.apply(tuple, axis=1).tolist()
print(L_TOT)

G_TOT = nx.DiGraph()

for i in range(len(L_TOT)):
    G_TOT.add_edge(L_TOT[i][0], L_TOT[i][1], weight = L_TOT[i][2])

edges_TOT = G_TOT.edges()
weights_TOT = [G_TOT[u][v]['weight'] for u, v in edges_TOT]

nx.draw(G_TOT, node_size=800, with_labels=True, node_color='white', width = weights_TOT)
plt.gca().collections[0].set_edgecolor('black') # sets the edge color of the nodes to black
plt.title("Pass network for Tottenham vs Manchester United", size = 20)
plt.show()

events_pn = TOT_UNI_events[['minute', 'second', 'team', 'type', 'location','player','shot_outcome','pass_shot_assist']]
events_TOT = events_pn[events_pn['team'] == 'Tottenham Hotspur']

events_pn_TOT = events_TOT[events_TOT['type'] == 'Shot']
print(events_pn_TOT.to_markdown())
events_pn_TOT=events_TOT[events_TOT['type']=='Pass']
events_pn_TOT=events_TOT[events_TOT['pass_shot_assist'].isnull()!=True].reset_index()
print(events_pn_TOT.to_markdown())

sourceFile.close()