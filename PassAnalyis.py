from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

comp = sb.competitions()

#print(comp.to_markdown())

match=sb.matches(competition_id=2,season_id=27)
#print(match.to_markdown())
match_count=0
for i in range(380):
    if match['home_team'][i]=='Chelsea' and match['away_team'][i]=='Tottenham Hotspur':
        print(match['home_team'][i],' ',match['away_team'][i],' ',match['match_id'][i])
        match_count+=1

TOT_CHELS_events=sb.events(match_id=3754092)
print(TOT_CHELS_events.head(10).to_markdown())


tact = TOT_CHELS_events[TOT_CHELS_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_TOT = tact[tact['team'] == 'Tottenham Hotspur']
tact_CHEL = tact[tact['team'] == 'Chelsea']
tact_TOT = tact_TOT['tactics']
tact_CHEL = tact_CHEL['tactics']

print(tact_TOT.to_markdown())
print(tact_CHEL.to_markdown())

dict_CHEL = tact_CHEL[0]['lineup']
dict_TOT = tact_TOT[1]['lineup']

lineup_TOT = pd.DataFrame.from_dict(dict_TOT)
lineup_CHEL=pd.DataFrame.from_dict((dict_CHEL))
print(lineup_TOT.to_markdown())
print(lineup_CHEL.to_markdown())

players_CHEL = {}
for i in range(len(lineup_CHEL)):
    key = lineup_CHEL.player[i]['name']
    val = lineup_CHEL.jersey_number[i]
    players_CHEL[key] = str(val)
print("\n",players_CHEL)

players_TOT = {}
for i in range(len(lineup_TOT)):
    key = lineup_TOT.player[i]['name']
    val = lineup_TOT.jersey_number[i]
    players_TOT[key] = str(val)
print("\n",players_TOT)

events_pn = TOT_CHELS_events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player']]

events_CHEL = events_pn[events_pn['team'] == 'Chelsea']
events_TOT = events_pn[events_pn['team'] == 'Tottenham Hotspur']

print(events_TOT.head(10).to_markdown())
print(events_CHEL.head(10).to_markdown())

events_pn_CHEL = events_CHEL[events_CHEL['type'] == 'Pass']
events_pn_TOT = events_TOT[events_TOT['type'] == 'Pass']

print(events_pn_TOT.head(10).to_markdown())
print(events_pn_CHEL.head(10).to_markdown())

events_pn_CHEL['pass_maker'] = events_pn_CHEL['player']
events_pn_CHEL['pass_receiver'] = events_pn_CHEL['player'].shift(-1)

events_pn_TOT['pass_maker'] = events_pn_TOT['player']
events_pn_TOT['pass_receiver'] = events_pn_TOT['player'].shift(-1)

print(events_pn_TOT.head(10).to_markdown())
print(events_pn_CHEL.head(10).to_markdown())

events_pn_CHEL = events_pn_CHEL[events_pn_CHEL['pass_outcome'].isnull() == True].reset_index()
events_pn_TOT= events_pn_TOT[events_pn_TOT['pass_outcome'].isnull() == True].reset_index()
print(events_pn_TOT.head(10).to_markdown())

substitution_CHEL = events_CHEL[events_CHEL['type'] == 'Substitution']
substitution_TOT = events_TOT[events_TOT['type'] == 'Substitution']

substitution_CHEL_minute = np.min(substitution_CHEL['minute'])
substitution_CHEL_minute_data = substitution_CHEL[substitution_CHEL['minute'] == substitution_CHEL_minute]
substitution_CHEL_second = np.min(substitution_CHEL_minute_data['second'])
print("minute =", substitution_CHEL_minute, "second =",  substitution_CHEL_second)

substitution_TOT_minute = np.min(substitution_TOT['minute'])
substitution_TOT_minute_data = substitution_TOT[substitution_TOT['minute'] == substitution_TOT_minute]
substitution_TOT_second = np.min(substitution_TOT_minute_data['second'])
print("minute =", substitution_TOT_minute, "second =",  substitution_TOT_second)

events_pn_CHEL = events_pn_CHEL[(events_pn_CHEL['minute'] <= 45)]

events_pn_TOT = events_pn_TOT[(events_pn_TOT['minute'] <= substitution_TOT_minute)]

print("hello\n",events_pn_CHEL.to_markdown())



Loc = events_pn_CHEL['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_CHEL['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_CHEL['pass_maker_x'] = Loc['pass_maker_x']
events_pn_CHEL['pass_maker_y'] = Loc['pass_maker_y']
events_pn_CHEL['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_CHEL['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_CHEL = events_pn_CHEL[['index', 'minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y']]

print(events_pn_CHEL.head(10).to_markdown())

Loc = events_pn_TOT['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_TOT['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_TOT['pass_maker_x'] = Loc['pass_maker_x']
events_pn_TOT['pass_maker_y'] = Loc['pass_maker_y']
events_pn_TOT['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_TOT['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_TOT = events_pn_TOT[['index', 'minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y']]

print(events_pn_TOT.head(10).to_markdown())

av_loc_CHEL = events_pn_CHEL.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
av_loc_CHEL.columns = ['pass_maker_x', 'pass_maker_y', 'count']

av_loc_TOT = events_pn_TOT.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
av_loc_TOT.columns = ['pass_maker_x', 'pass_maker_y', 'count']
print(av_loc_CHEL.to_markdown(),"\n")
print(av_loc_TOT.to_markdown())

pass_CHEL = events_pn_CHEL.groupby(['pass_maker', 'pass_receiver']).index.count().reset_index()
pass_TOT = events_pn_TOT.groupby(['pass_maker', 'pass_receiver']).index.count().reset_index()
print(pass_CHEL.head(10).to_markdown(),"\n",pass_TOT.head(10).to_markdown())

pass_CHEL.rename(columns = {'index':'number_of_passes'}, inplace = True)
pass_TOT.rename(columns = {'index':'number_of_passes'}, inplace = True)
print(pass_CHEL.head(10).to_markdown(),"\n",pass_TOT.head(10).to_markdown())

pass_CHEL = pass_CHEL.merge(av_loc_CHEL, left_on = 'pass_maker', right_index = True)
pass_TOT = pass_TOT.merge(av_loc_TOT, left_on = 'pass_maker', right_index = True)

print(pass_CHEL.head(10).to_markdown(),"\n",pass_TOT.head(10).to_markdown())

pass_CHEL = pass_CHEL.merge(av_loc_CHEL, left_on = 'pass_receiver', right_index = True, suffixes = ['', '_receipt'])
pass_CHEL.rename(columns = {'pass_maker_x_receipt':'pass_receiver_x', 'pass_maker_y_receipt':'pass_receiver_y', 'count_receipt':'number_of_passes_received'}, inplace = True)
pass_CHEL = pass_CHEL[pass_CHEL['pass_maker'] != pass_CHEL['pass_receiver']].reset_index()

pass_TOT = pass_TOT.merge(av_loc_TOT, left_on = 'pass_receiver', right_index = True, suffixes = ['', '_receipt'])
pass_TOT.rename(columns = {'pass_maker_x_receipt':'pass_receiver_x', 'pass_maker_y_receipt':'pass_receiver_y', 'count_receipt':'number_of_passes_received'}, inplace = True)
pass_TOT = pass_TOT[pass_TOT['pass_maker'] != pass_TOT['pass_receiver']].reset_index()
print(pass_CHEL.to_markdown(),"\n",pass_TOT.to_markdown())

pass_CHEL_new = pass_CHEL.replace({"pass_maker": players_CHEL, "pass_receiver": players_CHEL})
pass_TOT_new = pass_TOT.replace({"pass_maker": players_TOT, "pass_receiver": players_TOT})
print(pass_CHEL_new.to_markdown(),"\n",pass_TOT_new.to_markdown())

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
plt.title("Pass network for Tottenham against Chelsea", size = 20)
plt.show()





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
plt.title("Pass network for Tottenham vs Chelsea", size = 20)
plt.show()

deg_TOT = dict(nx.degree(G_TOT)) # prepares a dictionary with jersey numbers as the node ids, i.e, the dictionary keys and degrees as the dictionary values
degree_TOT = pd.DataFrame.from_dict(list(deg_TOT.items())) # convert a dictionary to a pandas dataframe
degree_TOT.rename(columns = {0:'jersey_number', 1: 'node_degree'}, inplace = True)
print(degree_TOT.to_markdown())

X = list(deg_TOT.keys())
Y = list(deg_TOT.values())
sns.barplot(x = Y, y = X, palette = "magma")
plt.xticks(range(0, max(Y)+5, 2))
plt.ylabel("Player Jersey number")
plt.xlabel("degree")
plt.title("Player pass degrees for Tottenham vs Chelsea", size = 16)
plt.show()


indeg_TOT = dict(G_TOT.in_degree())
indegree_TOT = pd.DataFrame.from_dict(list(indeg_TOT.items()))
indegree_TOT.rename(columns = {0:'jersey_number', 1: 'node_indegree'}, inplace = True)
print(indegree_TOT.to_markdown())
X = list(indeg_TOT.keys())
Y = list(indeg_TOT.values())
sns.barplot(x = Y, y = X, palette = "hls")
plt.xticks(range(0, max(Y)+5, 2))
plt.ylabel("Player Jersey number")
plt.xlabel("indegree")
plt.title("Player pass indegrees for Tottenham vs Liverpool", size = 16)
plt.show()

outdeg_TOT = dict(G_TOT.out_degree())
outdegree_TOT = pd.DataFrame.from_dict(list(outdeg_TOT.items()))
outdegree_TOT.rename(columns = {0:'jersey_number', 1: 'node_outdegree'}, inplace = True)
print(outdegree_TOT.to_markdown())
X = list(outdeg_TOT.keys())
Y = list(outdeg_TOT.values())
sns.barplot(x = Y, y = X, palette = "hls")
plt.xticks(range(0, max(Y)+5, 2))
plt.ylabel("Player Jersey number")
plt.xlabel("outdegree")
plt.title("Player pass outdegrees for Tottenham vs Chelsea", size = 16)
plt.show()


A_TOT = nx.adjacency_matrix(G_TOT)
A_TOT = A_TOT.todense()

sns.heatmap(A_TOT, annot = True, cmap ='gnuplot')
plt.title("Adjacency matrix for Real Madrid's pass network")
plt.show()

r_TOT = nx.degree_pearson_correlation_coefficient(G_TOT, weight = 'weight')
print(r_TOT)

pass_TOT_mod = pass_TOT_new[['pass_maker', 'pass_receiver']]
pass_TOT_mod['1/nop'] = 1/pass_TOT_new['number_of_passes']
print(pass_TOT_mod.head(5).to_markdown())



L_TOT_mod = pass_TOT_mod.apply(tuple, axis=1).tolist()

G_TOT_mod = nx.DiGraph()

for i in range(len(L_TOT_mod)):
    G_TOT_mod.add_edge(L_TOT_mod[i][0], L_TOT_mod[i][1], weight = L_TOT_mod[i][2])

edges_TOT_mod = G_TOT_mod.edges()
weights_TOT_mod = [G_TOT_mod[u][v]['weight'] for u, v in edges_TOT_mod]

nx.draw(G_TOT_mod, node_size=800, with_labels=True, node_color='white', width = weights_TOT_mod)
plt.gca().collections[0].set_edgecolor('black')
plt.title("Modified pass network for Real Madrid vs Liverpool", size = 20)
plt.show()