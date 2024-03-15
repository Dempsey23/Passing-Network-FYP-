from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

import SpectralClustering

match=sb.matches(competition_id=2,season_id=27)
sourceFile = open('TOT_V_NEW.txt', 'w')
print('\n Tottenham VS Newcastle United\n--------------------------------',file=sourceFile)

#print(match.to_markdown())

TOT_NEW_events=sb.events(match_id=3754276)
print(TOT_NEW_events.head(10).to_markdown())

tact = TOT_NEW_events[TOT_NEW_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_TOT = tact[tact['team'] == 'Tottenham Hotspur']
tact_NEW = tact[tact['team'] == 'Newcastle United']
tact_TOT = tact_TOT['tactics']
tact_NEW = tact_NEW['tactics']
print(tact_TOT.to_markdown())
print(tact_NEW.to_markdown())

dict_TOT = tact_TOT[0]['lineup']
dict_NEW = tact_NEW[1]['lineup']

lineup_TOT = pd.DataFrame.from_dict(dict_TOT)
lineup_NEW=pd.DataFrame.from_dict((dict_NEW))
print(lineup_TOT.to_markdown())
print(lineup_NEW.to_markdown())

players_TOT = {}
for i in range(len(lineup_TOT)):
    key = lineup_TOT.player[i]['name']
    val = lineup_TOT.jersey_number[i]
    players_TOT[key] = str(val)
print("\n",players_TOT)

players_NEW = {}
for i in range(len(lineup_NEW)):
    key = lineup_NEW.player[i]['name']
    val = lineup_NEW.jersey_number[i]
    players_NEW[key] = str(val)
print("\n",players_NEW)

#Events for TOT & UNITED

events_pn = TOT_NEW_events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player','pass_length']]
#print(TOT_UNI_events.columns)
events_TOT = events_pn[events_pn['team'] == 'Tottenham Hotspur']
events_NEW = events_pn[events_pn['team'] == 'Newcastle United']

events_pn_NEW = events_NEW[events_NEW['type'] == 'Pass']
events_pn_TOT = events_TOT[events_TOT['type'] == 'Pass']

#Passing Events
events_pn_TOT['pass_maker'] = events_pn_TOT['player']
events_pn_TOT['pass_receiver'] = events_pn_TOT['player'].shift(-1)

events_pn_NEW['pass_maker'] = events_pn_NEW['player']
events_pn_NEW['pass_receiver'] = events_pn_NEW['player'].shift(-1)

print(events_pn_TOT.head(10).to_markdown())
print(events_pn_NEW.head(10).to_markdown())

events_pn_TOT2=events_pn_TOT
events_pn_NEW = events_pn_NEW[events_pn_NEW['pass_outcome'].isnull() == True].reset_index()
events_pn_TOT= events_pn_TOT[events_pn_TOT['pass_outcome'].isnull() == True].reset_index()
print(events_pn_TOT.head(10).to_markdown())

#Substitutes
substitution_NEW = events_NEW[events_NEW['type'] == 'Substitution']
substitution_TOT = events_TOT[events_TOT['type'] == 'Substitution']

substitution_TOT_minute = np.min(substitution_TOT['minute'])
substitution_TOT_minute_data = substitution_TOT[substitution_TOT['minute'] == substitution_TOT_minute]
substitution_TOT_second = np.min(substitution_TOT_minute_data['second'])
print("minute =", substitution_TOT_minute, "second =",  substitution_TOT_second)

substitution_NEW_minute = np.min(substitution_NEW['minute'])
substitution_NEW_minute_data = substitution_NEW[substitution_NEW['minute'] == substitution_NEW_minute]
substitution_NEW_second = np.min(substitution_NEW_minute_data['second'])
print("minute =", substitution_NEW_minute, "second =",  substitution_NEW_second)

events_pn_UNI = events_pn_NEW[(events_pn_NEW['minute'] <= substitution_NEW_minute)]

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
plt.title("Pass network for Tottenham against Newcastle United", size = 20)
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
def player_degree():
    deg_TOT = dict(nx.degree(G_TOT))  # prepares a dictionary with jersey numbers as the node ids, i.e, the dictionary keys and degrees as the dictionary values
    degree_TOT = pd.DataFrame.from_dict(list(deg_TOT.items()))  # convert a dictionary to a pandas dataframe
    degree_TOT.rename(columns={0: 'jersey_number', 1: 'node_degree'}, inplace=True)
    X = list(deg_TOT.keys())
    Y = list(deg_TOT.values())
    sns.barplot(x=Y, y=X, palette="magma")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("degree")
    plt.title("Player pass degrees for Tottenham VS United", size=16)
    plt.show()

    indeg_TOT = dict(G_TOT.in_degree())
    indegree_TOT = pd.DataFrame.from_dict(list(indeg_TOT.items()))
    indegree_TOT.rename(columns={0: 'jersey_number', 1: 'node_indegree'}, inplace=True)
    print(indegree_TOT.to_markdown())
    X = list(indeg_TOT.keys())
    Y = list(indeg_TOT.values())
    sns.barplot(x=Y, y=X, palette="hls")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("indegree")
    plt.title("Player pass indegrees for Tottenham vs Newcastle", size=16)
    plt.show()

    outdeg_TOT = dict(G_TOT.out_degree())
    outdegree_TOT = pd.DataFrame.from_dict(list(outdeg_TOT.items()))
    outdegree_TOT.rename(columns={0: 'jersey_number', 1: 'node_outdegree'}, inplace=True)
    print(outdegree_TOT.to_markdown())
    X = list(outdeg_TOT.keys())
    Y = list(outdeg_TOT.values())
    sns.barplot(x=Y, y=X, palette="hls")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("outdegree")
    plt.title("Player pass outdegrees for Tottenham vs Newcastle", size=16)
    plt.show()
def modified_graph():
    pass_TOT_mod = pass_TOT_new[['pass_maker', 'pass_receiver']]
    pass_TOT_mod['1/nop'] = 1/pass_TOT_new['number_of_passes']
    L_TOT_mod = pass_TOT_mod.apply(tuple, axis=1).tolist()
    G_TOT_mod = nx.DiGraph()

    for i in range(len(L_TOT_mod)):
        G_TOT_mod.add_edge(L_TOT_mod[i][0], L_TOT_mod[i][1], weight = L_TOT_mod[i][2])

    edges_TOT_mod = G_TOT_mod.edges()
    weights_TOT_mod = [G_TOT_mod[u][v]['weight'] for u, v in edges_TOT_mod]

    nx.draw(G_TOT_mod, node_size=800, with_labels=True, node_color='white', width = weights_TOT_mod)
    plt.gca().collections[0].set_edgecolor('black')
    plt.title("Modified pass network for Tottenham vs Newcastle", size = 20)
    plt.show()
    E_TOT = nx.eccentricity(G_TOT_mod)
    print(E_TOT)
    print('\nEccentricity value of Modified Tottenham Passing Graph',E_TOT,file=sourceFile)
    av_E_TOT = sum(list(E_TOT.values())) / len(E_TOT)
    print(av_E_TOT)
    print('Average Eccentricity Values for Tottenham Passing Graph:',av_E_TOT,file=sourceFile)
    cc_TOT = nx.average_clustering(G_TOT, weight='weight')
    print(cc_TOT)
    print('Average Clustering of Passing Network:',cc_TOT,file=sourceFile)
    closeness_centrality = nx.closeness_centrality(G_TOT)
    # Print closeness centrality for each node
    for node, closeness in closeness_centrality.items():
        print(f"Node {node}: Closeness Centrality = {closeness}",file=sourceFile)
    betweeness=nx.betweenness_centrality(G_TOT, weight = 'weight')
    max_bc = max(betweeness, key = betweeness.get)
    print('Max Betweeness:',max_bc,file=sourceFile)
    # Calculate the density of the graph
    density = nx.density(G_TOT)
    print("Graph Density:", density,file=sourceFile)
    
def goal():
    print('\nGoals\n--------------------------------', file=sourceFile)
    events_pn = TOT_NEW_events[['minute', 'second', 'team', 'type', 'location','shot_end_location','player','shot_outcome']]
    events_pn = events_pn[events_pn['team'] == 'Tottenham Hotspur']

    events_pn = events_pn[events_pn['type'] == 'Shot']
    print(events_pn.to_markdown())
    shot_loc=events_pn['location']
    shot_loc = pd.DataFrame(shot_loc.to_list(), columns=['location_x', 'location_y'])
    print(shot_loc.to_markdown())
    print('Total Shots taken:', len(events_pn), '\n', file=sourceFile)

    shot_end_loc=events_pn['shot_end_location']
    shot_end_loc=pd.DataFrame(shot_end_loc.to_list(), columns=['shot_end_location_x', 'shot_end_location_y', 'shot_end_location_z'])
    print(shot_end_loc.to_markdown())

    events_pn = events_pn.reset_index()
    events_pn['location_x'] = shot_loc['location_x']
    events_pn['location_y'] = shot_loc['location_y']
    events_pn['shot_end_location_x'] = shot_end_loc['shot_end_location_x']
    events_pn['shot_end_location_y'] = shot_end_loc['shot_end_location_y']
    print(events_pn.to_markdown())

    events_pn = events_pn[
        ['team', 'minute', 'player', 'location_x', 'location_y', 'shot_end_location_x', 'shot_end_location_y',
         'shot_outcome']]
    print(events_pn.to_markdown())

    events_shot_Goal = events_pn[events_pn['shot_outcome'] == 'Goal']
    events_shot_off_wayward = events_pn[(events_pn['shot_outcome'] == 'Off T') | (events_pn['shot_outcome'] == 'Wayward')]
    events_shot_saved_blocked = events_pn[(events_pn['shot_outcome'] == 'Saved') | (events_pn['shot_outcome'] == 'Blocked')]
    print('Shots on target:',len(events_shot_saved_blocked)+len(events_shot_Goal),'\n',file=sourceFile)
    print('Shots off target:',len(events_shot_off_wayward),'\n',file=sourceFile)
    # Pitch drawing code
    pitch = Pitch(pitch_color='green', line_color='white', goal_type='box')
    fig, ax = pitch.draw()

    # Heat map code
    res = sns.kdeplot(x=events_pn['location_x'], y=events_pn['location_y'], fill=True,thresh=0.05, alpha=0.5, levels=10,cmap=sns.color_palette("magma", as_cmap=True))

    # Pass map code
    pitch.arrows(events_shot_Goal.location_x, events_shot_Goal.location_y,
                 events_shot_Goal.shot_end_location_x, events_shot_Goal.shot_end_location_y, ax=ax,
                 color='green', width=3, label='Goals')
    pitch.scatter(events_shot_Goal.location_x, events_shot_Goal.location_y, ax=ax, color='green')
    pitch.arrows(events_shot_off_wayward.location_x, events_shot_off_wayward.location_y,
                 events_shot_off_wayward.shot_end_location_x, events_shot_off_wayward.shot_end_location_y,
                 ax=ax, color='red', width=3, label='Off T / Wayward')
    pitch.scatter(events_shot_off_wayward.location_x, events_shot_off_wayward.location_y, ax=ax, color='red')
    pitch.arrows(events_shot_saved_blocked.location_x, events_shot_saved_blocked.location_y,
                 events_shot_saved_blocked.shot_end_location_x, events_shot_saved_blocked.shot_end_location_y,
                 ax=ax, color='orange', width=3, label='Saved / Blocked')
    pitch.scatter(events_shot_saved_blocked.location_x, events_shot_saved_blocked.location_y, ax=ax,
                  color='orange')

    # General plot code
    ax.legend(handlelength=3, edgecolor='None', fontsize=10)
    plt.title("Tottenham shot and heat map")
    plt.show()
#Adjancency matrix
A_TOT = nx.adjacency_matrix(G_TOT)
A_TOT=A_TOT.todense()
sns.heatmap(A_TOT, annot = True, cmap ='gnuplot')
plt.title("Adjacency matrix for Tottenham vs Newcaslte pass network")
plt.show()
goal()
player_degree()
SpectralClustering.spectral(A_TOT,G_TOT)
modified_graph()
sourceFile.close()