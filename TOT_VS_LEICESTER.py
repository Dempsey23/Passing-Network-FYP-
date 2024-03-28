from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

import SpectralClustering

match=sb.matches(competition_id=2,season_id=27)
sourceFile = open('TOT_V_LEI.txt', 'w')
print('Tottenham VS Leicester City\n--------------------------------',file=sourceFile)

#print(match.to_markdown())

TOT_LEI_events=sb.events(match_id=3754065)
for i in range(380):
    if match['match_id'][i] == 3754065:
        print(match['home_team'][i], ':', match['home_score'][i], '-', match['away_team'][i], ':', match['away_score'][i],file=sourceFile)

print(TOT_LEI_events.head(10).to_markdown())

tact = TOT_LEI_events[TOT_LEI_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_TOT = tact[tact['team'] == 'Tottenham Hotspur']
tact_LEI = tact[tact['team'] == 'Leicester City']
tact_TOT = tact_TOT['tactics']
tact_LEI = tact_LEI['tactics']
print(tact_TOT.to_markdown())
print(tact_LEI.to_markdown())
dict_TOT = tact_TOT[0]['lineup']
dict_LEI = tact_LEI[1]['lineup']

lineup_TOT = pd.DataFrame.from_dict(dict_TOT)
lineup_LEI=pd.DataFrame.from_dict((dict_LEI))
print(lineup_TOT.to_markdown())
print(lineup_LEI.to_markdown())

players_TOT = {}
for i in range(len(lineup_TOT)):
    key = lineup_TOT.player[i]['name']
    val = lineup_TOT.jersey_number[i]
    players_TOT[key] = str(val)
print("\n",players_TOT)

players_LEI = {}
for i in range(len(lineup_LEI)):
    key = lineup_LEI.player[i]['name']
    val = lineup_LEI.jersey_number[i]
    players_LEI[key] = str(val)
print("\n",players_LEI)

#Events for TOT & LEITED

events_pn = TOT_LEI_events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player','pass_length']]
#print(TOT_LEI_events.columns)
events_TOT = events_pn[events_pn['team'] == 'Tottenham Hotspur']
events_LEI = events_pn[events_pn['team'] == 'Leicester City']

events_pn_LEI = events_LEI[events_LEI['type'] == 'Pass']
events_pn_TOT = events_TOT[events_TOT['type'] == 'Pass']
print(len(events_pn_LEI),'\n',events_pn_LEI.to_markdown())
#Passing Events
events_pn_TOT['pass_maker'] = events_pn_TOT['player']
events_pn_TOT['pass_receiver'] = events_pn_TOT['player'].shift(-1)

events_pn_LEI['pass_maker'] = events_pn_LEI['player']
events_pn_LEI['pass_receiver'] = events_pn_LEI['player'].shift(-1)

print(events_pn_TOT.head(10).to_markdown())
print(events_pn_LEI.head(10).to_markdown())

events_pn_TOT2=events_pn_TOT
events_pn_LEI2=events_pn_LEI
print('hello',events_pn_LEI2.head(20).to_markdown())
events_pn_LEI = events_pn_LEI[events_pn_LEI['pass_outcome'].isnull() == True].reset_index()
events_pn_TOT = events_pn_TOT[events_pn_TOT['pass_outcome'].isnull() == True].reset_index()
print(events_pn_TOT.head(10).to_markdown())

#Substitutes
substitution_LEI = events_LEI[events_LEI['type'] == 'Substitution']
substitution_TOT = events_TOT[events_TOT['type'] == 'Substitution']

substitution_TOT_minute = np.min(substitution_TOT['minute'])
substitution_TOT_minute_data = substitution_TOT[substitution_TOT['minute'] == substitution_TOT_minute]
substitution_TOT_second = np.min(substitution_TOT_minute_data['second'])
print("minute =", substitution_TOT_minute, "second =",  substitution_TOT_second)

substitution_LEI_minute = np.min(substitution_LEI['minute'])
substitution_LEI_minute_data = substitution_LEI[substitution_LEI['minute'] == substitution_LEI_minute]
substitution_LEI_second = np.min(substitution_LEI_minute_data['second'])
print("minute =", substitution_LEI_minute, "second =",  substitution_LEI_second)

events_pn_LEI = events_pn_LEI[(events_pn_LEI['minute'] <= substitution_LEI_minute)]
events_pn_LEI2= events_pn_LEI2[(events_pn_LEI2['minute'] <= substitution_LEI_minute)]

events_pn_TOT = events_pn_TOT[(events_pn_TOT['minute'] <= 69)]
events_pn_TOT2 = events_pn_TOT2[(events_pn_TOT2['minute'] <= 69)]

#Passing Locations (Splitting X & Y locations TOTTENHAM
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

#Passing Locations (Splitting X & Y locations Leicester
Loc = events_pn_LEI['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_LEI['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_LEI['pass_maker_x'] = Loc['pass_maker_x']
events_pn_LEI['pass_maker_y'] = Loc['pass_maker_y']
events_pn_LEI['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_LEI['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_LEI = events_pn_LEI[['index','minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y','pass_length']]

print('hello\n',len(events_pn_LEI2),events_pn_LEI2.head(10).to_markdown())

events_pn_LEI_PASS = events_pn_LEI2[events_pn_LEI2['type'] == 'Pass']
LEI_PASS_SUCCESS=events_pn_LEI_PASS[events_pn_LEI_PASS['pass_outcome'].isnull() == True]
LEI_PASS_INCOMPLETE=events_pn_LEI_PASS[events_pn_LEI_PASS['pass_outcome'].isnull()==False]




#print(events_pn_TOT_PASS.to_markdown())
#Completed Passes
print('\n',len(events_pn_TOT2),TOT_PASS_SUCCESS.head(10).to_markdown())
print('\n TOTTENHAM- Number of Complete Passes:',len(TOT_PASS_SUCCESS),file=sourceFile)
print(TOT_PASS_INCOMPLETE.head(10).to_markdown())
print('\n TOTTENHAM- Number of Incomplete Passes:',len(TOT_PASS_INCOMPLETE),file=sourceFile)

print(LEI_PASS_SUCCESS.head(10).to_markdown())
print('\n LEICESTER- Number of Complete Passes:',len(LEI_PASS_SUCCESS),file=sourceFile)
print(LEI_PASS_INCOMPLETE.head(10).to_markdown())
print('\n LEICESTER- Number of Incomplete Passes:',len(LEI_PASS_INCOMPLETE),file=sourceFile)

#average Passing distance
avg_len_pass=TOT_PASS_SUCCESS.loc[:,'pass_length'].mean()
print('\n TOTTENHAM- Average length of Passes:',avg_len_pass,file=sourceFile)

avg_len_pass=LEI_PASS_SUCCESS.loc[:,'pass_length'].mean()
print('\n LEICESTER- Average length of Passes:',avg_len_pass,file=sourceFile)

#Pass Success rate
pass_success_rate=100-((len(TOT_PASS_INCOMPLETE)/len(TOT_PASS_SUCCESS))*100)
print('\n TOTTENHAM- Pass Success rate:',pass_success_rate,file=sourceFile)
print(len(events_pn_TOT_PASS))

pass_success_rate=100-((len(LEI_PASS_INCOMPLETE)/len(LEI_PASS_SUCCESS))*100)
print('\n LEICESTER- Pass Success rate:',pass_success_rate,file=sourceFile)
print(len(events_pn_LEI_PASS))
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

print(pass_TOT.head(10).to_markdown())
print(players_TOT)
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
print('playesr',players_TOT)



for index, row in av_loc_TOT.iterrows():
    pitch.annotate(players_TOT[row.name], xy=(row.pass_maker_x, row.pass_maker_y), c='black', va='center', ha='center',
                   size=10, ax=ax)
plt.title("Pass network for Tottenham against Leicester City", size = 20)
plt.savefig("TOT_VS_LEI_PASS_GRAPH.png")
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
plt.title("Pass network for Tottenham vs Leicester City", size = 20)
plt.savefig("TOT_VS_LEI_WEIGHTED_PASS_GRAPH.png")
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
    plt.title("Player pass degrees for Tottenham VS Leicester", size=16)
    plt.savefig("TOT_VS_LEI_DEGREE.png")
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
    plt.title("Player pass indegrees for Tottenham vs Leicester", size=16)
    plt.savefig("TOT_VS_LEI_INDEGREE.png")
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
    plt.title("Player pass outdegrees for Tottenham vs Leicester", size=16)
    plt.savefig("TOT_VS_LEI_OUTDEGREE.png")
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
    plt.title("Modified pass network for Tottenham vs Leicester", size = 20)
    plt.savefig("TOT_VS_LEI_MODIFIED_GRAPH.png")
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
    #Eigenvector
    eigenvector=nx.eigenvector_centrality(G_TOT)
    sorted_eigen=sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
    # Print eigenvector centrality values
    print("\n------------\nEigenvector Centrality (Highest to Lowest):",file=sourceFile)
    for node, centrality in sorted_eigen:
        print(f"Node {node}: {centrality}",file=sourceFile)
    shortest_path=nx.shortest_path(G_TOT_mod,weight='weight')
    return shortest_path

def goal():
    print('\nGoals\n--------------------------------', file=sourceFile)
    events_pn = TOT_LEI_events[['minute', 'second', 'team', 'type', 'location','shot_end_location','player','shot_outcome']]
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
    events_shot_off_wayward = events_pn[(events_pn['shot_outcome'] == 'Off T') | (events_pn['shot_outcome'] == 'Wayward')| (events_pn['shot_outcome'] == 'Saved to Post')]
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
                 ax=ax, color='red', width=3, label='Post/Off T / Wayward')
    pitch.scatter(events_shot_off_wayward.location_x, events_shot_off_wayward.location_y, ax=ax, color='red')
    pitch.arrows(events_shot_saved_blocked.location_x, events_shot_saved_blocked.location_y,
                 events_shot_saved_blocked.shot_end_location_x, events_shot_saved_blocked.shot_end_location_y,
                 ax=ax, color='orange', width=3, label='Saved / Blocked')
    pitch.scatter(events_shot_saved_blocked.location_x, events_shot_saved_blocked.location_y, ax=ax,
                  color='orange')

    # General plot code
    ax.legend(handlelength=3, edgecolor='None', fontsize=10)
    plt.title("Tottenham shot and heat map")
    plt.savefig("TOT_VS_LEI_GOAL.png")
    plt.show()

#Adjacency Matrix
A_TOT = nx.adjacency_matrix(G_TOT)
A_TOT=A_TOT.todense()
sns.heatmap(A_TOT, annot = True, cmap ='gnuplot')
plt.title("Adjacency matrix for Tottenham vs Leicester pass network")
plt.show()
goal()
SpectralClustering.spectral(A_TOT,G_TOT,"TOT_VS_LEI_SPECTRAL")
player_degree()
short_path=modified_graph()
sourceFile.close()