from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis

import SpectralClustering

match=sb.matches(competition_id=2,season_id=27)
sourceFile = open('LEI_V_ARS.txt', 'w')
print('Leicester VS Arsenal\n--------------------------------',file=sourceFile)

#print(match.to_markdown())

LEI_ARS_events=sb.events(match_id=3754174)
for i in range(380):
    if match['match_id'][i] == 3754174:
        print(match['home_team'][i], ':', match['home_score'][i], '-', match['away_team'][i], ':', match['away_score'][i],file=sourceFile)

print(LEI_ARS_events.head(10).to_markdown())

tact = LEI_ARS_events[LEI_ARS_events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())

tact = tact[tact['type'] == 'Starting XI']
tact_LEI = tact[tact['team'] == 'Leicester City']
tact_ARS = tact[tact['team'] == 'Arsenal']
tact_LEI = tact_LEI['tactics']
tact_ARS = tact_ARS['tactics']
print(tact_LEI.to_markdown())
print(tact_ARS.to_markdown())

dict_LEI = tact_LEI[0]['lineup']
dict_ARS = tact_ARS[1]['lineup']

lineup_LEI = pd.DataFrame.from_dict(dict_LEI)
lineup_ARS=pd.DataFrame.from_dict(dict_ARS)
print(lineup_LEI.to_markdown())
print(lineup_ARS.to_markdown())

players_LEI = {}
for i in range(len(lineup_LEI)):
    key = lineup_LEI.player[i]['name']
    val = lineup_LEI.jersey_number[i]
    players_LEI[key] = str(val)
print("\n",players_LEI)

players_ARS = {}
for i in range(len(lineup_ARS)):
    key = lineup_ARS.player[i]['name']
    val = lineup_ARS.jersey_number[i]
    players_ARS[key] = str(val)
print("\n",players_ARS)

#Events for LEI & ARSTED

events_pn = LEI_ARS_events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player','pass_length']]
#print(LEI_ARS_events.columns)
events_LEI = events_pn[events_pn['team'] == 'Leicester City']
events_ARS = events_pn[events_pn['team'] == 'Arsenal FC']

events_pn_ARS = events_ARS[events_ARS['type'] == 'Pass']
events_pn_LEI = events_LEI[events_LEI['type'] == 'Pass']

#Passing Events
events_pn_LEI['pass_maker'] = events_pn_LEI['player']
events_pn_LEI['pass_receiver'] = events_pn_LEI['player'].shift(-1)

events_pn_ARS['pass_maker'] = events_pn_ARS['player']
events_pn_ARS['pass_receiver'] = events_pn_ARS['player'].shift(-1)

print(events_pn_LEI.head(10).to_markdown())
print(events_pn_ARS.head(10).to_markdown())

events_pn_LEI2=events_pn_LEI
events_pn_ARS = events_pn_ARS[events_pn_ARS['pass_outcome'].isnull() == True].reset_index()
events_pn_LEI= events_pn_LEI[events_pn_LEI['pass_outcome'].isnull() == True].reset_index()
print(events_pn_LEI.head(10).to_markdown())

#Substitutes
substitution_ARS = events_ARS[events_ARS['type'] == 'Substitution']
substitution_LEI = events_LEI[events_LEI['type'] == 'Substitution']

substitution_LEI_minute = np.min(substitution_LEI['minute'])
substitution_LEI_minute_data = substitution_LEI[substitution_LEI['minute'] == substitution_LEI_minute]
substitution_LEI_second = np.min(substitution_LEI_minute_data['second'])
print("minute =", substitution_LEI_minute, "second =",  substitution_LEI_second)

substitution_ARS_minute = np.min(substitution_ARS['minute'])
substitution_ARS_minute_data = substitution_ARS[substitution_ARS['minute'] == substitution_ARS_minute]
substitution_ARS_second = np.min(substitution_ARS_minute_data['second'])
print("minute =", substitution_ARS_minute, "second =",  substitution_ARS_second)

events_pn_ARS = events_pn_ARS[(events_pn_ARS['minute'] <= substitution_ARS_minute)]

events_pn_LEI = events_pn_LEI[(events_pn_LEI['minute'] <= substitution_LEI_minute)]
events_pn_LEI2 = events_pn_LEI2[(events_pn_LEI2['minute'] <= substitution_LEI_minute)]

#Passing Locations (Splitting X & Y locations
Loc = events_pn_LEI['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_LEI['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_LEI['pass_maker_x'] = Loc['pass_maker_x']
events_pn_LEI['pass_maker_y'] = Loc['pass_maker_y']
events_pn_LEI['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_LEI['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_LEI = events_pn_LEI[['index','minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y','pass_length']]

print('hello\n',events_pn_LEI.head(10).to_markdown(),'\n',len(events_pn_LEI2))

events_pn_LEI_PASS = events_pn_LEI2[events_pn_LEI2['type'] == 'Pass']
LEI_PASS_SUCCESS=events_pn_LEI_PASS[events_pn_LEI_PASS['pass_outcome'].isnull() == True]
LEI_PASS_INCOMPLETE=events_pn_LEI_PASS[events_pn_LEI_PASS['pass_outcome']=='Incomplete']





#print(events_pn_LEI_PASS.to_markdown())
print(LEI_PASS_SUCCESS.head(10).to_markdown())
print('\n Number of Complete Passes:',len(LEI_PASS_SUCCESS),file=sourceFile)
print(LEI_PASS_INCOMPLETE.head(10).to_markdown())
print('\n Number of Incomplete Passes:',len(LEI_PASS_INCOMPLETE),file=sourceFile)

#average Passing distance
avg_len_pass=LEI_PASS_SUCCESS.loc[:,'pass_length'].mean()
print('\n Average length of Passes:',avg_len_pass,file=sourceFile)

#Pass Success rate
pass_success_rate=100-((len(LEI_PASS_INCOMPLETE)/len(LEI_PASS_SUCCESS))*100)
print('\n Pass Success rate:',pass_success_rate,file=sourceFile)
print(len(events_pn_LEI_PASS))

#average passing location
av_loc_LEI = events_pn_LEI.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
av_loc_LEI.columns = ['pass_maker_x', 'pass_maker_y', 'count']
print(av_loc_LEI.to_markdown())

pass_LEI = events_pn_LEI.groupby(['pass_maker', 'pass_receiver']).index.count().reset_index()
pass_LEI.rename(columns = {'index':'number_of_passes'}, inplace = True)
pass_LEI = pass_LEI.merge(av_loc_LEI, left_on = 'pass_maker', right_index = True)
pass_LEI = pass_LEI.merge(av_loc_LEI, left_on = 'pass_receiver', right_index = True, suffixes = ['', '_receipt'])
pass_LEI.rename(columns = {'pass_maker_x_receipt':'pass_receiver_x', 'pass_maker_y_receipt':'pass_receiver_y', 'count_receipt':'number_of_passes_received'}, inplace = True)
pass_LEI = pass_LEI[pass_LEI['pass_maker'] != pass_LEI['pass_receiver']].reset_index()
print(pass_LEI.to_markdown())

#Using player Jersey Number
pass_LEI_new = pass_LEI.replace({"pass_maker": players_LEI, "pass_receiver": players_LEI})
print(players_LEI)
print(pass_LEI_new.to_markdown())

#Passing graph on pitch
pitch = Pitch(pitch_color='grass', goal_type='box', line_color='white', stripe=True)
fig, ax = pitch.draw()
arrows = pitch.arrows(pass_LEI.pass_maker_x, pass_LEI.pass_maker_y,
                      pass_LEI.pass_receiver_x, pass_LEI.pass_receiver_y, lw=5,
                      color='black', zorder=1, ax=ax)
nodes = pitch.scatter(av_loc_LEI.pass_maker_x, av_loc_LEI.pass_maker_y,
                      s=350, color='white', edgecolors='black', linewidth=1, alpha=1, ax=ax)

for index, row in av_loc_LEI.iterrows():
    pitch.annotate(players_LEI[row.name], xy=(row.pass_maker_x, row.pass_maker_y), c='black', va='center', ha='center',
                   size=10, ax=ax)
plt.title("Pass network for Leicester against Arsenal FC", size = 20)
plt.savefig("LEI_VS_ARS__PASS_GRAPH.png")
plt.show()

#Graph of Passing Network
pass_LEI_new = pass_LEI_new[['pass_maker', 'pass_receiver', 'number_of_passes']]
print(pass_LEI_new.to_markdown())

L_LEI = pass_LEI_new.apply(tuple, axis=1).tolist()
print(L_LEI)

G_LEI = nx.DiGraph()

for i in range(len(L_LEI)):
    G_LEI.add_edge(L_LEI[i][0], L_LEI[i][1], weight = L_LEI[i][2])

edges_LEI = G_LEI.edges()
weights_LEI = [G_LEI[u][v]['weight'] for u, v in edges_LEI]

nx.draw(G_LEI, node_size=800, with_labels=True, node_color='white', width = weights_LEI)
plt.gca().collections[0].set_edgecolor('black') # sets the edge color of the nodes to black
plt.title("Pass network for Leicester vs Stoke City", size = 20)
plt.savefig("LEI_VS_ARS_WEIGHTED_PASS_GRAPH.png")
plt.show()

def player_degree():
    deg_LEI = dict(nx.degree(G_LEI))  # prepares a dictionary with jersey numbers as the node ids, i.e, the dictionary keys and degrees as the dictionary values
    degree_LEI = pd.DataFrame.from_dict(list(deg_LEI.items()))  # convert a dictionary to a pandas dataframe
    degree_LEI.rename(columns={0: 'jersey_number', 1: 'node_degree'}, inplace=True)
    X = list(deg_LEI.keys())
    Y = list(deg_LEI.values())
    sns.barplot(x=Y, y=X, palette="magma")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("degree")
    plt.title("Player pass degrees for Leicester VS Stoke", size=16)
    plt.savefig("LEI_VS_ARS_DEGREE.png")
    plt.show()

    indeg_LEI = dict(G_LEI.in_degree())
    indegree_LEI = pd.DataFrame.from_dict(list(indeg_LEI.items()))
    indegree_LEI.rename(columns={0: 'jersey_number', 1: 'node_indegree'}, inplace=True)
    print(indegree_LEI.to_markdown())
    X = list(indeg_LEI.keys())
    Y = list(indeg_LEI.values())
    sns.barplot(x=Y, y=X, palette="hls")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("indegree")
    plt.title("Player pass indegrees for Leicester vs Stoke", size=16)
    plt.savefig("LEI_VS_ARS_INDEGREE.png")
    plt.show()

    outdeg_LEI = dict(G_LEI.out_degree())
    outdegree_LEI = pd.DataFrame.from_dict(list(outdeg_LEI.items()))
    outdegree_LEI.rename(columns={0: 'jersey_number', 1: 'node_outdegree'}, inplace=True)
    print(outdegree_LEI.to_markdown())
    X = list(outdeg_LEI.keys())
    Y = list(outdeg_LEI.values())
    sns.barplot(x=Y, y=X, palette="hls")
    plt.xticks(range(0, max(Y) + 5, 2))
    plt.ylabel("Player Jersey number")
    plt.xlabel("outdegree")
    plt.title("Player pass outdegrees for Leicester vs Stoke", size=16)
    plt.savefig("LEI_VS_ARS_OUTDEGREE.png")
    plt.show()
def modified_graph():
    pass_LEI_mod = pass_LEI_new[['pass_maker', 'pass_receiver']]
    pass_LEI_mod['1/nop'] = 1/pass_LEI_new['number_of_passes']
    L_LEI_mod = pass_LEI_mod.apply(tuple, axis=1).tolist()
    G_LEI_mod = nx.DiGraph()

    for i in range(len(L_LEI_mod)):
        G_LEI_mod.add_edge(L_LEI_mod[i][0], L_LEI_mod[i][1], weight = L_LEI_mod[i][2])

    edges_LEI_mod = G_LEI_mod.edges()
    weights_LEI_mod = [G_LEI_mod[u][v]['weight'] for u, v in edges_LEI_mod]

    nx.draw(G_LEI_mod, node_size=800, with_labels=True, node_color='white', width = weights_LEI_mod)
    plt.gca().collections[0].set_edgecolor('black')
    plt.title("Modified pass network for Leicester vs Tottenham", size = 20)
    plt.savefig("LEI_VS_ARS_MODIFIED.png")
    plt.show()
    E_LEI = nx.eccentricity(G_LEI_mod)
    print(E_LEI)
    print('\nEccentricity value of Modified Leicester Passing Graph',E_LEI,file=sourceFile)
    av_E_LEI = sum(list(E_LEI.values())) / len(E_LEI)
    print(av_E_LEI)
    print('Average Eccentricity Values for Leicester Passing Graph:',av_E_LEI,file=sourceFile)
    cc_LEI = nx.average_clustering(G_LEI, weight='weight')
    print(cc_LEI)
    print('Average Clustering of Passing Network:', cc_LEI, file=sourceFile)
    betweeness=nx.betweenness_centrality(G_LEI, weight = 'weight')
    max_bc = max(betweeness, key = betweeness.get)
    print('Max Betweeness:',max_bc,file=sourceFile)
    # Calculate the density of the graph
    density = nx.density(G_LEI)
    print("Graph Density:", density,file=sourceFile)
#Graph of Passing Network
pass_LEI_new = pass_LEI_new[['pass_maker', 'pass_receiver', 'number_of_passes']]
print(pass_LEI_new.to_markdown())

L_LEI = pass_LEI_new.apply(tuple, axis=1).tolist()
print(L_LEI)

G_LEI = nx.DiGraph()

for i in range(len(L_LEI)):
    G_LEI.add_edge(L_LEI[i][0], L_LEI[i][1], weight = L_LEI[i][2])

edges_LEI = G_LEI.edges()
weights_LEI = [G_LEI[u][v]['weight'] for u, v in edges_LEI]

nx.draw(G_LEI, node_size=800, with_labels=True, node_color='white', width = weights_LEI)
plt.gca().collections[0].set_edgecolor('black') # sets the edge color of the nodes to black
plt.title("Pass network for Leicester vs Arsenal FC", size = 20)
plt.show()

def goal():
    print('\nGoals\n--------------------------------',file=sourceFile)
    events_pn = LEI_ARS_events[['minute', 'second', 'team', 'type', 'location','shot_end_location','player','shot_outcome']]
    events_pn = events_pn[events_pn['team'] == 'Leicester City']

    events_pn = events_pn[events_pn['type'] == 'Shot']
    print(events_pn.to_markdown())
    shot_loc=events_pn['location']
    shot_loc = pd.DataFrame(shot_loc.to_list(), columns=['location_x', 'location_y'])
    print(shot_loc.to_markdown())
    print('Total Shots taken:',len(events_pn),'\n',file=sourceFile)

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
    events_shot_off_wayward = events_pn[(events_pn['shot_outcome'] == 'Off T') | (events_pn['shot_outcome'] == 'Wayward')| (events_pn['shot_outcome'] == 'Post')]
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
    plt.title("Leicester shot and heat map")
    plt.savefig("LEI_VS_ARS_GOAL.png")
    plt.show()


#Adjancency matrix
A_LEI = nx.adjacency_matrix(G_LEI)
A_LEI=A_LEI.todense()
sns.heatmap(A_LEI, annot = True, cmap ='gnuplot')
plt.title("Adjacency matrix for Leicester vs Arsenal pass network")
plt.show()
goal()
player_degree()
SpectralClustering.spectral(A_LEI,G_LEI,"LEI_VS_ARS_SPECTRAL")
modified_graph()
sourceFile.close()