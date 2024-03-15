import numpy as np
import pandas as pd
from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns

comp=sb.competitions()

print(comp.to_markdown())

match=sb.matches(competition_id=2,season_id=27)
print(match.to_markdown())
print(match['home_team'])
match_count=0
for i in range(380):
    if match['home_team'][i]=='Chelsea' and match['away_team'][i]=='Tottenham Hotspur':
        print(match['home_team'][i],' ',match['away_team'][i],' ',match['match_id'][i])
        match_count+=1

print(match_count)

TOT_CHELS_events=sb.events(match_id=3754092)

eh = TOT_CHELS_events.head() # shows the first few rows
print(eh.to_markdown())

print(TOT_CHELS_events.columns)

events_pass = TOT_CHELS_events[['team', 'type', 'minute', 'location', 'pass_end_location', 'pass_outcome', 'player']]

e1 = events_pass.head(10) # extracts the first 10 rows
print(e1.to_markdown())

e2 = events_pass.tail(10) # extracts the last 10 rows
print(e2.to_markdown())

events_pass_p1 = events_pass[events_pass['player'] == 'Francesc Fàbregas i Soler']

print(events_pass_p1.to_markdown())

events_pass_p1 = events_pass_p1[events_pass_p1['type'] == 'Pass'].reset_index()
print(events_pass_p1.to_markdown())

events_pass_p1['pass_outcome'] = events_pass_p1['pass_outcome'].fillna('Successful')
print(events_pass_p1.to_markdown())

Loc = events_pass_p1['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['location_x', 'location_y'])
print(Loc.to_markdown())

Loc_end = events_pass_p1['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_end_location_x', 'pass_end_location_y'])
print(Loc_end.to_markdown())

events_pass_p1['location_x'] = Loc['location_x']
events_pass_p1['location_y'] = Loc['location_y']
events_pass_p1['pass_end_location_x'] = Loc_end['pass_end_location_x']
events_pass_p1['pass_end_location_y'] = Loc_end['pass_end_location_y']
print(events_pass_p1.to_markdown())

events_pass_p1 = events_pass_p1[['minute', 'location_x', 'location_y', 'pass_end_location_x', 'pass_end_location_y', 'pass_outcome']]
print(events_pass_p1.to_markdown())

pitch = Pitch(pitch_color = 'black', line_color = 'white',goal_type = 'box')
fig, ax = pitch.draw()
for i in range(len(events_pass_p1)):
    if events_pass_p1.pass_outcome[i] == 'Successful':
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='green', width = 3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color = 'green')
    else:
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='red', width=3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color='red')
plt.show()

# Pitch drawing code
pitch = Pitch(pitch_color='black', line_color='white', goal_type='box')
fig, ax = pitch.draw()

# Heat map code
res = sns.kdeplot(x=events_pass_p1['location_x'], y=events_pass_p1['location_y'], fill=True,
                  thresh=0.05, alpha=0.5, levels=10, cmap='Purples_r')

# Pass map code
for i in range(len(events_pass_p1)):
    if events_pass_p1.pass_outcome[i] == 'Successful':
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i],
                     events_pass_p1.pass_end_location_y[i], ax=ax, color='green', width=3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax=ax, color='green')
    else:
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i],
                     events_pass_p1.pass_end_location_y[i], ax=ax, color='red', width=3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax=ax, color='red')

# General plot code
plt.title("Cesc Fabregas pass and heat map")
plt.show()

events_pass_p1['pass_outcome'].value_counts(normalize=True).mul(100)

events_pass_p1['pass_outcome'].value_counts(normalize=True).mul(100).plot.bar()