import os, json
import numpy as np
import networkx as nx
from networkx.readwrite import edgelist, json_graph

# Required:
# 1. Load the graph
# 2. Add validation and testing attributes to the graph
# 3. Create the features
# 4. Save the data in the proper format

# Load graph
G = edgelist.read_edgelist('graph.txt')

# Load labels
labels = {}
for line in open('classes.txt'):
    k, v = line.split()
    labels[k] = v

# Names and ages are node attributes
# Load names
names = {}
for line in open('names.txt'):
    k, v = line.split()
    names[k] = v

# Load ages
ages = {}
for line in open('ages.txt'):
    k, v = line.split()
    ages[k] = int(v)


# Prepare graph
nx.set_node_attributes(G, 'val', False)
nx.set_node_attributes(G, 'test', False)

val = ['32', '17', '6']
test = ['19', '10', '10', '6']

for node in val:
    G.node[node]['val'] = True

for node in test:
    G.node[node]['test'] = True

# Generate features
feats = []

names_dict = {name: idx for idx, name in enumerate(set(names.values()))}
# print(names_dict)

for node in G:
    name = names[node]
    name_arr = [0]*len(names_dict)
    name_arr[names_dict[name]] = 1
    # 7-dim
    # print(name_arr)

    age = ages[node]
    degree = G.degree()[node]

    feat = name_arr + [age] + [degree]
    # 9-dim
    # print(feat)

    feats.append(feat)

print(feats)
# Writing:
# Required files:
# 1. -G.json
# 2. -id_map.json
# 3. -class_map.json
# 4. -feats.npy

# Make directory for processed data
os.makedirs('processed', exist_ok=True)

# 1. Write graph
data = json_graph.node_link_data(G)
with open('processed/karate-G.json', 'w') as f:
    json.dump(data, f)

# 2. Write id_map
id2idx = {n: idx for idx,n in enumerate(G.nodes())}
with open('processed/karate-id_map.json' ,'w') as f:
    json.dump(id2idx, f)

# 3. Write class_map
with open('processed/karate-class_map.json', 'w') as f:
    json.dump(labels, f)

# 4. Write features
#feats = np.array(feats)
feats = np.arange(1,35)
feats = feats.reshape(-1,1)
# shape: [number of nodes, number of features]
np.save('processed/karate-feats.npy', feats)
