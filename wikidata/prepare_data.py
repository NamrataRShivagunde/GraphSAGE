import os
import json
import numpy as np
import networkx as nx
from networkx.readwrite import edgelist, json_graph

# Required:
# 1. Load the graph
# 2. Add validation and testing attributes to the graph
# 3. Create the features
# 4. Save the data in the proper format


# Load labels
labels = {}
train_nodes = []
for line in open('train.txt'):
    k, v = line.split()
    labels[k] = v
    train_nodes.append(str(k))
train_nodes = set(train_nodes)
print("Loaded train.txt")

val_nodes = []
for line in open('val.txt'):
    node, label = line.split()
    val_nodes.append(str(node))
    labels[node] = label
val_nodes = set(val_nodes)
print("Loaded val.txt")

# Without labels
test_nodes = []
for node in open('test.txt'):
    testNode = str(node).strip('\n')
    test_nodes.append(testNode)
    if node not in labels:
        labels[testNode] = "0"

test_nodes = set(test_nodes)
print("Loaded test.txt")

sub_graph = {}
for line in open("network.txt"):
    e1, e2 = line.split()
    # add e1,e2 pair to subgraph if e1 and e2 both exist in either train, test or val
    if e1 in labels and e2 in labels:
        sub_graph[e1] = e2

with open('network-filtered.txt', 'a') as f:
    for e1, e2 in sub_graph.items():
        f.write(e1 + " " + e2 + "\n")


G = edgelist.read_edgelist('network-filtered.txt')
print("G loaded")

feats = []
i = 1
for node in G:
    if node in val_nodes:
        G.node[node]['val'] = True
    else:
        G.node[node]['val'] = False

    if node in test_nodes:
        G.node[node]['test'] = True
    else:
        G.node[node]['test'] = False

    feat = i
    feats.append(feat)
    i = i+1

print("Featues created")


# print(feats)
# Writing:
# Required files:
# 1. -G.json
# 2. -id_map.json
# 3. -class_map.json
# 4. -feats.npy

# Make directory for processed data
os.makedirs('processed_wiki', exist_ok=True)

# 1. Write graph
data = json_graph.node_link_data(G)
with open('processed_wiki/wiki-G.json', 'w') as f:
    json.dump(data, f)
print("data created")
# 2. Write id_map
id2idx = {n: idx for idx, n in enumerate(G.nodes())}
with open('processed_wiki/wiki-id_map.json', 'w') as f:
    json.dump(id2idx, f)
print("id2idx created")
# 3. Write class_map
with open('processed_wiki/wiki-class_map.json', 'w') as f:
    json.dump(labels, f)
print("class_map created")
# 4. Write features
feats = np.array(feats)
feats = feats.reshape(-1, 1)
# shape: [number of nodes, number of features]
np.save('processed_wiki/wiki-feats.npy', feats)
print("feats numpy created")
