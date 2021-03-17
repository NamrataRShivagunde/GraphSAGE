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
#G = edgelist.read_edgelist('graph.txt')
G = edgelist.read_edgelist('network.txt')
print("G loaded")

#Load labels
labels = {}
for line in open('train.txt'):
   k, v = line.split()
   labels[k] = v
print("Loaded train.txt")

val_nodes = []
for line in open('val.txt'):
    node,label = line.split()
    val_nodes.append(str(node))
val_nodes =set(val_nodes)
print("Loaded val.txt")

#Without labels
test_nodes = []
for node in open('test.txt'):
    test_nodes.append(str(node).strip('\n'))
test_nodes = set(test_nodes)
print("Loaded test.txt")

# Prepare graph
#nx.set_node_attributes(G, 'val', False)
#nx.set_node_attributes(G, 'test', False)


#for node in val_nodes:
 #   G.node[node]['val'] = True

#for node in test_nodes:
 #  G.node[node]['test'] = True

feats = []
i=1
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
    i= i+1

print("Featues created")
#val = ['398', '154', '369']
#test = ['398', '999', '445', '110']

# Generate features
#feats = []

#names_dict = {name: idx for idx, name in enumerate(set(names.values()))}
# print(names_dict)


#for node in G:
   # name = names[node]
   # name_arr = [0]*len(names_dict)
   # name_arr[names_dict[name]] = 1
    # 7-dim
    # print(name_arr)

   # age = ages[node]
    #degree = G.degree()[node]

    #feat = name_arr + [age] + [degree]
    # 9-dim
    # print(feat)
    #feats.append(feat)


#print(feats)
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
id2idx = {n: idx for idx,n in enumerate(G.nodes())}
with open('processed_wiki/wiki-id_map.json' ,'w') as f:
    json.dump(id2idx, f)
print("id2idx created")
# 3. Write class_map
with open('processed_wiki/wiki-class_map.json', 'w') as f:
    json.dump(labels, f)
print("class_map created")
# 4. Write features
feats = np.array(feats)
# shape: [number of nodes, number of features]
np.save('processed_wiki/wiki-feats.npy', feats)
print("feats numpy created")