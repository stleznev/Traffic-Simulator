import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import numpy as np

#Create weighted graph
def create_weighted_random_graph(n_nodes, n_edges, weights_from, weights_to, time_slot_size):
	
	G = nx.DiGraph()

	nodes = list(range(1,n_nodes + 1))
	edges = [tuple(rd.sample(nodes, k=2)) for i in range(n_edges)]

	edges = list(set(edges))
	weights = [rd.randint(weights_from, weights_to) for i in range(len(edges))]

	for i in range(len(edges)):
		G.add_edge(edges[i][0], edges[i][1], weight = weights[i], navigator_weight = weights[i])
		G.add_edge(edges[i][1], edges[i][0], weight = weights[i], navigator_weight = weights[i])

	return G

#Draw weighted graph
#Does not support weights as lists
def draw_graph(graph):
	#pos = nx.spring_layout(graph)
	#pos = nx.circular_layout(graph)
	#pos = nx.fruchterman_reingold_layout(graph)
	pos = nx.spectral_layout(graph)
	nx.draw_networkx(graph, pos = pos, node_size = 250, font_size = 10)
	labels = nx.get_edge_attributes(graph, 'weight')
	nx.draw_networkx_edge_labels(graph, pos = pos, edge_labels = labels)
	plt.show()

#G = create_weighted_random_graph(10,20,10,30)
#draw_graph(G)
#print(G.nodes())


