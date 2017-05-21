import random as rd

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_weighted_random_graph(n_nodes, n_edges, 
                                 weights_from, weights_to):
    """ Function that creates a weighted random graph with "n_nodes"
    nodes, "n_edges" edges and random weights ranging from 
    "weights_from" to "weights_to".

    The graph is fully connected and each edge of it has 2 
    directions.

    The actual number of edges is random as well, but generally
    close to the specified in "n_edges" one.
    """
    
    G = nx.DiGraph()

    nodes = list(range(1,n_nodes + 1))
    edges = [tuple(rd.sample(nodes, k=2)) for i in range(n_edges)]
    edges = list(set(edges))
    weights = [rd.randint(weights_from, weights_to) for i in range(len(edges))]

    for i in range(len(edges)):
        G.add_edge(u=edges[i][0], v=edges[i][1], 
                   weight=weights[i], navigator_weight=weights[i])
        G.add_edge(u=edges[i][1], v=edges[i][0], 
                   weight=weights[i], navigator_weight=weights[i])

    return G

def draw_graph(graph, layout='spectral'):
    """ A simple function to automatically draws any graph
    specified in "graph" and uses layout specified in "layout"
    """

    if layout == 'spectral':
        pos = nx.spectral_layout(G=graph)
    elif layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(graph)

    nx.draw_networkx(G=graph, pos=pos, node_size=250, font_size=10)
    labels = nx.get_edge_attributes(G=graph, name='weight')
    nx.draw_networkx_edge_labels(G=graph, pos=pos, edge_labels=labels)
    plt.show()

