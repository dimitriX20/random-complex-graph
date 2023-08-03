import numpy as np
import networkx as nx
from scipy.stats import powerlaw
from DSU import dsu
import plotly.graph_objects as go
import random  

class RandomGeometricGraph:
    def __init__(self, num_nodes, gamma, alpha, func):
        self.num_nodes = num_nodes
        self.dimension = 2   
        self.alpha = alpha
        self.func = func
        self.graph = nx.Graph()   
        self._place_nodes()
        self.weights = {node: random.paretovariate(gamma) for node in self.graph.nodes}
        self.largestComponentSz = 1
        self.dsu = dsu(num_nodes)
        self.m = 0 
        self.amountComponents = num_nodes
        self.maxEdgeLength = 0
        
    def _place_nodes(self):
        for i in range(self.num_nodes):
            pos = (np.random.random(), np.random.random())
            self.graph.add_node(i, pos=pos)

    def add_edges(self):
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]

                weight1 = self.weights[node1]
                weight2 = self.weights[node2]
                
                pos1 = np.array(self.graph.nodes[node1]['pos'])
                pos2 = np.array(self.graph.nodes[node2]['pos'])
                distance = np.linalg.norm(pos1 - pos2)

                if self.func(weight1, weight2, distance) >= self.alpha:
                    self.graph.add_edge(node1, node2, weight=1)
                    self.amountComponents -= self.dsu.union(node1, node2)
                    self.m += 1 
                    self.maxEdgeLength = max(self.maxEdgeLength, distance)
        
        self.distance_matrix = nx.floyd_warshall_numpy(self.graph)

    def getAverageDistance(self): 
        valid_distances = self.distance_matrix[np.isfinite(self.distance_matrix) & (self.distance_matrix != 0)]
        if valid_distances.size == 0:
            return 0
        return round(np.mean(valid_distances), 2)

    def getDiameter(self): 
        valid_distances = self.distance_matrix[np.isfinite(self.distance_matrix)]
        if valid_distances.size == 0:
            return 0
        return np.max(valid_distances)

    def getLargestComponent(self):
        for i in range(self.num_nodes): 
            self.largestComponentSz = max(self.largestComponentSz, self.dsu.sz(i))
        return self.largestComponentSz

    def getClusteringCoeffcient(self):  
        return round(nx.average_clustering(self.graph), 3)

    def isConnected(self): 
        return self.amountComponents == 1

    def plot_degree_distribution(self): 
        degrees = [deg for _, deg in self.graph.degree()]
        fig = go.Figure(data=[go.Histogram(x=degrees, histnorm='')])
        
        fig.update_layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Number of Nodes",
            bargap=0.2
        )
 
        fig.show()

    def plot_weights(self): 
        gewichte = list(self.weights.values())  
        fig = go.Figure(data=[go.Histogram(x=gewichte, histnorm='')])

        fig.update_layout(
            title="Distribution of Weights",
            xaxis_title="weight", 
            yaxis_title="amount of nodes",
            bargap=0.2
        )

        fig.show()

