import numpy as np
import networkx as nx
from scipy.stats import powerlaw 
import plotly.graph_objects as go
import random 

# wechsle zum torus  
# schwellenwert für die Existenz einer giant component ermitteln 

class dsu(object):
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.size = [1] * n  # initialize the size of each element to 1
    
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        rootA, rootB = self.find(x), self.find(y)
        if rootA != rootB:
            # merge the smaller set into the larger set
            if self.size[rootA] < self.size[rootB]:
                self.parents[rootA] = rootB
                self.size[rootB] += self.size[rootA]
            else:
                self.parents[rootB] = rootA
                self.size[rootA] += self.size[rootB]
            return True
        return False 

    def sz(self, node):
        root = self.find(node)
        return self.size[root]

class RandomGeometricGraph:
    def __init__(self, num_nodes, gamma, alpha, func):
        self.num_nodes = num_nodes
        self.dimension = 2  # unit square
        self.alpha = alpha
        self.func = func
        self.graph = nx.Graph()  # empty graph to begin with
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
                #print(distance)

                if self.func(weight1, weight2, distance) >= self.alpha:
                    self.graph.add_edge(node1, node2, weight=weight1 * weight2)
                    self.amountComponents -= self.dsu.union(node1, node2)
                    self.m += 1 
                    self.maxEdgeLength = max(self.maxEdgeLength, distance)
        self.distance_matrix = nx.floyd_warshall_numpy(self.graph)
    
    def isConnected(self): 
        return self.amountComponents == 1
    
    def getAverageDistance(self): 
        valid_distances = self.distance_matrix[np.isfinite(self.distance_matrix) & (self.distance_matrix != 0)]
        if valid_distances.size == 0:
            return 0
        return round(np.mean(valid_distances), 2)

    def getDiameter(self): 
        valid_distances = self.distance_matrix[np.isfinite(self.distance_matrix)]
        if valid_distances.size == 0:
            return 0
        return round(np.max(valid_distances), 5)

    def getLargestComponent(self):
        for i in range(self.num_nodes): 
            self.largestComponentSz = max(self.largestComponentSz, self.dsu.sz(i))
        return self.largestComponentSz

    def getClusteringCoeffcient(self):  
        return round(nx.average_clustering(self.graph), 5)

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

func = eval("lambda x, y, z: x*y / (z**2)")
gamma = 2.5    
k = 15
alpha = -5
    
alphas = []
probabilities = []
giant_component_sizes = []
clust_coefficients = []
diameter_sizes = []
N = 100
    
while (alpha < 5 * N):
    alpha += 5
    probConnected = 0
    avg_giant_component_size = 0
    avg_cluster = 0
    avg_diameter = 0
    for i in range(k): 
        rgg = RandomGeometricGraph(N, gamma, alpha, func)
        rgg.add_edges() 
        probConnected += rgg.isConnected()
        avg_giant_component_size += rgg.getLargestComponent()
        avg_cluster += rgg.getClusteringCoeffcient()
        avg_diameter += rgg.getDiameter() 
        
    avg_giant_component_size /= k   
    giant_component_sizes.append(avg_giant_component_size)
    
    probConnected /= k 
    probabilities.append(probConnected)
    
    avg_diameter /= k
    diameter_sizes.append(avg_diameter)
    
    avg_cluster /= k
    clust_coefficients.append(avg_cluster)
    
    alphas.append(alpha) 
    
fig = go.Figure(data=go.Scatter(x=alphas, y=probabilities, mode='lines+markers'))

fig.update_layout(
    title="Probability of Connectivity vs. Alpha for N = " + str(N) + " and γ = " + str(gamma),
    xaxis_title="Alpha",
    yaxis_title="Probability of Connectivity",
    showlegend=False
)

fig.show()

fig2 = go.Figure(data=go.Scatter(x=alphas, y=giant_component_sizes, mode='lines+markers'))

fig2.update_layout(
    title="Largest Component Size vs. Alpha for N = " + str(N) + " and γ = " + str(gamma),
    xaxis_title="Alpha",
    yaxis_title="Largest Component Size",
    showlegend=False
)

fig2.show()
 
fig3 = go.Figure(data=go.Scatter(x=alphas, y=diameter_sizes, mode='lines+markers'))

fig3.update_layout(
    title="Diameter vs. Alpha for N = " + str(N) + " and γ = " + str(gamma),
    xaxis_title="Alpha",
    yaxis_title="Diameter",
    showlegend=False
)

fig3.show()    

fig4 = go.Figure(data=go.Scatter(x=alphas, y=clust_coefficients, mode='lines+markers'))

fig4.update_layout(
    title = "Cluster Coefficient vs. Alpha for N = " + str(N) + " and γ = " + str(gamma),
    xaxis_title="Alpha",
    yaxis_title="Cluster Coefficient",
    showlegend=False
)

fig4.show()  