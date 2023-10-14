import numpy as np
import networkx as nx
from scipy.stats import powerlaw
from DSU import dsu
import plotly.graph_objects as go 
from collections import Counter
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
        self.diameter = 0
        self.root = max(self.weights, key=self.weights.get) 
        
    def _place_nodes(self):
        for i in range(self.num_nodes):
            pos = (np.random.uniform(0, 1), np.random.uniform(0, 1))
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

    def computeDistanceMatrix(self): 
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
        self.diameter = np.max(valid_distances)
        return self.diameter

    def get_diameter_path(self): 
        max_dist = -np.inf  # Starting with negative infinity to ensure any real distance is greater.
        max_indices = (0, 0)  

        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if i != j and self.distance_matrix[i][j] != np.inf and self.distance_matrix[i][j] > max_dist:
                    max_dist = self.distance_matrix[i][j]
                    max_indices = (i, j)
        return nx.shortest_path(self.graph, max_indices[0], max_indices[1])

    def getLargestComponent(self):
        for i in range(self.num_nodes): 
            self.largestComponentSz = max(self.largestComponentSz, self.dsu.sz(i))
        return self.largestComponentSz

    def getClusteringCoeffcient(self):  
        return round(nx.average_clustering(self.graph), 3)

    def isConnected(self): 
        return self.amountComponents == 1

    def verticesWithoutBiggerNeighbor(self): 
        no_bigger_neighbors = []
        for node in self.graph.nodes():
            weight = self.weights[node]
            if all(self.weights[n] <= weight for n in self.graph.neighbors(node)):
                no_bigger_neighbors.append(node)
        return no_bigger_neighbors
        
    def distance_from_root(self): 
        return nx.single_source_shortest_path_length(self.graph, self.root)

    def vertex_in_largest_component(self):
        # Bestimme Wurzel jedes Knotens und zähle Größe der Zusammenhangskomponenten
        root_to_size = {}
        for node in range(self.num_nodes):
            root = self.dsu.find(node)
            if root not in root_to_size:
                root_to_size[root] = self.dsu.sz(root)

        # Finde die Wurzel der größten Zusammenhangskomponente
        largest_root = max(root_to_size, key=root_to_size.get)
 
        return largest_root


    def naive_greedy_path(self, x, y): 
        visited_from_x = {x}
        visited_from_y = {y}
        queue_x = [x]
        queue_y = [y]
        
        while True:
            if not queue_x and not queue_y:
                return None

            if set(queue_x) & set(queue_y):
                break  

            if queue_x:
                current_x = queue_x.pop(0)
                # Wähle Nachbarn von current_x mit dem höchsten Gewicht, der noch nicht besucht wurde.
                neighbors_x = [(neighbor, self.weights[neighbor]) for neighbor in self.graph.neighbors(current_x) if neighbor not in visited_from_x]
                if neighbors_x:
                    next_x = max(neighbors_x, key=lambda x: x[1])[0]
                    visited_from_x.add(next_x)
                    queue_x.append(next_x)

            if queue_y:
                current_y = queue_y.pop(0)
                # Wählen Nachbarn von current_y mit dem höchsten Gewicht, der noch nicht besucht wurde.
                neighbors_y = [(neighbor, self.weights[neighbor]) for neighbor in self.graph.neighbors(current_y) if neighbor not in visited_from_y]
                if neighbors_y:
                    next_y = max(neighbors_y, key=lambda x: x[1])[0]
                    visited_from_y.add(next_y)
                    queue_y.append(next_y)

        intersections = list(set(queue_x) & set(queue_y))
        if intersections:
            intersection = intersections[0]
            # Erstelle Pfad von x nach y über den Schnittpunkt
            path_from_x = nx.shortest_path(self.graph, x, intersection, weight=None)
            path_from_y = nx.shortest_path(self.graph, y, intersection, weight=None)
            return path_from_x + path_from_y[1:]
        else:
            return None 

    def measure_naiv_greedy_success_probability(self):
        successful_attempts = 0

        largestRoot = self.vertex_in_largest_component()
        # Finde alle Ecken, die sich in der gleichen Zusammenhangskomponente wie die Wurzel befinden.
        nodes_in_largest_component = [node for node in self.graph.nodes if self.dsu.find(node) == self.dsu.find(largestRoot)]

        if len(nodes_in_largest_component) < 2: 
            return 1

        for _ in range(10): 
            i, j = random.sample(nodes_in_largest_component, 2)

            path = self.naive_greedy_path(i, j)
            if path:
                successful_attempts += 1
                 
        probability = successful_attempts / 10.0
        return probability

 
    def reconstruct_path(self, parent, start, end):
        path = []
        while end:
            path.append(end)
            end = parent[end]
        path.reverse()
        return path

    def improved_greedy_path(self, x, y):
        # check whether both are neighbors or have a common neighbor 
        #self.computeDistanceMatrix() // we compute this in app.py

        visited_from_x = {x}
        visited_from_y = {y}
        queue_x = [x]
        queue_y = [y]
        parent_from_x = {x: None}
        parent_from_y = {y: None}

        while True:
            if not queue_x and not queue_y:
                return None

            if set(queue_x) & set(queue_y):
                break

            if queue_x and queue_y:
                current_x = queue_x.pop(0)
                current_y = queue_y.pop(0)
                
                neighbors_x = [neighbor for neighbor in self.graph.neighbors(current_x) if neighbor not in visited_from_x]
                neighbors_y = [neighbor for neighbor in self.graph.neighbors(current_y) if neighbor not in visited_from_y]

                if neighbors_x and neighbors_y:
                    pairs = [(u, v) for u in neighbors_x for v in neighbors_y]
                    next_x, next_y = max(pairs, key=lambda pair: self.func(self.weights[pair[0]], self.weights[pair[1]], self.distance_matrix[pair[0]][pair[1]]))
                    
                    visited_from_x.add(next_x)
                    visited_from_y.add(next_y)
                    
                    parent_from_x[next_x] = current_x
                    parent_from_y[next_y] = current_y
                    
                    queue_x.append(next_x)
                    queue_y.append(next_y)

        intersections = list(set(queue_x) & set(queue_y))
        if intersections:
            intersection = intersections[0]
            path_from_x = self.reconstruct_path(parent_from_x, x, intersection)
            path_from_y = self.reconstruct_path(parent_from_y, y, intersection)
            #print(path_from_x + path_from_y[1:])
            return path_from_x + path_from_y[1:]
        else:
            return None


    def improved_greedy_path_one_way(self, x, y):
        # Initialize
        visited_from_x = {x}
        queue_x = [x]
        queue_y = [y]
        parent_from_x = {x: None}
        
        while True:
            if not queue_x:
                return None

            if set(queue_x) & set(queue_y):
                break

            if queue_x:
                current_x = queue_x.pop(0)
                neighbors_x = [neighbor for neighbor in self.graph.neighbors(current_x) if neighbor not in visited_from_x]
                
                if neighbors_x:
                    pairs = [(u, y) for u in neighbors_x]
                    next_x, next_y = max(pairs, key=lambda pair: self.func(self.weights[pair[0]], self.weights[pair[1]], self.distance_matrix[pair[0]][pair[1]]))
                    
                    visited_from_x.add(next_x)
                    parent_from_x[next_x] = current_x
                    
                    queue_x.append(next_x)

        intersections = list(set(queue_x) & set(queue_y))
        if intersections:
            intersection = intersections[0]
            path_from_x = self.reconstruct_path(parent_from_x, x, intersection)
           # print(x, y, intersection, path_from_x)
            return path_from_x
        else:
            return None

        
    def get_stretch_two_way(self): 
        successful_attempts = 0
        largestRoot = self.vertex_in_largest_component()
        nodes_in_largest_component = [node for node in self.graph.nodes if self.dsu.find(node) == self.dsu.find(largestRoot)]
        avg = 0 

        if len(nodes_in_largest_component) < 2:
            return 1

        while successful_attempts < 10: 
            i, j = random.sample(nodes_in_largest_component, 2)
            path = self.improved_greedy_path(i, j)
            if path:
                successful_attempts += 1
                avg += len(path) / self.distance_matrix[i][j]
        return round(avg / successful_attempts, 2)        

    def get_stretch_one_way(self):
        successful_attempts = 0
        largestRoot = self.vertex_in_largest_component()
        nodes_in_largest_component = [node for node in self.graph.nodes if self.dsu.find(node) == self.dsu.find(largestRoot)]
        avg = 0 

        if len(nodes_in_largest_component) <= 2:
            return 1

        while successful_attempts < 10: 
            i, j = random.sample(nodes_in_largest_component, 2)
            path = self.improved_greedy_path_one_way(i, j)
            if path:
                successful_attempts += 1
                avg += len(path) / self.distance_matrix[i][j]
                if len(path) / self.distance_matrix[i][j] < 1: 
                    print(i, j, path)
        return round(avg / successful_attempts, 2)

    def measure_improved_greedy_success_probability(self):
        successful_attempts = 0
        largestRoot = self.vertex_in_largest_component()
        nodes_in_largest_component = [node for node in self.graph.nodes if self.dsu.find(node) == self.dsu.find(largestRoot)]

        if len(nodes_in_largest_component) < 2:
            return 1

        for _ in range(10): 
            i, j = random.sample(nodes_in_largest_component, 2)
            path = self.improved_greedy_path(i, j)
            if path:
                successful_attempts += 1

        probability = successful_attempts / 10.0
        return probability

    def measure_improved_greedy_one_way_success_probability(self):
        successful_attempts = 0
        largestRoot = self.vertex_in_largest_component()
        nodes_in_largest_component = [node for node in self.graph.nodes if self.dsu.find(node) == self.dsu.find(largestRoot)]

        if len(nodes_in_largest_component) < 2:
            return 1

        for _ in range(10): 
            i, j = random.sample(nodes_in_largest_component, 2)
            path = self.improved_greedy_path_one_way(i, j)
            if path:
                successful_attempts += 1

        probability = successful_attempts / 10.0
        return probability


    def plot_distances_from_root(self): 
        distances_root = self.distance_from_root()

        # Calculate d' and d
        d_prime = max(distances_root.values())
        d = self.getDiameter()

        # Calculate the ratio d'/d
        ratio = d_prime / d

        fig = go.Figure(data=[go.Histogram(x=list(distances_root.values()), histnorm='')])

        # Update the layout to include the ratio in the title
        fig.update_layout(
            title=f"Distances from Root Distribution (d'/diameter = {ratio:.2f}) where d' is the maximum distance of a node from the root",
            xaxis_title="Distance from Root",
            yaxis_title="Number of Nodes",
            bargap=0.2
        )
        
        fig.show()

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
  

    def plot_degree_distribution_with_highlight(self):
        """
        Plots the degree distribution of the graph, highlighting nodes without a bigger neighbor in weight.
        """
        
        # Get all degrees
        degrees = [deg for _, deg in self.graph.degree()]
        
        # Get nodes without a bigger neighbor in terms of weight
        nodes_without_bigger = self.verticesWithoutBiggerNeighbor()
        
        # Split degrees into standard and highlighted based on nodes_without_bigger
        standard_degrees = [deg for node, deg in self.graph.degree() if node not in nodes_without_bigger]
        highlighted_degrees = [deg for node, deg in self.graph.degree() if node in nodes_without_bigger]
        
        # Aggregate degree counts
        standard_degree_counts = Counter(standard_degrees)
        highlighted_degree_counts = Counter(highlighted_degrees)
        
        # Convert the count dictionaries to two lists for x and y values
        standard_x, standard_y = zip(*standard_degree_counts.items())
        highlighted_x, highlighted_y = zip(*highlighted_degree_counts.items())
        
        # Create the figure
        fig = go.Figure()
        
        # Add bar traces for standard and highlighted nodes
        fig.add_trace(go.Bar(x=standard_x, y=standard_y, name="Standard Nodes"))
        fig.add_trace(go.Bar(x=highlighted_x, y=highlighted_y, name="Nodes w/o Bigger Neighbor", marker=dict(color='red')))
        
        # Update the layout
        fig.update_layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Number of Nodes",
            bargap=0.2,
            barmode='overlay'
        )
        
        # Show the plot
        fig.show()
 

