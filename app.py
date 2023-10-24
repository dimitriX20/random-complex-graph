from randomGeometricGraph import RandomGeometricGraph
from flask import Flask, render_template, jsonify, request

import networkx as nx

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graph', methods=['GET'])
def get_graph_data():
    n = request.args.get('n', default=500, type=int)
    gamma = request.args.get('gamma', default=2.5, type=float)
    alpha = request.args.get('alpha', default=500, type=float)

    func_str = request.args.get('func', default="lambda x, y, z: x*y / (z**2)")
    func = eval(func_str)

    rgg = RandomGeometricGraph(n, gamma, alpha, func)
    rgg.add_edges()
    
    rgg.computeDistanceMatrix()
    diam = rgg.getDiameter()
    avgDistance = rgg.getAverageDistance()

    rgg.getLargestComponent()

    diam_path_edges = set([(rgg.get_diameter_path()[i], rgg.get_diameter_path()[i+1]) for i in range(len(rgg.get_diameter_path())-1)])

    no_bigger_neighbors = rgg.verticesWithoutBiggerNeighbor()
    nodes = [{'id': node, 'weight': rgg.weights[node], 'x': pos[0], 'y': pos[1], 'has_bigger_neighbor': node in no_bigger_neighbors} for node, pos in nx.get_node_attributes(rgg.graph, 'pos').items()]

    edges = [{'source': edge[0], 'target': edge[1], 'is_diam_path': edge in diam_path_edges or edge[::-1] in diam_path_edges} for edge in rgg.graph.edges]   
    largestComponentSz = rgg.largestComponentSz
    mxEdge = round(rgg.maxEdgeLength, 3)

    succNaivGreedy = rgg.measure_naiv_greedy_success_probability()
    
    succGreedy = rgg.measure_improved_greedy_success_probability()
    stretchOneWay = rgg.get_stretch_one_way()

    succGreedyOneWay = rgg.measure_improved_greedy_one_way_success_probability()
    stretchTwoWay = rgg.get_stretch_two_way()

    specialPairs = rgg.find_special_pairs_count()

    #rgg.plot_degree_distribution()
    #rgg.plot_weights()
    
    #rgg.plot_distances_from_root()
    #rgg.plot_degree_distribution_with_highlight()


    return jsonify({'nodes': nodes, 'edges': edges, 'mxEdge': mxEdge, 'gamma': gamma, 'alpha': alpha, 'largestComponentSz': largestComponentSz, 'diameter': diam, 'avgDist': avgDistance, 'amountComp': rgg.amountComponents, 'clusteringCoef': rgg.getClusteringCoeffcient(), 'func': func_str, 'nrNoBiggerNeighbor' : len(no_bigger_neighbors), 'succNaivGreedy' : succNaivGreedy, 'succGreedy' : succGreedy, 'succGreedyOneWay' : succGreedyOneWay, 'stretchOneWay' : stretchOneWay, 'stretchTwoWay' : stretchTwoWay, 'specialPairs': specialPairs})

if __name__ == '__main__':
    app.run(debug=True)
