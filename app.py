from randomGeometricGraph import RandomGeometricGraph
from flask import Flask, render_template, jsonify, request

import networkx as nx

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graph', methods=['GET'])
def get_graph_data():
    n = request.args.get('n', default=2000, type=int)
    gamma = request.args.get('gamma', default=2.5, type=float)
    alpha = request.args.get('alpha', default=1600, type=float)

    func_str = request.args.get('func', default="lambda x, y, z: x*y / (z**2)")
    func = eval(func_str)

    rgg = RandomGeometricGraph(n, gamma, alpha, func)
    rgg.add_edges()
    rgg.getLargestComponent()
    
    # prepare data for JSON
    nodes = [{'id': node, 'weight': rgg.weights[node], 'x': pos[0], 'y': pos[1]} for node, pos in nx.get_node_attributes(rgg.graph, 'pos').items()]    
    edges = [{'source': edge[0], 'target': edge[1]} for edge in rgg.graph.edges]
    largestComponentSz = rgg.largestComponentSz
    diam = rgg.getDiameter()
    avgDistance = rgg.getAverageDistance()
    mxEdge = round(rgg.maxEdgeLength, 3)
    rgg.plot_degree_distribution()
    #rgg.plot_weights()
    return jsonify({'nodes': nodes, 'edges': edges, 'mxEdge': mxEdge, 'gamma': gamma, 'alpha': alpha, 'largestComponentSz': largestComponentSz, 'diameter': diam, 'avgDist': avgDistance, 'amountComp': rgg.amountComponents, 'clusteringCoef': rgg.getClusteringCoeffcient()})

if __name__ == '__main__':
    app.run(debug=True)
