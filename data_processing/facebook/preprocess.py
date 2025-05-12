import torch
import networkx as nx
from torch_geometric.utils import from_networkx

def extract_node_features(graph: nx.Graph):
    features = {}

    # Degree
    degree_dict = dict(graph.degree())
    nx.set_node_attributes(graph, degree_dict, "degree")

    # PageRank
    pagerank_dict = nx.pagerank(graph, alpha=0.85)
    nx.set_node_attributes(graph, pagerank_dict, "pagerank")

    # Clustering Coefficient
    clustering_dict = nx.clustering(graph)
    nx.set_node_attributes(graph, clustering_dict, "clustering")

    return graph

def convert_to_pyg(graph: nx.Graph):
    graph = extract_node_features(graph)

    # Convert node features into tensors
    for node in graph:
        attrs = graph.nodes[node]
        graph.nodes[node]['x'] = torch.tensor([
            attrs["degree"],
            attrs["pagerank"],
            attrs["clustering"]
        ], dtype= torch.float)

    # Convert to PyG format
        pyg_daa = from_networkx(graph)
        pyg_daa.x = torch.stack([graph.nodes[n]['x'] for n in graph.nodes])

        return pyg_daa