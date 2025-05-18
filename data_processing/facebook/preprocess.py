"""Preprocess Facebook graph and convert to PyG format with node features"""

import gzip
import torch
import networkx as nx
from torch_geometric.utils import from_networkx

def load_graph_from_gz(path: str) -> nx.Graph:
    print("Loading graph from:", path)
    G = nx.Graph()
    with gzip.open(path, 'rt') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def extract_node_features(graph: nx.Graph) -> nx.Graph:
    print("Extracting node features...")

    # Degree
    degree_dict = dict(graph.degree())
    nx.set_node_attributes(graph, degree_dict, "degree")

    # PageRank
    pagerank_dict = nx.pagerank(graph, alpha=0.85)
    nx.set_node_attributes(graph, pagerank_dict, "pagerank")

    # Clustering Coefficient
    clustering_dict = nx.clustering(graph)
    nx.set_node_attributes(graph, clustering_dict, "clustering")

    print("Node features extracted.")
    return graph

def convert_to_pyg(graph: nx.Graph, output_path="facebook_graph_data.pt"):
    print("Converting NetworkX graph to PyG format")

    graph = extract_node_features(graph)

    # Add 'x' feature vector to every node
    for node in graph.nodes():
        attrs = graph.nodes[node]
        graph.nodes[node]['x'] = torch.tensor([
            float(attrs["degree"]),
            float(attrs["pagerank"]),
            float(attrs["clustering"])
        ], dtype=torch.float)

    # Convert to PyG format
    pyg_data = from_networkx(graph)

    # Build feature matrix manually
    pyg_data.x = torch.stack([graph.nodes[n]["x"] for n in graph.nodes()])

    # Save to disk
    torch.save(pyg_data, output_path)
    print(f"Graph data saved to {output_path}")
    return pyg_data

if __name__ == "__main__":
    input_path = "/Users/santiagog/Desktop/facebook_data/facebook_combined.txt.gz"
    output_path = "/Users/santiagog/Desktop/Python/machine_learning/raw_data/facebook/facebook_graph_data.pt"

    try:
        graph = load_graph_from_gz(input_path)
        pyg_data = convert_to_pyg(graph, output_path)
        print("Preprocessing Complete")
    except Exception as e:
        print("An error occurred during preprocessing:")
        print(str(e))
