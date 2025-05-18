"""Preprocess Facebook graph and convert to PyG format with node features"""

import gzip
import torch
import networkx as nx
import random
from torch_geometric.utils import from_networkx, to_undirected, negative_sampling

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

def split_edges_for_link_prediction(data, val_ratio=0.05, test_ratio=0.1, seed=42):
    assert 'edge_index' in data, "data must contain 'edge_index'"

    edge_index = to_undirected(data.edge_index)
    num_edges = edge_index.size(1)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test

    torch.manual_seed(seed)
    perm = torch.randperm(num_edges)

    train_edges = edge_index[:, perm[:num_train]]
    val_edges = edge_index[:, perm[num_train:num_train+num_val]]
    test_edges = edge_index[:, perm[num_train+num_val:]]

    # Negative sampling for val/test
    neg_train = negative_sampling(edge_index=train_edges, num_nodes=data.num_nodes, num_neg_samples=train_edges.size(1), method='sparse')
    neg_val = negative_sampling(edge_index=train_edges, num_nodes=data.num_nodes, num_neg_samples=val_edges.size(1), method='sparse')
    neg_test = negative_sampling(edge_index=train_edges, num_nodes=data.num_nodes, num_neg_samples=test_edges.size(1), method='sparse')

    split_data = {
        "train_pos_edge_index": train_edges,
        "train_neg_edge_index": neg_train,
        "val_pos_edge_index": val_edges,
        "val_neg_edge_index": neg_val,
        "test_pos_edge_index": test_edges,
        "test_neg_edge_index": neg_test
    }

    print("Edge splitting completed:")
    print(f" Train edges: {train_edges.size(1)}")
    print(f" Val edges: {val_edges.size(1)} (with {neg_val.size(1)} negative samples)")
    print(f" Test edges: {test_edges.size(1)} (with {neg_test.size(1)} negative samples)")

    return split_data



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

    # Split edges for link prediction
    split_data = split_edges_for_link_prediction(pyg_data)

    # Attach split edge indices to pyg_data
    pyg_data.train_pos_edge_index = split_data["train_pos_edge_index"]
    pyg_data.train_neg_edge_index = split_data["train_neg_edge_index"]
    pyg_data.val_pos_edge_index = split_data["val_pos_edge_index"]
    pyg_data.val_neg_edge_index = split_data["val_neg_edge_index"]
    pyg_data.test_pos_edge_index = split_data["test_pos_edge_index"]
    pyg_data.test_neg_edge_index = split_data["test_neg_edge_index"]


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
