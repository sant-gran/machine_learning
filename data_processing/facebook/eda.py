"""Here we are going to perform Exploratory Data Analysis """

import gzip
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from sympy.printing.pretty.pretty_symbology import line_width

# 1 BASIC GRAPH STATS

file_path = "/Users/santiagog/Desktop/facebook_data/facebook_combined.txt.gz"

# Load the edge list into a NetworkX Graph
def load_facebook_graph(file_path):
    with gzip.open(file_path, 'rt') as f:
        G = nx.Graph()
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    return G

# Calculate Basic Graph Statistics
def graph_basic_statistics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    density = nx.density(G)
    return num_nodes, num_edges, is_connected, num_components, density

# Load the graph and get stats
G = load_facebook_graph(file_path)
num_nodes, num_edges, is_connected, num_components, density = graph_basic_statistics(G)

# Print Stats
print("Basic Graph Stats:")
print(f" Number of nodes: {num_nodes}")
print(f" Number of edges: {num_edges}")
print(f" Is the Graph connected? {'Yes' if is_connected else 'No'}")
print(f" Number of connected components: {num_components}")
print(f" Graph Density: {density:.6f}")

#2 DEGREE ANALYSIS

def analyze_degree_distribution(G):
    degrees = [deg for _, deg in G.degree()]

    # 1 Basic statistics

    min_deg = min(degrees)
    max_deg = max(degrees)
    avg_deg = sum(degrees) / len(degrees)


    print("Degree Stats:")
    print(f"Min: {min_deg}")
    print(f"Max: {max_deg}")
    print(f"Avg: {avg_deg:.2f}")


    # 2 Histogram of degree distribution
    plt.figure(figsize = (10,5))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution (Friend Count Per User)")
    plt.xlabel("Number of Friends")
    plt.ylabel("Number of Users")
    plt.grid(True)
    plt.show()

def plot_log_log_degree_distribution(G):
    # Step 1: Get all node degrees
    degrees = [d for _, d in G.degree()]

    # Step 2: Prepare data for plotting
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    # Step 3: Prepare data for plotting
    x = list(degree_counts.keys()) # degrees
    y = list(degree_counts.values()) # How many users have that degree

    # Step 4: Log-Log plot
    plt.figure(figsize=(15,10))
    plt.loglog(x,y, marker='o', linestyle='None', color='darkred')
    plt.title("Log-Log Degree Distribution (Power-Law Check)")
    plt.xlabel("Degree (Number of Friends)")
    plt.ylabel("Number of Users")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

analyze_degree_distribution(G)
plot_log_log_degree_distribution(G)



