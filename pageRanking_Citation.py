import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to calculate eigenvector centrality for citation ranking
def eigenvector_centrality(citation_matrix):
    # Using the NetworkX library for eigenvector centrality
    G = nx.DiGraph(citation_matrix)  # Create a directed graph from the citation matrix
    centrality = nx.eigenvector_centrality_numpy(G)  # Compute eigenvector centrality
    return G,centrality

# Example citation matrix for papers P1, P2, P3, and P4
# Rows are citing papers, and columns are cited papers
citation_matrix = np.array([
    [0, 1, 1, 0],  # P1 is cited by P2 and P3
    [1, 0, 0, 1],  # P2 is cited by P1 and P4
    [1, 0, 0, 0],  # P3 is cited by P1
    [0, 1, 0, 0]   # P4 is cited by P2
])


# Run the eigenvector centrality calculation
G,citation_ranks = eigenvector_centrality(citation_matrix)

# Display the citation ranking results
papers = ['P1', 'P2', 'P3', 'P4']
for paper, rank in citation_ranks.items():
    print(f"Paper {papers[paper]}: Influence Score = {rank:.4f}")

# Create a layout for our graph visualization
pos = nx.spring_layout(G)

# Plot the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, labels={i: papers[i] for i in range(len(papers))}, 
        node_size=2000, node_color='lightblue', font_size=12, font_color='black', 
        arrows=True, arrowstyle='->', arrowsize=20)

# Add the centrality scores as node labels
centrality_labels = {i: f'{papers[i]}\n{citation_ranks[i]:.4f}' for i in range(len(papers))}
nx.draw_networkx_labels(G, pos, labels=centrality_labels, font_size=10)

plt.title("Citation Network with Eigenvector Centrality Scores")
plt.show()