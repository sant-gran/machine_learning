""" Here we are going to write the algorithm for calculating the similarity between to nodes"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class HybridSimilarityScorer(nn.Module):
    """
    Computes a hybrid similarity score between two embeddings using a learnable weighted combination of cosine, euclidean, and pearson similarities

    """

    def __init__(self, embedding_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3) # Outputs: alpha, beta, gamma
        )

    def forward(self, h_u, h_v):
        # Feature engineering concat (h_u, h_v, |h_u - H_v|)
        features = torch.cat([h_u, h_v, torch.abs(h_u - h_v)], dim=-1)

        # Learnable weights for similarity components
        weights = F.softmax(self.mlp(features), dim=-1)
        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]

        # Cosine similarity
        cos_sim = F.cosine_similarity(h_u, h_v, dim=-1)

        # Euclidean similarity (1 / (1 + distance))
        eucl_dist = torch.norm(h_u - h_v, dim=-1)
        eucl_sim = 1 /(1+eucl_dist)

        # Pearson correlation
        h_u_centered = h_u - h_u.mean(dim=-1, keepdim=True)
        h_v_u_centered = h_v - h_v.mean(dim=-1, keepdim=True)
        numerator = (h_u_centered * h_v_u_centered).sum(dim=-1)
        denominator = torch.norm(h_u_centered, dim =-1) * torch.norm(h_v_u_centered, dim=-1)
        pearson_sim = numerator / (denominator + 1e-8) # Prevent division by zero

        # Final hybrid similarity score
        sim = alpha * cos_sim + beta * eucl_sim + gamma * pearson_sim
        return sim

    def score_components(self, h_u, h_v):  # Return individual components for analysis/debugging

        cos_sim = F.cosine_similarity(h_u, h_v, dim=-1)
        eucl_sim = 1 /(1 + torch.norm(h_u - h_v, dim=-1))

        h_u_centered = h_u - h_u.mean(dim=-1, keepdim=True)
        h_v_u_centered = h_v - h_v.mean(dim=-1, keepdim=True)
        numerator = (h_u_centered * h_v_u_centered).sum(dim=-1)
        denominator = torch.norm(h_u_centered, dim=-1) * torch.norm(h_v_u_centered, dim=-1)
        pearson_sim = numerator / (denominator + 1e-8)  # Prevent division by zero

        return cos_sim, eucl_sim, pearson_sim

if __name__ == "__main__":

    embedding_dim = 16
    scorer = HybridSimilarityScorer(embedding_dim=embedding_dim)
    h_u = torch.rand((4, embedding_dim))
    h_v = torch.rand((4, embedding_dim))


    sim = scorer(h_u, h_v)
    print("Hybrid similarity scores:", sim)

    cos, eucl, pear = scorer.score_components(h_u, h_v)
    print("Cosine:", cos)
    print("Euclidean:", eucl)
    print("Pearson:", pear)

    # Visualization of learned weights
    with torch.no_grad():
        features = torch.cat([h_u, h_v, torch.abs(h_u - h_v)], dim=-1)
        weights = F.softmax(scorer.mlp(features), dim=-1)
        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]

    labels = [f"Pair {i+1}" for i in range(h_u.size(0))]
    x = torch.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8,8))
    plt.bar(x - width, alpha, width, label ='Cosine (α)')
    plt.bar(x, width, beta, width, label='Euclidean (β)')
    plt.bar(x + width, gamma, width, label='Pearson (γ)')

    plt.xlabel("embedding Pairs")
    plt.ylabel("Learned Weight")
    plt.title("Learned Similarity Weights per Embedding Pair")
    plt.xticks(x , labels)
    plt.legend()
    plt.tight_layout()
    plt.show()












