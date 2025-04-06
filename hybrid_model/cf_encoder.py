import torch
import torch.nn as nn



class CFEncoder(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, dropout=0.2):
        super(CFEncoder, self).__init__()
        mlp_hidden_sizes = [128, 64, 1]

        # Apply learnable embeddings for users and movies
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # MLP to combine and process embeddings
        layers = []
        input_dim = 2 * embedding_dim # because we concatenate user and movie vectors

        for hidden_dim in mlp_hidden_sizes:
            layers. append(nn.Linear(input_dim, hidden_dim)) # fully connected layer
            layers.append(nn.ReLU()) # activation function
            layers.append(nn.Dropout(dropout)) # regularization
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward (self, user_idx, movie_idx):
        # Pass indices into embeddings
        user_embed = self.user_embedding(user_idx) # shape: [batch_size, embedding_dim]
        movie_embed = self.movie_embedding(movie_idx) # shape: [batch_size, embedding_dim]

        # Combine embeddings as input to MLP
        x = torch.cat([user_embed, movie_embed], dim=1) # shape: [batch_size, 2 * embedding_dim]

        # Pass through MLP to get score
        output = self.mlp(x) # shape: [batch_size, 1]

        return output.squeeze() # removes extra dimension -> shape: [batch-size]








