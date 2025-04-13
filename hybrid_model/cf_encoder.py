import torch
import torch.nn as nn



class CFEncoder(nn.Module):
    def __init__(self, num_users, num_movies, mlp_hidden_sizes = [128, 64, 1], embedding_dim=64, dropout=0.2):
        super(CFEncoder, self).__init__()


        # Add user and movie bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)


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
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Apply Xavier Initialization
        self._init_weights()


    def _init_weights(self):

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        # Initialize each linear layer
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)



    def forward (self, user_idx, movie_idx):
        # Pass indices into embeddings
        user_embed = self.user_embedding(user_idx) # shape: [batch_size, embedding_dim]
        movie_embed = self.movie_embedding(movie_idx) # shape: [batch_size, embedding_dim]
        bias = self.user_bias(user_idx) + self.movie_bias(movie_idx) # Shape: [batch_size, 1]


        # Combine embeddings as input to MLP
        x = torch.cat([user_embed, movie_embed], dim=1) # shape: [batch_size, 2 * embedding_dim]

        # Pass through MLP to get score
        output = self.mlp(x) + bias # shape: [batch_size, 1]
        output = torch.sigmoid(output)


        return output.squeeze(), user_embed, movie_embed # removes extra dimension -> shape: [batch-size]








