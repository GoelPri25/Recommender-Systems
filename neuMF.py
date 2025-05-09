import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10]):
        super(NeuMF, self).__init__()

        # GMF Embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_dim)

        # MLP Embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, layers[0] // 2)

        # Initialize embedding weights
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        # MLP Layers
        self.mlp_layers = nn.Sequential()
        input_dim = layers[0]  # Initial input size (concatenated user & item embeddings)
        for i in range(1, len(layers)):
            self.mlp_layers.add_module(f"fc{i}", nn.Linear(input_dim, layers[i]))
            self.mlp_layers.add_module(f"relu{i}", nn.ReLU())
            input_dim = layers[i]

        # Output layer: combines GMF and MLP outputs
        self.fc_output = nn.Linear(mf_dim + layers[-1], 1)  # GMF (mf_dim) + MLP (last layer size)



    def forward(self, user_indices, item_indices):
        """ Forward pass for NeuMF model """

        # GMF Forward Pass: Element-wise multiplication
        user_latent_gmf = self.user_embedding_gmf(user_indices)
        item_latent_gmf = self.item_embedding_gmf(item_indices)
        gmf_out = torch.mul(user_latent_gmf, item_latent_gmf)  # Element-wise multiplication

        # MLP Forward Pass: Concatenate embeddings and pass through MLP layers
        user_latent_mlp = self.user_embedding_mlp(user_indices)
        item_latent_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat((user_latent_mlp, item_latent_mlp), dim=-1)  # Concatenation
        mlp_out = self.mlp_layers(mlp_input)

        # Combine GMF and MLP outputs
        combined = torch.cat((gmf_out, mlp_out), dim=-1)
        prediction = torch.sigmoid(self.fc_output(combined))  # Final prediction

        return prediction