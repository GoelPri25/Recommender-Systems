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
        # nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        # nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        # nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        # nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        nn.init.kaiming_normal_(self.user_embedding_gmf.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_embedding_gmf.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.user_embedding_mlp.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_embedding_mlp.weight, nonlinearity='relu')

        
        
        # MLP Layers
        self.mlp_layers = nn.Sequential()
        input_dim = layers[0]  # Initial input size (concatenated user & item embeddings)
        # for i in range(1, len(layers)):
        #     self.mlp_layers.add_module(f"fc{i}", nn.Linear(input_dim, layers[i]))
        #     self.mlp_layers.add_module(f"relu{i}", nn.ReLU())
        #     input_dim = layers[i]
        
        for i in range(1, len(layers)):
            self.mlp_layers.add_module(f"fc{i}", nn.Linear(input_dim, layers[i]))
            # batch norm
            # self.mlp_layers.add_module(f"batchnorm{i}", nn.BatchNorm1d(layers[i])) 
            self.mlp_layers.add_module(f"relu{i}", nn.ReLU())
            # self.mlp_layers.add_module(f"dropout{i}", nn.Dropout(p=0.1))
            input_dim = layers[i]


        # Output layer: combines GMF and MLP outputs
        self.fc_output = nn.Linear(mf_dim + layers[-1], 1)  



    def forward(self, user_indices, item_indices):

        user_latent_gmf = self.user_embedding_gmf(user_indices)
        item_latent_gmf = self.item_embedding_gmf(item_indices)
        gmf_out = torch.mul(user_latent_gmf, item_latent_gmf) 
        
        #self.batch_norm =  nn.BatchNorm1d(mf_dim)
        #gmf_out = self.batch_norm(gmf_out)
        
        user_latent_mlp = self.user_embedding_mlp(user_indices)
        item_latent_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat((user_latent_mlp, item_latent_mlp), dim=-1) 
        mlp_out = self.mlp_layers(mlp_input)

        combined = torch.cat((gmf_out, mlp_out), dim=-1)
        # prediction = torch.sigmoid(self.fc_output(combined)) 
        prediction = self.fc_output(combined)
        return prediction