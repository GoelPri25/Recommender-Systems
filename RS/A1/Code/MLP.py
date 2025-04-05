import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        nn.init.kaiming_normal_(self.user_embedding.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_embedding.weight, nonlinearity='relu')

        self.mlp = nn.Sequential()
        input_dim = layers[0]
        for i in range(1, len(layers)):
            self.mlp.add_module(f'fc{i}', nn.Linear(input_dim, layers[i]))
            self.mlp.add_module(f'relu{i}', nn.ReLU())
            input_dim = layers[i]
        
        self.fc_output = nn.Linear(input_dim, 1)

    def forward(self, user_indices, item_indices):
        user_latent = self.user_embedding(user_indices).to(self.user_embedding.weight.device)
        item_latent = self.item_embedding(item_indices).to(self.user_embedding.weight.device)
        concatenated = torch.cat((user_latent, item_latent), dim=1)
        mlp_out = self.mlp(concatenated)
        prediction = self.fc_output(mlp_out)
        # prediction = torch.sigmoid(self.fc_output(mlp_out))
        return prediction