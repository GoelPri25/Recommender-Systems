import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        nn.init.kaiming_normal_(self.user_embedding.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_embedding.weight, nonlinearity='relu')

        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, user_indices, item_indices):
        device = next(self.parameters()).device

        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        user_latent = self.user_embedding(user_indices).to(self.user_embedding.weight.device)
        item_latent = self.item_embedding(item_indices).to(self.user_embedding.weight.device)
        
        # 计算用户和物品的交互
        prediction = user_latent * item_latent  # 形状为 (batch_size, latent_dim)
        
        # 直接传给 fc 层
        prediction = self.fc(prediction)  # fc 需要形状为 (batch_size, latent_dim)
        
        return prediction