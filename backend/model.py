import torch
import torch.nn as nn

# train.py에 있던 TwoTowerModel 클래스를 그대로 정의합니다.
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, precomputed_item_embeddings, text_embedding_dim, final_embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        self.final_embedding_dim = final_embedding_dim

        # User Tower
        self.user_embedding = nn.Embedding(num_users, final_embedding_dim)

        # Item Tower
        self.item_text_embedding = nn.Embedding.from_pretrained(precomputed_item_embeddings, freeze=True)
        self.item_mlp = nn.Sequential(
            nn.Linear(text_embedding_dim + 1, (text_embedding_dim + 1) // 2),
            nn.ReLU(),
            nn.Linear((text_embedding_dim + 1) // 2, final_embedding_dim)
        )

    def get_user_vector(self, user_ids):
        return self.user_embedding(user_ids)

    def get_item_vector(self, item_ids, item_prices):
        text_vecs = self.item_text_embedding(item_ids)
        combined_vec = torch.cat([text_vecs, item_prices.unsqueeze(1)], dim=1)
        return self.item_mlp(combined_vec) 