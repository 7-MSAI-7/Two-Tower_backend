import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# --- 0. 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
artifacts_folder = "inference_artifacts"
print(f"Using device: {device}")

# --- 1. 모델 아키텍처 정의 (train.py와 동일해야 함) ---
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

def predict():
    # --- 2. 아티팩트 로드 ---
    print("\n--- 2. Loading Artifacts ---")
    if not os.path.exists(artifacts_folder):
        print(f"Error: Artifacts folder '{artifacts_folder}' not found. Please run the updated train.py first.")
        return

    # 매핑 정보 로드
    with open(os.path.join(artifacts_folder, "mappings.pkl"), 'rb') as f:
        mappings = pickle.load(f)
    
    # 아이템 텍스트 임베딩 로드
    item_text_embeddings = torch.load(os.path.join(artifacts_folder, "item_text_embeddings.pt"), map_location=device)
    
    # 가격 스케일러 로드 (predict에서는 사용되지 않지만, 완전성을 위해 로드)
    with open(os.path.join(artifacts_folder, "price_scaler.pkl"), 'rb') as f:
        price_scaler = pickle.load(f)

    # 테스트 데이터 및 전체 상호작용 데이터 로드
    test_df = pd.read_pickle(os.path.join(artifacts_folder, "test_df.pkl"))
    # 과거 이력 조회를 위해 전체 interaction 데이터가 필요하면 train.py에서 저장해야 함.
    # 여기서는 test_df를 사용해 랜덤 유저를 뽑는 용도로만 사용.
    
    num_users = len(mappings['user_categories'])
    num_items = len(mappings['item_categories'])
    final_embedding_dim = mappings['final_embedding_dim']
    text_embedding_dim = mappings['text_embedding_dim']

    # 모델 가중치 로드
    model = TwoTowerModel(num_users, item_text_embeddings, text_embedding_dim, final_embedding_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(artifacts_folder, "two_tower_model.pth"), map_location=device))
    model.eval()
    print("Artifacts loaded successfully.")

    # --- 3. 랜덤 사용자 3명에 대한 추천 생성 ---
    print("\n--- 3. Generating Recommendations for 3 Random Users ---")
    if test_df.empty:
        print("Test set is empty, cannot generate recommendations.")
        return
        
    random_user_indices = np.random.choice(test_df['user_idx'].unique(), min(3, len(test_df['user_idx'].unique())), replace=False)
    
    # 모든 아이템의 가격 정보 준비
    all_item_prices_dict = mappings['item_prices']
    all_item_prices_tensor = torch.tensor([all_item_prices_dict.get(i, 0.0) for i in range(num_items)], dtype=torch.float).to(device)
    all_item_ids_tensor = torch.LongTensor(range(num_items)).to(device)

    with torch.no_grad():
        # 모든 아이템 벡터 미리 계산
        print("Pre-calculating all item vectors...")
        all_item_vectors = model.get_item_vector(all_item_ids_tensor, all_item_prices_tensor)

        for user_idx in random_user_indices:
            original_user_id = mappings['user_categories'][user_idx]
            print(f"\n" + "="*80)
            print(f"User ID: {original_user_id} (Internal Index: {user_idx})")
            
            # (참고) 사용자의 과거 행동 이력 확인 로직은 생략되었습니다.
            # 필요 시, train.py에서 전체 interaction_df를 저장하고 여기서 로드하여 구현할 수 있습니다.

            # 2. 모든 아이템에 대한 점수 계산
            user_tensor = torch.LongTensor([user_idx]).to(device)
            user_vector = model.get_user_vector(user_tensor)
            
            # 사용자 벡터와 모든 아이템 벡터 간의 내적(dot product)으로 점수 계산
            scores = torch.matmul(all_item_vectors, user_vector.T).squeeze().cpu().numpy()
            
            top_k_indices = np.argsort(scores)[-15:][::-1] # 상위 15개 후보 추출
            
            recommendations = []
            for item_idx in top_k_indices:
                if len(recommendations) >= 10: break
                # 여기서는 과거 상호작용 제외 로직을 생략 (필요 시 추가)
                recommendations.append({'item_idx': item_idx, 'score': scores[item_idx]})

            # 4. 결과 출력
            print("\n--- Top 10 Recommended Items ---")
            for i, item in enumerate(recommendations):
                item_idx = item['item_idx']
                item_info = (f"Rank {i+1}: {mappings['item_titles'].get(item_idx, 'N/A')} (Score: {item['score']:.4f})\n"
                             f"     (Category: {mappings['item_idx_to_c1'].get(item_idx, 'N/A')}, "
                             f"{mappings['item_idx_to_c2'].get(item_idx, 'N/A')})\n"
                             f"     (Brand: {mappings['item_idx_to_brand'].get(item_idx, 'N/A')}, "
                             f"Condition: {mappings['item_idx_to_condition'].get(item_idx, 'N/A')})")
                print(item_info)

if __name__ == '__main__':
    predict() 