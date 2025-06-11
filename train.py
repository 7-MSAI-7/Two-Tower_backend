import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import warnings
import os
import pickle
import multiprocessing
import glob

warnings.filterwarnings('ignore')

# 3-1. 모델 아키텍처 (가격 피처 결합)
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, precomputed_item_embeddings, text_embedding_dim, final_embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        self.final_embedding_dim = final_embedding_dim

        # User Tower
        self.user_embedding = nn.Embedding(num_users, final_embedding_dim)

        # Item Tower
        self.item_text_embedding = nn.Embedding.from_pretrained(precomputed_item_embeddings, freeze=True)
        # 텍스트 임베딩과 가격 피처(1차원)를 합친 후 최종 임베딩 차원으로 매핑하는 MLP
        self.item_mlp = nn.Sequential(
            nn.Linear(text_embedding_dim + 1, (text_embedding_dim + 1) // 2),
            nn.ReLU(),
            nn.Linear((text_embedding_dim + 1) // 2, final_embedding_dim)
        )

    def get_user_vector(self, user_ids):
        return self.user_embedding(user_ids)

    def get_item_vector(self, item_ids, item_prices):
        text_vecs = self.item_text_embedding(item_ids)
        # 가격 피처를 [batch_size, 1] 형태로 만들어 텍스트 임베딩과 결합
        combined_vec = torch.cat([text_vecs, item_prices.unsqueeze(1)], dim=1)
        return self.item_mlp(combined_vec)

# 3-2. 네거티브 샘플링을 포함한 데이터셋 클래스
class TripletDataset(Dataset):
    def __init__(self, interactions_df, item_prices_dict, all_item_indices):
        self.users = torch.LongTensor(interactions_df['user_idx'].values)
        self.pos_items = torch.LongTensor(interactions_df['item_idx'].values)
        self.item_prices = item_prices_dict
        self.all_item_indices = all_item_indices
        self.num_items = len(all_item_indices)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        pos_item_id = self.pos_items[idx]

        # 네거티브 샘플링
        while True:
            neg_item_id = np.random.randint(0, self.num_items)
            # 여기서는 단순 랜덤 샘플링을 사용 (개선 가능)
            if neg_item_id != pos_item_id.item():
                break
        
        pos_item_price = torch.tensor(self.item_prices.get(pos_item_id.item(), 0.0), dtype=torch.float)
        neg_item_price = torch.tensor(self.item_prices.get(neg_item_id, 0.0), dtype=torch.float)
        
        return user_id, pos_item_id, neg_item_id, pos_item_price, neg_item_price

def main():
    print("--- Python script started ---")
    # --- 0. 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_folder_path = "Data"
    artifacts_folder = "inference_artifacts"
    os.makedirs(artifacts_folder, exist_ok=True)

    # --- 1. 데이터 로드 및 전처리 ---
    print("\n--- 1. Loading and Preprocessing Data ---")
    
    parquet_files = [f"{data_folder_path}/{i:012d}.parquet" for i in range(4)]

    print(f"Attempting to load {len(parquet_files)} parquet files...")
    try:
        # 필요한 컬럼만 읽어와 메모리 사용량 최적화
        columns_to_read = ['session_id', 'item_id', 'name', 'price', 'c1_name', 'c2_name', 'brand_name', 'item_condition_name']
        dataset = load_dataset('parquet', data_files={'train': parquet_files}, split='train', columns=columns_to_read)
        df = dataset.to_pandas()
        print(f"Successfully loaded {len(df)} rows from {len(parquet_files)} Parquet files.")
        
        # Null 값 처리
        for col in ['c1_name', 'c2_name', 'brand_name', 'item_condition_name']:
            df[col] = df[col].fillna('Unknown')
        
        # 사용자/아이템 인덱스 생성
        df['user_idx'] = df['session_id'].astype('category').cat.codes
        df['item_idx'] = df['item_id'].astype('category').cat.codes
        
        num_users = len(df['user_idx'].unique())
        num_items = len(df['item_idx'].unique())
        print(f"Unique Users: {num_users}, Unique Items: {num_items}")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # --- 2. 피처 엔지니어링 ---
    print("\n--- 2. Feature Engineering ---")

    # 2-1. 가격(Price) 피처 처리
    print("Processing price feature...")
    # 가격이 0 이하인 경우 이상치로 간주하고 1로 처리 후 로그 변환
    df['price'] = df['price'].apply(lambda x: max(x, 1.0))
    df['log_price'] = np.log1p(df['price'])

    price_scaler = StandardScaler()
    df['scaled_price'] = price_scaler.fit_transform(df[['log_price']])

    # 아이템별 가격 정보 저장
    item_prices = df.drop_duplicates('item_idx').set_index('item_idx')['scaled_price'].to_dict()

    # 2-2. 텍스트(Text) 임베딩 생성 (기능 강화)
    print("Creating enhanced item embeddings...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 아이템별 메타데이터 딕셔너리 생성
    item_meta = df.drop_duplicates('item_idx').set_index('item_idx')
    item_titles = item_meta['name'].to_dict()
    item_idx_to_c1 = item_meta['c1_name'].to_dict()
    item_idx_to_c2 = item_meta['c2_name'].to_dict()
    item_idx_to_brand = item_meta['brand_name'].to_dict()
    item_idx_to_condition = item_meta['item_condition_name'].to_dict()

    # 모든 피처를 결합한 상세 설명 생성
    item_descriptions = []
    for i in range(num_items):
        name = item_titles.get(i, 'N/A')
        c1 = item_idx_to_c1.get(i, 'N/A')
        c2 = item_idx_to_c2.get(i, 'N/A')
        brand = item_idx_to_brand.get(i, 'N/A')
        condition = item_idx_to_condition.get(i, 'N/A')
        description = (f"Name: {name}. Category: {c1}, {c2}. "
                       f"Brand: {brand}. Condition: {condition}.")
        item_descriptions.append(description)

    # 텍스트 임베딩 생성
    item_text_embeddings = text_model.encode(item_descriptions, show_progress_bar=True, convert_to_tensor=True, device=device)
    text_embedding_dim = item_text_embeddings.shape[1]
    print("Item embeddings generated.")

    # --- 3. 모델 및 데이터셋 정의 ---
    print("\n--- 3. Defining Model and Dataset ---")

    # --- 4. 학습 및 검증 ---
    print("\n--- 4. Training and Validation ---")

    # 4-1. 데이터 분할 (Train, Validation, Test)
    df_interactions = df[['user_idx', 'item_idx']].drop_duplicates().reset_index(drop=True)
    train_val_df, test_df = train_test_split(df_interactions, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    all_items = np.arange(num_items)
    train_dataset = TripletDataset(train_df, item_prices, all_items)
    val_dataset = TripletDataset(val_df, item_prices, all_items)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # 4-2. 모델 학습 루프 (검증 및 조기 종료 포함)
    final_embedding_dim = 128
    model = TwoTowerModel(num_users, item_text_embeddings, text_embedding_dim, final_embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # TripletLoss: anchor와 positive는 가깝게, anchor와 negative는 멀게 만듦
    criterion = nn.TripletMarginLoss(margin=1.0)

    num_epochs = 20
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            user_ids, pos_item_ids, neg_item_ids, pos_prices, neg_prices = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            user_vec = model.get_user_vector(user_ids)
            pos_item_vec = model.get_item_vector(pos_item_ids, pos_prices)
            neg_item_vec = model.get_item_vector(neg_item_ids, neg_prices)
            
            loss = criterion(user_vec, pos_item_vec, neg_item_vec)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # 검증
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                user_ids, pos_item_ids, neg_item_ids, pos_prices, neg_prices = [b.to(device) for b in batch]
                user_vec = model.get_user_vector(user_ids)
                pos_item_vec = model.get_item_vector(pos_item_ids, pos_prices)
                neg_item_vec = model.get_item_vector(neg_item_ids, neg_prices)
                loss = criterion(user_vec, pos_item_vec, neg_item_vec)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 조기 종료 및 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(artifacts_folder, "best_model.pth"))
            print(f"Validation loss improved. Saved best model to '{artifacts_folder}/best_model.pth'")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Model training complete.")

    # --- 5. 아티팩트 저장 ---
    print("\n--- 5. Saving Artifacts ---")
    # 1. 가장 성능이 좋았던 모델 가중치 이름 변경
    if os.path.exists(os.path.join(artifacts_folder, "best_model.pth")):
        os.rename(os.path.join(artifacts_folder, "best_model.pth"), os.path.join(artifacts_folder, "two_tower_model.pth"))
        print("Saved best model as 'two_tower_model.pth'")

    # 2. 아이템 텍스트 임베딩 저장
    torch.save(item_text_embeddings, os.path.join(artifacts_folder, "item_text_embeddings.pt"))

    # 3. 가격 스케일러 저장
    with open(os.path.join(artifacts_folder, "price_scaler.pkl"), 'wb') as f:
        pickle.dump(price_scaler, f)
        
    # 4. 매핑 및 데이터 저장
    mappings = {
        'user_categories': df['session_id'].astype('category').cat.categories,
        'item_categories': df['item_id'].astype('category').cat.categories,
        'idx_to_item_original': {idx: original_id for idx, original_id in enumerate(df['item_id'].astype('category').cat.categories)},
        'item_titles': item_titles,
        'item_idx_to_c1': item_idx_to_c1,
        'item_idx_to_c2': item_idx_to_c2,
        'item_idx_to_brand': item_idx_to_brand,
        'item_idx_to_condition': item_idx_to_condition,
        'item_prices': item_prices,
        'final_embedding_dim': final_embedding_dim,
        'text_embedding_dim': text_embedding_dim
    }
    with open(os.path.join(artifacts_folder, "mappings.pkl"), 'wb') as f:
        pickle.dump(mappings, f)

    # 5. 테스트 데이터 저장 (최종 성능 평가용)
    test_df.to_pickle(os.path.join(artifacts_folder, "test_df.pkl"))

    print(f"Artifacts saved to '{artifacts_folder}' folder.")
    print("\n--- Training Script Finished ---")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()