import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import pickle
import numpy as np
from collections import defaultdict, deque
from typing import List, Optional, Dict
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 모델 클래스 임포트
from model import TwoTowerModel

# --- 전역 변수 및 시뮬레이션용 데이터 저장소 ---
# 서버가 시작될 때 단 한 번만 실행되어 모델과 데이터를 메모리에 올립니다.
# 이렇게 하면 매 요청마다 파일을 읽는 비효율을 막을 수 있습니다.

# [시뮬레이션] Redis를 대체할 인메모리 딕셔너리
# 사용자의 최근 행동(클릭한 상품명, 검색어) 10개를 저장합니다.
USER_RECENT_ACTIONS = defaultdict(lambda: deque(maxlen=10))

# [히스토리] 사용자의 행동 이력을 저장
USER_INTERACTION_HISTORY: Dict[str, List[Dict[str, str]]] = defaultdict(list)

# artifacts_folder 경로를 main.py 파일 기준으로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_folder = os.path.join(script_dir, "inference_artifacts")
device = torch.device("cpu") # 서빙 환경에서는 CPU를 사용하는 경우가 많습니다.

# --- FastAPI 애플리케이션 및 전역 변수 초기화 ---
app = FastAPI(title="Real-time Vector Update Recommendation API")

# CORS 미들웨어 추가
# 모든 출처에서의 요청을 허용합니다 (개발 환경용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션에서는 특정 도메인만 허용해야 합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 어플리케이션 시작 시 로드될 전역 변수들
SERVER_IS_READY = False
model = None
mappings = None
ALL_ITEM_VECTORS = None
SESSION_TO_IDX = None
ITEM_ID_TO_IDX = None
CATEGORY_TO_ITEMS = None
tfidf_vectorizer = None
tfidf_matrix = None
all_titles = None
num_items = 0

@app.on_event("startup")
def load_artifacts_on_startup():
    global SERVER_IS_READY, model, mappings, ALL_ITEM_VECTORS, SESSION_TO_IDX, ITEM_ID_TO_IDX
    global CATEGORY_TO_ITEMS, tfidf_vectorizer, tfidf_matrix, all_titles, num_items

    print("Loading artifacts...")
    try:
        # 매핑 정보 로드
        with open(os.path.join(artifacts_folder, "mappings.pkl"), 'rb') as f:
            mappings = pickle.load(f)
        
        item_text_embeddings = torch.load(os.path.join(artifacts_folder, "item_text_embeddings.pt"), map_location=device, weights_only=True)

        num_users = len(mappings['user_categories'])
        num_items = len(mappings['item_categories'])
        final_embedding_dim = mappings['final_embedding_dim']
        text_embedding_dim = mappings['text_embedding_dim']

        model = TwoTowerModel(num_users, item_text_embeddings, text_embedding_dim, final_embedding_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(artifacts_folder, "two_tower_model.pth"), map_location=device, weights_only=True))
        model.eval()

        print("Pre-calculating all item vectors...")
        with torch.no_grad():
            all_item_prices_dict = mappings['item_prices']
            all_item_prices_tensor = torch.tensor([all_item_prices_dict.get(i, 0.0) for i in range(num_items)], dtype=torch.float).to(device)
            all_item_ids_tensor = torch.LongTensor(range(num_items)).to(device)
            ALL_ITEM_VECTORS = model.get_item_vector(all_item_ids_tensor, all_item_prices_tensor)
        
        SESSION_TO_IDX = {session: i for i, session in enumerate(mappings['user_categories'])}
        # item_id를 item_idx로 변환하기 위한 딕셔너리 추가
        ITEM_ID_TO_IDX = {item_id: i for i, item_id in enumerate(mappings['item_categories'])}

        print("Building category-to-item mapping...")
        CATEGORY_TO_ITEMS = defaultdict(list)
        item_idx_to_c1 = mappings.get('item_idx_to_c1', {})
        for item_idx, cat in item_idx_to_c1.items():
            if cat and cat != 'Unknown':
                CATEGORY_TO_ITEMS[cat].append(item_idx)

        # 사용 가능한 세션 ID 예시 출력
        print("--- Available Session IDs (sample) ---")
        sample_sessions = list(SESSION_TO_IDX.keys())[:5]
        for s_id in sample_sessions:
            print(s_id)
        print("------------------------------------")
        
        # --- 내부 아이템 검색 엔진 초기화 (TF-IDF) ---
        print("Initializing internal item search engine...")
        all_titles = [mappings['item_titles'].get(i, "") for i in range(num_items)]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_titles)
        print("Internal search engine ready.")

        print("Artifacts loaded successfully. Server is ready.")
        SERVER_IS_READY = True
    except FileNotFoundError:
        print("CRITICAL: Artifacts not found. Please run train.py and copy artifacts.")
        SERVER_IS_READY = False

# --- 유틸리티 함수 ---
def find_internal_item_vector_by_title(title: str) -> Optional[torch.Tensor]:
    """외부 상품 제목과 가장 유사한 내부 아이템을 찾아 그 벡터를 반환합니다."""
    if not SERVER_IS_READY: return None
    try:
        title_tfidf = tfidf_vectorizer.transform([title])
        similarities = cosine_similarity(title_tfidf, tfidf_matrix).flatten()
        most_similar_item_idx = similarities.argmax()
        
        # 유사도가 매우 낮으면 (e.g., 0.1 미만) 관련 없는 아이템으로 간주하고 무시
        if similarities[most_similar_item_idx] < 0.1:
            print(f"No relevant internal item found for title: '{title}'")
            return None
            
        print(f"Found internal item '{all_titles[most_similar_item_idx]}' for external title '{title}'")
        return ALL_ITEM_VECTORS[most_similar_item_idx]
    except Exception as e:
        print(f"Error finding internal item by title: {e}")
        return None

def get_base_user_vector(session_id: str) -> Optional[torch.Tensor]:
    """모델에서 사용자의 기본(장기) 벡터를 가져옵니다."""
    user_idx = SESSION_TO_IDX.get(session_id)
    if user_idx is None: return None
    with torch.no_grad():
        return model.get_user_vector(torch.LongTensor([user_idx]).to(device))

def get_user_vector(session_id: str) -> Optional[torch.Tensor]:
    """
    세션의 단기 사용자 벡터를 계산합니다.
    최근 행동(최대 10개)의 평균 벡터를 반환합니다.
    최근 행동이 없으면 None을 반환합니다.
    """
    recent_actions = USER_RECENT_ACTIONS.get(session_id)
    if not recent_actions:
        return None # 최근 행동이 없음

    # 최근 행동들의 아이템 벡터를 수집
    recent_item_vectors = []
    print(f"Calculating short-term vector from actions: {list(recent_actions)}")
    for action_title in recent_actions:
        item_vector = find_internal_item_vector_by_title(action_title)
        if item_vector is not None:
            recent_item_vectors.append(item_vector)

    if not recent_item_vectors:
        return None # 유효한 아이템 벡터가 없음

    # 아이템 벡터들의 평균을 계산하여 순수 단기 벡터 생성
    short_term_vector = torch.mean(torch.stack(recent_item_vectors), dim=0)
    return short_term_vector

# API 요청/응답을 위한 데이터 모델 정의
class Event(BaseModel):
    session_id: str
    event_type: str
    value: str

class HistoryEvent(BaseModel):
    event_type: str
    value: str

class Product(BaseModel):
    title: str
    link: str
    price: Optional[float] = None
    thumbnail: Optional[str] = None
    source: Optional[str] = None

class RecommendationResponse(BaseModel):
    session_id: str
    recommendations: List[Product]

class HistoryResponse(BaseModel):
    session_id: str
    history: List[HistoryEvent]

@app.get("/")
def read_root():
    return {"message": "API is running."}

@app.post("/events", status_code=202)
def track_event(event: Event):
    """사용자의 행동 (클릭, 검색) 이벤트를 기록합니다."""
    if not SERVER_IS_READY:
        raise HTTPException(status_code=503, detail="Server is not ready.")
    
    # 1. 모든 이벤트를 전체 히스토리에 기록
    USER_INTERACTION_HISTORY[event.session_id].append(
        {"event_type": event.event_type, "value": event.value}
    )

    # 2. 클릭/검색 이벤트를 단기 행동 기록에 추가
    if event.event_type == 'click' or event.event_type == 'search':
        # 벡터를 바로 업데이트하는 대신, 행동 자체를 기록
        USER_RECENT_ACTIONS[event.session_id].append(event.value)
        print(f"Action '{event.value}' added to recent history for session: {event.session_id}")
        
    return {"message": "Event tracked"}

@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    """사용자의 행동 이력을 반환합니다."""
    if not SERVER_IS_READY: raise HTTPException(status_code=503, detail="Server not ready")
    
    history = USER_INTERACTION_HISTORY.get(session_id, [])
    return HistoryResponse(session_id=session_id, history=history)

@app.get("/search", response_model=RecommendationResponse)
def search_products(q: str, session_id: Optional[str] = "anonymous"):
    if not SERVER_IS_READY: raise HTTPException(status_code=503, detail="Server not ready")
    track_event(Event(session_id=session_id, event_type="search", value=q))
    
    # 내부 아이템 검색 로직으로 변경
    try:
        query_tfidf = tfidf_vectorizer.transform([q])
        similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        # 상위 30개 결과의 인덱스 추출
        top_indices = similarities.argsort()[-30:][::-1]

        recommendations = []
        for idx in top_indices:
            # 유사도가 임계값 이하이면 결과에 포함하지 않음
            if similarities[idx] < 0.1:
                continue
            
            title = mappings['item_titles'].get(idx)
            if title:
                recommendations.append(Product(
                    title=title,
                    link="#", # 내부 아이템은 링크가 없음
                    price=mappings['item_prices'].get(idx),
                    thumbnail=None, # 내부 아이템은 썸네일이 없음
                    source="Internal DB"
                ))
        
        return RecommendationResponse(session_id=session_id, recommendations=recommendations)
    except Exception as e:
        print(f"Internal search failed for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Internal search failed")

@app.get("/recommend/{session_id}", response_model=RecommendationResponse)
def get_recommendations(session_id: str, long_term_ratio: float = 0.7, num_recs: int = 30):
    if not SERVER_IS_READY: raise HTTPException(status_code=503, detail="Server not ready")

    long_term_vector = get_base_user_vector(session_id)
    if long_term_vector is None:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")

    short_term_vector = get_user_vector(session_id) # 최신 단기 벡터

    with torch.no_grad():
        # 장기 벡터는 항상 존재
        long_term_scores = torch.matmul(ALL_ITEM_VECTORS, long_term_vector.T).squeeze()

        if short_term_vector is not None:
            # 단기 벡터가 있으면 단기 점수도 계산
            short_term_scores = torch.matmul(ALL_ITEM_VECTORS, short_term_vector.T).squeeze()
            num_long_term_recs = int(num_recs * long_term_ratio)
            num_short_term_recs = num_recs - num_long_term_recs
        else:
            # 단기 벡터가 없으면 장기 추천만 제공
            print(f"No short-term vector for {session_id}, returning long-term recs only.")
            short_term_scores = None
            num_long_term_recs = num_recs
            num_short_term_recs = 0

        long_term_indices = torch.topk(long_term_scores, k=num_long_term_recs * 2).indices.tolist()
        
        if short_term_scores is not None:
            short_term_indices = torch.topk(short_term_scores, k=num_short_term_recs * 2).indices.tolist()
        else:
            short_term_indices = []

    # 2. 두 추천 목록을 조합
    combined_indices = []
    seen_indices = set()

    # 단기 추천(최신 관심사)을 먼저 일부 추가
    for idx in short_term_indices:
        if idx not in seen_indices:
            combined_indices.append(idx)
            seen_indices.add(idx)
        if len(seen_indices) >= num_short_term_recs:
            break
            
    # 장기 추천으로 나머지 채우기
    for idx in long_term_indices:
        if idx not in seen_indices:
            combined_indices.append(idx)
            seen_indices.add(idx)
        if len(combined_indices) >= num_recs:
            break

    # 3. 조합된 인덱스로부터 최종 추천 목록 생성 (내부 데이터 사용)
    final_recommendations = []
    for idx in combined_indices:
        title = mappings['item_titles'].get(idx)
        if title:
            final_recommendations.append(Product(
                title=title,
                link="#", # 내부 아이템은 링크가 없음
                price=mappings['item_prices'].get(idx),
                thumbnail=None,
                source="Internal DB"
            ))

    internal_titles = [rec.title for rec in final_recommendations]
    print(f"Blended internal recommendations for {session_id}: {internal_titles}")

    return RecommendationResponse(session_id=session_id, recommendations=final_recommendations)

if __name__ == "__main__":
    import uvicorn
    # Uvicorn을 프로그래매틱하게 실행
    uvicorn.run(app, host="127.0.0.1", port=8000) 